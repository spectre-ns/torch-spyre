# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import itertools
import math
import sympy
from torch_spyre._inductor.logging_utils import get_inductor_logger

from enum import Enum

if TYPE_CHECKING:
    from torch_spyre._inductor.pass_utils import PerCoreView
    from torch_spyre._inductor.scratchpad.coarse_tiling import TileReads
    from torch_spyre._inductor.work_division import (
        OpSplitSpace,
        ResidencyEdge,
    )
    from torch_spyre._inductor.scratchpad.lx_relayout import (
        ChosenRelayout,
    )

logger = get_inductor_logger("scratchpad.plan_solver")


class SolveError(Exception):
    """Raised when a solver is unable to find a solution"""


class BufferType(Enum):
    Intermediate = 0
    Input = 1
    Output = 2


def ceil_div(a: int, b: int) -> int:
    """Integer ceiling division. Used wherever a footprint is divided down by a
    core count, so every such site rounds identically (no float intermediate)."""
    return -(-a // b)


@dataclass
class LifetimeBoundBuffer:
    """
    Defines the data fields required for a plan solver.

    ``uses`` is the strictly increasing list of operation indices at which the
    buffer is accessed (as returned by ``calculate_liveness``).  It is normally
    non-empty, and callers that read ``start_time``/``end_time`` require that,
    since those properties index into it; the FirstFit scoring divides
    by ``len(uses)`` plus a write bonus, which is non-zero for a computed buffer
    even when ``uses`` is empty.  Emptiness is nevertheless allowed at
    construction -- see :meth:`__post_init__` for the registration state that
    needs it.  ``first_use_is_read`` is True for graph inputs (all accesses are
    reads) and False for computed buffers (first access is a write, all
    subsequent accesses are reads).

    Both properties of ``uses`` are asserted in ``__post_init__``.  Strictness
    is what makes ``read_count`` trustworthy: one entry per accessing op means
    that for a computed buffer ``read_count == 0`` is exactly "written, never
    read", which the in-place invariants rely on (see
    :func:`check_in_place_parent_is_read`).  A repeated index would describe a
    buffer written and read by the same op, i.e. with a single live tick, and
    would let such a buffer pass as an in-place parent.

    ``start_time`` and ``end_time`` are convenience properties derived from
    ``uses`` and widened by optional counted-loop lifetime overrides.
    """

    name: str
    size: int
    uses: list[int]
    first_use_is_read: bool = False
    address: Optional[int] = None
    in_place_parents: list[str] = field(default_factory=list)
    # define the reason for excluding the buffer based on allocator
    # or solver logic paths.
    residency_reason: Optional[str] = None
    # Optional lifetime bounds for storage reused by a counted loop. Keep these
    # separate from ``uses``: they change address overlap, but must not
    # manufacture reads or inflate residency/spill benefit.
    lifetime_start_override: Optional[int] = None
    lifetime_end_override: Optional[int] = None
    # The physical per-core ownership accepted by the residency judge. Placement
    # writes this beside the LX address; it must never derive another view.
    lx_view: Optional["PerCoreView"] = field(default=None, repr=False, compare=False)
    # Written by the solver: why it left the buffer out of LX, when it says.
    spill_reason: Optional[str] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        # Not also asserted non-empty: buffers are sometimes registered before
        # their uses are known and filled in afterwards (see
        # ``make_buffer_registry`` in tests/inductor/test_scratchpad_patterns.py),
        # and an empty list is vacuously strictly increasing. Callers that read
        # ``start_time``/``end_time`` still require a non-empty list.
        #
        # This runs at construction only, so it does not see later mutation of
        # ``uses``; the in-place invariants therefore test the property they need
        # directly rather than inferring it from ``read_count``.
        assert all(a < b for a, b in zip(self.uses, self.uses[1:])), (
            f"buffer {self.name} has uses={self.uses}, which is not strictly "
            "increasing; uses carries one distinct index per accessing operation"
        )
        if self.lifetime_start_override is not None and self.uses:
            assert self.lifetime_start_override <= self.uses[0], (
                f"buffer {self.name} has lifetime_start_override="
                f"{self.lifetime_start_override} after nominal start {self.uses[0]}"
            )
        if self.lifetime_end_override is not None and self.uses:
            assert self.lifetime_end_override >= self.uses[-1] + 1, (
                f"buffer {self.name} has lifetime_end_override="
                f"{self.lifetime_end_override} before nominal exclusive end "
                f"{self.uses[-1] + 1}"
            )

    @property
    def read_count(self) -> int:
        """Number of reads.  For a computed buffer the first use is the producing
        write, so every use but that one is a read; when ``first_use_is_read``
        (a graph input) every use is a read.  Exact because ``uses`` holds one
        distinct index per accessing op.

        This counts the buffer's reads, not the reads residency would save: an
        input's first read is the clone-in that pinning cannot avoid, so a cost
        model has to discount it separately (see
        :func:`buffer_graph.spill_cost`).  The ``max`` only guards
        the transient empty-``uses`` state described in :meth:`__post_init__`.
        """
        return max(0, len(self.uses) - (0 if self.first_use_is_read else 1))

    @property
    def start_time(self) -> int:
        nominal = self.uses[0]
        override = self.lifetime_start_override
        return min(nominal, override if override is not None else nominal)

    @property
    def end_time(self) -> int:
        nominal = self.uses[-1] + 1
        return max(nominal, self.lifetime_end_override or nominal)

    @property
    def min_footprint(self) -> int:
        """Smallest LX footprint the buffer can take, for the capacity check"""
        return self.size

    def overlaps_in_time(self, other: "LifetimeBoundBuffer") -> bool:
        """Returns true iff self and other overlap in time."""
        return self.start_time < other.end_time and other.start_time < self.end_time

    @property
    def sym_is_lx(self) -> sympy.Symbol:
        return sympy.Symbol(f"is_lx_{self.name}", integer=True, nonnegative=True)


@dataclass(frozen=True)
class TileAxis:
    """One coarse-tiling level.

    ``host_dim`` is a *positional* index: into ``op_out_coords(op)`` for an
    output axis, or into the op's ordered reduction loop variables (see
    :func:`wsr.coarse_tile.reduction_loop_vars`) for a reduction axis.
    ``is_reduction`` selects which frame ``host_dim`` indexes. ``count`` is the
    split factor -- how many equal tiles the axis is cut into.
    """

    host_dim: int
    count: int
    is_reduction: bool = False


@dataclass(frozen=True)
class TileSpec:
    """An ordered, outermost-first tuple of :class:`TileAxis` levels.

    Ordered -- where the core-division splits are dicts and so order-free --
    because tile levels *nest*: swapping two levels is a different plan. Frozen
    and hashable. The empty spec is *untiled*, and is the inert default every
    :class:`CoreDivision` carries unless a solver chose otherwise.
    """

    axes: tuple[TileAxis, ...] = ()

    @property
    def is_untiled(self) -> bool:
        return not self.axes

    @property
    def depth(self) -> int:
        """Number of nested tile levels."""
        return len(self.axes)

    @property
    def tile_count(self) -> int:
        """Total number of loop tiles across every level (all axes)."""
        return math.prod(a.count for a in self.axes)

    @property
    def level_counts(self) -> tuple[int, ...]:
        """Trip count of each level, outermost first: the loop nest alone,
        without the dims it tiles. ``host_dim`` is positional in each op's own
        output, so two specs can be equal and tile different dims of a buffer
        they share, or differ and tile the same one. Ops therefore share a loop
        group on equal nests (``derive_tiling_groups``), and whether a consumer
        reads its producer tile by tile is checked per edge."""
        return tuple(a.count for a in self.axes)

    @property
    def output_tile_count(self) -> int:
        """Product of the split factors over output (non-reduction) axes only.

        This is the factor by which a tiled op's own per-tile scratch shrinks,
        and so the factor :attr:`CoreDivisionBuffer.min_footprint` divides by. A
        reduction-tiled level does *not* shrink that buffer -- the op's own
        output is the accumulator, which keeps the full output extent -- so
        reduction axes are excluded here even though they count in
        :attr:`tile_count`.
        """
        return math.prod(a.count for a in self.axes if not a.is_reduction)

    def count_for(self, key: tuple[int, bool]) -> int:
        """Total split factor this spec puts on the axis ``key``, a
        ``(host_dim, is_reduction)`` pair; 1 for an axis it does not tile."""
        return math.prod(
            a.count for a in self.axes if (a.host_dim, a.is_reduction) == key
        )

    @property
    def is_clean(self) -> bool:
        """True when no reduction axis is tiled, so every tile produces a
        finished slice of the output rather than a partial accumulation.

        Landed with stage 1 and removed again by #4519 as unused; the
        solver-driven tiling path in ``TableCandidates._tiling_candidates``
        is the caller that makes it live, and it is the single filter that keeps
        reduction tilings (numerically fragile on e.g. softmax's max/sum) out of
        the candidate menu."""
        return not any(a.is_reduction for a in self.axes)

    @property
    def label(self) -> str:
        if not self.axes:
            return "untiled"
        return "/".join(
            f"{'~' if a.is_reduction else ''}d{a.host_dim}:{a.count}" for a in self.axes
        )


@dataclass
class CoreDivision:
    """One permissible core-division of a buffer's producing op.

    ``splits`` is keyed by the producer's iteration symbols -- one entry per
    axis with a split factor.
    ``reduction_syms`` names the subset of those keys that split a reduction
    axis rather than an output axis.
    ``tiling`` pairs a coarse tiling onto this division as one candidate. The
    empty :class:`TileSpec` is untiled and inert.
    ``tile_splits`` is that tiling in the keys ``splits`` uses: one
    ``(iteration symbol, trip count)`` per level, outermost first. A tiling is
    a split in time the way a core division is a split in space, so a view can
    carry both and be matched the same way. Empty when untiled.
    """

    splits: dict[sympy.Symbol, int] = field(default_factory=dict)
    reduction_syms: frozenset[sympy.Symbol] = field(default_factory=frozenset)
    tiling: TileSpec = field(default_factory=TileSpec)
    tile_splits: tuple[tuple[sympy.Symbol, int], ...] = ()

    @property
    def cores_used(self) -> int:
        return math.prod(self.splits.values())

    @property
    def output_splits(self) -> dict[sympy.Symbol, int]:
        return {s: v for s, v in self.splits.items() if s not in self.reduction_syms}

    @property
    def reduction_splits(self) -> dict[sympy.Symbol, int]:
        return {s: v for s, v in self.splits.items() if s in self.reduction_syms}

    @property
    def output_partition(self) -> int:
        """How many cores the output buffer is sliced across."""
        return math.prod(self.output_splits.values())

    @property
    def label(self) -> str:
        """Human-readable rendering of this division's splits, e.g.
        ``"s0/4 ~s1/2"`` (output split by 4 on symbol 0, reduction split by 2
        on symbol 1), or ``"whole"`` for the untouched, undivided candidate."""
        out = ",".join(
            f"s{s}/{f}"
            for s, f in sorted(self.output_splits.items(), key=lambda i: str(i[0]))
        )
        red = ",".join(
            f"~s{s}/{f}"
            for s, f in sorted(self.reduction_splits.items(), key=lambda i: str(i[0]))
        )
        return " ".join(p for p in (out, red) if p) or "whole"


@dataclass
class CoreDivisionBuffer(LifetimeBoundBuffer):
    """A :class:`LifetimeBoundBuffer` carrying the joint core-division metadata
    (the buffer form of an open node)."""

    core_divisions: list[CoreDivision] = field(default_factory=list)
    # Producer buffer names; defines the producer->consumer edges for matching.
    parents: list[str] = field(default_factory=list[str])
    # parent_buf_name -> (parent_div_idx, this_div_idx) pairs that induce the
    # *same per-core slicing of the parent*, precomputed by the allocator via
    # ``_per_core_view_on_buf`` (physical device-dim view equality, correct
    # across reductions/reshapes). These are the sole slicing-match predicate;
    # an absent/empty entry means no compatible division, so the gate forbids
    # the merge/residency across that edge.
    cd_parent_matches: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # The same relation per candidate rather than per pair: one edge per divided
    # producer this buffer reads, keyed as ``cd_parent_matches`` is. A solver
    # that generates divisions asks these instead of indexing the table, and
    # constructs the division on the other end of an edge by inverting the view.
    # Empty where the allocator has not built them (they need the live ops).
    residency_edges: dict[str, "ResidencyEdge"] = field(default_factory=dict)
    # This buffer's producing op's legal divisions as a space to move in --
    # ``core_divisions`` without materializing it. ``None`` where the allocator
    # built none (see ``allocator._DivisionMap``).
    division_space: Optional["OpSplitSpace"] = None
    # Index of this buffer's producing operation in ``graph.operations``, which
    # is what a coarse-tiling *run* is measured over: a group has to occupy one
    # contiguous stretch of that list, and buffer order is not operation order
    # (input clones are prepended, and an operation producing no solver buffer
    # has no index at all). ``None`` where there is no producing operation -- an
    # input clone -- or where the caller does not supply one, which is what a
    # solver reads as "operation order is unknown here, so no tiling may span
    # more than nothing".
    op_position: Optional[int] = None
    # Parent name -> whether this buffer's op reads that computed parent tile
    # by tile under a pair of specs; a pair it does not breaks a run there
    # (``derive_tiling_groups``). Filled only where the solver chooses tilings.
    tile_reads: dict[str, "TileReads"] = field(default_factory=dict)
    chosen_division: Optional[int] = None
    # Solver-chosen relayouts feeding this consumer: parent_buf_name -> the
    # fired candidate with the destination address (bytes) of the group's copy.
    # Written back by the solver when this
    # consumer reads the parent through a resident copy. Every consumer served
    # by one copy carries the same address; the commit path materializes one
    # plan per fired group (``FiredRelayoutGroup.from_chosen``).
    chosen_relayouts: dict[str, "ChosenRelayout"] = field(default_factory=dict)
    boundary: BufferType = BufferType.Intermediate

    def per_core_size(self, division: int) -> int:
        """Bytes per core after output partitioning and output tiling.

        Reduction splits retain a full output part.
        """
        cd = self.core_divisions[division]
        return ceil_div(self.size, cd.output_partition * cd.tiling.output_tile_count)

    @property
    def min_footprint(self) -> int:
        """Smallest :meth:`per_core_size` any candidate division allows. With no
        candidates there is nothing to divide by, so it falls back to ``size``
        (the placement-only case ``SpyreOptimizationGraph.add_node`` commits)."""
        if not self.core_divisions:
            return self.size
        return min(self.per_core_size(i) for i in range(len(self.core_divisions)))

    @property
    def sym_cores(self) -> sympy.Symbol:
        return math.prod(self.sym_core_divs.values())

    @property
    def sym_division(self) -> sympy.Symbol:
        """The chosen index into ``core_divisions`` as an objective unknown.

        The per-axis split symbols (:attr:`sym_core_divs`) carry the division's
        *shape* into the cost model; this carries its *identity*, for terms that
        are tables over candidates rather than functions of the splits. Engines
        bind it to their division variable (CP-SAT) or to the chosen index (the
        annealer's ``chosen``)."""
        return division_symbol(self.name)

    @property
    def sym_core_divs(self) -> dict[sympy.Symbol, sympy.Symbol]:
        """See :func:`sym_core_divs_for`."""
        return sym_core_divs_for(self.name, self.core_divisions)

    @property
    def sym_tile_counts(self) -> dict[tuple[int, bool], sympy.Symbol]:
        """See :func:`sym_tile_counts_for`."""
        return sym_tile_counts_for(self.name, self.core_divisions, self.division_space)


def sym_core_divs_for(
    name: str, divisions: Sequence[CoreDivision]
) -> dict[sympy.Symbol, sympy.Symbol]:
    """Symbolic stand-in for a chosen ``op_it_space_splits``: one symbol per
    stride coefficient seen across buffer ``name``'s candidate divisions, so
    the cost model can carry an undecided split as an unknown rather than a
    concrete value."""
    keys = dict.fromkeys(itertools.chain.from_iterable(cd.splits for cd in divisions))
    return {
        key: sympy.Symbol(f"split_{name}_{key}", integer=True, positive=True)
        for key in keys
    }


def sym_tile_counts_for(
    name: str, divisions: Sequence[CoreDivision], space: Optional["OpSplitSpace"]
) -> dict[tuple[int, bool], sympy.Symbol]:
    """Symbolic stand-in for a chosen ``tiling``, the coarse-tiling sibling
    of :func:`sym_core_divs_for`: one symbol per tiled axis (a ``(host_dim,
    is_reduction)`` pair) seen across the candidate divisions, bound to that
    candidate's count on the axis, 1 where it leaves the axis untiled. Keyed on
    the axis rather than the nesting level, so the symbols do not depend on how
    a candidate orders its levels -- the cost of a nest (a product of per-level
    trip counts) does not either. Empty while no candidate is tiled.

    The SA co-optimizer generates its tilings from the op's split ``space``
    instead of listing them as candidates, so every output dim that space may
    tile is declared too."""
    keys = dict.fromkeys(
        (axis.host_dim, axis.is_reduction)
        for cd in divisions
        for axis in cd.tiling.axes
    )
    if space is not None and space.tiling is not None:
        keys.update(dict.fromkeys((dim, False) for dim in space.tiling.output_dims))
    return {
        (dim, is_red): sympy.Symbol(
            f"tile_{name}_{'r' if is_red else 'o'}{dim}",
            integer=True,
            positive=True,
        )
        for (dim, is_red) in keys
    }


def division_symbol(buffer_name: str) -> sympy.Symbol:
    """The objective symbol for ``buffer_name``'s chosen division index (see
    :attr:`CoreDivisionBuffer.sym_division`). One constructor so a term built
    from a buffer's *name* (a relayout copy pricing its source) and the
    engine's binding built from the buffer agree on name and assumptions."""
    return sympy.Symbol(f"division_{buffer_name}", integer=True, nonnegative=True)


def solved_bindings(buffers: Sequence["LifetimeBoundBuffer"]) -> dict:
    """The objective's symbols as the solved plan fixes them: ``is_lx`` is 1
    for a placed buffer and 0 for a spilled one; a core-division buffer with a
    chosen division binds its ``division`` index and each per-axis split
    symbol to that division's split (1 for an axis it does not split), and
    each tile-count symbol to that division's tile count (1 for an axis it
    does not tile). The same reading the annealer applies to a candidate
    plan."""
    bindings: dict = {}
    for buf in buffers:
        bindings[buf.sym_is_lx] = 1 if buf.address is not None else 0
        chosen = getattr(buf, "chosen_division", None)
        divisions = getattr(buf, "core_divisions", None)
        if chosen is None or not divisions or not 0 <= chosen < len(divisions):
            continue
        bindings[buf.sym_division] = chosen
        splits = divisions[chosen].splits
        for key, sym in buf.sym_core_divs.items():
            bindings[sym] = splits.get(key, 1)
        tiling = divisions[chosen].tiling
        for key, sym in buf.sym_tile_counts.items():
            bindings[sym] = tiling.count_for(key)
    return bindings


def _evaluate(expr: sympy.Expr, bindings: dict) -> float | None:
    try:
        return float(sympy.sympify(expr).xreplace(bindings).evalf())
    except (TypeError, ValueError, AttributeError):
        return None


def cost_expr_record(
    cost_expr: sympy.Expr,
    bundle_terms: Sequence[tuple[list[str], sympy.Expr]],
    buffers: Sequence["LifetimeBoundBuffer"],
    params: object = None,
    off_expression_ns: Mapping[str, float] | None = None,
    score_ns: float | None = None,
    *,
    context: dict | None = None,
) -> dict:
    """One dump record for a solved co-optimized graph: the objective's
    per-bundle terms as ``sympy.srepr`` strings (lossless,
    ``parse_expr`` restores them), the solved symbol bindings, and every term
    evaluated under them. ``buffers`` are the solver's returned buffers;
    ``buffers`` names (the graph's stores) are what a reader joins on.

    ``objective_ns`` is ``cost_expr`` alone. ``off_expression_ns`` is what the
    engine adds outside it for this plan
    (:meth:`GraphSolver.off_expression_ns`), and ``score_ns`` the total the
    engine minimized (:meth:`GraphSolver.score_ns`),
    ``objective_ns`` for an engine that minimizes ``cost_expr`` itself.

    ``divisions`` carries each buffer's candidate core counts, the one chosen,
    its producers, its residency ``reason`` when one kept it out of LX, and the
    division pairs the residency gate admitted on each incoming edge -- the
    alternatives a decision was made over, which the objective alone cannot
    show.

    ``context`` is whatever the caller knows and this function cannot see -- the
    environment the plan was made in, and how the solve went -- so a reader gets
    one self-describing record per solve rather than a set of numbers whose
    meaning depends on flags nobody wrote down. Its keys may not collide with
    the record's own."""
    import dataclasses

    bindings = solved_bindings(buffers)
    bundles = [
        {
            "ops": list(names),
            "expr": sympy.srepr(sympy.sympify(term)),
            "value_ns": _evaluate(term, bindings),
        }
        for names, term in bundle_terms
    ]
    objective_ns = _evaluate(cost_expr, bindings)
    record = {
        "buffers": [b.name for b in buffers],
        # Buffer sizes in bytes: with the names, a key that tells kernels apart
        # even though every kernel numbers its buffers from buf0.
        "buffer_sizes": {b.name: b.size for b in buffers},
        "params": dataclasses.asdict(params)
        if params is not None
        and dataclasses.is_dataclass(params)
        and not isinstance(params, type)
        else {},
        "bundles": bundles,
        # Reading why a division was chosen needs the alternatives it was
        # chosen over: per buffer the core count and split shape of every
        # candidate, the index the solver took, and the ``(parent, consumer)``
        # index pairs the residency gate admitted on each incoming edge. Pairs
        # are stored as INDICES into the two buffers' ``cores`` lists, so a
        # reader can render them as core counts without the record repeating
        # the divisions. Keyed by the CONSUMER, which is where ``parents`` and
        # ``cd_parent_matches`` are populated. A parent with an EMPTY pair list
        # divides no way this buffer can read locally; a parent absent from
        # ``matches`` altogether was refused an edge outright, which is the
        # louder of the two signals (issue #4655 was of that kind).
        "divisions": {
            b.name: {
                "cores": [cd.cores_used for cd in b.core_divisions],
                "labels": [cd.label for cd in b.core_divisions],
                "chosen": b.chosen_division,
                "parents": list(b.parents),
                # Why this buffer never reached the solver, in the allocator's
                # own words ("op not allowed", "partial/offset read"). None
                # means it DID reach the solver, and ``bindings`` says what the
                # solve then decided -- three outcomes a reader must not
                # conflate: excluded, weighed and declined, or resident.
                "reason": b.residency_reason,
                "matches": {
                    parent: [list(pair) for pair in pairs]
                    for parent, pairs in (b.cd_parent_matches or {}).items()
                },
            }
            for b in buffers
            if isinstance(b, CoreDivisionBuffer) and b.core_divisions
        },
        "bindings": {str(k): v for k, v in bindings.items()},
        "objective_ns": objective_ns,
        "off_expression_ns": dict(off_expression_ns or {}),
        "score_ns": objective_ns if score_ns is None else score_ns,
    }
    # Additive only: context describes the record, it does not get to redefine
    # it. Without this a caller key named `bundles` would replace the terms.
    clashes = set(context or ()) & set(record)
    assert not clashes, f"context may not override record keys: {sorted(clashes)}"
    record.update(context or {})
    # A term that would not evaluate under the solved bindings reads in the JSON
    # exactly like one deliberately left unpriced. The difference matters: the
    # second is normal, the first means the objective and the bindings have
    # drifted apart -- a cost-model change introducing a symbol no engine binds,
    # say. Say so once, where the CP-SAT path already logs "cannot linearize".
    unpriced = sum(1 for entry in bundles if entry["value_ns"] is None)
    if unpriced or objective_ns is None:
        logger.warning(
            "cost dump: %d of %d terms did not evaluate under the solved "
            "bindings%s; objective and bindings may have drifted",
            unpriced,
            len(bundles),
            "" if objective_ns is not None else " (whole objective too)",
        )
    return record


def check_in_place_parent_is_read(
    parent: "LifetimeBoundBuffer", child_name: str
) -> None:
    """Reject an in-place parent whose storage is never read before handover.

    The child takes the parent's storage over at the parent's last use, so that
    use has to be a read. For a computed buffer the first use is the write, so a
    single use means the buffer is written and never read -- handing it to a
    child would overwrite data nothing ever consumed, and it would make parent
    and child come alive on the same tick while sharing storage. Graph inputs are
    exempt: all their uses are reads, so one use is enough.

    Split out of :func:`_check_in_place_relationships` because the
    permutation-based layout solvers call it directly, where they resolve
    declared pairs (``_compute_inplace_partners``), alongside their own copy of
    the abutment check -- their incremental machinery samples the contact
    profiles at the single tick the pair overlaps and so cannot re-derive a
    longer overlap. The size invariant is the one they do *not* take as a
    precondition: an oversized child is simply not placed in-place (see
    ``_can_inplace``), so checking it here would reject inputs they handle
    correctly.
    """
    # Tested as "a use strictly after the first" rather than via ``read_count``:
    # the two agree whenever ``uses`` is strictly increasing, but ``uses`` is
    # validated at construction and can be mutated afterwards, and this way a
    # repeated index cannot pass as a read.
    has_read_after_write = len(parent.uses) > 1 and parent.uses[-1] > parent.uses[0]
    if not (parent.first_use_is_read or has_read_after_write):
        raise ValueError(
            f"In-place parent {parent.name} is a computed buffer that is never "
            f"read (uses={parent.uses}), so it cannot hand its storage to child "
            f"{child_name}"
        )


def _check_in_place_relationships(
    buffers: Sequence["LifetimeBoundBuffer"],
) -> None:
    """Reject any declared in-place pair that violates a required invariant."""
    buf_by_name = {b.name: b for b in buffers}
    for child in buffers:
        for parent_name in child.in_place_parents:
            parent = buf_by_name.get(parent_name)
            if parent:
                if parent.end_time != child.start_time + 1:
                    raise ValueError(
                        f"In-place parent {parent_name}.end_time={parent.end_time} "
                        f"must equal child {child.name}.start_time+1="
                        f"{child.start_time + 1}"
                    )
                check_in_place_parent_is_read(parent, child.name)
                # With core_divisions ``size`` is the *total* footprint, so a static
                # size check doesn't apply; the per-core match is enforced against the
                # chosen division in ``TableStrategy.encode_merge``. Only
                # the division-fixed case (plain ``LifetimeBoundBuffer``, no
                # ``core_divisions``) keeps the static check.
                if not (
                    getattr(parent, "core_divisions", None)
                    or getattr(child, "core_divisions", None)
                ):
                    if child.size > parent.size:
                        raise ValueError(
                            f"In-place child {child.name}.size={child.size} "
                            f"must be <= parent {parent_name}.size={parent.size}"
                        )
