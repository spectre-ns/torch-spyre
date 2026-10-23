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

"""The LX layout solve as a graph of resource nodes.

A :class:`ResourceNode` is one buffer an op needs (an op output or a graph
input). Nodes hold incoming data about the buffer and its op, and the
decisions the solve makes about it (``work_split``, ``resident``, ``offset``),
each pinned before the solve or ``None`` for the solve to fill.
:class:`DependencyEdge` joins a producer to a consumer that reads it (and may
take over its storage, ``in_place``); :class:`ExecutionEdge` joins op outputs
that are consecutive in execution order. :class:`SpyreOptimizationGraph` owns
the nodes and runs the solve through an ``EncodingStrategy``
(``encoding_strategy.py``), which holds every candidate, relation and CP
variable: nodes and edges are pure data.

``SpyreOptimizationGraph.solve`` encodes the graph (``_finalize``), runs the
objective ladder and fills every open decision. The ladder is one solve of the
priced cost expression when one is given and lowers, else a lexicographic
fallback: residency (HBM traffic), coarse-tiling cuts, parallelism, division
shape and tile count, each stage holding the previous optimum.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, ClassVar, Optional, TYPE_CHECKING

import sympy
import torch

from torch_spyre._inductor import config
from torch_spyre._inductor.scratchpad.plan_solver import (
    BufferType,
    CoreDivision,
    CoreDivisionBuffer,
    LifetimeBoundBuffer,
    RelayoutCopyBuffer,
    SolveError,
    TileSpec,
    ceil_div,
)

if TYPE_CHECKING:
    from torch._inductor.ir import Operation

    from torch_spyre._inductor.scratchpad.encoding_strategy import EncodingStrategy
    from torch_spyre._inductor.work_division import OpSplitSpace

logger = logging.getLogger(__name__)

SplitKey = Any  # CoreDivision.splits keys: sympy symbols (ints/strs in tests)
TileKey = tuple[int, bool]  # sym_tile_counts keys: (host_dim, is_reduction)

# Spill cause for a buffer the solver chose to leave out of LX, as opposed to
# one barred up front (its residency_reason).
SOLVER_CHOSE_SPILL = "spilled by solver (no residency benefit / no room)"


@dataclass(frozen=True)
class WorkSplit:
    """How one op's work is split: across cores (a split factor per loop
    symbol) and into coarse tiles. Decisions speak in work splits, never in
    menu indices. :attr:`COMMITTED` is a split fixed upstream by work
    division, which the solve does not see."""

    splits: tuple[tuple[SplitKey, int], ...]
    tiling: Optional[TileSpec]
    committed: bool = False
    COMMITTED: ClassVar[WorkSplit]

    @staticmethod
    def of(cd: CoreDivision) -> WorkSplit:
        return WorkSplit(tuple(sorted(cd.splits.items(), key=str)), cd.tiling)


WorkSplit.COMMITTED = WorkSplit(splits=(), tiling=None, committed=True)


@dataclass(eq=False)
class DependencyEdge:
    """``dst``'s op reads ``src``; with ``in_place`` it may also take over
    ``src``'s storage at the tick they share. The one object sits in
    ``src.outputs`` and ``dst.inputs``. ``reads`` counts the times the child
    lists the parent among its ``parents`` (an op may read one buffer twice),
    each one a read relation; it is 0 for an in-place edge whose read the
    allocator does not model, which gates the merge, not residency or cuts."""

    src: ResourceNode
    dst: ResourceNode
    in_place: bool = False
    reads: int = 1
    # decisions, filled by solve(): an in-place edge whether the child took
    # over the parent's storage; the transform this read goes through (None:
    # it reads ``src`` directly)
    merged: Optional[bool] = None
    transform: Optional[Transform] = None

    def __repr__(self) -> str:
        kind = "in_place" if self.in_place else "read"
        return f"{self.src.name}->{self.dst.name}:{kind}"


@dataclass(eq=False)
class ExecutionEdge:
    """Two op outputs consecutive in execution order. ``contiguous`` is false
    across a tick gap (or a shared tick)."""

    before: ResourceNode
    after: ResourceNode
    contiguous: bool

    def __repr__(self) -> str:
        return f"{self.before.name}=>{self.after.name}"


@dataclass(eq=False)
class ResourceNode:
    """One buffer an op needs: its incoming data, and its decisions (``None``
    = open). After a successful solve no decision is ``None`` (``offset``
    only when resident, ``spill_reason`` only when not)."""

    name: str
    size: int  # alignment units
    start_time: int  # [start, end) in ticks, overrides applied
    end_time: int
    read_count: int  # every reader, extern ones included; not the write
    first_use_is_read: bool
    boundary: BufferType
    sym_core_divs: Mapping[SplitKey, sympy.Symbol]
    sym_tile_counts: Mapping[TileKey, sympy.Symbol]
    # Tick of the producing write (uses[0]); None for a graph input. Not
    # start_time, which a start override can move earlier.
    produced_at: Optional[int]
    op: Optional[Operation] = None
    space: Optional[OpSplitSpace] = None  # None: committed upstream
    residency_reason: Optional[str] = None  # applied by encode_op_constraints
    # The access ticks, the size in bytes, and the committed per-core view.
    uses: tuple[int, ...] = ()
    nbytes: int = 0
    lx_view: Any = None
    # decisions
    work_split: Optional[WorkSplit] = None
    resident: Optional[bool] = None
    offset: Optional[int] = None  # alignment units
    spill_reason: Optional[str] = None
    # structure
    prev: Optional[ExecutionEdge] = None
    next: Optional[ExecutionEdge] = None
    inputs: list[DependencyEdge] = field(default_factory=list)
    outputs: list[DependencyEdge] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"ResourceNode({self.name})"

    @property
    def tiling(self) -> Optional[TileSpec]:
        return self.work_split.tiling if self.work_split else None

    @property
    def committed(self) -> bool:
        return self.work_split is not None and self.work_split.committed

    @property
    def sym_is_lx(self) -> sympy.Symbol:
        return sympy.Symbol(f"is_lx_{self.name}", integer=True, nonnegative=True)

    def chosen_transforms(self) -> list[Transform]:
        """The distinct transforms this output's reads go through, in copy
        order: what realization inserts."""
        chosen = {e.transform: None for e in self.outputs if e.transform is not None}
        return sorted(chosen, key=lambda t: t.group)

    def add_child(
        self, child: ResourceNode, in_place: bool = False, reads: int = 1
    ) -> DependencyEdge:
        """The only way to add a dependency edge; refuses a second edge between
        the same pair."""
        assert all(e.dst is not child for e in self.outputs), (
            f"{self.name} -> {child.name} already has an edge"
        )
        assert in_place or reads, "an edge either reads or is in place"
        edge = DependencyEdge(self, child, in_place, reads)
        self.outputs.append(edge)
        child.inputs.append(edge)
        return edge

    def add_successor(self, after: ResourceNode) -> ExecutionEdge:
        """The only way to add an execution edge."""
        assert self.next is None, f"{self.name} already has a successor"
        assert after.prev is None, f"{after.name} already has a predecessor"
        assert self.produced_at is not None and after.produced_at is not None
        edge = ExecutionEdge(
            self, after, contiguous=after.produced_at == self.produced_at + 1
        )
        self.next = edge
        after.prev = edge
        return edge


def _integer_objective(cp: Any) -> Any:
    """Re-post ``cp``'s objective with integer coefficients and return it,
    so a later stage can hold its optimum (CP-SAT takes no float constraint).
    A float objective is scaled the way CP-SAT scales one internally: by the
    largest power of two that keeps its activity within 2**53."""
    fo = cp.proto.floating_point_objective
    obj = cp.proto.objective
    pairs = (
        list(zip(fo.vars, fo.coeffs)) if fo.vars else list(zip(obj.vars, obj.coeffs))
    )
    if not pairs:
        return None

    def var(index: int) -> Any:
        if index >= 0:
            return cp.get_int_var_from_proto_index(index)
        return 1 - cp.get_int_var_from_proto_index(-index - 1)

    scale = 1
    if fo.vars:
        activity = sum(
            abs(c)
            * max(abs(d) for d in cp.proto.variables[v if v >= 0 else -v - 1].domain)
            for v, c in pairs
        )
        scale = 2 ** max(0, min(40, 53 - math.ceil(math.log2(max(activity, 1.0)))))
    expr = sum(round(c * scale) * var(v) for v, c in pairs if round(c * scale))
    cp.minimize(expr)
    return expr


def _unposted(*args, **kwargs) -> int:
    raise AssertionError("the solver sets the solve settings")


def empty_graph(capacity_bytes: int, alignment: int = 128) -> SpyreOptimizationGraph:
    """A graph to build from the IR: ``alignment`` bytes per unit (one stick,
    what every solver aligns to); the solver sets the solve settings."""
    return SpyreOptimizationGraph(
        alignment=alignment,
        capacity=capacity_bytes // alignment,
        settings=SolveSettings(None, False, _unposted),
    )


def spill_cost(item: Any, size: Optional[int] = None) -> int:
    """Differential HBM traffic a spill adds over residency, in units of
    ``size`` (default ``item.size``): the reads residency would serve, plus
    the producer's write, which residency turns into a free LX write. A graph
    input has no write to save and a graph output's write-out is unavoidable,
    so only an intermediate counts it. An input's first read is the clone-in
    residency cannot avoid, so it is not served."""
    first_read = getattr(item, "first_use_is_read", False)
    boundary = getattr(item, "boundary", None)
    is_intermediate = (
        boundary == BufferType.Intermediate if boundary is not None else not first_read
    )
    reads_served = item.read_count - (1 if first_read else 0)
    return (reads_served + (1 if is_intermediate else 0)) * (
        item.size if size is None else size
    )


class TransformKind(Enum):
    RELAYOUT = auto()  # another per-core view, LX to LX
    COPY_OUT = auto()  # a cut tile's full buffer, to HBM
    TRANSPOSE = auto()  # future: another dim order
    RESTICKIFY = auto()  # future: another stick dimension


@dataclass(eq=False)
class Transform:
    """A converted copy of the producer's output that one or more edges read
    instead of the producer: one per (producer, kind, target), shared by every
    edge that needs that conversion and priced once. A relayout is held in LX
    from the first read it serves to the last. Candidates belong to the
    strategy; ``solve()`` fills the decisions of each one and points the
    edges that read through it at it (``edge.transform``)."""

    producer: Optional[ResourceNode]  # None: source not in this solve
    name: str
    group: int
    size: int  # alignment units
    read_count: int
    division: CoreDivision
    start_time: int
    end_time: int
    kind: TransformKind = TransformKind.RELAYOUT
    residency_reason: Optional[str] = None
    # The placement view of a pinned transform: access ticks, size in bytes,
    # its fixed per-core view.
    uses: tuple[int, ...] = ()
    nbytes: int = 0
    lx_view: Any = None
    # decisions, filled by solve()
    resident: Optional[bool] = None
    offset: Optional[int] = None
    spill_reason: Optional[str] = None

    def __repr__(self) -> str:
        return f"Transform({self.kind.name}, {self.name})"

    @property
    def sym_is_lx(self) -> sympy.Symbol:
        return sympy.Symbol(f"is_lx_{self.name}", integer=True, nonnegative=True)


@dataclass(frozen=True)
class SolveSettings:
    """How ``solve`` runs CP-SAT: the time limit, whether to bottom-justify
    offsets, the hook that solves one ladder stage and records its stats
    (``ModelSolver._solve_and_record``), and where a stats entry goes
    when the objective cannot be lowered."""

    time_limit_seconds: Optional[float]  # None or 0: no limit
    bottom_justify: bool
    record: Callable[..., int]
    note_stats: Callable[[dict], None] = lambda stats: None


@dataclass
class _PlacementUnit:
    """A connected component of in-place-merged buffers placed as one block."""

    members: list[str]
    footprint: int
    start_time: int
    end_time: int
    original_offset: int  # offset the solver chose, before bottom-justify
    justified_offset: int = 0  # final justified offset


def justify(units: list[_PlacementUnit], capacity: int) -> Optional[dict[str, int]]:
    """Slide each placement unit down to the lowest free address. Processing
    in current-base order and giving each the lowest non-conflicting slot
    preserves the relative stacking, so it mostly squeezes out the float gaps
    the search leaves. Returns a name -> address map, or ``None`` if the
    result would not fit in ``capacity``.

    A merged unit is coarsened to one rectangle spanning the union of its
    members' lifetimes at their largest footprint, which is conservative: it
    can make two units conflict here that did not conflict in the model, and
    the bump that resolves that conflict can push a unit's top past capacity.
    The caller then keeps the solver's own offsets, which the model
    constrained to fit. Returning ``None`` rather than clamping keeps this a
    pure optimisation -- it never decides residency, and never hands back an
    address outside the scratchpad."""
    placed: list[_PlacementUnit] = []
    offsets = {}
    for u in sorted(units, key=lambda u: (u.original_offset, u.start_time)):
        # lowest base whose [base, base+footprint) clears every already-placed
        # unit that overlaps this one in time. We don't need to worry about
        # tied offsets because blocks cannot have the same offset and also
        # overlap in time.
        obstacles = sorted(
            (p.justified_offset, p.justified_offset + p.footprint)
            for p in placed
            if u.start_time < p.end_time and p.start_time < u.end_time
        )
        base = 0
        for lo, hi in obstacles:
            if base + u.footprint <= lo:
                break  # fits in the gap below this obstacle
            if base < hi:
                base = hi  # otherwise bump above it
        if base + u.footprint > capacity:
            return None
        u.justified_offset = base
        placed.append(u)
        for n in u.members:
            offsets[n] = base
    return offsets


@dataclass(eq=False)
class SpyreOptimizationGraph:
    """The solve's input: the nodes (which own the edges) in ``add_node``
    order, the head of the execution chain, and the global context."""

    alignment: int  # bytes per unit
    capacity: int  # LX capacity in alignment units
    settings: SolveSettings
    nodes: dict[str, ResourceNode] = field(default_factory=dict)
    first: Optional[ResourceNode] = None
    solver: Any = None  # a Backend class or a GraphSolver; None: CpSat

    def add_ir_node(
        self,
        name: str,
        *,
        nbytes: int,
        uses: Sequence[int],
        first_use_is_read: bool = False,
        boundary: Optional[BufferType] = None,
        residency_reason: Optional[str] = None,
        start_override: Optional[int] = None,
        end_override: Optional[int] = None,
        op: Optional[Operation] = None,
        space: Optional[OpSplitSpace] = None,
        sym_core_divs: Mapping[SplitKey, sympy.Symbol] = {},
        sym_tile_counts: Mapping[TileKey, sympy.Symbol] = {},
        lx_view: Any = None,
        open_split: bool = False,
    ) -> ResourceNode:
        """A node for IR buffer ``name``: an op output, or a graph input read
        (``first_use_is_read``). Its lifetime is its ``uses`` widened by the
        overrides; ``nbytes`` is its LX footprint in bytes (the total, divided
        per candidate, when its work split is open). A work split is open only
        when the caller has candidates for it (``open_split``); else it was
        committed upstream."""
        assert name not in self.nodes, f"{name} added twice"
        uses = list(uses)
        start = min(uses[0], start_override) if start_override is not None else uses[0]
        end = uses[-1] + 1
        if end_override is not None:
            end = max(end, end_override)
        node = ResourceNode(
            name=name,
            size=ceil_div(nbytes, self.alignment),
            start_time=start,
            end_time=end,
            read_count=max(0, len(uses) - (0 if first_use_is_read else 1)),
            first_use_is_read=first_use_is_read,
            boundary=(
                boundary
                if boundary is not None
                else BufferType.Input
                if first_use_is_read
                else BufferType.Intermediate
            ),
            sym_core_divs=sym_core_divs,
            sym_tile_counts=sym_tile_counts,
            produced_at=uses[0] if uses and not first_use_is_read else None,
            op=op,
            space=space,
            residency_reason=residency_reason,
            uses=tuple(uses),
            nbytes=nbytes,
            lx_view=lx_view,
            work_split=None if open_split else WorkSplit.COMMITTED,
        )
        self.nodes[name] = node
        return node

    def add_reads(
        self,
        child: ResourceNode,
        parents: Sequence[str],
        in_place_parents: Sequence[str] = (),
    ) -> None:
        """The child's dependency edges: one per parent it reads (counted:
        an op may list a parent twice), ``in_place`` for one whose storage it
        may take over. In-place inputs come first, in their order, since the
        merge literals are made in it; a parent that is not a node is not in
        the solve."""
        reads = Counter(parents)
        for parent in in_place_parents:
            self.nodes[parent].add_child(child, in_place=True, reads=reads[parent])
        for parent, count in reads.items():
            if parent in self.nodes and parent not in in_place_parents:
                self.nodes[parent].add_child(child, reads=count)

    def add_node(
        self, source: LifetimeBoundBuffer, residency_reason: Optional[str] = None
    ) -> ResourceNode:
        """The buffer form of :meth:`add_ir_node`, for tests and capture
        replay: a node holding ``source``'s incoming data. A buffer with no
        candidate divisions is committed upstream."""
        assert not isinstance(source, RelayoutCopyBuffer), (
            "a relayout copy is a transform, not a node"
        )
        b = source
        divided = b if isinstance(b, CoreDivisionBuffer) and b.core_divisions else None
        return self.add_ir_node(
            b.name,
            nbytes=b.size,
            uses=b.uses,
            first_use_is_read=b.first_use_is_read,
            boundary=b.boundary if isinstance(b, CoreDivisionBuffer) else None,
            residency_reason=residency_reason,
            start_override=b.lifetime_start_override,
            end_override=b.lifetime_end_override,
            space=divided.division_space if divided else None,
            sym_core_divs=divided.sym_core_divs if divided else {},
            sym_tile_counts=divided.sym_tile_counts if divided else {},
            lx_view=b.lx_view,
            open_split=divided is not None,
        )

    def get_nodes(self) -> Iterator[ResourceNode]:
        return iter(self.nodes.values())

    def execution_chain(self) -> Iterator[ExecutionEdge]:
        node = self.first
        while node is not None and node.next is not None:
            yield node.next
            node = node.next.after

    def link_execution_order(self) -> None:
        """Chain every op output in ``produced_at`` order (ties in add order)."""
        ticks = {
            n: n.produced_at for n in self.nodes.values() if n.produced_at is not None
        }
        ops = sorted(ticks, key=ticks.__getitem__)
        self.first = ops[0] if ops else None
        for before, after in zip(ops, ops[1:]):
            before.add_successor(after)

    def set_solver(self, solver: Any) -> None:
        """What ``solve`` runs on: a backend class (the posted model's solver)
        or a graph solver, which solves the graph itself."""
        self.solver = solver

    # ---- the solve ----

    def solve(
        self,
        model: Any = None,
        cost_expr: Optional[sympy.Expr] = None,
        strategy: Optional[EncodingStrategy] = None,
    ) -> list[ResourceNode]:
        """Encode, run the objective ladder, and fill every open decision on
        the nodes, transforms and edges. ``model`` is the backend to post
        to and solve with (a bare ``CpModel`` is wrapped in ``CpSat``; none:
        an instance of :meth:`set_solver`'s backend class). A graph solver set
        with :meth:`set_solver` solves the graph itself instead. Returns the
        op outputs in execution order. Raises :class:`SolveError`, writing
        nothing."""
        if strategy is None:
            from torch_spyre._inductor.scratchpad.encoding_strategy import (
                EnumerationFactory,
            )

            strategy = EnumerationFactory().strategy()
        if self.solver is not None and not isinstance(self.solver, type):
            if not self.solver.accepts(strategy):
                raise ValueError(
                    f"{type(self.solver).__name__} cannot solve a "
                    f"{type(strategy).__name__}"
                )
            return self.solver.solve(self, strategy, cost_expr)

        from ortools.sat.python import cp_model

        from torch_spyre._inductor.scratchpad.backend import CpSat

        if model is None:
            model = (self.solver or CpSat)()
        backend = CpSat(model) if isinstance(model, cp_model.CpModel) else model
        self._finalize(backend, strategy)
        solver = backend.solver
        backend.record = self.settings.record
        settings = self.settings
        if settings.time_limit_seconds:
            solver.parameters.max_time_in_seconds = float(settings.time_limit_seconds)
        # Presolve runs on every model, priced or not; the copy-count threshold
        # is an opt-in escape hatch (off by default).
        free_copies = sum(
            t.producer is not None and strategy.chooses_split(t.producer)
            for t in strategy.transforms
        )
        max_copies = config.lx_solver_relayout_presolve_max_copies
        if max_copies > 0 and free_copies > max_copies:
            solver.parameters.cp_model_presolve = False
            logger.info(
                "[CP-SAT layout solver] %d relayout copies exceed the presolve "
                "threshold of %d; solving without presolve",
                free_copies,
                max_copies,
            )
        solver.parameters.num_search_workers = (
            1
            if torch.are_deterministic_algorithms_enabled()
            else config.get_cpu_count()
        )
        # Fixed seed so a given worker configuration is reproducible run-to-run.
        solver.parameters.random_seed = 0
        # Justifying would move a caller-pinned offset.
        justify_offsets = settings.bottom_justify and all(
            n.offset is None for n in self.get_nodes()
        )
        self._ladder(backend, strategy, cost_expr)
        self._extract(backend, strategy, justify_offsets)
        return self.execution_plan()

    def execution_plan(self) -> list[ResourceNode]:
        """The op outputs along the execution chain."""
        node = self.first
        plan = []
        while node is not None:
            plan.append(node)
            node = node.next.after if node.next else None
        return plan

    def _finalize(self, model: Any, strategy: EncodingStrategy) -> None:
        """Walk the graph in emission order, calling the strategy once per
        node, edge or transform, and record which call posted what."""
        nodes = list(self.get_nodes())
        assert all(n.produced_at is None or n is self.first or n.prev for n in nodes)
        strategy.prepare(self)
        transforms = strategy.transforms
        record = _ProvenanceRecorder(model, strategy)

        for n in nodes:  # 1. vars
            record(strategy.encode_node, n)
        for t in transforms:
            record(strategy.encode_transform, t)
        merges = [e for n in nodes for e in n.inputs if e.in_place]
        for e in merges:  # 2. merges
            record(strategy.encode_merge, e)
        for n in nodes:
            record(strategy.limit_merges, n, incoming=True)
        for n in dict.fromkeys(e.src for e in merges):
            record(strategy.limit_merges, n, incoming=False)
        for item in (*nodes, *transforms):  # 3. placement
            record(strategy.encode_capacity, item)
        record(strategy.encode_no_overlap, self)
        for item in (*nodes, *transforms):  # 4. op constraints
            record(strategy.encode_op_constraints, item)
        for n in nodes:  # 5. reads
            for e in n.outputs:
                record(strategy.encode_read, e)
        for t in transforms:  # 6. transforms
            record(strategy.encode_transform_use, t)
        divided = [n for n in nodes if strategy.chooses_split(n)]
        for item in (*divided, *transforms):  # 7. coarse tiling
            record(strategy.encode_tiling_vars, item)
        segments = strategy.tiling_segments(self)
        if self.first is not None:
            record(strategy.encode_chain_head, self.first)
        for edge in self.execution_chain():
            record(strategy.encode_nest, edge)
        for item in (*divided, *transforms):
            record(strategy.encode_cuts, item, segments)
        for n in nodes:
            record(strategy.encode_tile_counts, n)
        for n in nodes:  # 8. clones
            if n.produced_at is None:
                record(strategy.encode_clone, n)
        for item in (*nodes, *transforms):  # 9. lifetimes
            record(strategy.encode_lifetime, item)
        # Read by the equivalence harness, next to each stage's proto.
        setattr(model.model, "_spyre_provenance", strategy.provenance)

    def _ladder(
        self,
        model: Any,
        strategy: EncodingStrategy,
        cost_expr: Optional[sympy.Expr],
    ) -> None:
        """One priced solve, or the fallback ladder when there is no cost
        expression or it cannot be lowered. Raises :class:`SolveError` when a
        stage ends without a plan."""
        from ortools.sat.python import cp_model

        solver = model
        items: list[ResourceNode | Transform] = [
            *self.get_nodes(),
            *strategy.transforms,
        ]
        cut_terms = strategy.cut_terms
        if cut_terms:
            logger.debug(
                "[CP-SAT layout solver] cut tiebreak over %d candidate cut(s)",
                len(cut_terms),
            )
        tile_terms = strategy.tile_terms

        def solve_stage(stage: str, objective: bool = False) -> int:
            result = model.Solve(objective=objective)
            if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                raise SolveError(
                    f"CP-SAT returned {solver.StatusName(result)} without a plan "
                    f"after {solver.WallTime():.2f}s ({stage})"
                )
            return result

        status = None
        core_terms = None
        occupancy: Optional[int] = None
        if cost_expr is not None:
            priced = self._minimize_cost_expr(model, strategy, cost_expr)
            if priced is not None:
                status, held = priced
                # Splits the cost model prices alike (a bandwidth-bound op
                # with nothing resident) tie; break the tie as the ladder
                # does, holding the priced optimum.
                terms = [strategy.objective_terms(item) for item in items]
                if any(c is not None for c, _ in terms):
                    if held is not None:
                        model.Add(held <= model.Value(held))
                    # The tie is among splits: residency stays as priced.
                    for item in items:
                        lit = strategy.vars[item].in_buffer
                        model.Add(lit == int(model.BooleanValue(lit)))
                    core_terms, occupancy, status = self._tiebreak(
                        model, terms, tile_terms, solve_stage, objective=True
                    )

        if status is None:
            # One lexicographic ladder, in priority order: LX residency (HBM
            # traffic), cut count, parallelism, division shape, tile count.
            # Each stage holds the previous optimum, so a later stage only
            # breaks ties the earlier ones leave open.

            # The traffic objective knows no relayout price, and an unpriced
            # shuffle looks free, so no relayout may fire under it.
            with _ProvenanceRecorder(model, strategy).section("ladder:copy pin"):
                for t in strategy.transforms:
                    model.Add(strategy.vars[t].in_buffer == 0)

            vars_ = strategy.vars
            hbm_terms = [
                spill_cost(item) * (1 - vars_[item].in_buffer) for item in items
            ]
            status = cp_model.INFEASIBLE
            terms = [strategy.objective_terms(item) for item in items]
            if hbm_terms:
                model.Minimize(sum(hbm_terms))
                status = solve_stage("residency")
                # Lock the traffic optimum (not just the count); the objective
                # is a sum of ints, so the round is exact.
                if cut_terms or any(cores is not None for cores, _ in terms):
                    model.Add(sum(hbm_terms) <= round(solver.ObjectiveValue()))

            if cut_terms:
                model.Minimize(sum(cut_terms))
                status = solve_stage("cut tiebreak")
                cuts = round(solver.ObjectiveValue())
                logger.debug(
                    "[CP-SAT layout solver] cut tiebreak: %d cut(s) at the "
                    "residency optimum",
                    cuts,
                )

            if any(cores is not None for cores, _ in terms):
                if cut_terms:
                    model.Add(sum(cut_terms) <= cuts)
                core_terms, occupancy, status = self._tiebreak(
                    model, terms, tile_terms, solve_stage
                )

        if logger.isEnabledFor(logging.DEBUG):
            if status is None:
                status = cp_model.INFEASIBLE
            spilled = [
                item.name
                for item in items
                if not solver.BooleanValue(strategy.vars[item].in_buffer)
            ]
            if core_terms and tile_terms:
                final_objective = "tile_count"
            elif core_terms:
                final_objective = "balance"
            else:
                final_objective = "hbm_traffic"
            logger.debug(
                "[CP-SAT layout solver] tensors=%d resident=%d %s=%d "
                "occupancy=%s status=%s walltime=%.2f ms",
                len(items),
                len(items) - len(spilled),
                final_objective,
                round(solver.ObjectiveValue()),
                occupancy if occupancy is not None else "n/a",
                solver.StatusName(status),
                solver.WallTime() * 1e3,
            )
            reasons = {item.name: item.residency_reason for item in items}
            for name in sorted(spilled):
                logger.debug(
                    "[CP-SAT layout solver]   %s -> HBM: %s",
                    name,
                    reasons[name] or SOLVER_CHOSE_SPILL,
                )

    @staticmethod
    def _tiebreak(
        model: Any,
        terms: list[tuple[Any, Any]],
        tile_terms: list,
        solve_stage: Callable[..., int],
        objective: bool = False,
    ) -> tuple[list, int, int]:
        """The ladder's last stages, each holding the previous optimum:
        parallelism (most cores), division shape, then the fewest tiles.
        Returns the core terms, the occupancy and the last status."""
        core_terms = [cores for cores, _ in terms if cores is not None]
        core_cost_terms = [cost for _, cost in terms if cost is not None]
        model.Maximize(sum(core_terms))
        model.HintSolution()
        solve_stage("parallelism", objective)
        occupancy = round(model.ObjectiveValue())
        model.Add(sum(core_terms) >= occupancy)
        model.Minimize(sum(core_cost_terms))
        model.HintSolution()
        status = solve_stage("division shape", objective)
        # Tilings that differ only in count tie on every stage above; holding
        # the shape optimum, take the fewest tiles.
        if tile_terms:
            model.Add(sum(core_cost_terms) <= round(model.ObjectiveValue()))
            model.Minimize(sum(tile_terms))
            model.HintSolution()
            status = solve_stage("tile count", objective)
        return core_terms, occupancy, status

    def _minimize_cost_expr(
        self,
        model: Any,
        strategy: EncodingStrategy,
        cost_expr: sympy.Expr,
    ) -> Optional[tuple[int, Any]]:
        """Solve once under the lowered cost expression: the status, and the
        objective as an integer expression a later stage can hold (``None``
        for a constant cost). ``None`` (with the reason in the stats) when
        it cannot be lowered."""
        from ortools.sat.python import cp_model

        from torch_spyre._inductor.scratchpad.cpsat_expr_printer import (
            cost_expr_to_cpsat,
        )

        sym_map, buffer_map = strategy.cost_bindings()
        try:
            with _ProvenanceRecorder(model, strategy).section("ladder:cost_expr"):
                cp_cost = cost_expr_to_cpsat(
                    model.model, sym_map, buffer_map, cost_expr
                )
                extra = strategy.extra_cost_terms()
                if extra:
                    cp_cost = cp_cost + sum(extra)
                held = None
                if not isinstance(cp_cost, (int, float)):
                    # A constant cost leaves any solution optimal.
                    model.Minimize(cp_cost)
                    held = _integer_objective(model.model)
            solver = model
            status = model.Solve(objective=True)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                raise SolveError(
                    f"CP-SAT returned {solver.StatusName(status)} without a plan "
                    f"after {solver.WallTime():.2f}s"
                )
            return status, held
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                "[CP-SAT layout solver] cannot linearize the sympy expr: %s", exc
            )
            if not config._cpsat_warn_on_cost_expr:
                raise
            # A fallback solve follows and records over this; it stands only
            # if no fallback runs, and says why.
            self.settings.note_stats(
                {
                    "status": "NOT_LINEARIZABLE",
                    "error": str(exc),
                    "objective_used": False,
                }
            )
            return None

    def _extract(
        self,
        solver: Any,
        strategy: EncodingStrategy,
        justify_offsets: bool,
    ) -> None:
        """Fill every open decision from the solution. With
        ``justify_offsets``, each in-place-merged placement unit slides down to the lowest
        free address; if that cannot keep every unit inside capacity the
        solver's own offsets are kept, since those are always legal."""
        every: list[ResourceNode | Transform] = [
            *self.get_nodes(),
            *strategy.transforms,
        ]
        items = {item.name: item for item in every}
        vars_ = {name: strategy.vars[item] for name, item in items.items()}
        spilled = {
            name for name, v in vars_.items() if not solver.BooleanValue(v.in_buffer)
        }
        footprint = {
            name: strategy.footprint(item, solver) for name, item in items.items()
        }
        offsets: Optional[dict[str, int]] = None
        if justify_offsets:
            # A placement unit is a connected component of active merges: its
            # members share one base, so it slides as a single block.
            resident = [n for n in items if n not in spilled]
            parent = {n: n for n in resident}

            def find(x: str) -> str:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for dst, v in vars_.items():
                for src, lit in v.merges.items():
                    if solver.BooleanValue(lit):
                        parent[find(src)] = find(dst)

            components: dict[str, list[str]] = {}
            for n in resident:
                components.setdefault(find(n), []).append(n)
            units = [
                _PlacementUnit(
                    members=names,
                    footprint=max(footprint[n] for n in names),
                    start_time=min(items[n].start_time for n in names),
                    end_time=max(items[n].end_time for n in names),
                    original_offset=solver.Value(vars_[names[0]].offset),
                )
                for names in components.values()
            ]
            offsets = justify(units, self.capacity)
        if offsets is None:
            offsets = {
                name: solver.Value(v.offset)
                for name, v in vars_.items()
                if name not in spilled
            }

        for name, item in items.items():
            if isinstance(item, ResourceNode):
                split = strategy.chosen_split(item, solver)
                if item.work_split is None:
                    item.work_split = split
            item.resident = name not in spilled
            item.offset = offsets.get(name)
            item.spill_reason = (
                (item.residency_reason or SOLVER_CHOSE_SPILL)
                if name in spilled
                else None
            )
        for node in self.get_nodes():
            for edge in node.outputs:
                if edge.in_place:
                    edge.merged = solver.BooleanValue(
                        vars_[edge.dst.name].merges[node.name]
                    )
                fired = [
                    t
                    for t, lit in strategy.served.get(edge, ())
                    if solver.BooleanValue(lit)
                ]
                if not fired:
                    continue
                assert len(fired) == 1, (
                    f"{edge.dst.name} reads {node.name} through {len(fired)} "
                    "copies at once"
                )
                assert fired[0].resident, (
                    f"{edge.dst.name} reads {node.name} from a spilled copy "
                    f"{fired[0].name}"
                )
                edge.transform = fired[0]
                strategy.record_fired(edge, solver)


class _ProvenanceRecorder:
    """Records, around one call, the model's variable and constraint ranges
    it added, as a :class:`Provenance` entry on the strategy."""

    def __init__(self, model: Any, strategy: EncodingStrategy) -> None:
        self.model = model
        self.strategy = strategy

    def _counts(self) -> tuple[int, int]:
        return self.model.NumVariables(), self.model.NumConstraints()

    def __call__(self, method: Callable, element: Any, *args, **kwargs) -> None:
        with self.section(method.__name__, element):
            method(element, *args, model=self.model, **kwargs)

    def section(self, source: str, element: Any = "graph") -> _Section:
        return _Section(self, source, element)


class _Section:
    def __init__(self, recorder: _ProvenanceRecorder, source: str, element) -> None:
        self.recorder, self.source, self.element = recorder, source, element

    def __enter__(self) -> None:
        self.before = self.recorder._counts()

    def __exit__(self, *exc) -> None:
        from torch_spyre._inductor.scratchpad.encoding_strategy import Provenance

        after = self.recorder._counts()
        if after == self.before:
            return
        element = (
            self.element.name
            if hasattr(self.element, "name")
            else repr(self.element)
            if not isinstance(self.element, SpyreOptimizationGraph)
            else "graph"
        )
        self.recorder.strategy.provenance.append(
            Provenance(
                self.source,
                element,
                range(self.before[0], after[0]),
                range(self.before[1], after[1]),
            )
        )


def build_graph(
    buffers: Sequence[LifetimeBoundBuffer],
    alignment: int,
    capacity: int,
    settings: SolveSettings,
    reasons: Mapping[str, str] = {},
) -> SpyreOptimizationGraph:
    """The graph of ``buffers`` (relayout copies excluded; they are the
    strategy's candidate transforms), sized in ``alignment`` units. ``reasons`` is each
    barred buffer's residency reason (``MemoryPlanSolver.excluded``).

    One edge per (producer, consumer) pair the consumer reads (``parents``),
    ``in_place`` when the producer is one of its ``in_place_parents``; in-place
    inputs come first, in that order, since the merge literals are made in it."""
    graph = SpyreOptimizationGraph(
        alignment=alignment, capacity=capacity, settings=settings
    )
    real = [b for b in buffers if not isinstance(b, RelayoutCopyBuffer)]
    for b in real:
        graph.add_node(b, reasons.get(b.name))
    nodes = graph.nodes
    for b in real:
        graph.add_reads(nodes[b.name], getattr(b, "parents", ()), b.in_place_parents)
    graph.link_execution_order()
    return graph
