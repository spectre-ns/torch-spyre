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

"""Joint core-division + LX-placement solver built on OR-Tools CP-SAT
(``config.layout_solver == "cpsat"``).

Selects each buffer's core division and its LX scratchpad placement in one
constraint model over :class:`CoreDivisionBuffer`s:

* **Joint core-division.** ``size`` is the *total* device footprint; a ``div``
  var indexes the buffer's candidate divisions (from
  ``enumerate_work_division_candidates``) and ``AddElement`` ties the chosen
  index to the per-core footprint (``eff_size = size / output_partition``) and
  total core usage (``cores = cores_used``, including any reduction-axis split).
* **Slicing-match residency gate.** A resident buffer's division must induce the
  same per-core slicing as *every* consumer's, using the precomputed
  ``cd_parent_matches`` pairs over the ``parents`` (producer/consumer) edges; a
  buffer with no consumer, or a consumer with no compatible pair, can never
  reside (``_CoreDivisionBufferWithCpVars.constrain_residency``).
* **Placement** is a global ``AddNoOverlap2D`` over optional rectangles
  (``[start_time, end_time) x [offset, offset + eff_size)``, present iff
  resident). In-place reuse (``in_place_parents`` -> per-edge ``merge_vars``) is
  encoded by *shortening the child's lifetime* by the single handoff tick when
  the merge fires, so the parent and its in-place child abut in time and may
  legally share an offset; the single-tick-overlap invariant
  (``_assert_in_place_relationships``) makes this exact. The parent keeps its
  full lifetime, so the footprint above a smaller child stays protected on the
  handoff tick (``_add_no_overlap_2d``).
* **Objective** (``_objective``). Always a sympy expression written in the
  buffers' own symbols (``sym_is_lx``, ``sym_cores``, ``sym_inv_cores``), lowered
  onto the model and minimized in a **single phase**. This module's share of that
  is one step, ``_bind_symbols``: tie each symbol to the variable that decides it
  (``cost_symbols``). The conversion itself belongs to
  :func:`~torch_spyre._inductor.scratchpad.cost_expr_ortools.lower`, which takes
  only a model, an expression and that binding -- so the objective is data this
  solver supplies variables for, not a shape built into it. Anything the lowering
  does not understand raises ``CostExpressionError`` rather than being
  approximated.

  The caller may supply that expression as ``plan_layout_and_core_divisions``'
  ``cost_expr`` -- typically the latency the cost model predicts for the graph --
  and it is then the *whole* objective; nothing else is optimized alongside it.
  With no ``cost_expr`` the default (``default_cost_expr``) is built the same
  way from the same symbols, so the injected and default paths are one path and
  the default is simply the expression on it.

  That default is ``weight * sum_b spill_cost(b) * (1 - is_lx_b) - sum_b cores_b``.
  The first sum is total **HBM transfer traffic**: the *differential* traffic a
  spill adds over residency, so resident buffers contribute 0. An intermediate
  costs ``(num_consumers + 1) * size`` (the producer's HBM write, which residency
  turns into a free LX write, plus one re-read per consumer); a graph input drops
  the producer write it never had and the clone-in read residency cannot avoid
  (``(num_consumers - 1) * size``); a graph output drops its unavoidable
  write-out (``num_consumers * size``). The second sum is total core usage, so
  every buffer -- resident or spilled, the latter free of the slicing gate --
  takes its most parallel division, which the allocator commits. ``weight``
  exceeds the whole range of the core sum, which makes the collapse of the
  earlier two-phase lexicographic solve exact rather than approximate:
  *residency stays the hard priority*, since no amount of parallelism can pay for
  the smallest traffic increase. So the default still puts as much in LX as
  possible, choosing whatever division serves that (even no split, if that is
  what lets a buffer match its consumers and reside), and only then parallelises.

After the solve, ``_justify`` slides each in-place-merged placement unit down to
the lowest free address, squeezing out float gaps the search leaves. It coarsens
a merged unit to one rectangle over the union of its members' lifetimes, which is
conservative enough that the squeeze can occasionally need more room than the
solver's own answer; when it would not fit, the solver's offsets are kept.

The same model also serves plain :class:`LifetimeBoundBuffer`s via
``plan_layout`` (the ``MemoryPlanSolver`` contract the placement-only allocator
calls). Those buffers carry no candidate divisions, so the division-dependent
pieces -- per-core sizing, the slicing-match gate, the merge division gate and
the default objective's parallelism term -- simply drop out: the footprint is the
buffer's ``size`` and the solve reduces to minimising HBM traffic under the 2D
no-overlap with in-place reuse. Residency is then gated only by capacity and by
the allocator's own ``residency_reason`` bars (which both paths honour, since
that field lives on the base buffer). That specialisation
lives on the buffer wrappers (``_LifetimeBufferWithCpVars`` and its joint
subclass ``_CoreDivisionBufferWithCpVars``), so the solver methods below are
written once against whichever wrapper ``_wrap`` chose.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional, TypeVar, cast
import sympy
import torch
from torch.utils._sympy.value_ranges import ValueRanges


if TYPE_CHECKING:
    from ortools.sat.python import cp_model

    from torch_spyre._inductor.scratchpad.cost_expr_ortools import Val, lower
else:
    try:
        from ortools.sat.python import cp_model

        # Sibling module, but it imports ortools unguarded, so it belongs in the
        # same try: without ortools this module must still import (the friendly
        # error is raised from ``CpSatLayoutSolver.__init__``, not here).
        from torch_spyre._inductor.scratchpad.cost_expr_ortools import Val, lower

    except ImportError:  # pragma: no cover - exercised only when ortools is absent
        cp_model = None
        Val = lower = None

from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivisionBuffer,
    ceil_div,
    CoreDivisionLayoutSolver,
    LifetimeBoundBuffer,
    LifetimeBoundBufferWithSolverVars,
    SolveError,
    BufferType,
    _assert_in_place_relationships,
)

__all__ = ["CpSatLayoutSolver"]

logger = logging.getLogger(__name__)

# Drop cause for a buffer the solver chose to spill (rather than one pinned out
# up front by _add_core_division): it fit but residency gave no benefit, or
# there was no room once higher-value buffers were placed. Shared so the DEBUG
# log and the reasons surfaced to the allocator agree.
_SOLVER_CHOSE_SPILL = "spilled by solver (no residency benefit / no room)"

# Fixed-point scale for ``sym_inv_cores``. That symbol means the exact rational
# ``1 / cores_used``, which an integer variable cannot hold, so the variable
# carries ``_INV_CORES_SCALE // cores_used`` and every objective coefficient on
# the symbol is divided by the same scale on the way in (``_bind_symbols``), which
# leaves the product -- the only thing the objective reads -- unchanged. A power
# of two above the 32-core maximum: exact on the powers of two a work division
# actually produces, and under 0.2% quantisation error on the rest (worst case
# 1024/31 -> 33).
_INV_CORES_SCALE = 1024

# Buffer type the wrapper carries: the base placement wrapper holds any
# LifetimeBoundBuffer; the joint subclass binds this to CoreDivisionBuffer.
_BufT = TypeVar("_BufT", bound=LifetimeBoundBuffer)


@dataclass
class _PlacementUnit:
    """A connected component of in-place-merged buffers placed as one block."""

    members: list[str]
    footprint: int
    start_time: int
    end_time: int
    original_offset: int  # offset the solver chose, before bottom-justify
    justified_offset: int = 0  # final justified offset


def default_cost_expr(tensors: dict[str, _LifetimeBufferWithCpVars]) -> sympy.Expr:
    """The objective for a solve the caller gave no ``cost_expr``, written
    in the same symbols an injected one is -- so there is a single objective
    path and the default is simply the expression on it.

    It is the two-phase lexicographic solve this replaces (minimize HBM
    traffic; then, holding that optimum, maximize core usage) collapsed into
    one weighted expression, and the collapse is *exact* rather than a
    reweighting: traffic is integer-valued, so a weight above the whole span
    the core sum can swing across makes the smallest traffic increase cost
    more than every core gain available put together. Minimizing the sum
    therefore has precisely the lexicographic solve's optimum set --
    parallelism still can never buy a spill -- and each buffer's span comes
    from its own candidate divisions (``parallelism_term``), so a solve with
    nothing to parallelise weights the traffic by 1 and a solve with more to
    trade weights it by more.
    """
    parallelism = [sb.parallelism_term() for sb in tensors.values()]
    weight = sum(span for _, span in parallelism) + 1
    traffic = sympy.Add(*(sb.spill_term() for sb in tensors.values()))
    return weight * traffic + sympy.Add(*(term for term, _ in parallelism))


def _bind_symbols(
    tensors: dict[str, _LifetimeBufferWithCpVars],
    cost_expr: sympy.Expr,
) -> tuple[sympy.Expr, dict[sympy.Symbol, "Val"]]:
    """Build the symbol environment a cost expression is lowered against, and
    put the expression in the units that environment measures in.

    This is the whole of what the objective needs from *this* solver: which
    variable decides which symbol. Each buffer contributes the symbols for the
    decisions it exposes (``cost_symbols``), and only those, so an expression
    naming anything else is rejected by name rather than quietly interpreted.

    The one unit conversion is ``sym_inv_cores``: the symbol is the exact
    rational ``1 / cores_used`` but its variable holds
    ``_INV_CORES_SCALE // cores_used``, so substituting
    ``sym_inv_cores -> sym_inv_cores / _INV_CORES_SCALE`` moves the scale onto
    the coefficient standing next to it, where the integer rescale inside
    ``lower`` absorbs it. Doing it here rather than at the binding is what lets a
    cost model write ``work * sym_inv_cores`` and mean it: the fixed point is
    this solver's representation choice, not part of the objective's language.
    """
    wanted = cost_expr.free_symbols
    env: dict[sympy.Symbol, "Val"] = {}
    for sb in tensors.values():
        env.update(sb.cost_symbols(wanted))

    inv_scale = {
        sym: sym / _INV_CORES_SCALE for sym in env if sym.name.startswith("inv_cores_")
    }
    return cost_expr.xreplace(inv_scale), env


def _objective(
    model: "cp_model.CpModel",
    tensors: dict[str, _LifetimeBufferWithCpVars],
    cost_expr: sympy.Expr | None,
) -> tuple[object, str]:
    """The expression this solve minimizes, lowered onto ``model``, plus its
    name for the DEBUG summary.

    The caller's ``cost_expr`` when there is one, :func:`default_cost_expr`
    otherwise; from there the two are the same code, which is what makes the
    default an injected objective rather than a second mechanism beside one.
    Two steps: bind the model's decisions to the symbols the expression is
    written in (:func:`_bind_symbols`, the only solver-aware part), then hand
    expression and environment to :func:`~cost_expr_ortools.lower`, which owns
    the conversion itself and raises ``CostExpressionError`` on anything it
    cannot convert.

    An injected expression with no free symbols is the one case that falls back
    rather than raising: it is a constant, so minimizing it would leave the
    residency and division variables entirely unconstrained by any objective,
    and the solver would return the first feasible plan it found -- typically
    the all-spilled one. That is a cost model that failed to see any decision
    (every op's features came back concrete), not a request to pack arbitrarily,
    so the default objective takes over and the caller gets a warning. (The
    default itself may legitimately be constant -- no buffer with any traffic to
    save or division to choose -- and is then minimized as one, leaving the
    solve to return any feasible packing, which is all such a model can
    distinguish anyway.)
    """
    name = "injected_cost"
    if cost_expr is not None:
        cost_expr = sympy.sympify(cost_expr)
        if not cost_expr.free_symbols:
            logger.warning(
                "[CP-SAT layout solver] injected cost expression %s names no "
                "solver decision; falling back to the default objective",
                cost_expr,
            )
            cost_expr = None
    if cost_expr is None:
        name = "default_cost"
        cost_expr = default_cost_expr(tensors)

    cost_expr, env = _bind_symbols(tensors, cost_expr)
    root, analysis = lower(model, cost_expr, env)
    logger.debug(
        "[CP-SAT layout solver] %s objective: %d symbols, %d aux vars, scale=%d",
        name,
        len(env),
        len(analysis.aux),
        analysis.scale,
    )
    return root.e, name


def _gate_divisions(model, compatible, src_div, dst_div, enforce_lit) -> None:
    """Enforce, when ``enforce_lit`` is true, that ``(src_div, dst_div)`` is
    one of the ``compatible`` (i, j) pairs. With no compatible pairs the
    relation is unsatisfiable, so ``enforce_lit`` is forced false."""
    if not compatible:
        model.Add(enforce_lit == 0)
        return
    pair_lits = []
    for i, j in compatible:
        lit = model.NewBoolVar("")
        model.Add(src_div == i).OnlyEnforceIf(lit)
        model.Add(dst_div == j).OnlyEnforceIf(lit)
        pair_lits.append(lit)
    model.AddBoolOr(pair_lits).OnlyEnforceIf(enforce_lit)


@dataclass
class _LifetimeBufferWithCpVars(LifetimeBoundBufferWithSolverVars):
    """A :class:`LifetimeBoundBuffer` bundled with the CP-SAT variables the
    solver creates for it, so one object flows through the solve instead of a
    buffer list shadowed by a parallel ``name -> {var}`` dict.

    This is the *placement-only* wrapper backing :meth:`plan_layout`: the
    buffer's core division is already fixed upstream, so its footprint is the
    constant ``size`` (which the 2D no-overlap and capacity constraints accept
    wherever a var would go) and there is no division to choose. Every
    division-aware hook below is therefore a no-op or a fixed-size answer;
    :class:`_CoreDivisionBufferWithCpVars` overrides them to add the joint
    core-division model. Keeping the hooks on the wrapper is what lets ``_run``
    and its helpers serve both entry points unchanged.

    The buffer spans ``[buffer.start_time, buffer.end_time)``; the vars encode
    where (``offset``) and whether (``in_buffer``) it resides in LX.
    ``merge_vars`` maps each in-place parent name to the merge bool for that
    parent->this edge.

    CP-SAT variables must be created against a model, so this wrapper takes the
    model and the unit capacity ``M`` and creates only the variables here; the
    constraints tying them together are added by the solver methods."""

    model: "cp_model.CpModel"

    def __post_init__(self):
        b = self.buffer
        m = self.model
        M = self.capacity_units

        self.in_buffer = m.new_bool_var(f"in_buffer_{b.name}")
        # offset domain [0, M-1]; the resident => offset+eff_size<=M bound is
        # added in the in-place relaxation pass.
        self.offset = m.new_int_var(0, max(0, M - 1), f"off_{b.name}")
        # Fixed footprint -- no division to pick, so a constant stands in for
        # the joint solver's eff_size var.
        self.eff_size = b.size
        self.merge_vars = {
            parent: m.new_bool_var(f"merge_{parent}_{b.name}")
            for parent in b.in_place_parents
        }

    # -- producer/consumer edges (joint model only; none when division-fixed) --
    @property
    def parents(self) -> list[str]:
        return []

    def match_pairs(self, parent: str) -> list[tuple[int, int]]:
        return []

    # ------------------------------ residency ------------------------------
    def spill_cost(self) -> int:
        """Differential HBM traffic a spill adds over residency: the reads
        residency would have served from LX (``read_count``, computed by the
        allocator) plus the producer's write, which residency turns into a free
        LX write. A graph input has no producer write to save; a graph output's
        write-out is unavoidable either way, so it too cancels. Both cases are
        exactly ``boundary != Intermediate`` -- for a plain
        :class:`LifetimeBoundBuffer`, whose boundary is not tracked,
        ``first_use_is_read`` marks the same distinction for inputs."""
        b = self.buffer
        boundary = getattr(b, "boundary", None)
        is_intermediate = (
            boundary == BufferType.Intermediate
            if boundary is not None
            else not b.first_use_is_read
        )
        return (b.read_count + (1 if is_intermediate else 0)) * b.size

    def constrain_residency(self, model, kids, bufs) -> None:
        """Placement-only: any buffer may reside, so there is no slicing gate."""

    def constrain_merge(self, model, parent: "_LifetimeBufferWithCpVars", edge) -> None:
        """Extra conditions on an active in-place merge. None when the division
        is fixed: ``_assert_in_place_relationships`` already checks the child
        fits in the parent's slot."""

    # ------------------------------ objective ------------------------------
    def cost_symbols(self, wanted: set[sympy.Symbol]) -> dict[sympy.Symbol, "Val"]:
        """This buffer's slice of the objective's symbol environment: each
        symbol a cost expression may mention, bound to the solver variable that
        decides it and the range that variable lives in.

        Residency is the one decision every buffer exposes, so it is all the
        placement-only wrapper binds; the joint wrapper adds its division-derived
        scalars. A cost expression naming anything else is rejected by ``lower``
        against this environment, which is the point of building it per buffer:
        the objective can only speak about decisions the model actually makes.

        ``wanted`` is the set of symbols the expression actually names, for
        bindings that have to *build* something to offer (the joint wrapper's
        per-division scalars). Residency needs no such filter -- the variable
        already exists -- and binding it either way keeps every buffer's name in
        the "bound:" list an unbound-symbol error prints."""
        return {self.buffer.sym_is_lx: Val(self.in_buffer, ValueRanges(0, 1))}

    def spill_term(self) -> sympy.Expr:
        """This buffer's share of the default objective's traffic sum: the HBM
        traffic a spill adds over residency, charged only when it is spilled."""
        return self.spill_cost() * (1 - self.buffer.sym_is_lx)

    def parallelism_term(self) -> tuple[sympy.Expr, int]:
        """This buffer's share of the default objective's parallelism sum, and
        how much that share can vary between plans.

        There is nothing to parallelise without candidate divisions, so the
        placement-only wrapper contributes neither a term nor any span for the
        traffic weight to have to dominate. That is the explicit skip standing in
        for the ``sb.cores is None`` test the old phase 2 used: the wrapper that
        has no division to choose says so, rather than the solver inferring it
        from a null variable."""
        return sympy.Integer(0), 0

    # ------------------------------- extract -------------------------------
    def footprint(self, solver: "cp_model.CpSolver") -> int:
        return self.buffer.size

    def record_division(self, solver: "cp_model.CpSolver") -> None:
        """Write the chosen division back onto the buffer (nothing to record
        when the division is fixed)."""


@dataclass
class _CoreDivisionBufferWithCpVars(_LifetimeBufferWithCpVars):
    """The joint-model wrapper: a :class:`CoreDivisionBuffer` plus the vars for
    its chosen core division (``division``) and the per-core footprint that
    division implies (``eff_size``). Its core usage is division-derived too, but
    only an objective reads it, so it is bound on demand in
    :meth:`cost_symbols` rather than created up front.

    On top of the base placement vars it supplies the division-aware pieces of
    the model: the slicing-match residency gate, the division gate on an
    in-place merge, and the edge-counted spill cost. The ``buffer`` field is
    narrowed to :class:`CoreDivisionBuffer` via the base's type parameter."""

    def __post_init__(self):
        super().__post_init__()
        b = self.buffer
        m = self.model

        per_core = [ceil_div(b.size, cd.output_partition) for cd in b.core_divisions]
        self.division = m.new_int_var(0, len(b.core_divisions) - 1, f"div_{b.name}")
        self.eff_size = m.new_int_var(0, max(per_core), f"eff_size_{b.name}")

        # tie the per-core footprint (output split only) to the chosen division
        # index. Core usage is division-derived too, but only an objective reads
        # it, so it is bound on demand in ``cost_symbols`` rather than here.
        m.add_element(self.division, per_core, self.eff_size)

    @property
    def parents(self) -> list[str]:
        return self.buffer.parents

    def match_pairs(self, parent: str) -> list[tuple[int, int]]:
        return self.buffer.cd_parent_matches.get(parent, [])

    def constrain_residency(self, model, kids, bufs) -> None:
        """Slicing-consistency gate: a resident buffer's division must match
        *every* consumer's division under the ``cd_parent_matches`` pairs.

        This is the part of residency that genuinely depends on the solver's
        free variables, so it stays here as a constraint. The precomputable
        parts -- having no LX reader at all, or a consumer with no compatible
        pair -- are decided by the allocator and arrive as ``read_count`` /
        ``residency_reason``. A consumer with no compatible pair still lands
        correctly if it slips through: ``_gate_divisions`` forces ``in_buffer``
        false when the pair list is empty."""
        for child, compatible in kids:
            _gate_divisions(
                model, compatible, self.division, bufs[child].division, self.in_buffer
            )

    def _cores_table(self) -> list[int]:
        """Total cores the op runs on under each candidate division -- includes
        any reduction-axis split, so a reduction-parallel division counts its
        full parallelism (``output_partition`` alone would score it as 1 core)."""
        return [cd.cores_used for cd in self.buffer.core_divisions]

    def cost_symbols(self, wanted: set[sympy.Symbol]) -> dict[sympy.Symbol, "Val"]:
        """Residency, plus the core count the chosen division implies and the
        reciprocal of that count.

        Each is one ``AddElement`` over the same per-division table the rest of
        the model is indexed by, which is what keeps a symbol's cost to a single
        lookup (R3.7). ``sym_cores`` is the count itself; ``sym_inv_cores`` means
        the exact rational ``1 / cores_used``, so its table holds the fixed-point
        ``_INV_CORES_SCALE // cores_used`` and the scale comes back out of the
        objective's coefficients in :func:`_bind_symbols`.

        Built here, and only for the symbols the objective actually names, rather
        than in ``__post_init__``: they exist to be read by an objective, and a
        solve whose objective does not price parallelism should not carry a
        variable and an element constraint per buffer for nothing."""
        env = super().cost_symbols(wanted)
        b = self.buffer
        cores = self._cores_table()
        tables = {
            b.sym_cores: cores,
            b.sym_inv_cores: [_INV_CORES_SCALE // c for c in cores],
        }
        for sym, table in tables.items():
            if sym not in wanted:
                continue
            lo, hi = min(table), max(table)
            var = self.model.new_int_var(lo, hi, sym.name)
            self.model.add_element(self.division, table, var)
            env[sym] = Val(var, ValueRanges(lo, hi))
        return env

    def parallelism_term(self) -> tuple[sympy.Expr, int]:
        """Minus the core count of the chosen division -- the default objective
        minimizes, and more cores is better -- and the span between this
        buffer's least and most parallel candidate division, which is all its
        term can contribute to the swing the traffic weight has to outrank."""
        cores = self._cores_table()
        return -self.buffer.sym_cores, max(cores) - min(cores)

    def constrain_merge(self, model, parent, edge) -> None:
        """An active merge means the child reuses the parent's exact per-core
        storage, so their chosen divisions must have equal per-core footprints
        and must induce the same per-core slicing of that storage (the
        ``cd_parent_matches`` pairs; no pairs => merge forbidden)."""
        model.add(self.eff_size == parent.eff_size).OnlyEnforceIf(edge)
        _gate_divisions(
            model,
            self.match_pairs(parent.name),
            parent.division,
            self.division,
            edge,
        )

    def footprint(self, solver: "cp_model.CpSolver") -> int:
        t = self.buffer
        cd = t.core_divisions[solver.Value(self.division)]
        return ceil_div(t.size, cd.output_partition)

    def record_division(self, solver: "cp_model.CpSolver") -> None:
        self.buffer.chosen_division = solver.Value(self.division)


class CpSatLayoutSolver(CoreDivisionLayoutSolver):
    """Joint core-division + LX placement via an OR-Tools CP-SAT search
    (``config.layout_solver == "cpsat"``). See the module docstring for the
    model (joint division, slicing-match residency gate, 2D no-overlap with
    in-place lifetime shortening) and for the single-phase objective, injected
    by the caller or defaulted to traffic-over-parallelism.
    """

    def __init__(
        self,
        size: int,
        alignment: int = 128,
        time_limit_seconds: float = 600.0,
        bottom_justify: bool = True,
    ) -> None:
        if cp_model is None:
            raise ImportError(
                "The 'cpsat' layout solver requires the 'ortools' package, "
                "which is not installed. Install it with 'pip install ortools' "
                "or select a different layout_solver (e.g. 'greedy')."
            )
        super().__init__(size, alignment)
        # The solver works in alignment-sized units so every offset it picks is
        # automatically aligned; plan_layout scales sizes/offsets in and out.
        self._capacity_units = self.limit // self.alignment
        self._time_limit_seconds = time_limit_seconds
        self._bottom_justify = bottom_justify

    def plan_layout(
        self, buffers: Sequence[LifetimeBoundBuffer], log_lx_usage: bool = False
    ) -> list[LifetimeBoundBuffer]:
        """Place buffers on their already-fixed core divisions (placement-only).

        Same model as :meth:`plan_layout_and_core_divisions` minus the joint
        division choice: each buffer's footprint is its ``size``, so there is no
        slicing gate on residency and no parallelism phase -- the solve reduces
        to minimising HBM traffic under the 2D no-overlap with in-place reuse.
        Dispatch is per buffer and keys on whether it carries candidate
        divisions, not on its class, so a :class:`CoreDivisionBuffer` with an
        empty candidate list is placed here rather than divided."""
        return cast(
            "list[LifetimeBoundBuffer]", list(self._plan_layout_generic(buffers))
        )

    def plan_layout_and_core_divisions(
        self,
        buffers: Sequence[CoreDivisionBuffer],
        cost_expr: sympy.Expr | None = None,
    ) -> list[CoreDivisionBuffer]:
        """Jointly choose each buffer's core division and its LX placement.

        The full model described in the module docstring. Every buffer must
        carry enumerated candidate divisions; the chosen index is written back
        to ``chosen_division`` for the allocator to commit."""
        assert all(len(b.core_divisions) != 0 for b in buffers), (
            "All buffers must have at least 1 valid core division"
        )
        return cast(
            "list[CoreDivisionBuffer]",
            list(self._plan_layout_generic(buffers, cost_expr=cost_expr)),
        )

    def _wrap(
        self, model: "cp_model.CpModel", buffer: LifetimeBoundBuffer
    ) -> _LifetimeBufferWithCpVars:
        """Bundle a *copy* of ``buffer`` with its CP-SAT vars, scaled into the
        alignment units the solver works in.

        A buffer carrying enumerated core divisions gets the joint wrapper (its
        ``size`` is the total device footprint, divided down by the chosen
        division); anything else -- a plain :class:`LifetimeBoundBuffer`, or a
        :class:`CoreDivisionBuffer` with nothing to choose from -- gets the
        placement-only wrapper, whose footprint is ``size`` as given."""
        units = ceil_div(buffer.size, self.alignment)
        if isinstance(buffer, CoreDivisionBuffer) and buffer.core_divisions:
            return _CoreDivisionBufferWithCpVars(
                buffer=replace(buffer, size=units),
                capacity_units=self._capacity_units,
                model=model,
            )
        return _LifetimeBufferWithCpVars(
            buffer=replace(buffer, size=units),
            capacity_units=self._capacity_units,
            model=model,
        )

    def _plan_layout_generic(
        self,
        buffers: Sequence[LifetimeBoundBuffer | CoreDivisionBuffer],
        log_lx_usage: bool = False,
        cost_expr: sympy.Expr | None = None,
    ) -> list[LifetimeBoundBuffer | CoreDivisionBuffer]:
        if not buffers:
            return []
        assert all(b.address is None for b in buffers), (
            "Buffers cannot be previously or partially planned"
        )

        _assert_in_place_relationships(buffers)

        # Declarative exclusion, shared with every other solver: whatever the
        # allocator barred (each buffer's ``residency_reason``), plus the
        # no-LX-reader and capacity checks. Unlike the gap solvers -- which
        # ``partition`` these out -- we still hand the barred buffers to the
        # model (they must stay available for slicing matching and in-place
        # chains) but pin them non-resident below, so we only need the reasons.
        forced_reasons = dict(self.record_exclusions(buffers))

        model = cp_model.CpModel()
        # Solve on copies so we never mutate the caller's buffers.
        working = {b.name: self._wrap(model, b) for b in buffers}

        solved = self._run(model, working, forced_reasons, cost_expr=cost_expr)
        # Surface a drop cause for every spilled buffer: the pre-solve forced
        # reason when we have one, otherwise the solver chose to spill it.
        self.spill_reasons = {
            name: forced_reasons.get(name, _SOLVER_CHOSE_SPILL)
            for name, sb in solved.items()
            if sb.address is None
        }

        # Copy the solved results back onto the caller's buffers. Offsets come
        # back in alignment units (the solver works in aligned units), so scale
        # the address to bytes on the way out.
        for b in buffers:
            sb = solved[b.name]
            b.address = None if sb.address is None else sb.address * self.alignment
            if isinstance(b, CoreDivisionBuffer) and isinstance(sb, CoreDivisionBuffer):
                b.chosen_division = sb.chosen_division
        return list(buffers)

    # ------------------------------------------------------------------
    # Model build + solve
    # ------------------------------------------------------------------
    def _run(
        self,
        model: "cp_model.CpModel",
        tensors: dict[str, _LifetimeBufferWithCpVars],
        forced_reasons: dict[str, str],
        cost_expr: sympy.Expr | None,
    ) -> dict[str, LifetimeBoundBuffer]:
        children_of = self._get_children(tensors)
        self._add_inplace_relaxation(model, tensors)
        self._add_core_division(model, tensors, children_of, forced_reasons)

        solver = cp_model.CpSolver()
        if self._time_limit_seconds:
            solver.parameters.max_time_in_seconds = float(self._time_limit_seconds)
        solver.parameters.num_search_workers = (
            1 if torch.are_deterministic_algorithms_enabled() else (os.cpu_count() or 1)
        )
        # Fixed seed so a given worker configuration is reproducible run-to-run.
        solver.parameters.random_seed = 0

        # R3.2: one objective, minimized in one phase -- no lexicographic
        # sequence and no per-phase locking. The caller's expression when there
        # is one, the solver's own default otherwise; both are lowered the same
        # way, so this is the single place a plan's objective is decided.
        objective, objective_name = _objective(model, tensors, cost_expr)
        model.minimize(objective)
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise SolveError("CP-SAT memory planner found no feasible plan")

        final_tensors = self._extract(solver, tensors)

        if logger.isEnabledFor(logging.DEBUG):
            spilled = [n for n, t in final_tensors.items() if t.address is None]
            logger.debug(
                "[CP-SAT layout solver] tensors=%d resident=%d %s=%d "
                "status=%s walltime=%.2f ms",
                len(tensors),
                len(tensors) - len(spilled),
                objective_name,
                round(solver.ObjectiveValue()),
                solver.StatusName(status),
                solver.WallTime() * 1e3,
            )
            # Per-buffer drop cause: a pre-solve forced reason when we have one,
            # otherwise the solver chose to spill it (residency gave no benefit,
            # or there was no room once higher-value buffers were placed).
            for name in sorted(spilled):
                logger.debug(
                    "[CP-SAT layout solver]   %s -> HBM: %s",
                    name,
                    forced_reasons.get(name, _SOLVER_CHOSE_SPILL),
                )

        return final_tensors

    def _add_inplace_relaxation(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _LifetimeBufferWithCpVars],
    ) -> None:
        """In-place reuse as a relaxation of the no-overlap constraint: each
        parent->child edge gets a merge bool that, when active, pins the pair to
        one shared base. Rather than lifting a pairwise no-overlap, an active
        merge *shortens the child's lifetime by the single handoff tick* it
        shares with the parent (``_assert_in_place_relationships`` guarantees the
        overlap is exactly that one tick): the two then become time-adjacent
        rectangles that may legally sit at the same offset under the global 2D
        no-overlap (see ``_add_no_overlap_2d``). Chains are induced transitively
        by the shared-offset equalities -- no merge groups, no path enumeration.
        The per-buffer ``merge_vars`` bools are read back in ``_extract`` to
        reconstruct placement units."""
        M = self._capacity_units

        # A storage slot is handed off linearly, so a buffer reuses at most one
        # parent and is reused by at most one child. ``incoming`` also drives the
        # lifetime shortening in ``_add_no_overlap_2d``.
        incoming: dict[str, list] = {}
        outgoing: dict[str, list] = {}
        for dst, c in bufs.items():
            for src, edge in c.merge_vars.items():
                src_v, dst_v = bufs[src], bufs[dst]
                # active merge => shared base and both endpoints resident
                model.add(src_v.offset == dst_v.offset).OnlyEnforceIf(edge)
                model.add_implication(edge, src_v.in_buffer)
                model.add_implication(edge, dst_v.in_buffer)
                # active merge => the child must be able to take over the
                # parent's exact storage (joint model: equal per-core footprints
                # under slicing-compatible divisions; nothing extra when the
                # division is fixed).
                dst_v.constrain_merge(model, src_v, edge)
                outgoing.setdefault(src, []).append(edge)
                incoming.setdefault(dst, []).append(edge)

        for ms in (*incoming.values(), *outgoing.values()):
            if len(ms) > 1:
                model.add_at_most_one(ms)

        for sb in bufs.values():
            # if a buffer is resident its top must be below the peak usage.
            model.add(sb.offset + sb.eff_size <= M).OnlyEnforceIf(sb.in_buffer)

        self._add_no_overlap_2d(model, bufs, incoming)

    def _add_no_overlap_2d(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _LifetimeBufferWithCpVars],
        incoming: dict[str, list],
    ) -> None:
        """Global 2D no-overlap: each resident buffer is an optional rectangle
        ``[start_time, end_time) x [offset, offset + eff_size)`` and no two may
        intersect (touching edges are allowed). Residency is the interval
        presence (``in_buffer``), so spilled buffers drop out for free.

        In-place reuse is handled *inside* this constraint rather than by
        relaxing it: an active incoming merge shortens the child's time interval
        by the single handoff tick it shares with the parent
        (``start -> start + 1``). The parent and child then abut in time at the
        same offset (pinned equal by the merge), which the 2D constraint accepts
        as non-overlapping -- so the child legally reuses the parent's slot. With
        no active merge the child keeps its full lifetime and the shared-offset
        placement is correctly forbidden, exactly as the pairwise encoding did.

        It is the *child* that gives up the tick, never the parent: the parent's
        rectangle has to keep covering the handoff tick at full footprint. The
        child may be smaller than its parent (the placement-only model only
        requires ``child.size <= parent.size``), and the bytes above the child
        are still holding parent data that is read on that tick, so they are not
        free for a third buffer. Shortening the parent instead would expose them
        -- and a parent whose whole lifetime is that one tick would drop out of
        the propagator entirely, exposing its full slot.

        ``AddAtMostOne`` on the incoming edges bounds the shortening at one tick.
        A child whose entire lifetime is the handoff tick degenerates to a
        zero-width box the 2D propagator ignores, which is safe here: the tick is
        covered by the parent's box, whose footprint contains the child's at the
        shared offset."""
        x_intervals = []
        y_intervals = []
        for sb in bufs.values():
            ins = incoming.get(sb.name, [])
            if ins:
                # at most one incoming merge is active (AddAtMostOne), so the
                # sum is 0 or 1: shorten the child by the handoff tick exactly
                # when it takes over a parent's slot.
                start_var = model.new_int_var(
                    sb.start_time, sb.end_time, f"start_{sb.name}"
                )
                model.add(start_var == sb.start_time + sum(ins))
                x_start: object = start_var
                x_size: object = sb.end_time - start_var
            else:
                x_start = sb.start_time
                x_size = sb.end_time - sb.start_time
            x_intervals.append(
                model.new_optional_interval_var(
                    x_start, x_size, sb.end_time, sb.in_buffer, f"x_{sb.name}"
                )
            )
            # An interval's ``end`` must be affine (a single var), so the address
            # top ``offset + eff_size`` (a sum of two vars) needs its own var; the
            # interval ties it to start+size whenever the buffer is resident.
            y_end = model.new_int_var(0, self._capacity_units, f"top_{sb.name}")
            y_intervals.append(
                model.new_optional_interval_var(
                    sb.offset,
                    sb.eff_size,
                    y_end,
                    sb.in_buffer,
                    f"y_{sb.name}",
                )
            )
        model.add_no_overlap_2d(x_intervals, y_intervals)

    def _get_children(
        self, bufs: dict[str, _LifetimeBufferWithCpVars]
    ) -> dict[str, list[tuple[str, list[tuple[int, int]]]]]:
        """parent name -> list of (child name, match_pairs), where ``match_pairs``
        is the child's ``cd_parent_matches[parent]`` (empty when the edge has no
        compatible division). The child's ``parents`` define the edges; a
        placement-only buffer declares none, so the map is empty there."""
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]] = {}
        for sb in bufs.values():
            for parent in sb.parents:
                children_of.setdefault(parent, []).append(
                    (sb.name, sb.match_pairs(parent))
                )
        return children_of

    def _add_core_division(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _LifetimeBufferWithCpVars],
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
        forced: dict[str, str],
    ) -> None:
        """Pin out every buffer ``forced`` non-resident (decided declaratively by
        :meth:`MemoryPlanSolver.partition`) and install the per-buffer residency
        gate. In the joint model that gate is the slicing match, driven entirely
        by the precomputed ``cd_parent_matches`` pairs; placement-only buffers
        have no gate."""
        for name in forced:
            model.add(bufs[name].in_buffer == 0)
        for sb in bufs.values():
            sb.constrain_residency(model, children_of.get(sb.name, []), bufs)

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------
    def _extract(
        self,
        solver: "cp_model.CpSolver",
        bufs: dict[str, _LifetimeBufferWithCpVars],
    ) -> dict[str, LifetimeBoundBuffer]:
        """Read the solution back onto each buffer and return ``name -> buffer``.

        Every buffer gets its ``chosen_division`` (a no-op for a placement-only
        buffer, whose division was fixed upstream) and, when resident, its LX
        ``address`` (in alignment units, as the solver works them; the caller
        scales to bytes). A spilled buffer gets ``address = None``. When
        bottom_justify is set, each in-place-merged placement unit is slid down
        to the lowest free address (preserving merges); if that squeeze cannot
        keep every unit inside capacity the solver's own offsets are kept, since
        those are always legal."""
        by_name = {name: sb.buffer for name, sb in bufs.items()}
        spilled = {
            name for name, sb in bufs.items() if not solver.BooleanValue(sb.in_buffer)
        }
        footprint = {name: sb.footprint(solver) for name, sb in bufs.items()}

        offsets: Optional[dict[str, int]] = None
        if self._bottom_justify:
            # A placement unit is a connected component of active merge edges: its
            # members share one base (the merge equalities), so the component
            # slides as a single block and in-place reuse is preserved.
            resident = [n for n in by_name if n not in spilled]
            parent = {n: n for n in resident}

            def find(x: str) -> str:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for dst, c in bufs.items():
                for src, edge in c.merge_vars.items():
                    if solver.BooleanValue(edge):
                        parent[find(src)] = find(dst)

            components: dict[str, list[str]] = {}
            for n in resident:
                components.setdefault(find(n), []).append(n)

            units = [
                _PlacementUnit(
                    members=names,
                    footprint=max(footprint[n] for n in names),
                    start_time=min(by_name[n].start_time for n in names),
                    end_time=max(by_name[n].end_time for n in names),
                    original_offset=solver.Value(bufs[names[0]].offset),
                )
                for names in components.values()
            ]
            offsets = self._justify(units, self._capacity_units)

        if offsets is None:
            offsets = {
                name: solver.Value(sb.offset)
                for name, sb in bufs.items()
                if name not in spilled
            }

        for name, sb in bufs.items():
            t = sb.buffer
            sb.record_division(solver)
            if name in spilled:
                t.address = None
            else:
                t.address = offsets[name]
        return by_name

    @staticmethod
    def _justify(
        units: list[_PlacementUnit], capacity: int
    ) -> Optional[dict[str, int]]:
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
