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

"""What the allocator solves with (Phase 3a, 3d, 3g): a graph solver. It
fills every open decision of a :class:`SpyreOptimizationGraph`, either by
posting the graph through the strategy to a backend
(``ilp_solver_ortools.ModelSolver``) or by reading the nodes itself.

Each declares its :class:`Capabilities`, which the allocator, the candidate
generator and the enumeration factory read; an impossible combination is
refused when it is declared. :class:`PlacementSolver` runs a placement
algorithm (greedy, first-fit, best-fit, simulated annealing) over each node's
*placement view* -- name, size in bytes, uses, in-place parents and the
transforms pinned with it. :class:`DecomposedPlacement` is the exhaustive
search over the open nodes' menus. :class:`BufferSolver` runs the annealing
co-optimizer over the buffer form of the graph and candidates.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Optional

import sympy

from torch_spyre._inductor.scratchpad.buffer_graph import (
    SOLVER_CHOSE_SPILL,
    ResourceNode,
    SpyreOptimizationGraph,
    Transform,
    WorkSplit,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivisionBuffer,
    CoreDivisionLayoutSolver,
    LifetimeBoundBuffer,
    MemoryPlanSolver,
    ceil_div,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Capabilities:
    """What a solver can realize, asked before any solve.

    - ``chooses_core_divisions``: chooses each node's core division jointly
      with its placement; the open division mode needs it.
    - ``supports_paired_buffers``: places a committed relayout's source and
      destinations all-or-nothing; only such a solver gets relayout plans.
    - ``decides_lx_relayouts``: places candidate relayout copies, so the
      candidate generator offers them.
    - ``generates_divisions``: generates divisions from an op's split space
      (``node.space``) instead of only choosing among the menu; with
      ``chooses_tilings`` the allocator applies the tilings it picks.
    - ``chooses_tilings``: chooses coarse tilings jointly with divisions.
    - ``tilings_from_menu``: takes tilings off the menu (tiled candidates).
    - ``replans_after_tiling``: needs a second, placement solve once its
      tilings are applied.
    - ``linear_cost_only``: minimizes only a cost expression that lowers to
      linear terms; withheld under ``auto_coarse_tiling``.
    - ``posts_model``: encodes the graph through a strategy, so it realizes
      every enumeration the factory allows; the others read the table model.
    """

    chooses_core_divisions: bool = False
    supports_paired_buffers: bool = False
    decides_lx_relayouts: bool = False
    generates_divisions: bool = False
    chooses_tilings: bool = False
    tilings_from_menu: bool = False
    replans_after_tiling: bool = False
    linear_cost_only: bool = False
    posts_model: bool = False

    def __post_init__(self) -> None:
        for implied, prerequisite in (
            ("decides_lx_relayouts", "chooses_core_divisions"),
            ("generates_divisions", "chooses_core_divisions"),
            ("chooses_tilings", "chooses_core_divisions"),
            ("linear_cost_only", "chooses_core_divisions"),
            ("tilings_from_menu", "chooses_tilings"),
            ("replans_after_tiling", "chooses_tilings"),
        ):
            if getattr(self, implied) and not getattr(self, prerequisite):
                raise ValueError(f"{implied} requires {prerequisite}")


class GraphSolver:
    """Fills every open decision of a graph: ``solve`` takes the strategy
    holding the graph's candidates and the priced cost expression (``None``:
    the solver's own objective). The stats and the off-expression terms
    describe the last solve, for the cost-expression dump."""

    capabilities = Capabilities()

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity  # bytes
        self.last_solve_stats: dict = {}

    @property
    def name(self) -> str:
        return type(self).__name__

    def accepts(self, strategy: Any) -> bool:
        return True

    def solve(
        self,
        graph: SpyreOptimizationGraph,
        strategy: Any,
        cost_expr: Optional[sympy.Expr] = None,
    ) -> list[ResourceNode]:
        raise NotImplementedError

    def off_expression_ns(self) -> dict[str, float]:
        """The terms the objective adds outside ``cost_expr`` for the solved
        plan, in ns."""
        return {}

    def score_ns(self) -> Optional[float]:
        """The total minimized for the solved plan, in ns, where that is not
        ``cost_expr`` alone."""
        return None


def placement_view(item: ResourceNode | Transform) -> LifetimeBoundBuffer:
    """What a placement-only solver reads about a node or a pinned transform:
    name, size in bytes, uses (with the lifetime it was given), in-place
    parents and residency reason."""
    uses = list(item.uses)
    nominal_start, nominal_end = uses[0], uses[-1] + 1
    in_place = (
        [e.src.name for e in item.inputs if e.in_place]
        if isinstance(item, ResourceNode)
        else []
    )
    return LifetimeBoundBuffer(
        item.name,
        item.nbytes,
        uses,
        first_use_is_read=getattr(item, "first_use_is_read", False),
        in_place_parents=in_place,
        residency_reason=item.residency_reason,
        lifetime_start_override=(
            item.start_time if item.start_time < nominal_start else None
        ),
        lifetime_end_override=item.end_time if item.end_time > nominal_end else None,
        lx_view=item.lx_view,
    )


def placement_views(graph: SpyreOptimizationGraph) -> list[LifetimeBoundBuffer]:
    """Every node's placement view in ``add_node`` order, each pinned
    transform just before its source, which carries them as its paired
    members."""
    views: list[LifetimeBoundBuffer] = []
    for node in graph.get_nodes():
        pinned = [placement_view(t) for t in node.pinned_transforms]
        view = placement_view(node)
        view.paired_with = pinned
        view.lx_relayout_plans = list(node.lx_relayout_plans)
        views.extend(pinned)
        views.append(view)
    return views


def _write_back(
    item: ResourceNode | Transform, address: Optional[int], spill: Optional[str], unit
) -> None:
    item.resident = address is not None
    if address is not None:
        assert address % unit == 0, f"{item.name} placed off alignment: {address}"
        item.offset = address // unit
        item.spill_reason = None
    else:
        item.offset = None
        item.spill_reason = spill or item.residency_reason or SOLVER_CHOSE_SPILL


class PlacementSolver(GraphSolver):
    """Placement only: every node's work split is committed upstream, so the
    placement algorithm ``inner`` packs the nodes' placement views."""

    def __init__(self, inner: type[MemoryPlanSolver], capacity: int) -> None:
        super().__init__(capacity)
        self.inner = inner
        self.last: list[LifetimeBoundBuffer] = []
        self.capabilities = Capabilities(supports_paired_buffers=inner.pairs_buffers)

    @property
    def name(self) -> str:
        return self.inner.__name__

    def accepts(self, strategy: Any) -> bool:
        return strategy.relayout != "lazy"

    def solve(
        self,
        graph: SpyreOptimizationGraph,
        strategy: Any,
        cost_expr: Optional[sympy.Expr] = None,
    ) -> list[ResourceNode]:
        open_ = [n.name for n in graph.get_nodes() if not n.committed]
        assert not open_, f"{self.inner.__name__} cannot choose work splits: {open_}"
        views = placement_views(graph)
        planned = self.inner(views, self.capacity, graph.alignment).plan_layout(
            log_lx_usage=True
        )
        by_name = {v.name: v for v in planned}
        for node in graph.get_nodes():
            items: list[ResourceNode | Transform] = [node, *node.pinned_transforms]
            for t in items:
                v = by_name[t.name]
                _write_back(t, v.address, v.spill_reason, graph.alignment)
        self.last = planned
        return graph.execution_plan()


def buffers_of(graph: SpyreOptimizationGraph, tables: Any) -> list[CoreDivisionBuffer]:
    """The buffer form of the graph's nodes and the table strategy's
    candidates: what a solver that still reads buffers (the annealing
    co-optimizer) is handed. Each node's reads keep their count, and its
    lifetime its overrides."""
    out = []
    for node in graph.get_nodes():
        uses = list(node.uses)
        keys = [(e.src.name, node.name) for e in node.inputs]
        b = CoreDivisionBuffer(
            name=node.name,
            size=node.nbytes,
            uses=uses,
            first_use_is_read=node.first_use_is_read,
            in_place_parents=[e.src.name for e in node.inputs if e.in_place],
            core_divisions=list(tables.menus.get(node.name, ())),
            parents=[e.src.name for e in node.inputs for _ in range(e.reads)],
            cd_parent_matches={
                k[0]: tables.matches[k] for k in keys if k in tables.matches
            },
            cd_parent_relayouts={
                k[0]: tables.relayouts[k] for k in keys if k in tables.relayouts
            },
            residency_edges={
                k[0]: tables.residency[k] for k in keys if k in tables.residency
            },
            residency_reason=node.residency_reason,
            lifetime_start_override=node.start_time
            if node.start_time < uses[0]
            else None,
            lifetime_end_override=(
                node.end_time if node.end_time > uses[-1] + 1 else None
            ),
            op_position=tables.positions.get(node.name),
            boundary=node.boundary,
        )
        b.division_space = node.space
        b.tile_reads = dict(tables.tile_reads.get(node.name, {}))
        out.append(b)
    return out


class BufferSolver(GraphSolver):
    """The annealing co-optimizer, which reads buffers: it solves the buffer
    form of the graph's nodes and the strategy's candidates, and fills the
    nodes from what it chose."""

    capabilities = Capabilities(
        chooses_core_divisions=True, generates_divisions=True, chooses_tilings=True
    )

    def __init__(self, inner: type[CoreDivisionLayoutSolver], capacity: int) -> None:
        super().__init__(capacity)
        self.inner = inner
        self.last: list[LifetimeBoundBuffer] = []
        self._engine: Optional[CoreDivisionLayoutSolver] = None

    @property
    def name(self) -> str:
        return self.inner.__name__

    def accepts(self, strategy: Any) -> bool:
        return strategy.division == "table" and strategy.relayout == "data"

    def off_expression_ns(self) -> dict[str, float]:
        return {} if self._engine is None else self._engine.off_expression_ns()

    def score_ns(self) -> Optional[float]:
        return None if self._engine is None else self._engine.score_ns()

    def solve(
        self,
        graph: SpyreOptimizationGraph,
        strategy: Any,
        cost_expr: Optional[sympy.Expr] = None,
    ) -> list[ResourceNode]:
        solver = self.inner(
            buffers_of(graph, strategy.tables), self.capacity, graph.alignment
        )
        solver.plan_layout_and_core_divisions(cost_expr)
        self._engine = solver
        self.last = list(solver.buffers)
        by_name = {b.name: b for b in self.last}
        # A solver that generates divisions appends the ones it chose that the
        # menu lacked (a generated tiling); the chosen index is into that menu.
        for b in self.last:
            menu = tuple(b.core_divisions)
            if len(menu) > len(strategy.tables.menus.get(b.name, ())):
                strategy.tables.menus[b.name] = menu
                strategy.menus[b.name] = menu
        for node in graph.get_nodes():
            b = by_name[node.name]
            _write_back(node, b.address, b.spill_reason, graph.alignment)
            if node.work_split is None and b.core_divisions:
                assert b.chosen_division is not None
                node.work_split = WorkSplit.of(b.core_divisions[b.chosen_division])
                strategy.solved[node] = b.chosen_division
        return graph.execution_plan()


class DecomposedPlacement(GraphSolver):
    """Enumerates the ``Table`` axes (each open node's candidate menu) and
    solves every leaf's placement with an inner placement-only solver,
    keeping the leaf with the least HBM traffic (the total bytes of the
    nodes left out of LX): ``backend.Decomposed`` with a placement heuristic
    inside, which is the exhaustive search. A leaf bars a producer whose
    division no consumer's matches, and drops an in-place parent smaller per
    core than its child."""

    capabilities = Capabilities(chooses_core_divisions=True)

    def __init__(self, inner: type[MemoryPlanSolver], capacity: int) -> None:
        super().__init__(capacity)
        self.inner = inner
        self.last: list[LifetimeBoundBuffer] = []

    @property
    def name(self) -> str:
        return f"ExhaustiveSearchSolver[{self.inner.__name__}]"

    def accepts(self, strategy: Any) -> bool:
        return hasattr(strategy, "menus") and not getattr(strategy, "residency", {})

    def solve(
        self,
        graph: SpyreOptimizationGraph,
        strategy: Any,
        cost_expr: Optional[sympy.Expr] = None,
    ) -> list[ResourceNode]:
        nodes = list(graph.get_nodes())
        menus = {n.name: strategy.menus.get(n.name, ()) for n in nodes}
        if any(not cd.tiling.is_untiled for m in menus.values() for cd in m):
            raise ValueError("Exhaustive search requires untiled candidates")
        chosen = {n.name: 0 for n in nodes}
        variable = [n for n in nodes if len(menus[n.name]) > 1]

        def per_core(node: ResourceNode, i: int) -> int:
            menu = menus[node.name]
            if not menu:
                return node.nbytes
            cd = menu[i]
            return ceil_div(
                node.nbytes, cd.output_partition * cd.tiling.output_tile_count
            )

        def reason(node: ResourceNode) -> Optional[str]:
            if node.residency_reason is not None:
                return node.residency_reason
            for edge in node.outputs:
                if not edge.reads:
                    continue
                pairs = strategy._pairs(edge)
                pair = (chosen[node.name], chosen[edge.dst.name])
                if pair not in pairs:
                    return "core div mismatch"
            return None

        def views() -> list[LifetimeBoundBuffer]:
            out = []
            for node in nodes:
                view = placement_view(node)
                size = per_core(node, chosen[node.name])
                view.size = size
                view.residency_reason = reason(node)
                view.in_place_parents = [
                    p
                    for p in view.in_place_parents
                    if size <= per_core(graph.nodes[p], chosen[p])
                ]
                out.append(view)
            return out

        best_total = math.inf
        best = dict(chosen)

        def score() -> int:
            allocation = self.inner(views(), self.capacity).plan_layout()
            pinned = {a.name for a in allocation if a.address is not None}
            return sum(n.nbytes for n in nodes if n.name not in pinned)

        def recurse(k: int) -> None:
            nonlocal best_total, best
            if k == len(variable):
                hbm = score()
                if hbm < best_total:
                    best_total, best = hbm, dict(chosen)
                return
            node = variable[k]
            prev = chosen[node.name]
            for i in range(len(menus[node.name])):
                chosen[node.name] = i
                recurse(k + 1)
            chosen[node.name] = prev

        t = time.perf_counter()
        recurse(0)
        logger.info(
            "co-opt search: %d paths in %.1fms; winner=%s",
            math.prod(max(1, len(m)) for m in menus.values()),
            (time.perf_counter() - t) * 1e3,
            {
                n.name: menus[n.name][best[n.name]].label
                for n in variable
                if best[n.name]
            },
        )
        chosen.update(best)
        self.last = self.inner(views(), self.capacity).plan_layout()
        final = {v.name: v for v in self.last}
        for node in nodes:
            if node.work_split is None and menus[node.name]:
                node.work_split = WorkSplit.of(menus[node.name][chosen[node.name]])
            v = final[node.name]
            _write_back(node, v.address, v.spill_reason, graph.alignment)
        strategy.solved.update({n: chosen[n.name] for n in nodes if menus[n.name]})
        return graph.execution_plan()


def graph_solver_for(solver: Any, capacity: int) -> GraphSolver:
    """The graph solver a solver class names (a graph solver is returned as
    given): ``CpSatLayoutSolver`` posts a model, ``ExhaustiveSearchSolver
    .wrapping(placement)`` searches the menus, another joint solver reads
    buffers, and a placement algorithm packs placement views."""
    if isinstance(solver, GraphSolver):
        return solver
    from torch_spyre._inductor.scratchpad.exhaustive_search import (
        ExhaustiveSearchSolver,
    )
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
        CpSatLayoutSolver,
        ModelSolver,
    )

    if issubclass(solver, CpSatLayoutSolver):
        return ModelSolver(capacity)
    if issubclass(solver, ExhaustiveSearchSolver):
        return DecomposedPlacement(solver._inner_factory, capacity)
    if issubclass(solver, CoreDivisionLayoutSolver):
        return BufferSolver(solver, capacity)
    return PlacementSolver(solver, capacity)


def _unused(*args, **kwargs) -> int:
    raise AssertionError("a graph solver posts no model")


def solve_buffers(
    solver: MemoryPlanSolver, cost_expr: Optional[sympy.Expr] = None
) -> list:
    """The buffer form of a graph solver (tests, capture replay): the graph
    of ``solver``'s buffers, solved by the graph solver its class names, the
    decisions written back onto the buffers. Raises ``SolveError`` as the
    solve does."""
    spec = graph_solver_for(type(solver), solver.limit)
    from torch_spyre._inductor.scratchpad.buffer_graph import SolveSettings, build_graph

    graph = build_graph(
        solver.buffers,
        solver.alignment,
        solver.limit // solver.alignment,
        SolveSettings(time_limit_seconds=None, bottom_justify=False, record=_unused),
        # The allocator's reasons; the capacity check is the solver's own.
        {b.name: b.residency_reason for b in solver.buffers if b.residency_reason},
    )
    from torch_spyre._inductor.scratchpad.encoding_strategy import (
        EnumerationFactory,
    )

    strategy = EnumerationFactory().strategy(solver.buffers)
    spec.solve(graph, strategy, cost_expr)
    by_name = {b.name: b for b in solver.buffers}
    # The inner solver's own reasons: none for a buffer that did not fit.
    last = getattr(spec, "last", [])
    reasons = {v.name: v.spill_reason for v in last}
    for node in graph.get_nodes():
        items: list[ResourceNode | Transform] = [node, *node.pinned_transforms]
        for item in items:
            b = by_name[item.name]
            b.address = None if item.offset is None else item.offset * graph.alignment
            b.spill_reason = reasons.get(item.name)
        if node in strategy.solved:
            by_name[node.name].chosen_division = strategy.solved[node]
    return list(solver.buffers)
