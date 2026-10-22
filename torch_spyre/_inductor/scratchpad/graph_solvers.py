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

"""Graph solvers (Phase 3a, 3d): solvers that work on the graph directly
instead of a posted model. ``SpyreOptimizationGraph.solve`` runs one set with
``set_solver`` in place of encoding and the objective ladder.

:class:`PlacementSolver` runs a placement-only solver (greedy, first-fit,
best-fit, simulated annealing) over each node's *placement view* -- name, size
in bytes, uses, in-place parents and the transforms pinned with it -- and
writes the decisions back onto the nodes. :class:`BufferSolver` runs a joint
solver that still reads the allocator's buffers (the simulated-annealing
co-optimizer, exhaustive search) and fills the nodes from what it chose.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

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
)


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


class PlacementSolver:
    """A placement-only solver as a graph solver: every node's work split is
    committed upstream, so it packs the nodes' placement views."""

    def __init__(self, inner: type[MemoryPlanSolver], capacity: int) -> None:
        self.inner = inner
        self.capacity = capacity  # bytes
        self.last: list[LifetimeBoundBuffer] = []

    def accepts(self, strategy: Any) -> bool:
        return strategy.relayout != "lazy"

    def solve(
        self,
        graph: SpyreOptimizationGraph,
        strategy: Any,
        cost_expr: Optional[sympy.Expr],
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


class BufferSolver:
    """A joint solver that reads the allocator's buffers (the annealing
    co-optimizer, exhaustive search) as a graph solver: it solves its own
    buffers and fills the graph's nodes from the choices it wrote back."""

    def __init__(self, solver: CoreDivisionLayoutSolver) -> None:
        self.solver = solver
        self.result: Sequence[Any] = ()

    def accepts(self, strategy: Any) -> bool:
        return strategy.division == "table" and strategy.relayout == "data"

    def solve(
        self,
        graph: SpyreOptimizationGraph,
        strategy: Any,
        cost_expr: Optional[sympy.Expr],
    ) -> list[ResourceNode]:
        self.result = self.solver.plan_layout_and_core_divisions(cost_expr)
        by_name = {b.name: b for b in self.solver.buffers}
        for node in graph.get_nodes():
            b = by_name[node.name]
            _write_back(node, b.address, b.spill_reason, graph.alignment)
            if node.work_split is None and isinstance(b, CoreDivisionBuffer):
                node.work_split = (
                    WorkSplit.of(b.core_divisions[b.chosen_division])
                    if b.core_divisions and b.chosen_division is not None
                    else WorkSplit.COMMITTED
                )
        return graph.execution_plan()


def _unused(*args, **kwargs) -> int:
    raise AssertionError("a graph solver posts no model")


def solve_buffers(
    solver: MemoryPlanSolver,
    cost_expr: Optional[sympy.Expr] = None,
    *,
    joint: bool = False,
) -> list:
    """The one way the allocator solves: the graph of ``solver``'s buffers,
    solved with what the solver names (``graph_solver``), its decisions
    written back onto the buffers. A backend encodes the graph through a
    strategy, which the solver's own facade sets up; a graph solver gets the
    graph built here. Raises ``SolveError`` as the solve does."""
    spec = solver.graph_solver()
    if isinstance(spec, type):
        if joint:
            assert isinstance(solver, CoreDivisionLayoutSolver)
            return solver.plan_layout_and_core_divisions(cost_expr)
        return solver.plan_layout(log_lx_usage=True)
    from torch_spyre._inductor.scratchpad.buffer_graph import SolveSettings, build_graph

    graph = build_graph(
        solver.buffers,
        solver.alignment,
        solver.limit // solver.alignment,
        SolveSettings(time_limit_seconds=None, bottom_justify=False, record=_unused),
        # The allocator's reasons; the capacity check is the solver's own.
        {b.name: b.residency_reason for b in solver.buffers if b.residency_reason},
    )
    graph.set_solver(spec)
    graph.solve(cost_expr=cost_expr)
    if isinstance(spec, BufferSolver):
        return list(spec.result)
    by_name = {b.name: b for b in solver.buffers}
    for node in graph.get_nodes():
        items: list[ResourceNode | Transform] = [node, *node.pinned_transforms]
        for item in items:
            b = by_name[item.name]
            b.address = None if item.offset is None else item.offset * graph.alignment
            b.spill_reason = item.spill_reason if not item.resident else None
    return list(solver.buffers)
