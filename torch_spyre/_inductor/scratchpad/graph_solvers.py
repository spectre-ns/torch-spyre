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
refused when it is declared. :class:`BufferSolver` runs the annealing
co-optimizer over the buffer form of the graph and candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional, TYPE_CHECKING

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
    LifetimeBoundBuffer,
)

if TYPE_CHECKING:
    from torch_spyre._inductor.scratchpad.sa_cooptimizer import SaCoOptimizingSolver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Capabilities:
    """What a solver can realize, asked before any solve.

    - ``chooses_core_divisions``: chooses each node's core division jointly
      with its placement; the open division mode needs it.
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
    the solver's own objective) and returns every node, its decisions on it.
    The stats and the off-expression terms describe the last solve, for the
    cost-expression dump."""

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


def buffers_of(nodes: Iterable[ResourceNode], tables: Any) -> list[CoreDivisionBuffer]:
    """The buffer form of ``nodes`` and the table strategy's candidates: what a solver that still reads buffers (the annealing
    co-optimizer) is handed. Each node's reads keep their count, and its
    lifetime its overrides."""
    out = []
    for node in nodes:
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
            # Its place on the chain of execution edges.
            op_position=node.produced_at,
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

    def __init__(self, inner: type[SaCoOptimizingSolver], capacity: int) -> None:
        super().__init__(capacity)
        self.inner = inner
        self.last: list[LifetimeBoundBuffer] = []
        self._engine: Optional[SaCoOptimizingSolver] = None

    @property
    def name(self) -> str:
        return self.inner.__name__

    def accepts(self, strategy: Any) -> bool:
        from torch_spyre._inductor.scratchpad.lazy_strategy import LazyStrategy

        return not isinstance(strategy, LazyStrategy)

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
            buffers_of(graph.get_nodes(), strategy.tables),
            self.capacity,
            graph.alignment,
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
        return list(graph.get_nodes())
