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

:class:`ModelSolver` is the graph solver the allocator configures for
``layout_solver="cpsat"``. The graph (``buffer_graph.py``) has one :class:`ResourceNode` per op output or graph input,
a :class:`DependencyEdge` per (producer, consumer) read (``in_place`` when the
consumer may take over the producer's storage), and :class:`ExecutionEdge`\\ s
chaining op outputs in execution order. Each node's decisions -- its work
split (core division and coarse tiling), whether it is resident in LX, and its
offset -- are open for the solve to fill. ``graph.solve`` encodes the graph
through an :class:`EncodingStrategy` (``encoding_strategy.py``), runs the
objective ladder and fills the decisions.

The model the default (table) strategy encodes:

* **Joint core-division.** A node's ``size`` is its *total* device footprint;
  a ``div`` var indexes its candidate divisions and element constraints tie
  the chosen index to the per-core footprint (``eff_size = size /
  output_partition``, further divided by any coarse tiling) and total core
  usage (``cores = cores_used``, including any reduction-axis split). A node
  whose division was committed upstream (a plain :class:`LifetimeBoundBuffer`,
  or a buffer with no candidates) has a fixed footprint and no division.
* **Slicing-match residency gate.** A resident producer's division must induce
  the same per-core slicing as *every* consumer's (the ``cd_parent_matches``
  pairs over each read edge), or the consumer must read it through a resident
  relayout copy. A missing pair list forbids residency across that edge.
* **Placement** is a global 2-D no-overlap over optional rectangles
  (``[start_time, end_time) x [offset, offset + eff_size)``, present iff
  resident). In-place reuse shortens the child's lifetime by the single
  handoff tick when its merge fires, so parent and child abut in time and may
  share an offset; ``_check_in_place_relationships`` guarantees the overlap is
  exactly that tick.
* **Objective.** The priced cost expression when one is given and lowers;
  otherwise a lexicographic ladder, each stage holding the previous optimum:
  HBM traffic (``spill_cost * (1 - in_buffer)``, residency is the hard
  priority), coarse-tiling cuts, parallelism (total cores used), division
  balance (summed squared split factors, ``2x2`` over ``4x1``) and tile count.

After the solve, each in-place-merged placement unit slides down to the lowest
free address (``buffer_graph.justify``); when that would not fit, the solver's
own offsets are kept.

**LX relayouts.** An edge the candidate generator offers a relayout
(``RelayoutEdge``) may be read through a copy of its source in another
per-core view (``lazy_relayout``): one copy per (source, destination view),
shared by every consumer that reads that view, a :class:`Transform` whose
residency IS the decision to shuffle and whose rectangle sits in the same
no-overlap as every node; its price is a strategy objective term. Under the
fallback ladder a shuffle is unpriced and would look free, so every copy is
pinned out there.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from ortools.sat.python import cp_model

    _CP_MODEL_IMPORT_ERROR = ""
else:
    # Why ortools failed to import: often a missing dependency of it, not ortools itself.
    _CP_MODEL_IMPORT_ERROR = ""
    try:
        from ortools.sat.python import cp_model

    except ImportError as exc:  # pragma: no cover - only when ortools is absent
        cp_model = None
        _CP_MODEL_IMPORT_ERROR = str(exc)

from torch_spyre._inductor.scratchpad.buffer_graph import (
    SOLVER_CHOSE_SPILL,
    ResourceNode,
    SolveSettings,
)
from torch_spyre._inductor.scratchpad.encoding_strategy import (
    _gate_divisions,
)
from torch_spyre._inductor.scratchpad.graph_solvers import Capabilities, GraphSolver
from torch_spyre._inductor.scratchpad.plan_solver import (
    ceil_div,
)
from torch_spyre._inductor import config

__all__ = ["ModelSolver", "_gate_divisions"]

logger = logging.getLogger(__name__)


def _backend():
    """The backend the graph posts to. Prototype knob: ``SPYRE_LX_BACKEND`` =
    ``cpsat`` (default), ``mip:<SCIP|HIGHS|CBC>`` or ``bruteforce``."""
    import os

    from torch_spyre._inductor.scratchpad.backend import BruteForce, CpSat, Mip

    choice = os.environ.get("SPYRE_LX_BACKEND", "cpsat")
    if choice.startswith("mip"):
        return Mip(choice.partition(":")[2] or "SCIP")
    if choice == "bruteforce":
        return BruteForce()
    if choice.startswith("anneal"):
        from torch_spyre._inductor.scratchpad.backend import Annealing

        return Annealing(budget_s=float(choice.partition(":")[2] or 10))
    return CpSat()


# Drop cause for a buffer the solver chose to spill, as opposed to one barred
# up front (its residency reason).
_SOLVER_CHOSE_SPILL = SOLVER_CHOSE_SPILL


class ModelSolver(GraphSolver):
    """Posts the graph through its strategy to a backend (CP-SAT unless
    ``SPYRE_LX_BACKEND`` names another), runs the objective ladder and fills
    every open decision. See the module docstring for the model."""

    capabilities = Capabilities(
        chooses_core_divisions=True,
        decides_lx_relayouts=True,
        chooses_tilings=True,
        tilings_from_menu=True,
        replans_after_tiling=True,
        linear_cost_only=True,
        posts_model=True,
    )

    def __init__(
        self,
        capacity: int,
        time_limit_seconds: Optional[float] = None,
        bottom_justify: bool = True,
    ) -> None:
        if cp_model is None:
            raise ImportError(
                "The 'cpsat' layout solver requires the 'ortools' package, which "
                f"could not be imported ({_CP_MODEL_IMPORT_ERROR or 'not installed'}). "
                "Install it with 'pip install ortools' "
                "or select layout_solver='simulated_annealing'."
            )
        super().__init__(capacity)
        self.time_limit_seconds = (
            config.cpsat_time_limit_seconds
            if time_limit_seconds is None
            else time_limit_seconds
        )
        self.bottom_justify = bottom_justify

    def solve(self, graph, strategy, cost_expr=None) -> list[ResourceNode]:
        """A node the allocator did not bar is barred here when even its
        smallest per-core footprint exceeds LX."""
        for node in graph.get_nodes():
            if node.residency_reason is not None:
                continue
            menu = strategy.menus.get(node.name, ())
            footprint = min(
                (
                    ceil_div(
                        node.nbytes, cd.output_partition * cd.tiling.output_tile_count
                    )
                    for cd in menu
                ),
                default=node.nbytes,
            )
            if footprint > self.capacity:
                node.residency_reason = (
                    f"min footprint {footprint} B > LX capacity {self.capacity} B"
                )
        graph.settings = SolveSettings(
            time_limit_seconds=self.time_limit_seconds,
            bottom_justify=self.bottom_justify,
            record=self._solve_and_record,
            note_stats=self._note_stats,
        )
        return graph.solve(_backend(), cost_expr, strategy)

    def _note_stats(self, stats: dict) -> None:
        self.last_solve_stats = stats

    def _solve_and_record(
        self,
        solver: "cp_model.CpSolver",
        model: "cp_model.CpModel",
        *,
        objective: bool = False,
    ) -> int:
        """Solve one ladder stage, and keep what it cost and returned for the
        cost-expression dump. Bound to solving, so the last record always
        describes the solve that produced the plan."""
        status = solver.Solve(model)
        self.last_solve_stats = {
            "status": solver.StatusName(status),
            "solve_s": round(solver.WallTime(), 3),
            "variables": len(model.proto.variables),
            "constraints": len(model.proto.constraints),
            "limit_s": solver.parameters.max_time_in_seconds or None,
            "objective_used": objective,
        }
        return status
