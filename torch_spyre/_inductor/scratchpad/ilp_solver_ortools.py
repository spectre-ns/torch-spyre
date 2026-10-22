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

The allocator's buffers become a :class:`SpyreOptimizationGraph`
(``buffer_graph.py``): one :class:`ResourceNode` per op output or graph input,
a :class:`DependencyEdge` per (producer, consumer) read (``in_place`` when the
consumer may take over the producer's storage), and :class:`ExecutionEdge`\\ s
chaining op outputs in execution order. Each node's decisions -- its work
split (core division and coarse tiling), whether it is resident in LX, and its
offset -- are open for the solve to fill. ``graph.solve`` encodes the graph
through an :class:`EncodingStrategy` (``encoding_strategy.py``), runs the
objective ladder and fills the decisions; this module builds the graph and the
strategy from the allocator's buffers and writes the decisions back.

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

**LX relayouts.** The allocator hands the solver one ``RelayoutCopyBuffer`` per
relayout group (a source and one destination per-core view, however many
consumers read it); the strategy holds each as a candidate
:class:`Transform`, live from the group's first consumer to its last. The
copy's residency IS the decision to shuffle, its rectangle sits in the same
no-overlap as every node, and its price is an ordinary term of the cost
expression (``RelayoutCopyBuffer.cost_term``). A resident copy needs its
source resident under a division it was priced for, a consumer reads the copy
only under a division pair its candidates list, and a resident copy serves at
least one consumer. Under the fallback ladder a shuffle is unpriced and would
look free, so every copy is pinned out there.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, cast

import sympy


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
    build_graph,
)
from torch_spyre._inductor.scratchpad.encoding_strategy import (
    EnumerationFactory,
    TableStrategy,
    _gate_divisions,
)
from torch_spyre._inductor.scratchpad.lx_relayout import ChosenRelayout
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivisionBuffer,
    CoreDivisionLayoutSolver,
    LifetimeBoundBuffer,
    RelayoutCopyBuffer,
    _check_in_place_relationships,
    build_relayout_copy,
)
from torch_spyre._inductor import config

__all__ = ["CpSatLayoutSolver", "_gate_divisions"]

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


class CpSatLayoutSolver(CoreDivisionLayoutSolver):
    """Joint core-division + LX placement via an OR-Tools CP-SAT search
    (``config.layout_solver == "cpsat"``). See the module docstring for the
    model (joint division, slicing-match residency gate, 2D no-overlap with
    in-place lifetime shortening) and the lexicographic objective
    (residency, then parallelism, then division balance).
    """

    @classmethod
    def supports_paired_buffers(cls) -> bool:
        return False

    @classmethod
    def decides_lx_relayouts(cls) -> bool:
        return True

    @classmethod
    def generates_divisions(cls) -> bool:
        return False

    @classmethod
    def chooses_tilings(cls) -> bool:
        return True

    @classmethod
    def tilings_from_menu(cls) -> bool:
        return True

    @classmethod
    def replans_after_tiling(cls) -> bool:
        return True

    @classmethod
    def linear_cost_only(cls) -> bool:
        return True

    def __init__(
        self,
        buffers: Sequence[LifetimeBoundBuffer],
        size: int,
        alignment: int = 128,
        time_limit_seconds: Optional[float] = None,
        bottom_justify: bool = True,
    ) -> None:
        if cp_model is None:
            raise ImportError(
                "The 'cpsat' layout solver requires the 'ortools' package, which "
                f"could not be imported ({_CP_MODEL_IMPORT_ERROR or 'not installed'}). "
                "Install it with 'pip install ortools' "
                "or select a different layout_solver (e.g. 'greedy')."
            )
        super().__init__(buffers, size, alignment)
        # What the last solve cost and returned, for the cost-expression dump.
        # Here rather than on the base class: this is the only solver that
        # reports it, and the allocator reads it with a default, so the base
        # contract does not change. Empty until a solve, so a reader can tell
        # "not recorded" from "no solve".
        self.last_solve_stats: dict = {}
        # The solver works in alignment-sized units so every offset it picks is
        # automatically aligned; plan_layout scales sizes/offsets in and out.
        self._capacity_units = self.limit // self.alignment
        self._time_limit_seconds = (
            config.cpsat_time_limit_seconds
            if time_limit_seconds is None
            else time_limit_seconds
        )
        self._bottom_justify = bottom_justify

    def plan_layout(self, log_lx_usage: bool = False) -> list[LifetimeBoundBuffer]:
        """Place buffers on their already-fixed core divisions (placement-only).

        Same model as :meth:`plan_layout_and_core_divisions` minus the joint
        division choice: each buffer's footprint is its ``size``, so there is no
        slicing gate on residency and no parallelism step -- the solve reduces
        to minimising HBM traffic under the 2D no-overlap with in-place reuse.
        Dispatch is per buffer and keys on whether it carries candidate
        divisions, not on its class, so a :class:`CoreDivisionBuffer` with an
        empty candidate list is placed here rather than divided."""
        return cast("list[LifetimeBoundBuffer]", list(self._plan_layout_generic()))

    def plan_layout_and_core_divisions(
        self, cost_expr: sympy.Expr | None = None
    ) -> list[CoreDivisionBuffer]:
        """Jointly choose each buffer's core division and its LX placement.

        The full model described in the module docstring. Every buffer must
        carry enumerated candidate divisions; the chosen index is written back
        to ``chosen_division`` for the allocator to commit."""
        buffers = cast("Sequence[CoreDivisionBuffer]", self.buffers)
        assert all(len(b.core_divisions) != 0 for b in buffers), (
            "All buffers must have at least 1 valid core division"
        )
        return cast(
            "list[CoreDivisionBuffer]",
            list(self._plan_layout_generic(cost_expr=cost_expr)),
        )

    posts_model = True

    def graph_solver(self) -> Any:
        """The CP-SAT backend: the graph is encoded through a strategy and
        solved by the objective ladder."""
        from torch_spyre._inductor.scratchpad.backend import CpSat

        return CpSat

    def _plan_layout_generic(
        self,
        log_lx_usage: bool = False,
        cost_expr: sympy.Expr | None = None,
    ) -> list[LifetimeBoundBuffer | CoreDivisionBuffer]:
        buffers = self.buffers
        if not buffers:
            return []
        assert all(b.address is None for b in buffers), (
            "Buffers cannot be previously or partially planned"
        )

        _check_in_place_relationships(buffers)

        # Declarative exclusion, shared with every other solver: whatever the
        # allocator barred (each buffer's ``residency_reason``), plus the
        # capacity check. Barred buffers stay in the model (they must stay
        # available for slicing matching and in-place chains), pinned
        # non-resident.
        forced_reasons = dict(self.record_exclusions())
        settings = SolveSettings(
            time_limit_seconds=self._time_limit_seconds,
            bottom_justify=self._bottom_justify,
            record=self._solve_and_record,
            note_stats=self._note_stats,
        )
        graph = build_graph(
            buffers, self.alignment, self._capacity_units, settings, forced_reasons
        )
        strategy = EnumerationFactory.from_config().strategy(buffers, forced_reasons)
        graph.solve(_backend(), cost_expr, strategy)

        # Write the decisions back onto the caller's buffers, in bytes.
        order = {name: i for i, name in enumerate(graph.nodes)}
        transforms = {t.name: t for t in strategy.transforms}
        for b in buffers:
            item = graph.nodes.get(b.name) or transforms[b.name]
            b.address = None if item.offset is None else item.offset * self.alignment
            b.spill_reason = item.spill_reason
            if isinstance(b, RelayoutCopyBuffer):
                b.chosen_division = 0
            if not isinstance(item, ResourceNode) or not isinstance(
                b, CoreDivisionBuffer
            ):
                continue
            if item in strategy.solved:
                b.chosen_division = strategy.solved[item]
            chosen = dict(b.chosen_relayouts)
            for edge in sorted(item.inputs, key=lambda e: order[e.src.name]):
                if edge.transform is not None and edge.transform.offset is not None:
                    chosen[edge.src.name] = ChosenRelayout(
                        strategy.fired[edge], edge.transform.offset
                    )
            b.chosen_relayouts = {
                parent: c.scaled(self.alignment) for parent, c in chosen.items()
            }
        return [*buffers, *self._lazy_copies(graph, strategy, buffers)]

    def _lazy_copies(
        self, graph, strategy: TableStrategy, buffers: Sequence[LifetimeBoundBuffer]
    ) -> list[RelayoutCopyBuffer]:
        """A copy buffer per resident lazy relayout (2d), built from the
        candidates that fired through it as the table encoding builds one per
        group, and placed where the solve put it."""
        by_name = {b.name: b for b in buffers}
        ticks = {b.name: b.start_time for b in buffers}
        fired: dict = {}
        for node in graph.get_nodes():
            for edge in node.outputs:
                t = edge.transform
                if t is not None and t.name not in by_name:
                    fired.setdefault(t, []).append(strategy.fired[edge])
        copies = []
        for t, candidates in fired.items():
            assert t.producer is not None and t.offset is not None
            source = by_name[t.producer.name]
            assert isinstance(source, CoreDivisionBuffer)
            copy = build_relayout_copy(source, candidates[0].group, candidates, ticks)
            copy.address = t.offset * self.alignment
            copy.chosen_division = 0
            copies.append(copy)
        return copies

    def _note_stats(self, stats: dict) -> None:
        self.last_solve_stats = stats

    def _solve_and_record(
        self,
        solver: "cp_model.CpSolver",
        model: "cp_model.CpModel",
        *,
        objective: bool = False,
    ) -> int:
        """Solve, and stash what it cost and returned for the cost-expression
        dump. The one way this class solves.

        Recording is bound to solving rather than left to each call site: the
        graph solves in its own ladder stages when the cost expression does not
        lower, and a site that solved without recording would leave its plan
        described by an earlier call's numbers -- silently wrong data rather
        than an error. The two paths are exclusive, so the last record always
        describes the solve that produced this plan.

        Nothing else in the pipeline records this, so "why was that compile
        slow" currently has no artifact behind it.
        """
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
