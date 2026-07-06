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

"""Brute-force joint core-division + LX-placement solver
(``config.layout_solver == "dfs"``).

A pure-Python drop-in for :class:`CpSatLayoutSolver` (no OR-Tools dependency):
same :class:`CoreDivisionBuffer` input, same ``address`` / ``chosen_division`` /
``spill_reasons`` output. Instead of a constraint solve it enumerates the
cross-product of each buffer's candidate divisions with an explicit loop, scores
each assignment with the *same* two-phase objective CP-SAT uses (phase 1: minimize
HBM transfer traffic; phase 2: maximize total core usage), and delegates the
actual address packing of each assignment to an injected inner
:class:`MemoryPlanSolver` (greedy / first-fit / best-fit).

The search is bounded by ``K^N`` over the buffers with more than one candidate
division, so it is intended for the bounded ``axis_swaps`` / ``matmul_only`` search
spaces; a hard iteration cap guards the (rare) blow-up on the ``complete`` space.
When every buffer carries a single division the loop runs exactly once and the
solver reduces to a single inner-placement pass.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Sequence

from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivisionBuffer,
    CoOptimizingSolver,
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
    MemoryPlanSolver,
    _SOLVER_CHOSE_SPILL,
    _assert_core_divisions_enumerated,
    _assert_in_place_relationships,
)

__all__ = ["DfsLayoutSolver"]

logger = logging.getLogger(__name__)

# Hard cap on the number of division assignments the loop evaluates. The seed
# (all-index-0) assignment is always first, so a truncated search still returns a
# valid plan no worse than the seed. Bounded search spaces stay well under this.
_MAX_ASSIGNMENTS = 4096


class DfsLayoutSolver(CoOptimizingSolver):
    """Joint core-division + LX placement by brute-force search over candidate
    divisions, scored with CP-SAT's objective and packed by an inner placement
    solver. See the module docstring."""

    def __init__(
        self,
        size: int,
        alignment: int = 128,
        inner_solver: MemoryPlanSolver | None = None,
    ) -> None:
        super().__init__(size, alignment)
        # Placement oracle used to pack each fixed division assignment. Defaults to
        # greedy; select_allocator injects the config-chosen gap solver.
        self.inner = inner_solver or GreedyLayoutSolver(size, alignment)

    def plan_layout_and_core_divs(
        self, buffers: Sequence[CoreDivisionBuffer], log_lx_usage: bool = False
    ) -> list[CoreDivisionBuffer]:
        self.spill_reasons = {}
        if not buffers:
            return []
        assert all(b.address is None for b in buffers), (
            "Buffers cannot be previously or partially planned"
        )
        _assert_in_place_relationships(buffers)
        _assert_core_divisions_enumerated(buffers)

        by_name = {b.name: b for b in buffers}
        # parent name -> list of (child name, compatible (parent_div, child_div)
        # pairs); mirrors CpSatLayoutSolver._get_children.
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]] = {}
        for b in buffers:
            for parent in b.parents:
                children_of.setdefault(parent, []).append(
                    (b.name, b.cd_parent_matches.get(parent, []))
                )

        # Buffers that can never reside, regardless of the division assignment
        # (mirrors CP-SAT's _trim_oversized_tensors + the unconditional cases of
        # _implicate_core_division). Their division is still chosen (phase-2 cores
        # + child matching), but they are always spilled with the recorded reason.
        forced_reasons = self._forced_spill_reasons(buffers, children_of)

        best = self._search(buffers, by_name, children_of, forced_reasons)
        chosen, offsets = best

        for b in buffers:
            b.chosen_division = chosen[b.name]
            b.address = offsets.get(b.name)  # None => spilled

        self.spill_reasons = {
            b.name: forced_reasons.get(b.name, _SOLVER_CHOSE_SPILL)
            for b in buffers
            if b.address is None
        }

        if logger.isEnabledFor(logging.DEBUG):
            resident = sum(1 for b in buffers if b.address is not None)
            logger.debug(
                "[DFS layout solver] tensors=%d resident=%d inner=%s",
                len(buffers),
                resident,
                type(self.inner).__name__,
            )
            for b in buffers:
                if b.address is None:
                    logger.debug(
                        "[DFS layout solver]   %s -> HBM: %s",
                        b.name,
                        self.spill_reasons[b.name],
                    )

        return list(buffers)

    # ------------------------------------------------------------------
    # Forced (assignment-independent) spills
    # ------------------------------------------------------------------
    def _forced_spill_reasons(
        self,
        buffers: Sequence[CoreDivisionBuffer],
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
    ) -> dict[str, str]:
        """Reasons a buffer can never reside no matter which division is chosen:
        the allocator pinned it out (``not placement``), its *smallest* candidate
        footprint exceeds capacity, it has no consumer reading it from LX, or a
        consumer has no slicing-compatible division at all."""
        forced: dict[str, str] = {}
        for b in buffers:
            if not b.placement:
                forced[b.name] = (
                    b.residency_reason or "residency not allowed by allocator"
                )
                continue
            min_fp = min(
                math.ceil(b.size / cd.output_partition) for cd in b.core_divisions
            )
            if min_fp > self.limit:
                forced[b.name] = (
                    f"min per-core footprint {min_fp} > LX capacity {self.limit}"
                )
                continue
            kids = children_of.get(b.name, [])
            if not kids:
                forced[b.name] = "no consumer reads it from LX"
                continue
            for child, pairs in kids:
                if not pairs:
                    forced[b.name] = (
                        f"consumer {child} has no slicing-compatible core division"
                    )
                    break
        return forced

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _search(
        self,
        buffers: Sequence[CoreDivisionBuffer],
        by_name: dict[str, CoreDivisionBuffer],
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
        forced_reasons: dict[str, str],
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Enumerate the division cross-product, score each assignment, return the
        winning ``(chosen_div_by_name, offsets_by_name)`` (offsets omit spills).

        Score is lexicographic: minimize phase-1 HBM traffic, then maximize total
        cores (phase-2 parallelism) -- the same objective CP-SAT optimizes."""
        # Only buffers with a real choice branch; the rest are fixed at index 0.
        branch = [b for b in buffers if len(b.core_divisions) > 1]
        n_assignments = math.prod(len(b.core_divisions) for b in branch)
        capped = n_assignments > _MAX_ASSIGNMENTS
        if capped:
            logger.warning(
                "[DFS layout solver] %d division assignments exceeds cap %d; "
                "search truncated (seed assignment is evaluated first, so the "
                "result is no worse than the committed split).",
                n_assignments,
                _MAX_ASSIGNMENTS,
            )

        best_traffic = math.inf
        best_cores = -1
        best_chosen: dict[str, int] = {b.name: 0 for b in buffers}
        best_offsets: dict[str, int] = {}

        base_choice = {b.name: 0 for b in buffers}
        for count, combo in enumerate(
            itertools.product(*[range(len(b.core_divisions)) for b in branch])
        ):
            if count >= _MAX_ASSIGNMENTS:
                break
            chosen = dict(base_choice)
            for b, idx in zip(branch, combo):
                chosen[b.name] = idx

            offsets, traffic, cores = self._score(
                buffers, by_name, children_of, forced_reasons, chosen
            )
            if traffic < best_traffic or (
                traffic == best_traffic and cores > best_cores
            ):
                best_traffic, best_cores = traffic, cores
                best_chosen, best_offsets = chosen, offsets

        return best_chosen, best_offsets

    def _score(
        self,
        buffers: Sequence[CoreDivisionBuffer],
        by_name: dict[str, CoreDivisionBuffer],
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
        forced_reasons: dict[str, str],
        chosen: dict[str, int],
    ) -> tuple[dict[str, int], int, int]:
        """Place one division assignment via the inner solver and score it.

        Returns ``(offsets, traffic, cores)`` where ``offsets`` maps resident
        buffers to addresses, ``traffic`` is the phase-1 HBM cost
        (spilled pay ``_spill_cost``; resident pay ``boundary_cost``), and
        ``cores`` is the phase-2 total core usage of the assignment."""
        # Residency-eligible under this assignment: not force-spilled, chosen
        # footprint fits, and its division matches every child's division
        # (the cd_parent_matches gate CP-SAT enforces via _implicate_core_division).
        eligible: list[str] = []
        for b in buffers:
            if b.name in forced_reasons:
                continue
            cd = b.core_divisions[chosen[b.name]]
            if math.ceil(b.size / cd.output_partition) > self.limit:
                continue
            if all(
                (chosen[b.name], chosen[child]) in pairs
                for child, pairs in children_of.get(b.name, [])
            ):
                eligible.append(b.name)

        eligible_set = set(eligible)
        inner_buffers = [
            LifetimeBoundBuffer(
                name=b.name,
                size=math.ceil(
                    b.size / b.core_divisions[chosen[b.name]].output_partition
                ),
                uses=b.uses,
                first_use_is_read=b.first_use_is_read,
                # Restrict to eligible parents: the inner solver asserts every
                # in-place parent is present in its buffer list.
                in_place_parents=[p for p in b.in_place_parents if p in eligible_set],
            )
            for b in buffers
            if b.name in eligible_set
        ]
        placed = {
            b.name: b.address
            for b in self.inner.plan_layout(inner_buffers)
            if b.address is not None
        }

        traffic = 0
        cores = 0
        for b in buffers:
            cores += b.core_divisions[chosen[b.name]].cores_used
            if b.name in placed:
                traffic += b.boundary_cost
            else:
                traffic += self._spill_cost(b, len(children_of.get(b.name, [])))
        return placed, traffic, cores
