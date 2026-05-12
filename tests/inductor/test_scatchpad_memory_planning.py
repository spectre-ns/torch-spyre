# Copyright 2025 The Torch-Spyre Authors.
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

"""Tests for scratchpad memory planning solver."""

from unittest import TestCase
from torch_spyre._inductor.scratchpad_memory_plan_solver import (
    LifetimeBoundBuffer,
    OrToolsMemoryPlanSolver,
)


class TestOrToolsMemoryPlanSolver(TestCase):
    """Tests derived from the canonical planning demo scenarios."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _solve(
        self, buffers: list[LifetimeBoundBuffer], buffer_size: int
    ) -> dict[str, LifetimeBoundBuffer]:
        solver = OrToolsMemoryPlanSolver(buffer_size=buffer_size)
        return {b.name: b for b in solver.plan_layout(buffers)}

    def _peak_usage(self, results: dict[str, LifetimeBoundBuffer]) -> int:
        """High-water mark: max(offset + size) over all in-buffer tensors."""
        in_buf = [b for b in results.values() if b.address is not None]
        return max(b.address + b.size for b in in_buf if b.address is not None)

    def test_dag_branch_picks_one_inplace(self) -> None:
        """Solver picks exactly one of two competing in-place candidates for C."""
        # C = op(A, B); both A and B die when C is born.
        # Buffer is generous — no spill pressure — so secondary peak objective
        # drives the solver to activate an in-place (peak 200 vs 300).
        buffers = [
            LifetimeBoundBuffer("A", 100, 0, 3),
            LifetimeBoundBuffer("B", 100, 2, 3),
            LifetimeBoundBuffer("C", 100, 3, 5, inplace_parents=["A", "B"]),
            LifetimeBoundBuffer("OTHER", 100, 4, 5),
        ]
        results = self._solve(buffers, 500)

        # Buffer is large enough — no spills.
        for name, b in results.items():
            self.assertIsNotNone(b.address, f"{name} should not be spilled")

        # C shares its slot with exactly one of A or B.
        c_addr = results["C"].address
        self.assertTrue(
            c_addr == results["A"].address or c_addr == results["B"].address,
            "C must be in-placed from A or B",
        )

        # A and B overlap in time [2,3], so they cannot share an offset.
        self.assertNotEqual(results["A"].address, results["B"].address)

    def test_forced_spillage_spills_cheapest_tensor(self) -> None:
        """When concurrent tensors exceed the buffer, the cheapest one spills."""
        # At t=2–4 all three are alive: 400+400+200=1000 > 800.
        # Effective spill costs: BIG_A=400*1=400, BIG_B=400*1=400, SMALL_CHEAP=200*0.1=20.
        buffers = [
            LifetimeBoundBuffer("BIG_A", 400, 0, 5, heuristic=1.0),
            LifetimeBoundBuffer("BIG_B", 400, 1, 5, heuristic=1.0),
            LifetimeBoundBuffer("SMALL_CHEAP", 200, 2, 4, heuristic=0.1),
        ]
        results = self._solve(buffers, 800)

        self.assertIsNone(
            results["SMALL_CHEAP"].address, "SMALL_CHEAP should be spilled"
        )
        self.assertIsNotNone(results["BIG_A"].address, "BIG_A should be in buffer")
        self.assertIsNotNone(results["BIG_B"].address, "BIG_B should be in buffer")

    def test_inplace_chain_avoids_spillage(self) -> None:
        """In-place chain eliminates spillage that is unavoidable without it."""
        # Consecutive 300-byte tensors, each pair overlapping at one time step.
        # Without candidates: two live at once = 600 > 400 → must spill.
        # With chain T0→T1→T2→T3: all share one 300-byte slot → peak 300, no spill.
        no_ip = [
            LifetimeBoundBuffer("T0", 300, 0, 1),
            LifetimeBoundBuffer("T1", 300, 1, 2),
            LifetimeBoundBuffer("T2", 300, 2, 3),
            LifetimeBoundBuffer("T3", 300, 3, 4),
        ]
        with_ip = [
            LifetimeBoundBuffer("T0", 300, 0, 1),
            LifetimeBoundBuffer("T1", 300, 1, 2, inplace_parents=["T0"]),
            LifetimeBoundBuffer("T2", 300, 2, 3, inplace_parents=["T1"]),
            LifetimeBoundBuffer("T3", 300, 3, 4, inplace_parents=["T2"]),
        ]

        res_no = self._solve(no_ip, 400)
        res_yes = self._solve(with_ip, 400)

        # Without in-place: at least one tensor must spill.
        none_spilled = all(b.address is not None for _, b in res_no.items())
        self.assertFalse(
            none_spilled, "Without in-place, at least one tensor must spill"
        )

        # With in-place: no spills.
        none_spilled = all(b.address is not None for _, b in res_yes.items())
        self.assertTrue(none_spilled, "With in-place chain, nothing should spill")

        # The full chain collapses to a single 300-byte slot.
        addrs = [res_yes[f"T{i}"].address for i in range(4)]
        self.assertEqual(
            len(set(addrs)), 1, f"Chain must share one offset; got {addrs}"
        )

    def test_dag_with_branching_and_chaining(self) -> None:
        """Solver handles a candidate DAG with both branch and chain edges."""
        # A and B both compete to in-place into C; C can chain into D.
        # Max concurrent at t=2: A+B+C=300 bytes and at t=4: C+D+E=300 bytes.
        # Buffer=400 is sufficient — solver should place everything without spilling.
        buffers = [
            LifetimeBoundBuffer("A", 100, 0, 2),
            LifetimeBoundBuffer("B", 100, 1, 2),
            LifetimeBoundBuffer("C", 100, 2, 4, inplace_parents=["A", "B"]),
            LifetimeBoundBuffer("D", 100, 3, 5, inplace_parents=["C"]),
            LifetimeBoundBuffer("E", 100, 4, 5),
        ]
        results = self._solve(buffers, 400)

        for name, b in results.items():
            self.assertIsNotNone(b.address, f"{name} should not be spilled")

        # C must be in-placed from one of its parents (secondary peak drives this).
        c_addr = results["C"].address
        self.assertTrue(
            c_addr == results["A"].address or c_addr == results["B"].address,
            "C must be in-placed from A or B",
        )

    def test_branch_choice_minimizes_spill_cost(self) -> None:
        """Solver selects B→C over A→C because it results in 10× lower spill cost."""
        # WALL (150 bytes) occupies most of the 200-byte buffer.
        # During [3,4]: WALL + A + T_BLOCKER = 250 bytes — something must give.
        #   A→C branch: A is locked in a merge group, so T_BLOCKER (cost=500) spills.
        #   B→C branch: A is free, so A (cost=50) spills instead.
        # Optimal plan: pick B→C, spill A (total cost=50).
        buffers = [
            LifetimeBoundBuffer("WALL", 150, 0, 20, heuristic=10.0),
            LifetimeBoundBuffer("A", 50, 0, 6, heuristic=1.0),
            LifetimeBoundBuffer("B", 50, 5, 6, heuristic=1.0),
            LifetimeBoundBuffer(
                "C", 50, 6, 15, heuristic=10.0, inplace_parents=["A", "B"]
            ),
            LifetimeBoundBuffer("T_BLOCKER", 50, 3, 4, heuristic=10.0),
        ]
        results = self._solve(buffers, 200)

        self.assertIsNone(results["A"].address, "A should be spilled under B→C branch")
        self.assertIsNotNone(
            results["T_BLOCKER"].address, "T_BLOCKER should not be spilled"
        )
        self.assertIsNotNone(results["WALL"].address, "WALL should not be spilled")
        self.assertIsNotNone(results["C"].address, "C should not be spilled")
        self.assertEqual(
            results["B"].address,
            results["C"].address,
            "B and C must share an offset (in-place B→C selected)",
        )

    def test_only_atoC_candidate_is_declined(self) -> None:
        """With only A→C available, the solver declines it because not using it is cheaper.

        Using A→C would lock A into a merge group spanning [0,15], forcing both
        T_BLOCKER (cost=500) and B (cost=50) to spill — total 550.
        Declining A→C and spilling A+B costs only 100, so the solver rejects the
        candidate and spills A and B instead.
        """
        buffers = [
            LifetimeBoundBuffer("WALL", 150, 0, 20, heuristic=10.0),
            LifetimeBoundBuffer("A", 50, 0, 6, heuristic=1.0),
            LifetimeBoundBuffer("B", 50, 5, 6, heuristic=1.0),
            LifetimeBoundBuffer("C", 50, 6, 15, heuristic=10.0, inplace_parents=["A"]),
            LifetimeBoundBuffer("T_BLOCKER", 50, 3, 4, heuristic=10.0),
        ]
        results = self._solve(buffers, 200)

        # Solver declines A→C; spilling A+B (cost=100) beats using the candidate
        # and spilling T_BLOCKER+B (cost=550).
        self.assertIsNone(results["A"].address, "A should be spilled")
        self.assertIsNone(
            results["B"].address, "B should be spilled (conflicts with C at t=6)"
        )
        self.assertIsNotNone(
            results["T_BLOCKER"].address, "T_BLOCKER should not be spilled"
        )
        self.assertIsNotNone(results["WALL"].address, "WALL should not be spilled")
        self.assertIsNotNone(results["C"].address, "C should not be spilled")

    def test_force_cheap_inplace_branch(self) -> None:
        """Forcing B→C keeps T_BLOCKER in buffer and spills only the cheap A."""
        buffers = [
            LifetimeBoundBuffer("WALL", 150, 0, 20, heuristic=10.0),
            LifetimeBoundBuffer("A", 50, 0, 6, heuristic=1.0),
            LifetimeBoundBuffer("B", 50, 5, 6, heuristic=1.0),
            LifetimeBoundBuffer("C", 50, 6, 15, heuristic=10.0, inplace_parents=["B"]),
            LifetimeBoundBuffer("T_BLOCKER", 50, 3, 4, heuristic=10.0),
        ]
        results = self._solve(buffers, 200)

        self.assertIsNone(results["A"].address, "A should be spilled under B→C branch")
        self.assertIsNotNone(
            results["T_BLOCKER"].address, "T_BLOCKER should not be spilled"
        )
        self.assertEqual(
            results["B"].address,
            results["C"].address,
            "B and C must share an offset (in-place B→C)",
        )

    # ------------------------------------------------------------------
    # Example F: secondary peak objective compacts the layout
    # ------------------------------------------------------------------

    def test_secondary_peak_objective_collapses_chain(self) -> None:
        """Secondary peak objective causes the solver to fully collapse a chain."""
        # Buffer is very loose (1000); primary objective (spill cost) is 0 in any layout.
        # The secondary objective (minimize peak) selects the full chain collapse,
        # reducing peak from 200 (two consecutive solo tensors) to 100 (one shared slot).
        buffers = [
            LifetimeBoundBuffer("T0", 100, 0, 1),
            LifetimeBoundBuffer("T1", 100, 1, 2, inplace_parents=["T0"]),
            LifetimeBoundBuffer("T2", 100, 2, 3, inplace_parents=["T1"]),
            LifetimeBoundBuffer("T3", 100, 3, 4, inplace_parents=["T2"]),
        ]
        results = self._solve(buffers, 1000)

        for name, b in results.items():
            self.assertIsNotNone(b.address, f"{name} should not be spilled")

        addrs = [results[f"T{i}"].address for i in range(4)]
        self.assertEqual(
            len(set(addrs)), 1, f"Full chain must collapse to one offset; got {addrs}"
        )
        self.assertEqual(self._peak_usage(results), 100)

    def test_secondary_peak_compacts_realistic_schedule_without_inplace(self) -> None:
        """Realistic 17-buffer schedule with no in-place candidates fits in 150 bytes.

        Seventeen tensors with sizes 15–75 bytes and lifetimes spanning t=0..17
        are placed into a generous 225-byte buffer.  The max concurrent usage is
        120 bytes (at t=2: A+B+C and at t=14: G+J+N+O).  With no spill pressure
        the secondary peak objective compacts the layout to a 150-byte high-water
        mark despite the 225-byte budget.
        """
        buffers = [
            LifetimeBoundBuffer("A", 60, 0, 3),
            LifetimeBoundBuffer("B", 30, 1, 4),
            LifetimeBoundBuffer("C", 30, 2, 13),
            LifetimeBoundBuffer("D", 30, 3, 4),
            LifetimeBoundBuffer("E", 30, 4, 5),
            LifetimeBoundBuffer("F", 60, 5, 6),
            LifetimeBoundBuffer("G", 30, 6, 15),
            LifetimeBoundBuffer("H", 30, 7, 8),
            LifetimeBoundBuffer("I", 30, 8, 9),
            LifetimeBoundBuffer("J", 15, 9, 16),
            LifetimeBoundBuffer("K", 15, 10, 12),
            LifetimeBoundBuffer("L", 15, 11, 12),
            LifetimeBoundBuffer("M", 15, 12, 13),
            LifetimeBoundBuffer("N", 30, 13, 15),
            LifetimeBoundBuffer("O", 45, 14, 15),
            LifetimeBoundBuffer("P", 30, 15, 16),
            LifetimeBoundBuffer("Q", 75, 16, 17),
        ]
        results = self._solve(buffers, 225)

        for name, b in results.items():
            self.assertIsNotNone(b.address, f"{name} should not be spilled")

        self.assertEqual(self._peak_usage(results), 150)

    def test_inplace_reuse_enables_realistic_schedule_in_tight_buffer(self) -> None:
        """In-place reuse allows the 17-buffer schedule to fit in a 120-byte buffer.

        Same schedule as test_isuru_without_inplace with two changes: A's
        lifetime ends at t=2 (eliminating its overlap with C at t=2), and P
        declares G and N as in-place parents (both die at t=15 when P is born).
        The tighter 120-byte budget forces the solver to exploit the G→P or N→P
        in-place opportunity; it must find a valid layout with no spills and a
        peak of exactly 120 bytes.
        """
        buffers = [
            LifetimeBoundBuffer("A", 60, 0, 2),
            LifetimeBoundBuffer("B", 30, 1, 4),
            LifetimeBoundBuffer("C", 30, 2, 13),
            LifetimeBoundBuffer("D", 30, 3, 4),
            LifetimeBoundBuffer("E", 30, 4, 5),
            LifetimeBoundBuffer("F", 60, 5, 6),
            LifetimeBoundBuffer("G", 30, 6, 15),
            LifetimeBoundBuffer("H", 30, 7, 8),
            LifetimeBoundBuffer("I", 30, 8, 9),
            LifetimeBoundBuffer("J", 15, 9, 16),
            LifetimeBoundBuffer("K", 15, 10, 12),
            LifetimeBoundBuffer("L", 15, 11, 12),
            LifetimeBoundBuffer("M", 15, 12, 13),
            LifetimeBoundBuffer("N", 30, 13, 15),
            LifetimeBoundBuffer("O", 45, 14, 15),
            LifetimeBoundBuffer("P", 30, 15, 16, inplace_parents=["G", "N"]),
            LifetimeBoundBuffer("Q", 75, 16, 17),
        ]
        results = self._solve(buffers, 120)

        for name, b in results.items():
            self.assertIsNotNone(b.address, f"{name} should not be spilled")

        self.assertEqual(self._peak_usage(results), 120)


if __name__ == "__main__":
    import unittest

    unittest.main()
