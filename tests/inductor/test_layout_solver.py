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

"""Tests for layout solvers"""

from unittest import TestCase
from torch_spyre._inductor.layout_backend import (
    LifetimeBoundBuffer,
    GreedyLayoutSolver,
    SortedLayoutSolver
)

__LARGE_SIZE__ = 512
__SMALL_SIZE__ = 10
__ALIGNMENT__ = 128


class TestGreedySolver(TestCase):
    def verify_layout(
        self,
        input: LifetimeBoundBuffer,
        expectation: LifetimeBoundBuffer,
        size=10,
        alignment=1,
    ):
        result = GreedyLayoutSolver(size, alignment).plan_layout(input)
        for planned, expected in zip(result, expectation):
            self.assertEqual(planned.address, expected.address)

    def test_simple_layout(self):
        input = [
            LifetimeBoundBuffer("buffer0", 3, 0, 2, {}),
            LifetimeBoundBuffer("buffer1", 3, 0, 2, {}),
            LifetimeBoundBuffer("buffer2", 4, 0, 2, {}),
        ]

        expectation = [
            LifetimeBoundBuffer("buffer0", 3, 0, 2, {}, 0),
            LifetimeBoundBuffer("buffer1", 3, 0, 2, {}, 3),
            LifetimeBoundBuffer("buffer2", 4, 0, 2, {}, 6),
        ]
        self.verify_layout(input, expectation)

    def test_simple_layout_below_alignment(self):
        input = [
            LifetimeBoundBuffer("buffer0", 3, 0, 2, {}),
            LifetimeBoundBuffer("buffer1", 3, 0, 2, {}),
            LifetimeBoundBuffer("buffer2", 4, 0, 2, {}),
        ]

        expectation = [
            LifetimeBoundBuffer("buffer0", 3, 0, 2, {}, 0),
            LifetimeBoundBuffer("buffer1", 3, 0, 2, {}, None),
            LifetimeBoundBuffer("buffer2", 4, 0, 2, {}, None),
        ]
        self.verify_layout(input, expectation, alignment=__ALIGNMENT__)

    def test_alignment_enforced(self):
        input = [
            LifetimeBoundBuffer("buffer0", 3, 0, 2, {}),
            LifetimeBoundBuffer("buffer1", 3, 0, 2, {}),
            LifetimeBoundBuffer("buffer2", 4, 0, 2, {}),
        ]

        expectation = [
            LifetimeBoundBuffer("buffer0", 3, 0, 2, {}, 0),
            LifetimeBoundBuffer("buffer1", 3, 0, 2, {}, 128),
            LifetimeBoundBuffer("buffer2", 4, 0, 2, {}, 256),
        ]
        self.verify_layout(input, expectation, __LARGE_SIZE__, __ALIGNMENT__)

    def test_simple_evication_layout(self):
        input = [
            LifetimeBoundBuffer("buffer0", 7, 0, 2, {}),
            LifetimeBoundBuffer("buffer1", 4, 0, 2, {}),
            LifetimeBoundBuffer("buffer2", 3, 0, 2, {}),
        ]

        expectation = [
            LifetimeBoundBuffer("buffer0", 7, 0, 2, {}, 0),
            LifetimeBoundBuffer("buffer1", 4, 0, 2, {}, None),
            LifetimeBoundBuffer("buffer2", 3, 0, 2, {}, 7),
        ]
        self.verify_layout(input, expectation)

    def test_realloc(self):
        input = [
            LifetimeBoundBuffer("buffer0", 10, 0, 2, {}),
            LifetimeBoundBuffer("buffer1", 3, 2, 3, {}),
        ]

        expectation = [
            LifetimeBoundBuffer("buffer0", 10, 0, 2, {}, 0),
            LifetimeBoundBuffer("buffer1", 3, 2, 3, {}, 0),
        ]
        self.verify_layout(input, expectation)

    def test_realloc_between(self):
        input = [
            LifetimeBoundBuffer("buffer0", 3, 0, 4, {}),
            LifetimeBoundBuffer("buffer1", 3, 1, 3, {}),
            LifetimeBoundBuffer("buffer2", 3, 2, 4, {}),
            LifetimeBoundBuffer("buffer3", 3, 3, 4, {}),
        ]

        expectation = [
            LifetimeBoundBuffer("buffer0", 3, 0, 4, {}, 0),
            LifetimeBoundBuffer("buffer1", 3, 1, 3, {}, 3),
            LifetimeBoundBuffer("buffer2", 3, 2, 4, {}, 6),
            LifetimeBoundBuffer("buffer3", 3, 3, 4, {}, 3),
        ]
        self.verify_layout(input, expectation)

    def test_realloc_between_with_allocation(self):
        input = [
            LifetimeBoundBuffer("buffer0", 200, 0, 4, {}),
            LifetimeBoundBuffer("buffer1", 100, 1, 3, {}),
            LifetimeBoundBuffer("buffer2", 100, 2, 4, {}),
            LifetimeBoundBuffer("buffer3", 100, 3, 4, {}),
        ]

        expectation = [
            LifetimeBoundBuffer("buffer0", 200, 0, 4, {}, 0),
            LifetimeBoundBuffer("buffer1", 100, 1, 3, {}, 256),
            LifetimeBoundBuffer("buffer2", 100, 2, 4, {}, 384),
            LifetimeBoundBuffer("buffer3", 100, 3, 4, {}, 256),
        ]
        self.verify_layout(input, expectation, __LARGE_SIZE__, __ALIGNMENT__)


class TestSortedLayoutSolver(TestCase):
    def verify_layout(
        self,
        buffers: list[LifetimeBoundBuffer],
        expected_addresses: dict[str, int | None],
        capacity: int = 10,
        sorting_attribute: str = "size",
    ):
        result = SortedLayoutSolver(capacity, sorting_attribute).plan_layout(
            buffers
        )
        address_map = {b.name: b.address for b in result}
        for name, expected_addr in expected_addresses.items():
            self.assertEqual(address_map[name], expected_addr)

    def test_simple_layout(self):
        buffers = [
            LifetimeBoundBuffer("buffer0", 3, 0, 4, {}),
            LifetimeBoundBuffer("buffer1", 5, 0, 4, {}),
            LifetimeBoundBuffer("buffer2", 7, 0, 4, {}),
        ]
        # Sorted by size desc: buffer2(7)→0, buffer1(5)→7, buffer0(3)→12
        self.verify_layout(
            buffers,
            {"buffer0": 12, "buffer1": 7, "buffer2": 0},
            capacity=20,
        )

    def test_largest_placed_first(self):
        # With capacity=9 only buffer_large fits; size-descending sort ensures it
        # gets priority.  A smaller-first sort would evict buffer_large instead.
        buffers = [
            LifetimeBoundBuffer("buffer_small", 3, 0, 2, {}),
            LifetimeBoundBuffer("buffer_large", 8, 0, 2, {}),
        ]
        self.verify_layout(
            buffers,
            {"buffer_large": 0, "buffer_small": None},
            capacity=9,
        )

    def test_eviction_when_no_gap_fits(self):
        # buffer1 is evicted because no remaining gap is large enough for it after
        # buffer0 and buffer2 are placed.
        buffers = [
            LifetimeBoundBuffer("buffer0", 7, 0, 2, {}),
            LifetimeBoundBuffer("buffer1", 5, 0, 2, {}),
            LifetimeBoundBuffer("buffer2", 3, 0, 2, {}),
        ]
        # Sorted: buffer0(7)→0, buffer1(5) needs 5 but only [7,10) size 3 remains,
        # buffer2(3) fits in that gap.
        self.verify_layout(
            buffers,
            {"buffer0": 0, "buffer1": None, "buffer2": 7},
            capacity=10,
        )

    def test_realloc_shares_memory(self):
        # buffer1 starts exactly when buffer0 ends; no lifetime overlap means
        # buffer1 can reuse buffer0's address.
        buffers = [
            LifetimeBoundBuffer("buffer0", 10, 0, 2, {}),
            LifetimeBoundBuffer("buffer1", 5, 2, 4, {}),
        ]
        self.verify_layout(
            buffers,
            {"buffer0": 0, "buffer1": 0},
            capacity=15,
        )

    def test_realloc_between(self):
        # buffer3 reuses the address freed by buffer1 once buffer1 expires.
        buffers = [
            LifetimeBoundBuffer("buffer0", 3, 0, 4, {}),
            LifetimeBoundBuffer("buffer1", 3, 1, 3, {}),
            LifetimeBoundBuffer("buffer2", 3, 2, 4, {}),
            LifetimeBoundBuffer("buffer3", 3, 3, 4, {}),
        ]
        self.verify_layout(
            buffers,
            {"buffer0": 0, "buffer1": 3, "buffer2": 6, "buffer3": 3},
            capacity=10,
        )

    def test_best_fit_selects_tightest_gap(self):
        # At the time buf_target is placed two free gaps exist: [10,18) size 8 and
        # [24,30) size 6.  Best-fit selects the smaller gap (diff=1) over the
        # larger one (diff=3), so buf_target lands at 24 rather than 10.
        buffers = [
            LifetimeBoundBuffer("buf_perm_a", 10, 0, 4, {}),
            LifetimeBoundBuffer("buf_bridge", 8, 0, 2, {}),
            LifetimeBoundBuffer("buf_perm_b", 6, 0, 4, {}),
            LifetimeBoundBuffer("buf_target", 5, 2, 4, {}),
        ]
        # Sorted: perm_a(10)→0, bridge(8)→10 (expires at 2), perm_b(6)→18,
        # target(5): occupied=[(0,10),(18,24)], gaps=[(10,18),(24,30)]
        # best-fit for size 5: (24,30) diff=1 < (10,18) diff=3 → placed at 24
        self.verify_layout(
            buffers,
            {"buf_perm_a": 0, "buf_bridge": 10, "buf_perm_b": 18, "buf_target": 24},
            capacity=30,
        )

    def test_custom_sorting_attribute(self):
        # Sorting by start_time desc puts the later-starting buffer first, giving
        # it the lower address — the opposite of size-based sorting here.
        buffers = [
            LifetimeBoundBuffer("buf_early", 7, 0, 8, {}),
            LifetimeBoundBuffer("buf_late", 3, 5, 8, {}),
        ]
        # start_time sort desc: buf_late(5)→0, buf_early(0)→3
        self.verify_layout(
            buffers,
            {"buf_late": 0, "buf_early": 3},
            capacity=20,
            sorting_attribute="start_time",
        )

    def test_invalid_sorting_attribute_raises(self):
        with self.assertRaises(AssertionError):
            SortedLayoutSolver(100, "nonexistent_attribute")


if __name__ == "__main__":
    import unittest

    unittest.main()
