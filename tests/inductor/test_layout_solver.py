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
import torch_spyre._inductor.layout_backend as solvers

__LARGE_SIZE__ = 512
__SMALL_SIZE__ = 10
__ALIGNMENT__ = 128


class TestGreedySolver(TestCase):
    def verify_layout(
        self,
        input: solvers.LifetimeBoundBuffer,
        expectation: solvers.LifetimeBoundBuffer,
        size=10,
        alignment=1,
    ):
        result = solvers.GreedyLayoutSolver(size, alignment).plan_layout(input)
        for planned, expected in zip(result, expectation):
            self.assertEqual(planned.address, expected.address)

    def test_simple_layout(self):
        input = [
            solvers.LifetimeBoundBuffer("buffer0", 3, 0, 2, {}),
            solvers.LifetimeBoundBuffer("buffer1", 3, 0, 2, {}),
            solvers.LifetimeBoundBuffer("buffer2", 4, 0, 2, {}),
        ]

        expectation = [
            solvers.LifetimeBoundBuffer("buffer0", 3, 0, 2, {}, 0),
            solvers.LifetimeBoundBuffer("buffer1", 3, 0, 2, {}, 3),
            solvers.LifetimeBoundBuffer("buffer2", 4, 0, 2, {}, 6),
        ]
        self.verify_layout(input, expectation)

    def test_simple_layout_below_alignment(self):
        input = [
            solvers.LifetimeBoundBuffer("buffer0", 3, 0, 2, {}),
            solvers.LifetimeBoundBuffer("buffer1", 3, 0, 2, {}),
            solvers.LifetimeBoundBuffer("buffer2", 4, 0, 2, {}),
        ]

        expectation = [
            solvers.LifetimeBoundBuffer("buffer0", 3, 0, 2, {}, 0),
            solvers.LifetimeBoundBuffer("buffer1", 3, 0, 2, {}, None),
            solvers.LifetimeBoundBuffer("buffer2", 4, 0, 2, {}, None),
        ]
        self.verify_layout(input, expectation, alignment=__ALIGNMENT__)

    def test_alignment_enforced(self):
        input = [
            solvers.LifetimeBoundBuffer("buffer0", 3, 0, 2, {}),
            solvers.LifetimeBoundBuffer("buffer1", 3, 0, 2, {}),
            solvers.LifetimeBoundBuffer("buffer2", 4, 0, 2, {}),
        ]

        expectation = [
            solvers.LifetimeBoundBuffer("buffer0", 3, 0, 2, {}, 0),
            solvers.LifetimeBoundBuffer("buffer1", 3, 0, 2, {}, 128),
            solvers.LifetimeBoundBuffer("buffer2", 4, 0, 2, {}, 256),
        ]
        self.verify_layout(input, expectation, __LARGE_SIZE__, __ALIGNMENT__)

    def test_simple_evication_layout(self):
        input = [
            solvers.LifetimeBoundBuffer("buffer0", 7, 0, 2, {}),
            solvers.LifetimeBoundBuffer("buffer1", 4, 0, 2, {}),
            solvers.LifetimeBoundBuffer("buffer2", 3, 0, 2, {}),
        ]

        expectation = [
            solvers.LifetimeBoundBuffer("buffer0", 7, 0, 2, {}, 0),
            solvers.LifetimeBoundBuffer("buffer1", 4, 0, 2, {}, None),
            solvers.LifetimeBoundBuffer("buffer2", 3, 0, 2, {}, 7),
        ]
        self.verify_layout(input, expectation)

    def test_realloc(self):
        input = [
            solvers.LifetimeBoundBuffer("buffer0", 10, 0, 2, {}),
            solvers.LifetimeBoundBuffer("buffer1", 3, 2, 3, {}),
        ]

        expectation = [
            solvers.LifetimeBoundBuffer("buffer0", 10, 0, 2, {}, 0),
            solvers.LifetimeBoundBuffer("buffer1", 3, 2, 3, {}, 0),
        ]
        self.verify_layout(input, expectation)

    def test_realloc_between(self):
        input = [
            solvers.LifetimeBoundBuffer("buffer0", 3, 0, 4, {}),
            solvers.LifetimeBoundBuffer("buffer1", 3, 1, 3, {}),
            solvers.LifetimeBoundBuffer("buffer2", 3, 2, 4, {}),
            solvers.LifetimeBoundBuffer("buffer3", 3, 3, 4, {}),
        ]

        expectation = [
            solvers.LifetimeBoundBuffer("buffer0", 3, 0, 4, {}, 0),
            solvers.LifetimeBoundBuffer("buffer1", 3, 1, 3, {}, 3),
            solvers.LifetimeBoundBuffer("buffer2", 3, 2, 4, {}, 6),
            solvers.LifetimeBoundBuffer("buffer3", 3, 3, 4, {}, 3),
        ]
        self.verify_layout(input, expectation)

    def test_realloc_between_with_allocation(self):
        input = [
            solvers.LifetimeBoundBuffer("buffer0", 200, 0, 4, {}),
            solvers.LifetimeBoundBuffer("buffer1", 100, 1, 3, {}),
            solvers.LifetimeBoundBuffer("buffer2", 100, 2, 4, {}),
            solvers.LifetimeBoundBuffer("buffer3", 100, 3, 4, {}),
        ]

        expectation = [
            solvers.LifetimeBoundBuffer("buffer0", 200, 0, 4, {}, 0),
            solvers.LifetimeBoundBuffer("buffer1", 100, 1, 3, {}, 256),
            solvers.LifetimeBoundBuffer("buffer2", 100, 2, 4, {}, 384),
            solvers.LifetimeBoundBuffer("buffer3", 100, 3, 4, {}, 256),
        ]
        self.verify_layout(input, expectation, __LARGE_SIZE__, __ALIGNMENT__)


if __name__ == "__main__":
    import unittest

    unittest.main()
