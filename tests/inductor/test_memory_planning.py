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

import unittest
import torch_spyre._inductor.layout_backend as lb
from .tools.layout_visualization import plot_layout


class TestOps(unittest.TestCase):
    def test_annealing_layout(self):
        SPM_CAPACITY = 100

        buffer_list = [
            lb.Buffer(name="A", size=100, start_time=0, end_time=2),
            lb.Buffer(name="B", size=50, start_time=2, end_time=4),
            lb.Buffer(name="C", size=25, start_time=2, end_time=4),
            lb.Buffer(name="D", size=25, start_time=2, end_time=10),
            lb.Buffer(name="E", size=25, start_time=6, end_time=10),
            lb.Buffer(name="F", size=25, start_time=4, end_time=6),
            lb.Buffer(name="G", size=25, start_time=4, end_time=7),
            lb.Buffer(name="H", size=25, start_time=4, end_time=6),
            lb.Buffer(name="I", size=25, start_time=7, end_time=10),
            lb.Buffer(name="J", size=25, start_time=6, end_time=10),
        ]

        best_allocation = lb.allocate_buffers(SPM_CAPACITY, buffer_list, "annealing")
        plot_layout(SPM_CAPACITY, best_allocation)


if __name__ == "__main__":
    unittest.main()
