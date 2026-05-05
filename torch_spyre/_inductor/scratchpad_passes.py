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

from abc import abstractmethod
from torch._inductor.graph import GraphLowering


class SpyreLxOptimizationPass:
    @abstractmethod
    def apply_pass(self, graph: GraphLowering) -> None:
        """
        Accepts a candidate graph to be optimized and evaluated for lx memory allocation.
        `graph` will be mutated according in an implementation defined way. The order and
        number of nodes in the graph may change as a result of an optimization pass.

        Args:
            graph (GraphLowering): The graph to be optimized to LX memory allocation
        """
        pass
