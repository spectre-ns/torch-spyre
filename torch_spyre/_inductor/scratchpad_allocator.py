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

from typing import List
from abc import abstractmethod
from operation import Operation


class LxAllocator:
    def __init__(self, graph):
        self.graph = graph

    @abstractmethod
    def plan_allocation(self, operations: List[Operation]):
        """
        Interface method meant as a placeholder for the interface of LX
        allocator classes.

        Args:
            operations (List[Operation]): List of operations to be considered
                for LX planning.
        """        
        pass