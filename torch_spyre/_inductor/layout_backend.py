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


from dataclasses import dataclass
from typing import list, Optional, dict, str, float
from abc import abstractmethod


@dataclass
class LifetimeBoundBuffer:
    """
    Defines the data fields required for a layout solver.
    The required heuristics are implementation defined.
    """

    name: str
    size: int
    start_time: int
    end_time: int
    heuristic: dict[str, float] = {}
    address: Optional[int] = None
    spilled: Optional[bool] = None


class LayoutSolver:
    """
    An abstract class for defining algorithms which solve
    the memory layout patterns based on provided sizes, lifetimes,
    and optional heuristics based on the implementation inputs.
    """

    @abstractmethod
    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        """
        Utilizes an implementation defined algorithm to determine
        whether buffers should be placed in a memory layout based
        on their attributes.

        Args:
            buffers (list[LifetimeBoundBuffer]): The set of candidate buffers for the memory layout

        Returns:
            list[LifetimeBoundBuffer]: The set of buffer with their placements defined.
        """
        pass
