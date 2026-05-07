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


from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
import math


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
    heuristic: dict[str, float] = field(default_factory=dict)
    address: Optional[int] = None


class LayoutSolver(ABC):
    """
    An abstract class for defining algorithms which solve
    memory layout patterns based on provided sizes, lifetimes,
    and optional heuristics based on the implementation details.
    """

    @abstractmethod
    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        """
        Utilizes an implementation defined algorithm to determine
        if and where buffers should be placed in lx memory based
        on their attributes.

        Args:
            buffers (list[LifetimeBoundBuffer]): The set of candidate buffers for memory planning

        Returns:
            list[LifetimeBoundBuffer]: The set of buffers with their placements defined.
        """
        pass


class GreedyLayoutSolver(LayoutSolver):
    def __init__(self, size: int, alignment: int = 128):
        self.limit = size
        self.alignment = alignment
        self.usage: list[LifetimeBoundBuffer] = []

    def _get_lowest_addr_in_use(self):
        if self.usage:
            return min([rec.address for rec in self.usage])
        return 0

    def _get_highest_addr_in_use(self):
        if self.usage:
            return max([rec.address + rec.size for rec in self.usage]) - 1
        return 0

    def _get_available_total(self):
        total_avail = self.limit
        for rec in self.usage:
            total_avail -= rec.size
        return total_avail

    def _find_free_block(self, size_needed: int) -> Optional[int]:
        assert all(x.address is not None for x in self.usage)
        curr_lo = self._get_lowest_addr_in_use()
        curr_hi = self._get_highest_addr_in_use()
        if not self.usage or curr_lo >= size_needed:
            return 0
        elif curr_hi + size_needed < self.limit:
            address = math.ceil((curr_hi + 1) / self.alignment) * self.alignment
            if address < self.limit:
                return address
        elif self.usage:
            #force allignment here. It might be best to inflate the buffers to the alignment
            self.usage.sort(key=lambda x: (x.address is None, x.address))
            for i in range(len(self.usage) - 1):
                assert (current_address := self.usage[i].address) is not None
                assert (next_address := self.usage[i + 1].address) is not None
                frag_st = current_address + self.usage[i].size
                frag_st = math.ceil((frag_st) / self.alignment) * self.alignment
                if next_address - frag_st >= size_needed:
                    return frag_st
            return None

        # cannot find any free blocks
        return None

    def _try_allocate(self, buffer: LifetimeBoundBuffer):
        """
        _summary_

        Args:
            buffer (LifetimeBoundBuffer): _description_
        """
        # Decide whether to reuse.
        addr = self._find_free_block(buffer.size)

        if addr is not None:
            buffer.address = addr
            self.usage.append(buffer)
        else:
            buffer.address = None

    def _try_deallocate(self, bufs: list[LifetimeBoundBuffer] | LifetimeBoundBuffer):
        """
        _summary_

        Args:
            bufs (list[LifetimeBoundBuffer] | LifetimeBoundBuffer): _description_
        """
        if isinstance(bufs, LifetimeBoundBuffer):
            bufs = [bufs]

        for buf in bufs:
            if buf in self.usage:
                del self.usage[self.usage.index(buf)]

    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        """
        _summary_

        Args:
            buffers (list[LifetimeBoundBuffer]): _description_

        Returns:
            list[LifetimeBoundBuffer]: _description_
        """
        if not buffers:
            return []

        # walk through all the transition points once in 
        # chronological order
        times = [b.start_time for b in buffers]
        times.extend([b.end_time for b in buffers])
        times = list(set(times))
        times.sort()

        for idx in times:
            # attempt to allocate at based on time
            for buffer in buffers:
                if idx == buffer.end_time:
                    self._try_deallocate(buffer)

                if idx == buffer.start_time:
                    self._try_allocate(buffer)

        return buffers
