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
        self.usage: list[
            LifetimeBoundBuffer
        ] = []

    def get_lowest_addr_in_use(self):
        if self.usage:
            return min([rec.address for rec in self.usage])
        return 0

    def get_highest_addr_in_use(self):
        if self.usage:
            return max([rec.address + rec.size for rec in self.usage])
        return 0

    def get_available_total(self):
        total_avail = self.limit
        for rec in self.usage:
            total_avail -= rec.size
        return total_avail

    def find_free_block(self, size_needed: int) -> Optional[int]:
        # cannot perform defragmentation yet, will add more cases in the future
        curr_lo = self.get_lowest_addr_in_use()
        curr_hi = self.get_highest_addr_in_use()
        if not self.usage or curr_lo >= size_needed:
            # completely free or enough room at addr0
            return 0
        elif curr_hi + size_needed - 1 < self.limit:
            address = math.ceil(curr_hi / self.alignment) * self.alignment
            if address < self.limit:
                return address
        elif self.usage:
            # find a "hole" between lowest and highest (assume a block was dealloc'ed)
            rec_only = self.usage  # simply drop tensor names, not needed
            sorted_rec = sorted(rec_only, key=lambda rec: rec.address)
            for i in range(len(sorted_rec) - 1):
                frag_st = sorted_rec[i].address + sorted_rec[i].size
                frag_end = sorted_rec[i + 1].address
                if frag_end - frag_st >= size_needed:
                    return frag_st
            return None
        else:
            # cannot find any free blocks
            return None

    def try_allocate(self, buffer: LifetimeBoundBuffer):
        """
        Simple reuse rule:
        1. for an "input" tensor, found a matched tensor (name and size) on LX
        2. for an output tensor, if this op is on the "white list" => prep for pinning
            => alloc a new LX block for the "output" of the op
        If can_reuse => add lx info to corresponding buffer.layout
        NOTE: 1. if an op, e.g. max, occurs multiple times on graph, output buffers will
                 have different names -> end-of-life analysis will take care of dealloc
              2. prev Op's sdsc.out.out.out.json may have useful info, not needed yet
              3. may be able to generalize this decision in buf end-of-life analysis
              4. greedy alloc may cause fragments, can further improve
        """
        # Decide whether to reuse.
        addr = self.find_free_block(buffer.size)

        if addr is not None:
            buffer.address = addr
            self.usage.append(buffer)
        else:
            buffer.address = None
  

    def deallocate(self, bufs: list[LifetimeBoundBuffer] | LifetimeBoundBuffer):
        """Try to deallocate each of the buffers in a list, if exists."""
        if isinstance(bufs, LifetimeBoundBuffer):
            bufs = [bufs]

        for buf in bufs:
            if buf in self.usage:
                del self.usage[self.usage.index(buf)]

    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        if not buffers:
            return []

        max_time = max(b.end_time for b in buffers)
        for idx in range(max_time):
            # attempt to allocate at based on time
            for buffer in buffers:
                if idx == buffer.end_time:
                    self.deallocate(buffer)

                if idx == buffer.start_time:
                    self.try_allocate(buffer)

        return buffers
