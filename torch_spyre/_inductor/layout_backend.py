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
from operator import attrgetter
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

    def get_lowest_addr_in_use(self):
        if self.usage:
            return min([rec.address for rec in self.usage])
        return 0

    def get_highest_addr_in_use(self):
        if self.usage:
            return max([rec.address + rec.size for rec in self.usage]) - 1
        return 0

    def get_available_total(self):
        total_avail = self.limit
        for rec in self.usage:
            total_avail -= rec.size
        return total_avail

    def find_free_block(self, size_needed: int) -> Optional[int]:
        assert all(x.address is not None for x in self.usage)
        curr_lo = self.get_lowest_addr_in_use()
        curr_hi = self.get_highest_addr_in_use()
        if not self.usage or curr_lo >= size_needed:
            return 0
        elif curr_hi + size_needed < self.limit:
            address = math.ceil((curr_hi + 1) / self.alignment) * self.alignment
            if address < self.limit:
                return address
        elif self.usage:
            # force allignment here. It might be best to inflate the buffers to the alignment
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

    def try_allocate(self, buffer: LifetimeBoundBuffer):
        """
        _summary_

        Args:
            buffer (LifetimeBoundBuffer): _description_
        """
        # Decide whether to reuse.
        addr = self.find_free_block(buffer.size)

        if addr is not None:
            buffer.address = addr
            self.usage.append(buffer)
        else:
            buffer.address = None

    def try_deallocate(self, bufs: list[LifetimeBoundBuffer] | LifetimeBoundBuffer):
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
                    self.try_deallocate(buffer)

                if idx == buffer.start_time:
                    self.try_allocate(buffer)

        return buffers


class SortedLayoutSolver(LayoutSolver):
    def __init__(self, capacity: int, sorting_attribute="size"):
        assert sorting_attribute in LifetimeBoundBuffer.__dataclass_fields__, (
            "sorting_attribute must be a valid attribute of LifetimeBoundBuffer"
        )
        self.capacity = capacity
        self.sorting_attribute = sorting_attribute

    def allocate_sorted_global(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        buffers.sort(
            key=attrgetter(self.sorting_attribute),
            reverse=True,
        )

        allocated_buffers: list[LifetimeBoundBuffer] = []
        for target in buffers:
            self.place_buffer(target, allocated_buffers)

        return allocated_buffers

    def place_buffer(
        self, target: LifetimeBoundBuffer, allocated_buffers: list[LifetimeBoundBuffer]
    ):
        # Find all currently allocated buffers that overlap in TIME
        overlapping_in_time = []
        for alloc in allocated_buffers:
            if max(target.start_time, alloc.start_time) < min(
                target.end_time, alloc.end_time
            ):
                overlapping_in_time.append(alloc)

        # Extract their memory address ranges and sort them spatially
        occupied_spaces = sorted(
            [
                (b.address, b.address + b.size)
                for b in overlapping_in_time
                if b.address is not None
            ]
        )

        # Find all free gaps in memory during this time window
        free_gaps = self.find_free_gaps(occupied_spaces)

        # Filter gaps to only those large enough to hold the target buffer
        valid_gaps = [gap for gap in free_gaps if gap[1] - gap[0] >= target.size]

        if valid_gaps:
            # BEST-FIT LOGIC: Find the interval closest matching the needed space
            best_gap = min(valid_gaps, key=lambda gap: (gap[1] - gap[0]) - target.size)

            # Assign the address (starting at the bottom of the best-fit gap)
            target.address = best_gap[0]
        else:
            # No valid gap found, the buffer must be spilled to DRAM
            target.spilled = True
            target.address = None
        allocated_buffers.append(target)

    def find_free_gaps(self, occupied: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Given a sorted list of occupied memory intervals, return the free gaps."""
        gaps = []
        current_addr = 0

        for start, end in occupied:
            if start > current_addr:
                gaps.append((current_addr, start))
            current_addr = max(current_addr, end)

        if current_addr < self.capacity:
            gaps.append((current_addr, self.capacity))

        return gaps

    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        return self.allocate_sorted_global(buffers)
