import math
from dataclasses import dataclass
import copy
from itertools import combinations
from typing import List, Tuple
import numpy as np
from enum import Enum
from torch.utils._ordered_set import OrderedSet
from abc import abstractmethod


@dataclass
class LifetimeBoundBuffer:
    name: str
    size: int
    start_time: int
    end_time: int
    address: int = 0
    spilled: bool = False
    density: float = 0  # effectively normalized size for now


class BufferDeviceLayout:
    """This class mimics the FixedTiledLayout.device_layout field."""

    def __init__(self, size: int):
        self.device_size = [(size + 127) // 128, 128]


class BufferLayout:
    """This class mimics the TensorBox.layout field (a FixedTiledLayout)."""

    def __init__(self, size: int):
        self.device_layout = BufferDeviceLayout(size)
        self.size = size
        self.allocation = {}


class Buffer:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.layout = BufferLayout(size)
        self.data = self  # This helps 'scratchpad'


@dataclass
class ReadWrites:
    reads: OrderedSet[Buffer]
    writes: OrderedSet[Buffer]


@dataclass
class Operation:
    name: str
    inputs: list[str]
    outputs: list[str]
    _buffer_registry: dict[str, "Buffer"]

    # To make scratchpad.py work, we add origin_node and target fields that point to the op itself,
    # a field _opname that is the same as name, and a field op_it_space_splits that is used in core
    # division. (If the value of op_it_space_splits is different for operations in a sequence, that
    # blocks LX allocation, so we make sure it is always the same.)
    op_it_space_splits = None
    origin_node = None
    target = None
    _opname = None

    def __post_init__(self):
        self.op_it_space_splits = []
        self.origin_node = self
        self.target = self
        self._opname = self.name

    def get_read_writes(self) -> ReadWrites:
        # Returns a list of (buffer_name, "read" or "write") for all buffers used by this operation.
        reads = OrderedSet(
            self._buffer_registry[buffer_name] for buffer_name in self.inputs
        )
        writes = OrderedSet(
            self._buffer_registry[buffer_name] for buffer_name in self.outputs
        )
        return ReadWrites(reads=reads, writes=writes)

    def get_read_names(self):
        return self.inputs


class Component(Enum):
    LX = "LX"
    HBM = "HBM"


@dataclass
class Allocation:
    buffer: str
    component: Component = Component.LX
    # If the component is LX, then the address must be an integer. If the component is HBM, we don't
    # care about the address; this is encoded by the address being None. (This is enforced in
    # TestExamplePattern.verify_pattern.)
    address: int = 0


# A type alias for the result of an allocation. The ith entry in the list is the state during
# the ith operation. It maps each allocated buffer to the scratch pad address where it is
# allocated at that point in time.
AllocationResult = list[Allocation]

def get_component(spilled: bool) -> Component:
    if spilled:
        return Component.HBM
    return Component.LX

class LayoutSolver:
    @abstractmethod
    def plan_layout(self,
                    buffers: list[LifetimeBoundBuffer]) -> AllocationResult:
        pass


def normalize_buffer_sizes(buffers: list[LifetimeBoundBuffer]):
    if not buffers:
        return
    max_size = max(b.size for b in buffers)
    if max_size > 0:
        for b in buffers:
            b.density = b.size / max_size


class SortingSolver(LayoutSolver):
    def __init__(self, capacity: int):
        self.capacity = capacity

    def allocate_sorted_global(self,
                               buffers: list[LifetimeBoundBuffer]
                               ) -> list[LifetimeBoundBuffer]:
        normalize_buffer_sizes(buffers)

        buffers.sort(
            key=lambda item: ((item.end_time - item.start_time), item.size), reverse=True
        )

        allocated_buffers: list[LifetimeBoundBuffer] = []
        for target in buffers:
            self.place_buffer(target, allocated_buffers)

        return allocated_buffers

    def place_buffer(
            self,
            target: LifetimeBoundBuffer,
            allocated_buffers: list[LifetimeBoundBuffer]):
        # Find all currently allocated buffers that overlap in TIME
        overlapping_in_time = []
        for alloc in allocated_buffers:
            if max(target.start_time, alloc.start_time) < min(
                target.end_time, alloc.end_time
            ):
                overlapping_in_time.append(alloc)

        # Extract their memory address ranges and sort them spatially
        occupied_spaces = sorted(
            [(b.address, b.address + b.size) for b in overlapping_in_time]
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
            target.address = -1 # use the convension from elsewhere
        allocated_buffers.append(target)

    def find_free_gaps(
        self, occupied: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
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
    
    def plan_layout(self, 
                    buffers: list[LifetimeBoundBuffer]) -> AllocationResult:
        allocations = self.allocate_sorted_global(buffers)
        return [
            Allocation(
                allocation.name,
                get_component(allocation.spilled),
                allocation.address
                ) for allocation in allocations
        ]


class SimulatedAnnealingSolver(LayoutSolver):
    def __init__(
        self,
        capacity: int,
        initial_temp: float = 2,
        alpha: float = 0.9,
        convergence_criteria: float = 0.0001,
        num_iterations: int = 10000,
        collision_penalty: float = 100,
        eviction_penalty: float = 1.5
    ):
        self.capacity = capacity
        self.initial_temp = initial_temp
        self.alpha = alpha
        self.convergence_criteria = convergence_criteria
        self.num_iterations = num_iterations
        self.collision_penalty = collision_penalty
        self.eviction_penalty = eviction_penalty

    def allocate_sa(
        self,
        buffers: list[LifetimeBoundBuffer]
    ) -> AllocationResult:
        rng = np.random.default_rng(seed=10)
        overlapping_time: dict[str, list[tuple[str, int]]] = {}
        for buffer in buffers:
            overlapping_time[buffer.name] = []
            buffer.address = rng.integers(0, self.capacity - buffer.size) if self.capacity > buffer.size else 0

        normalize_buffer_sizes(buffers)

        for first, second in combinations(buffers, 2):
            time_overlap = min(second.end_time, first.end_time) - max(
                second.start_time, first.start_time
            )
            if time_overlap > 0:
                overlapping_time[first.name].append((second.name, time_overlap))

        for buffer in buffers:
            eviction_length = (
                -buffer.address
                if buffer.address < 0
                else np.clip(buffer.address + buffer.size - capacity, 0, buffer.size)
            )
            buffer.spilled = bool(eviction_length)

        def evaluate_objective(candidate_buffers: list[LifetimeBoundBuffer]) -> float:
            collision_term = 0.0
            for buffer in candidate_buffers:
                for second, time_overlap in overlapping_time[buffer.name]:
                    second_buffer = next((b for b in candidate_buffers if b.name == second))
                    end = min(
                        buffer.address + buffer.size,
                        second_buffer.address + second_buffer.size,
                    )
                    start = max(
                        buffer.address,
                        second_buffer.address,
                    )
                    if start < end:
                        area_overlap = (end - start) * time_overlap
                        collision_term += collision_penalty * area_overlap

            eviction_term = 0.0
            for alloc in candidate_buffers:
                if alloc.spilled:
                    eviction_term += eviction_penalty * (
                        np.abs(-alloc.address)
                        if alloc.address < 0
                        else np.clip(alloc.address + alloc.size - capacity, 0, np.inf)
                    )

            return collision_term + eviction_term

        # ==========================================
        # ANNEALING LOOP
        # ==========================================
        current_list = []

        best: list[LifetimeBoundBuffer] = buffers
        best_eval: float = evaluate_objective(best)

        current_eval: float = best_eval
        current: list[LifetimeBoundBuffer] = copy.deepcopy(best)

        reheating_step = initial_temp
        temp = initial_temp

        for i in range(num_iterations):
            idx = rng.integers(0, len(buffers))
            candidate: list[LifetimeBoundBuffer] = copy.deepcopy(current)
            buffer = candidate[idx]
            e: np.ndarray = rng.integers(-capacity, capacity) * temp
            buffer.address = buffer.address + e
            eviction_length = (
                -buffer.address
                if buffer.address < 0
                else np.clip(buffer.address + buffer.size - capacity, 0, buffer.size)
            )
            buffer.spilled = bool(eviction_length)

            candidate_eval: float = evaluate_objective(candidate)

            if candidate_eval < best_eval:
                best, best_eval = copy.deepcopy(candidate), candidate_eval

            diff: float = candidate_eval - current_eval
            metropolis: float = np.exp(-diff / temp) if temp > 0 else 0

            if diff < 0 or rng.random() < metropolis:
                current, current_eval = candidate, candidate_eval
                current_list.append(current_eval)

            overlapping = False
            for buffer in best:
                for second_buffer_name, time_overlap in overlapping_time[buffer.name]:
                    second_buffer = next((b for b in best if b.name == second_buffer_name))
                    end = min(
                        buffer.address + buffer.size,
                        second_buffer.address + second_buffer.size,
                    )
                    start = max(buffer.address, second_buffer.address)
                    if start < end:
                        overlapping = True
                        break

            if np.all([not b.spilled for b in best]) and not overlapping:
                break

            # Check for reheating / melting / restart condition if the gradient is not
            # descending fast enough relative to the energy in the current
            # result. Increase temp and revert back to best result for
            # search restart.
            if len(current_list) > 10:
                if (
                    np.abs(np.mean(np.gradient(current_list, 1)[-10:])) / current_list[-1]
                    < convergence_criteria
                ):
                    temp = temp + reheating_step
                    current = best
                    current_list = []

            temp = temp * alpha

        return best
    
    def plan_layout(self, buffers: dict) -> AllocationResult:
        return self.allocate_sa(buffers)


class GreedyLayoutSolver(LayoutSolver):
    def __init__(self, size: int):
        # scratch pad is 2MB = 2<<20 bytes in total. preserve total * DXP_LX_FRAC_AVAIL
        # for backend usage unless specified otherwise
        self.limit = size
        self.usage: dict = {}  # each record will be tensor_name:{"addr": yy, "size": zz}
        self.lx_usage_hist: list = []

    def get_lowest_addr_in_use(self):
        if len(self.usage) > 0:
            return min([rec["addr"] for rec in self.usage.values()])
        return None

    def get_highest_addr_in_use(self):
        if len(self.usage) > 0:
            return max([rec["addr"] + rec["size"] for rec in self.usage.values()])
        return None

    def get_available_total(self):
        total_avail = self.limit
        for rec in self.usage.values():
            total_avail -= rec["size"]
        return total_avail

    def find_free_block(self, size_needed: int):
        # cannot perform defragmentation yet, will add more cases in the future
        curr_lo = self.get_lowest_addr_in_use()
        curr_hi = self.get_highest_addr_in_use()
        if len(self.usage) == 0 or curr_lo >= size_needed:
            # completely free or enough room at addr0
            return 0
        elif curr_hi + size_needed < self.limit:
            # enough room at higher addr, return next 128-multiple
            return math.ceil(curr_hi / 128) * 128
        elif len(self.usage) > 1:
            # find a "hole" between lowest and highest (assume a block was dealloc'ed)
            rec_only = list(self.usage.values())  # simply drop tensor names, not needed
            sorted_rec = sorted(rec_only, key=lambda rec: rec["addr"])
            for i in range(len(sorted_rec) - 1):
                frag_st = sorted_rec[i]["addr"] + sorted_rec[i]["size"]
                frag_end = sorted_rec[i + 1]["addr"]
                if frag_end - frag_st >= size_needed:
                    return frag_st
            return -1
        else:
            # cannot find any free blocks
            return -1

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
        addr = -1
        tensor_on_lx = self.usage.get(buffer.name, {})
        size_match = tensor_on_lx.get("size", 0) == buffer.size

        if tensor_on_lx and size_match:
            addr = self.usage[buffer.name]["addr"]
        else:
            addr = self.find_free_block(buffer.size)

        # add lx info into V.graph.buffers.layout for later codegen use.
        if addr != -1:
            self.usage[buffer.name] = {"addr": addr, "size": buffer.size}

            # Record usage history for debugging
            self.lx_usage_hist.append(
                {
                    "tensor_name": buffer.name,
                    "addr": addr,
                    "size": buffer.size,
                }
            )
    
    def deallocate(self, bufs: list[str] | str):
        """Try to deallocate each of the buffers in a list, if exists."""
        if isinstance(bufs, str):
            bufs = [bufs]

        for buf in bufs:
            if buf in self.usage:
                del self.usage[buf]

    def plan_layout(
            self,
            buffers: list[LifetimeBoundBuffer]) -> AllocationResult:
        max_time = max(b.end_time for b in buffers)
        for idx in range(max_time):
            # attempt to allocate at based on time
            for buffer in buffers:
                if idx == buffer.start_time:
                    self.try_allocate(buffer)

                if idx == buffer.end_time:
                    self.deallocate(buffer.name)

        seen = set() 
        return [
                Allocation(allocation["tensor_name"], Component.LX, allocation["addr"])
                for allocation in self.lx_usage_hist
                if allocation["tensor_name"] not in seen
                and not seen.add(allocation["tensor_name"])
            ]
    