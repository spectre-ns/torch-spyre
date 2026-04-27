from dataclasses import dataclass
import copy
from itertools import combinations
from typing import List, Tuple, Literal, Optional
import numpy as np
from enum import Enum
from torch.utils._ordered_set import OrderedSet


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
    address: Optional[int] = None


# A type alias for the result of an allocation. The ith entry in the list is the state during
# the ith operation. It maps each allocated buffer to the scratch pad address where it is
# allocated at that point in time.
AllocationResult = list[dict[str, Allocation]]


def calculate_liveness(ops: list[Operation]) -> Tuple[dict[str, int], dict[str, int]]:
    # Verify that the actual run's allocation is valid. We assume that any allocation is "live"
    # during the entire liveness of the corresponding buffer.
    liveness_start = {}
    liveness_end = {}
    for i, op in enumerate(ops):
        for buffer_name in op.inputs + op.outputs:
            if buffer_name not in liveness_start:
                liveness_start[buffer_name] = i
            liveness_end[buffer_name] = i


def allocate_sorted_global(capacity: int, buffers: list[Buffer]) -> list[TimeBoundBuffer]:
    normalize_buffer_sizes(buffers)

    buffers.sort(
        key=lambda item: ((item.end_time - item.start_time), item.size), reverse=True
    )

    allocated_buffers: list[Buffer] = []
    for target in buffers:
        _place_buffer(capacity, target, allocated_buffers)

    return allocated_buffers


def _place_buffer(capacity: int, target: Buffer, allocated_buffers: list[Buffer]):
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
    free_gaps = _find_free_gaps(capacity, occupied_spaces)

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
        target.address = capacity
    allocated_buffers.append(target)


@dataclass
class TimeBoundBuffer:
    name: str
    size: int
    start_time: int
    end_time: int
    address: int = 0
    spilled: bool = False
    density: float = 0  # effectively normalized size for now


def _find_free_gaps(
    capacity: int, occupied: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Given a sorted list of occupied memory intervals, return the free gaps."""
    gaps = []
    current_addr = 0

    for start, end in occupied:
        if start > current_addr:
            gaps.append((current_addr, start))
        current_addr = max(current_addr, end)

    if current_addr < capacity:
        gaps.append((current_addr, capacity))

    return gaps


def normalize_buffer_sizes(buffers: list[Buffer]):
    if not buffers:
        return

    max_size = max(b.size for b in buffers)
    if max_size > 0:
        for b in buffers:
            b.density = b.size / max_size


def allocate_sa(
    capacity: int,
    buffers: list[Buffer],
    initial_temp: float = 2,
    alpha: float = 0.9,
    convergence_criteria: float = 0.0001,
    num_iterations: int = 10000,
    collision_penalty: float = 100,
    eviction_penalty: float = 1.5,
) -> list[LifetimeBoundBuffer]:
    rng = np.random.default_rng(seed=10)
    overlapping_time: dict[str, list[tuple[str, int]]] = {}
    for buffer in buffers:
        overlapping_time[buffer.name] = []
        buffer.address = rng.integers(0, capacity - buffer.size) if capacity > buffer.size else 0

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

    def evaluate_objective(candidate_buffers: list[Buffer]) -> float:
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

    best: list[Buffer] = buffers
    best_eval: float = evaluate_objective(best)

    current_eval: float = best_eval
    current: list[Buffer] = copy.deepcopy(best)

    reheating_step = initial_temp
    temp = initial_temp

    for i in range(num_iterations):
        idx = rng.integers(0, len(buffers))
        candidate: list[Buffer] = copy.deepcopy(current)
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


def allocate_buffers(
    capacity: int,
    ops: List[Operation],
    buffer_registry: dict[str, Buffer] = {},
    method: Literal["sorted-global", "annealing"] = "sorted-global",
    allow_fallback: bool = True,
    **kwargs,
) -> AllocationResult:
    # Check all buffers have a valid start and end time
    start_times, end_times = calculate_liveness(ops)
    buffers = [LifetimeBoundBuffer(
            name=n, 
            size=b.size, 
            start_time=start_times[n], 
            end_time=end_times[n], 
            address=0, 
            spilled=False, 
            density=0
        ) for n, b in buffer_registry]

    # Don't try to assign buffers which are bigger than capacity
    _buffers = [b for b in buffers if b.size <= capacity]

    result: None | AllocationResult = None
    match method:
        case "sorted-global":
            result = allocate_sorted_global(capacity, _buffers)
        case "annealing":
            result = allocate_sa(capacity, _buffers, **kwargs)
        case _:
            raise ValueError(f"Allocation method: {method} not not permitted")

    if allow_fallback and not np.all([b.spilled for b in result]):
        result = allocate_sa(capacity, _buffers, **kwargs)
        # TODO: compare results for optimality if no obvious improvement...
    
    allocation_result = [Allocation(b.name, Component.HBM if b.spilled else Component.LX, b.address) for b in result]
    return allocation_result
