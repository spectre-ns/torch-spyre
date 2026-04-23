from dataclasses import dataclass
import copy
from itertools import combinations
from typing import List, Tuple, Dict, Literal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


@dataclass
class ReadWrites:
    reads: list[str]
    writes: list[str]


@dataclass
class Buffer:
    name: str
    size: int
    start_time: int | None = None
    end_time: int | None = None
    address: float | None = None
    spilled: bool | None = None
    density: float | None = None  # effectively normalized size for now


@dataclass
class Operation:
    name: str
    inputs: list[str]
    outputs: list[str]
    _buffer_registry: dict[str, Buffer]

    # To make scratchpad.py work, we add an origin_node field that points to the op itself.
    origin_node = None

    def __post_init__(self):
        self.origin_node = self

    def get_read_writes(self) -> ReadWrites:
        # Returns a list of (buffer_name, "read" or "write") for all buffers used by this operation.
        reads = [self._buffer_registry[buffer_name] for buffer_name in self.inputs]
        writes = [self._buffer_registry[buffer_name] for buffer_name in self.outputs]
        return ReadWrites(reads=reads, writes=writes)


def plot_layout(capacity: int, buffers: list[Buffer]):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(
        y=capacity,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"SPM Ceiling ({capacity})",
    )
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2, label="SPM Floor")

    max_time = max(b.end_time for b in buffers)
    ax.set_xlim(0, max_time + 1)
    ax.set_ylim(-capacity, capacity * 2)
    ax.set_xlabel("Time (Logical Steps)")
    ax.set_ylabel("Memory Address (Bytes)")
    ax.set_title("Calculated Memory Layout")

    rects = {}
    texts = {}

    # Initialize the visual blocks
    for b in buffers:
        rect = patches.Rectangle(
            (b.start_time, 0),
            b.end_time - b.start_time,
            b.size,
            linewidth=1.5,
            edgecolor="black",
            facecolor="skyblue",
            alpha=0.8,
        )
        ax.add_patch(rect)
        rects[b.name] = rect

        txt = ax.text(
            0,
            0,
            f"Buffer {b.name}",
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
        )
        texts[b.name] = txt

    step_text = ax.text(
        0.02, 0.90, "", transform=ax.transAxes, fontsize=12, weight="bold"
    )
    fig.canvas.draw()
    fig.canvas.flush_events()

    msg = "FINISHED! Showing Final Configuration"
    for b in buffers:
        if not b.spilled:
            rects[b.name].set_visible(True)
            texts[b.name].set_visible(True)
            rects[b.name].set_y(b.address)
            texts[b.name].set_position(
                (
                    b.start_time + (b.end_time - b.start_time) / 2,
                    b.address + b.size / 2,
                )
            )
        else:
            rects[b.name].set_visible(False)
            texts[b.name].set_visible(False)

    # Add a warning box for spilled buffers
    spilled_buffers = [b.name + f" : {b.density:.2f}" for b in buffers if b.spilled]
    if spilled_buffers:
        msg += f"\nSpilled to DRAM: {', '.join(spilled_buffers)}"
    else:
        msg += "\n No evictions required"

    step_text.set_text(msg)
    plt.show()


def calculate_logical_lifetimes(ops: list[Operation]) -> list[Buffer]:
    pass


def allocate_greedy(capacity: int, buffers: list[Buffer]) -> list[Buffer]:
    # TODO: Adapt implementation from IBM as a baseline
    pass


def allocate_greedy_global(capacity: int, buffers: list[Buffer]) -> list[Buffer]:
    normalize_buffer_sizes(buffers)

    buffers.sort(
        key=lambda item: ((item.end_time - item.start_time), item.size), reverse=True
    )

    allocated_buffers = []
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


def normalize_buffer_sizes(buffers: dict[str, Buffer]):
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
) -> list[Buffer]:
    rng = np.random.default_rng(seed=10)
    overlapping_time = {}
    for buffer in buffers:
        overlapping_time[buffer.name] = []
        buffer.address = rng.integers(0, capacity - buffer.size)

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

    def evaluate_objective(candidate_buffers: dict[str, Buffer]) -> float:
        collision_term = 0.0
        for buffer in candidate_buffers:
            for second, time_overlap in overlapping_time[buffer.name]:
                second_buffer = next(
                    (b for b in candidate_buffers if b.name == second), None
                )
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

    best: Dict[str, Buffer] = buffers
    best_eval: float = evaluate_objective(best)

    current_eval: float = best_eval
    current: Dict[str, Buffer] = copy.deepcopy(best)

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
            for second, time_overlap in overlapping_time[buffer.name]:
                second_buffer = next((b for b in best if b.name == second), None)
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


# TODO: not sure how to handle this one yet
def calculate_useage():
    pass


def allocate_buffers(
    capacity: int,
    buffers: List[Buffer],
    method: Literal["greedy", "ordered-global", "annealing"],
    allow_fallback: bool = False,
    **kwargs,
):
    # Check all buffers have a valid start and end time
    for b in buffers:
        assert b.start_time is not None and b.end_time is not None, (
            "Start and end times must be evaluated prior to scheduling"
        )

        assert b.start_time < b.end_time, "Buffer must start before end"

    # Don't try to assign buffers which are bigger than capacity
    _buffers = [b for b in buffers if b.size < capacity]

    result: None | List[Buffer] = None
    match method:
        case "greedy":
            result = allocate_greedy(capacity, _buffers)
        case "ordered-global":
            result = allocate_greedy_global(capacity, _buffers)
        case "annealing":
            result = allocate_sa(capacity, _buffers, **kwargs)
        case _:
            raise ValueError(f"Allocation method: {method} not not permitted")

    if allow_fallback and not np.all([b.spilled for b in result]):
        fallback_result = allocate_sa(capacity, _buffers, **kwargs)
        # TODO: compare results for optimality if no obvious improvement...
        return fallback_result
    return result


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    SPM_CAPACITY = 1000

    buffer_list = [
        Buffer(name="A", size=750, start_time=0, end_time=3),
        Buffer(name="B", size=50, start_time=1, end_time=4),
        Buffer(name="C", size=800, start_time=3, end_time=5),
    ]

    best_allocation = allocate_buffers(
        SPM_CAPACITY, buffer_list, "ordered-global", allow_fallback=True
    )
    plot_layout(SPM_CAPACITY, best_allocation)
