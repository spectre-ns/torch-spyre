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


def calculate_logical_lifetimes(ops: list[Operation]) -> list[Buffer]:
    pass


def allocate_greedy(capacity: int, buffers: list[Buffer]) -> list[Buffer]:
    # TODO: Adapt implementation from IBM as a baseline
    pass


def allocate_greedy_global(capacity: int, buffers: list[Buffer]) -> list[Buffer]:
    # 1. Calculate density for all buffers
    for _, b in buffers.items():
        if not b.density:
            b.calculate_density()

    # 2. Sort buffers based on priority:
    # Priority 1: Pinned status
    # Priority 2: Highest Density
    # Priority 3: Largest Size (Tie-breaker)

    buffers = dict(
        sorted(
            buffers.items(),
            key=lambda item: (item[1].pinned, item[1].density, item[1].size),
            reverse=True,
        )
    )

    # 3. Place each buffer
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


def allocate_sa(capacity: int, buffers: list[Buffer], **kwargs) -> list[Buffer]:
    # Check all buffers have a valid start and end time
    for b in buffers:
        assert b.start_time is not None and b.end_time is not None, \
            "Start and end times must be evaluated prior to global scheduling"

    rng = np.random.default_rng(seed=0)
    overlapping_time = {}
    for buffer in buffers:
        overlapping_time[buffer.name] = []
        buffer.address = rng.integers(0, capacity)

    normalize_buffer_sizes(buffers)

    for first, second in combinations(buffers, 2):
        time_overlap = min(second.end_time, first.end_time) - max(
            second.start_time, first.start_time
        )
        if time_overlap > 0:
            overlapping_time[first.name].append((second.name, time_overlap))

    for buffer in buffers:
        eviction_length = -buffer.address if buffer.address < 0 else np.clip(buffer.address + buffer.size - capacity, 0, buffer.size)
        buffer.spilled = bool(eviction_length)

    def evaluate_objective(candidate_buffers: dict[str, Buffer]) -> float:
        __COLLISION_PENALTY__ = 100
        __EVICTION_PENALTY__ = 1.5

        collision_term = 0.0
        for buffer in candidate_buffers:
            for second, time_overlap in overlapping_time[buffer.name]:
                second_buffer = next((b for b in candidate_buffers if b.name == second), None)
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
                    collision_term += __COLLISION_PENALTY__ * area_overlap

        eviction_penalty = 0.0
        for alloc in candidate_buffers:
            if alloc.spilled:
                eviction_penalty += (
                    alloc.density
                    * __EVICTION_PENALTY__
                    * (
                        np.abs(-alloc.address)
                        if alloc.address < 0
                        else np.clip(
                            alloc.address + alloc.size - capacity, 0, np.inf
                        )
                    )
                    / (alloc.end_time - alloc.start_time)
                )

        return collision_term + eviction_penalty

    # ==========================================
    # LIVE PLOTTING SETUP
    # ==========================================
    plt.ion()  # Turn on interactive mode
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
    ax.set_title("Live Simulated Annealing")

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

    # ==========================================
    # ANNEALING LOOP
    # ==========================================
    e_list = []
    temp_list = []
    best_list = []

    best: Dict[str, Buffer] = buffers  # self.allocate_deterministic(buffers)
    best_eval: float = evaluate_objective(best)

    current_eval: float = best_eval
    current: Dict[str, Buffer] = copy.deepcopy(best)

    alpha = 0.0005
    initial_temp: float = 2
    num_iterations: int = 20000  # Shortened for demonstration purposes
    temp: float = initial_temp

    # IMPORTANT: How many iterations between screen updates
    update_interval = 50

    for i in range(num_iterations):
        candidate: Dict[str, Buffer] = copy.deepcopy(current)
        e: np.ndarray = np.array(
            [int(rng.integers(-capacity / 2, capacity / 2) * temp) for b in candidate]
        )

        for idx, buffer in enumerate(candidate):
            buffer.address = buffer.address + e[idx]
            eviction_length = (
                -buffer.address
                if buffer.address < 0
                else np.clip(buffer.address + buffer.size - capacity, 0, buffer.size)
            )
            buffer.spilled = bool(eviction_length)

        candidate_eval: float = evaluate_objective(candidate)

        if candidate_eval < best_eval:
            best, best_eval = copy.deepcopy(candidate), candidate_eval
            best_list.append(best_eval)

        diff: float = candidate_eval - current_eval
        temp = initial_temp * np.exp(-alpha * i)
        metropolis: float = np.exp(-diff / temp) if temp > 0 else 0

        if diff < 0 or rng.random() < metropolis:
            current, current_eval = candidate, candidate_eval

        e_list.append(np.sum(np.sqrt(e**2)))
        temp_list.append(temp)

        # --- LIVE UPDATE THE PLOT ---
        if i % update_interval == 0:
            step_text.set_text(f"Iteration: {i} / {num_iterations} | Temp: {temp:.2f}")

            # Show the 'current' state bouncing around
            for b in candidate:
                rects[b.name].set_visible(True)
                texts[b.name].set_visible(True)
                rects[b.name].set_y(b.address)

                cx = b.start_time + (b.end_time - b.start_time) / 2
                cy = b.address + b.size / 2
                texts[b.name].set_position((cx, cy))

            # Force matplotlib to draw the updated frame
            fig.canvas.draw()
            fig.canvas.flush_events()

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

    # Turn off interactive mode at the end and snap to the absolute best solution found
    plt.ioff()

    msg = "FINISHED! Showing Best Configuration"
    for b in best:
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
    spilled_buffers = [
        b.name + f" : {b.density:.2f}" for b in best if b.spilled
    ]
    if spilled_buffers:
        msg += f"\nSpilled to DRAM: {', '.join(spilled_buffers)}"
    else:
        msg += "\n No evictions required"

    step_text.set_text(msg)

    fig.canvas.draw()

    return best, e_list, temp_list, best_list


# TODO: not sure how to handle this one yet
def calculate_useage():
    pass


def allocate_buffers(
    capacity: int,
    buffers: List[Buffer],
    method: Literal["greedy", "greedy-global", "annealing"],
    **kwargs,
):
    result: None | List[Buffer] = None
    match method:
        case "greedy":
            result = allocate_greedy(capacity, buffers)
        case "greedy-global":
            result = allocate_greedy_global(capacity, buffers)
        case "annealing":
            result = allocate_sa(capacity, buffers, kwargs)
        case _:
            raise ValueError(f"Allocation method: {method} not not permitted")

    if "allow_fallback" in kwargs:
        if not np.all([b.spilled for b in result]) and kwargs["allow_fallback"]:
            fallback_result = allocate_sa(capacity, buffers, kwargs)
            # TODO: compare results for optimality...
            return fallback_result

    return result


class ScratchpadAllocator:
    def __init__(self, capacity: int):
        self.rng = np.random.default_rng(seed=70)
        self.capacity = capacity
        self.allocated_buffers: dict[Buffer] = {}
        self.overlapping_time: dict[str, List[Tuple[str, int]]] = {}


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    SPM_CAPACITY = 1000

    buffer_list = [
        Buffer(name="A", size=500, start_time=0, end_time=1),
        Buffer(name="B", size=400, start_time=0, end_time=4),
        Buffer(name="C", size=900, start_time=4, end_time=5),
        Buffer(name="D", size=500, start_time=1, end_time=4),
        Buffer(name="E", size=100, start_time=0, end_time=5),
    ]

    # This will now pop open a window and animate live
    best_allocation, e, temp, best_scores = allocate_sa(SPM_CAPACITY, buffer_list)

    # After it finishes, plot the metrics
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(12, 4))
    ax1.plot(e)
    ax1.set_title("Energy (E)")
    ax2.plot(temp)
    ax2.set_title("Temperature")
    ax3.semilogy(best_scores)
    ax3.set_title("Best Score")
    plt.tight_layout()

    # Keep the final windows open
    plt.show()
