import dataclasses
import copy
from itertools import combinations
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

@dataclasses.dataclass
class Buffer:
    reads: int
    writes: int
    size: int
    start_time: int
    end_time: int
    pinned: bool
    density: float | None = None
    address: float = 0.0
    spilled: bool = False

    def calculate_density(self):
        duration = max(1, self.end_time - self.start_time)
        self.density = self.size


class ScratchpadAllocator:
    def __init__(self, capacity: int):
        self.rng = np.random.default_rng(seed=70)
        self.capacity = capacity
        self.allocated_buffers: dict[Buffer] = {}
        self.overlapping_time: dict[str, List[Tuple[str, int]]] = {}

    def normalize_densities(self, buffers: dict[str, Buffer]):
        if not buffers:
            return

        max_density = max(b.density for _, b in buffers.items())
        if max_density > 0:
            for _, b in buffers.items():
                b.density /= max_density

    def allocate_annealing(self, buffers: dict[str, Buffer]):
        for key, buffer in buffers.items():
            self.overlapping_time[key] = []
            buffer.address = self.rng.integers(0, self.capacity)

        for _, b in buffers.items():
            b.calculate_density()

        self.normalize_densities(buffers)

        for first, second in combinations(buffers.keys(), 2):
            time_overlap = min(buffers[second].end_time, buffers[first].end_time) - max(buffers[second].start_time, buffers[first].start_time)
            if time_overlap > 0:
                self.overlapping_time[first].append((second, time_overlap))

        def evaluate_objective(candidate_buffers: dict[str, Buffer]) -> float:
            __COLLISION_PENALTY__ = 100
            __PINNED_PENALTY__ = 100000
            __EVICTION_PENALTY__ = 1.5

            collision_term = 0.0
            for first, overlapping_buffers in self.overlapping_time.items():
                for second, time_overlap in overlapping_buffers:
                    end = min(candidate_buffers[first].address + candidate_buffers[first].size,
                              candidate_buffers[second].address + candidate_buffers[second].size)
                    start = max(candidate_buffers[first].address, candidate_buffers[second].address)
                    if start < end:
                        area_overlap = (end - start) * time_overlap
                        collision_term += __COLLISION_PENALTY__ * area_overlap

            eviction_penalty = 0.0
            for key, alloc in candidate_buffers.items():
                if alloc.pinned and alloc.spilled:
                    eviction_penalty += __PINNED_PENALTY__
                elif alloc.spilled:
                    eviction_penalty += (
                        alloc.density * __EVICTION_PENALTY__ * 
                        (np.abs(-alloc.address) if alloc.address < 0 else np.clip(alloc.address + alloc.size - self.capacity, 0, np.inf)) 
                        / (alloc.end_time - alloc.start_time)
                        )

            return collision_term + eviction_penalty

        # ==========================================
        # LIVE PLOTTING SETUP
        # ==========================================
        plt.ion() # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axhline(y=self.capacity, color='red', linestyle='--', linewidth=2, label=f'SPM Ceiling ({self.capacity})')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='SPM Floor')

        max_time = max(b.end_time for b in buffers.values())
        ax.set_xlim(0, max_time + 1)
        ax.set_ylim(-self.capacity, self.capacity *2)
        ax.set_xlabel('Time (Logical Steps)')
        ax.set_ylabel('Memory Address (Bytes)')
        ax.set_title('Live Simulated Annealing')

        rects = {}
        texts = {}

        # Initialize the visual blocks
        for key, b in buffers.items():
            rect = patches.Rectangle((b.start_time, 0), b.end_time - b.start_time, b.size,
                                     linewidth=1.5, edgecolor='black', facecolor='skyblue', alpha=0.8)
            ax.add_patch(rect)
            rects[key] = rect

            txt = ax.text(0, 0, f"Buffer {key}", ha='center', va='center', fontsize=10, weight='bold')
            texts[key] = txt

        step_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12, weight='bold')
        fig.canvas.draw()
        fig.canvas.flush_events()

        # ==========================================
        # ANNEALING LOOP
        # ==========================================
        e_list = []
        temp_list = []
        best_list = []

        best: Dict[str, Buffer] = buffers #self.allocate_deterministic(buffers)
        best_eval: float = evaluate_objective(best)

        current_eval: float = best_eval
        current: Dict[str, Buffer] = copy.deepcopy(best)

        alpha = .0005
        initial_temp: float = 2
        num_iterations: int = 10000 # Shortened for demonstration purposes
        temp: float = initial_temp

        # IMPORTANT: How many iterations between screen updates
        update_interval = 50

        for i in range(num_iterations):
            candidate: Dict[str, Buffer] = copy.deepcopy(current)
            e: np.ndarray = np.array([int(self.rng.integers(-self.capacity/2, self.capacity/2) * temp) for _, b in candidate.items()])

            for idx, (_, buffer) in enumerate(candidate.items()):
                buffer.address = buffer.address + e[idx]
                eviction_length = -buffer.address if buffer.address < 0 else np.clip(buffer.address + buffer.size - self.capacity, 0, buffer.size)
                buffer.spilled = bool(eviction_length)

            candidate_eval: float = evaluate_objective(candidate)

            if candidate_eval < best_eval:
                best, best_eval = copy.deepcopy(candidate), candidate_eval
                best_list.append(best_eval)

            diff: float = candidate_eval - current_eval
            temp = initial_temp * np.exp(-alpha * i)
            metropolis: float = np.exp(-diff / temp) if temp > 0 else 0

            if diff < 0 or self.rng.random() < metropolis:
                current, current_eval = candidate, candidate_eval

            e_list.append(np.sum(np.sqrt(e**2)))
            temp_list.append(temp)

            # --- LIVE UPDATE THE PLOT ---
            if i % update_interval == 0:
                step_text.set_text(f"Iteration: {i} / {num_iterations} | Temp: {temp:.2f}")

                # Show the 'current' state bouncing around
                for key, b in candidate.items():
                    rects[key].set_visible(True)
                    texts[key].set_visible(True)
                    rects[key].set_y(b.address)

                    cx = b.start_time + (b.end_time - b.start_time) / 2
                    cy = b.address + b.size / 2
                    texts[key].set_position((cx, cy))

                # Force matplotlib to draw the updated frame
                fig.canvas.draw()
                fig.canvas.flush_events()


            overlapping = False
            for first, overlapping_buffers in self.overlapping_time.items():
                for second, time_overlap in overlapping_buffers:
                    end = min(best[first].address + best[first].size, best[second].address + best[second].size)
                    start = max(best[first].address, best[second].address)
                    if start < end:
                        overlapping = True
                        break

            if np.all([not b.spilled for _, b in best.items()]) and not overlapping:
                break

        # Turn off interactive mode at the end and snap to the absolute best solution found
        plt.ioff()

        msg = "FINISHED! Showing Best Configuration"
        for key, b in best.items():
            if not b.spilled:
                rects[key].set_visible(True)
                texts[key].set_visible(True)
                rects[key].set_y(b.address)
                texts[key].set_position((b.start_time + (b.end_time - b.start_time) / 2, b.address + b.size / 2))
            else:
                rects[key].set_visible(False)
                texts[key].set_visible(False)

        # Add a warning box for spilled buffers
        spilled_buffers = [key + f" : {b.density:.2f}" for key, b in best.items() if b.spilled]
        if spilled_buffers:
           msg += f"\nSpilled to DRAM: {', '.join(spilled_buffers)}"
        else:
           msg += "\n No evictions required"

        step_text.set_text(msg)

        fig.canvas.draw()


        return best, e_list, temp_list, best_list

    def allocate_deterministic(self, buffers: dict[Buffer]):
        # 1. Calculate density for all buffers
        for _, b in buffers.items():
            if not b.density:
                b.calculate_density()

        # 2. Sort buffers based on priority:
        # Priority 1: Pinned status
        # Priority 2: Highest Density
        # Priority 3: Largest Size (Tie-breaker)

        buffers = dict(sorted(buffers.items(), key=lambda item: (item[1].pinned, item[1].density, item[1].size), reverse=True))

        # 3. Place each buffer
        for key, target in buffers.items():
            self._place_buffer(key, target)

        return self.allocated_buffers

    def _place_buffer(self, key: str, target: Buffer):
        # Find all currently allocated buffers that overlap in TIME
        overlapping_in_time = []
        for _, alloc in self.allocated_buffers.items():
            if max(target.start_time, alloc.start_time) < min(target.end_time, alloc.end_time):
                overlapping_in_time.append(alloc)

        # Extract their memory address ranges and sort them spatially
        occupied_spaces = sorted([(b.address, b.address + b.size) for b in overlapping_in_time])

        # Find all free gaps in memory during this time window
        free_gaps = self._find_free_gaps(occupied_spaces)

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
            target.address = self.capacity
        self.allocated_buffers[key] = target

    def _find_free_gaps(self, occupied: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
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

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    SPM_CAPACITY = 1000

    buffer_list = {
        "A": Buffer(reads=1000, writes=1000, size=500, start_time=0, end_time=1, pinned=False),
        "B": Buffer(reads=1000, writes=1000, size=100, start_time=1, end_time=4, pinned=False),
        "C": Buffer(reads=1000, writes=1000, size=500, start_time=4, end_time=5, pinned=False),
        "D": Buffer(reads=1000, writes=1000, size=50, start_time=0, end_time=3, pinned=False),
        "E": Buffer(reads=1000, writes=1000, size=26, start_time=0, end_time=5, pinned=False),
        "F": Buffer(reads=1000, writes=1000, size=38, start_time=1, end_time=3, pinned=False),
        "G": Buffer(reads=1000, writes=1000, size=250, start_time=2, end_time=5, pinned=False),
        "H": Buffer(reads=1000, writes=1000, size=120, start_time=0, end_time=3, pinned=False),
        "I": Buffer(reads=1000, writes=1000, size=100, start_time=0, end_time=2, pinned=False)
    }
    
    allocator = ScratchpadAllocator(capacity=SPM_CAPACITY)

    # This will now pop open a window and animate live
    best_allocation, e, temp, best_scores = allocator.allocate_annealing(buffer_list)

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