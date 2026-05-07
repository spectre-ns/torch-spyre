


import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from matplotlib import patches
from typing import Optional, Protocol


# Break and dependency on device code by creating a protocol type

@dataclass
class LifetimeBoundBufferProtocol(Protocol):
    """
    Defines the data fields required for a layout solver.
    The required heuristics are implementation defined.
    """

    name: str
    size: int
    start_time: int
    end_time: int
    heuristic: dict[str, float]
    address: Optional[int]



def plot_layout(capacity: int, buffers: list[LifetimeBoundBufferProtocol]):
    """Creates a simple plot to visualize scratchpad usage vs time

    Args:
        capacity (int): Size of the scratchpad buffer
        buffers (list[LifetimeBoundBufferProtocol]): List of buffers to be visualized
    """
    assert np.all([b.start_time is not None for b in buffers]), (
        "Start time must be defined"
    )
    assert np.all([b.end_time is not None for b in buffers]), "End time must be defined"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(
        y=capacity,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"SPM Ceiling ({capacity})",
    )
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2, label="SPM Floor")

    max_time = max([b.end_time for b in buffers if b.end_time is not None])
    ax.set_xlim(0, max_time)
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
        if b.address is not None:
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
    spilled_buffers = [f"\n"+ b.name + f"\n\t -Heuristic: {b.heuristic}".expandtabs(4)  for b in buffers if b.address is None]
    if spilled_buffers:
        msg += f"\nSpilled to DRAM:".join(spilled_buffers)
    else:
        msg += "\n No evictions required"

    step_text.set_text(msg)
    plt.show()

if __name__ == "__main__":
    @dataclass
    class LifetimeBoundBuffer:
        name: str
        size: int
        start_time: int
        end_time: int
        heuristic: dict[str, float] = field(default_factory=dict)
        address: Optional[int] = None

    expectation = [
        LifetimeBoundBuffer("buffer0", 7, 0, 2, {}, 0),
        LifetimeBoundBuffer("buffer1", 4, 0, 2, {}, None),
        LifetimeBoundBuffer("buffer2", 3, 0, 2, {}, 7),
    ]

    plot_layout(10, expectation)