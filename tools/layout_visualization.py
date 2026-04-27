import matplotlib.pyplot as plt
from matplotlib import patches

def plot_layout(capacity: int, buffers: list[Buffer]):
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