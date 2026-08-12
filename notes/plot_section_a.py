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

"""Plot the GOLDEN profiler section-A size sweep (neg, gelu).

Reads the SUMMARY lines from a profile_sweep log and draws kernel device time
vs HBM I/O bytes, with a least-squares line through the origin region so the
slope reads directly as effective bandwidth. Memset overhead is overlaid to
show it is NOT a fixed term. Output: notes/section_a_sweep.png
"""

import regex as re

import matplotlib.pyplot as plt
import numpy as np

LOG = "haoyang_logs/profile_sweep_20260618_180841.log"
OUT = "notes/section_a_sweep.png"

PAT = re.compile(
    r"op=(\w+) rows=(\d+) cols=(\d+) elems=(\d+) kernel_us=([\d.]+) "
    r"memset_us=([\d.]+)"
)


def load():
    rows = {}
    with open(LOG) as f:
        for line in f:
            if not line.startswith("SUMMARY"):
                continue
            m = PAT.search(line)
            if not m:
                continue
            op, _r, _c, elems, kus, mus = m.groups()
            # fp16 = 2 B/elem; a 1R+1W op moves 2x the tensor (read + write).
            tensor_bytes = int(elems) * 2
            io_bytes = 2 * tensor_bytes  # 1 read + 1 write
            rows.setdefault(op, []).append(
                (io_bytes, float(kus), float(mus))
            )
    return rows


def fit(x, y):
    """Least-squares slope+intercept; slope -> GB/s as bytes/ns = bytes/(us*1e3)."""
    a, b = np.polyfit(x, y, 1)  # y = a*x + b  (us per byte, us)
    bw_gbps = 1.0 / (a * 1e3)  # bytes / (us*1e3) = bytes/ns = GB/s
    return a, b, bw_gbps


def main():
    rows = load()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    colors = {"neg": "tab:blue", "gelu": "tab:orange"}
    for op, recs in rows.items():
        recs.sort()
        x = np.array([r[0] for r in recs]) / 1e6  # MB
        k = np.array([r[1] for r in recs])  # kernel us
        a, b, bw = fit(x * 1e6, k)
        xs = np.linspace(0, x.max() * 1.05, 50)
        ax.scatter(x, k, color=colors.get(op), label=f"{op} kernel", zorder=3)
        ax.plot(
            xs, a * 1e6 * xs + b, color=colors.get(op), lw=1.4, alpha=0.8,
            label=f"{op} fit: {bw:.0f} GB/s, b={b:+.1f}us",
        )

    ax.set_xlabel("HBM I/O moved (MB)  [1R + 1W, device fp16]")
    ax.set_ylabel("kernel device time (us)")
    ax.set_title("Section A: kernel time is linear in I/O\n(intercept ~0 -> no fixed term)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: kernel vs memset overhead -> overhead is NOT fixed, it scales with size.
    for op, recs in rows.items():
        recs.sort()
        x = np.array([r[0] for r in recs]) / 1e6
        k = np.array([r[1] for r in recs])
        mset = np.array([r[2] for r in recs])
        ax2.plot(x, k, "o-", color=colors.get(op), label=f"{op} kernel")
        ax2.plot(
            x, mset, "s--", color=colors.get(op), alpha=0.6,
            label=f"{op} memset (overhead)",
        )
    ax2.set_xlabel("HBM I/O moved (MB)")
    ax2.set_ylabel("device time (us)")
    ax2.set_title("Kernel (golden) vs Memset overhead\n(memset scales too -> not fixed)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")
    # Echo the fits.
    for op, recs in rows.items():
        recs.sort()
        x = np.array([r[0] for r in recs])
        k = np.array([r[1] for r in recs])
        a, b, bw = fit(x, k)
        print(f"  {op}: BW={bw:.1f} GB/s  intercept={b:+.2f} us")


if __name__ == "__main__":
    main()
