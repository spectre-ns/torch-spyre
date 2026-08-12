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

"""Effective HBM bandwidth vs read-fraction, from the B-F golden sweep.

Each op's effective BW = (model HBM bytes) / (golden kernel time). We classify
ops by their HBM read:write pass mix and plot BW against read-fraction; the clean
two-rate model eff = (R+W)/(R/bw_r + W/bw_w) is overlaid. Output: notes/bw_readfrac.png
"""

import matplotlib.pyplot as plt
import numpy as np

# (label, reads, writes, effective_bw_gbps) -- reads/writes = HBM full-tensor passes.
# bw values are the bw_gbps printed in profile_sweep_20260618_210419.log.
POINTS = [
    # read-only (1R, 0W): reductions read the full input, write a tiny output.
    ("read@1k", 1, 0, 144.3), ("read@4k", 1, 0, 153.3),
    ("sumrow@1k", 1, 0, 145.7), ("sumrow@4k", 1, 0, 149.9),
    ("amax", 1, 0, 151.5), ("mean", 1, 0, 149.7), ("sumall", 1, 0, 155.2),
    # balanced 1R+1W
    ("neg@1k", 1, 1, 108.7), ("neg@4k", 1, 1, 104.8), ("exp", 1, 1, 104.1),
    # broadcast: 1 real read + 1 write (the cached operand is ~free)
    ("bcast", 1, 1, 123.9), ("bcastcol", 1, 1, 121.7),
    # multi-read fused (2-4 reads + 1 write); LX intermediates excluded
    ("mul", 2, 1, 116.4), ("add", 2, 1, 115.0),
    ("add3", 3, 1, 115.1), ("add4", 4, 1, 116.3),
    # write-dominated (both inputs cached -> ~1 write); noisy
    ("write@1k", 0, 1, 140.5), ("write@4k", 0, 1, 111.6),
    # cross-core reduction (reduces the partitioned row axis): combine penalty
    ("sumcol", 1, 0, 120.8),
]

# Two-rate fit from the cleanest cases: read-only pins bw_r; balanced neg pins bw_w.
BW_R = 150.0   # read-only rate (mean of the 1R/0W points ~ 150)
BW_W = 80.8    # solved from balanced neg eff=105: 2/(1/150 + 1/bw_w)=105


def eff(r, w):
    return (r + w) / (r / BW_R + w / BW_W) if (r + w) else 0.0


def main():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # color by class
    cls = {
        (1, 0): ("tab:green", "read-only (reductions)"),
        (1, 1): ("tab:blue", "balanced 1R+1W"),
        (2, 1): ("tab:orange", "multi-read fused"),
        (3, 1): ("tab:orange", None), (4, 1): ("tab:orange", None),
        (0, 1): ("tab:red", "write-dominated"),
    }
    seen = set()
    for label, r, w, bw in POINTS:
        rf = r / (r + w) if (r + w) else 0.0
        color, name = cls.get((r, w), ("tab:purple", "broadcast / cross-core"))
        if name is None or (r, w) in seen:
            name = None
        else:
            seen.add((r, w))
        # broadcast + sumcol are special: mark them distinctly
        if label in ("bcast", "bcastcol"):
            color, name = "tab:purple", ("broadcast (1 real R+W)"
                                         if "broadcast" not in seen else None)
            seen.add("broadcast")
            rf = 0.5
        if label == "sumcol":
            color, name = "tab:brown", "cross-core reduce (sumcol)"
            rf = 1.0
        ax.scatter(rf, bw, color=color, label=name, zorder=3, s=45)
        ax.annotate(label, (rf, bw), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")

    xs = np.linspace(0.5, 1.0, 50)
    ax.plot([r / (r + 1) for r in [1, 2, 3, 4]],
            [eff(r, 1) for r in [1, 2, 3, 4]], "k--", alpha=0.5, lw=1.2,
            label=f"two-rate fit (bw_r={BW_R:.0f}, bw_w={BW_W:.0f})")
    ax.axhline(BW_R, color="tab:green", ls=":", alpha=0.5)
    ax.text(0.52, BW_R + 1, f"read-only ~{BW_R:.0f}", color="tab:green", fontsize=8)

    ax.set_xlabel("read fraction  =  reads / (reads + writes)  [HBM passes]")
    ax.set_ylabel("effective HBM bandwidth (GB/s) = model bytes / kernel us")
    ax.set_title("Spyre effective HBM BW rises with read-fraction\n"
                 "(B-F golden sweep, rows=2048, fp16)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("notes/bw_readfrac.png", dpi=130)
    print("wrote notes/bw_readfrac.png")
    # echo the two-rate predictions vs measured for the multi-read ops
    for label, r, w, bw in POINTS:
        if w == 1 and r >= 1:
            print(f"  {label:9s} R={r} W={w}: pred {eff(r, w):6.1f}  meas {bw:6.1f}")


if __name__ == "__main__":
    main()
