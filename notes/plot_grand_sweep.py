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

"""Plot the comprehensive 5-hour grand sweep (run_5h_sweep.sh) into the figures.

Parses the SUMMARY lines (op/rows/cols/cores/tiles/lx/io_hbm_bytes/kernel_us/pred_us/
bw_gbps...) AND the per-run IO breakdown (to recover exact read/write bytes per role,
so the read fraction and the write-only point do NOT depend on a single bw_gbps). Emits:

  FIG 1  notes/fig1_broadcast.png   -- broadcast = one-time load (gap-vs-R, P1a)
  FIG 2  notes/fig2_bandwidth.png   -- I/O linearity + the V-shape turnaround (P1b/c/d)
  FIG 3  notes/fig3_lx_free.png     -- is LX ~free? LX on/off + fused-add arity (P2/P1c)
  FIG 4  notes/fig4_coarse_tile.png -- coarse-tiling model: K-sweep + pred-vs-measured

Usage:
    python notes/plot_grand_sweep.py [haoyang_logs/grand_sweep_*.log]
(defaults to the most recent grand_sweep log.)
"""

import glob
import sys

import regex as re

import matplotlib.pyplot as plt
import numpy as np

BW_PEAK, ALPHA = 150.0, 0.00574  # turnaround params (cost_model.CostParams defaults)

_KV = re.compile(r"(\w+)=(\S+)")
_IO_ARG = re.compile(
    r"^IO\s+(output|input)\s+.*\bin\s+(hbm|lx)\b.*hbm counted:\s*(\d+)\s*B"
)


def _num(s):
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return s  # e.g. cores="-"


def load(path):
    """Parse runs; each SUMMARY gets r_bytes/w_bytes from its preceding IO block."""
    runs = []
    r_bytes = w_bytes = 0
    for line in open(path):
        s = line.rstrip("\n")
        if s.startswith("IO -- device-layout"):
            r_bytes = w_bytes = 0  # new run's IO block
            continue
        m = _IO_ARG.match(s)
        if m:
            role, _mem, b = m.group(1), m.group(2), int(m.group(3))
            if role == "input":
                r_bytes += b
            else:
                w_bytes += b
            continue
        if s.startswith("SUMMARY"):
            d = {k: _num(v) for k, v in _KV.findall(s)}
            d["r_bytes"], d["w_bytes"] = r_bytes, w_bytes
            runs.append(d)
            r_bytes = w_bytes = 0
    return runs


def _eff_bw(d):  # effective HBM BW (GB/s) = total bytes / kernel ns
    tot = d["r_bytes"] + d["w_bytes"]
    return tot / (d["kernel_us"] * 1000) if d.get("kernel_us", 0) > 0 else 0.0


def _read_frac(d):
    tot = d["r_bytes"] + d["w_bytes"]
    return d["r_bytes"] / tot if tot else 0.0


def fig1_broadcast(runs):
    pts = [d for d in runs if d.get("op") in ("bcast", "bcastcol", "add")
           and str(d.get("cores")) == "32" and str(d.get("cols")) == "16384"]
    if not pts:
        print("FIG1: no P1a data, skipping")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"bcast": "tab:blue", "bcastcol": "tab:purple", "add": "tab:orange"}
    for op in ("bcast", "bcastcol", "add"):
        rows = sorted((d for d in pts if d["op"] == op), key=lambda d: d["rows"])
        if not rows:
            continue
        x = [d["rows"] for d in rows]
        y = [d["kernel_us"] for d in rows]
        ax.plot(x, y, "o-", color=colors[op], label=op)
    ax.set_xlabel("ROWS (R)  [cores=32, C=16384]")
    ax.set_ylabel("kernel device time (us)")
    ax.set_title("FIG 1: broadcast operand is a ONE-TIME load\n"
                 "bcast~bcastcol (2-pass) << add (3-pass) => NOT reloaded per core")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("notes/fig1_broadcast.png", dpi=130)
    print("wrote notes/fig1_broadcast.png")


# read-fraction class -> color, for the V-shape
_FRAC_OPS = {"read": "1R0W", "sumrow": "1R0W", "neg": "1R1W", "mul": "2R1W",
             "add3": "3R1W", "add4": "4R1W", "write": "0R1W"}


def fig2_bandwidth(runs):
    pw = [d for d in runs if d.get("op") in _FRAC_OPS
          and str(d.get("tiles")) in ("0", "1")]
    if not pw:
        print("FIG2: no pointwise/reduction data, skipping")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # left: kernel time linear in total I/O (no fixed term)
    x = np.array([(d["r_bytes"] + d["w_bytes"]) / 1e6 for d in pw])
    y = np.array([d["kernel_us"] for d in pw])
    ax1.scatter(x, y, c="tab:blue", s=30, zorder=3)
    if len(x) > 1:
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(0, x.max() * 1.05, 50)
        ax1.plot(xs, a * xs + b, "k--", alpha=0.6,
                 label=f"fit: {1000 / a:.0f} GB/s, intercept {b:+.1f}us")
        ax1.legend()
    ax1.set_xlabel("total HBM I/O moved (MB)")
    ax1.set_ylabel("kernel device time (us)")
    ax1.set_title("I/O linearity (fill ~ 0)")
    ax1.grid(True, alpha=0.3)
    # right: effective BW vs read fraction -> V-shape + turnaround overlay
    for d in pw:
        rf, bw = _read_frac(d), _eff_bw(d)
        c = "tab:red" if d["op"] == "write" else "tab:green"
        ax2.scatter(rf, bw, color=c, s=35, zorder=3)
        ax2.annotate(f"{d['op']}", (rf, bw), fontsize=6,
                     xytext=(3, 3), textcoords="offset points")
    ff = np.linspace(0, 1, 100)
    ax2.plot(ff, 1.0 / (1.0 / BW_PEAK + ALPHA * np.minimum(ff, 1 - ff)),
             "k-", alpha=0.6, label=f"turnaround (BW={BW_PEAK:.0f}, a={ALPHA})")
    ax2.set_xlabel("read fraction = R / (R+W)")
    ax2.set_ylabel("effective HBM BW (GB/s)")
    ax2.set_title("FIG 2: V-shape -> turnaround model\n"
                  "(red 'write' = broadcast-I/O caveat point)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("notes/fig2_bandwidth.png", dpi=130)
    print("wrote notes/fig2_bandwidth.png")


def fig3_lx_free(runs):
    ct = [d for d in runs if d.get("op") == "ctsum" and str(d.get("cols")) == "512"
          and int(_num(d.get("tiles", 0))) >= 2]
    arity = [d for d in runs if d.get("op") in ("neg", "mul", "add3", "add4")
             and str(d.get("tiles")) in ("0", "1")]
    if not ct and not arity:
        print("FIG3: no LX data, skipping")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # left: coarse-tile ctsum kernel vs K, LX off vs on -> the delta is combine traffic
    for lx, color in (("0", "tab:red"), ("1", "tab:green")):
        pts = sorted((d for d in ct if str(d.get("lx")) == lx),
                     key=lambda d: int(_num(d["tiles"])))
        if not pts:
            continue
        ax1.plot([int(_num(d["tiles"])) for d in pts], [d["kernel_us"] for d in pts],
                 "o-", color=color, label=f"LX_PLANNING={lx}")
    ax1.set_xlabel("tile count K")
    ax1.set_ylabel("kernel device time (us)")
    ax1.set_title("ctsum K-sweep: LX off vs on\n(partial in HBM vs LX ~free)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # right: fused-add arity -> kernel tracks HBM passes, not LX passes
    hbm_pass = {"neg": 2, "mul": 3, "add3": 4, "add4": 5}
    lx_pass = {"neg": 0, "mul": 0, "add3": 2, "add4": 4}
    for d in arity:
        op = d["op"]
        ax2.scatter(hbm_pass[op], d["kernel_us"], color="tab:blue", s=40, zorder=3)
        ax2.annotate(f"{op} (LX passes={lx_pass[op]})", (hbm_pass[op], d["kernel_us"]),
                     fontsize=7, xytext=(4, 0), textcoords="offset points")
    ax2.set_xlabel("HBM passes (reads + write)")
    ax2.set_ylabel("kernel device time (us)")
    ax2.set_title("FIG 3: kernel tracks HBM passes only\n"
                  "(LX intermediates add ~0 => LX ~free)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("notes/fig3_lx_free.png", dpi=130)
    print("wrote notes/fig3_lx_free.png")


def fig4_coarse_tile(runs):
    ct = [d for d in runs if str(d.get("op")).startswith("ct")]
    if not ct:
        print("FIG4: no coarse-tile data, skipping")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # left: K-sweep measured vs predicted (ctsum, both LX modes) -> calibrate c_loop
    ks = [d for d in ct if d["op"] == "ctsum" and str(d.get("cols")) == "512"
          and int(_num(d.get("tiles", 0))) >= 2]
    for lx, color in (("0", "tab:red"), ("1", "tab:green")):
        pts = sorted((d for d in ks if str(d.get("lx")) == lx),
                     key=lambda d: int(_num(d["tiles"])))
        if not pts:
            continue
        kk = [int(_num(d["tiles"])) for d in pts]
        ax1.plot(kk, [d["kernel_us"] for d in pts], "o-", color=color,
                 label=f"measured LX={lx}")
        if "pred_us" in pts[0]:
            ax1.plot(kk, [d["pred_us"] for d in pts], "x--", color=color, alpha=0.6,
                     label=f"model LX={lx}")
    ax1.set_xlabel("tile count K")
    ax1.set_ylabel("kernel us")
    ax1.set_title("Coarse-tile K-sweep: measured vs model\n(gap vs K -> c_loop)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    # right: predicted vs measured scatter for ALL coarse-tile runs
    have_pred = [d for d in ct if "pred_us" in d and d.get("kernel_us", 0) > 0]
    if have_pred:
        mx = [d["kernel_us"] for d in have_pred]
        px = [d["pred_us"] for d in have_pred]
        ax2.scatter(mx, px, c="tab:purple", s=35, zorder=3)
        lim = max(max(mx), max(px)) * 1.05
        ax2.plot([0, lim], [0, lim], "k--", alpha=0.5, label="ideal")
        ax2.set_xlim(0, lim)
        ax2.set_ylim(0, lim)
    ax2.set_xlabel("measured kernel us")
    ax2.set_ylabel("model pred us")
    ax2.set_title("FIG 4: coarse-tile model vs golden")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("notes/fig4_coarse_tile.png", dpi=130)
    print("wrote notes/fig4_coarse_tile.png")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        logs = sorted(glob.glob("haoyang_logs/grand_sweep_*.log"))
        if not logs:
            sys.exit("no grand_sweep_*.log found; pass a path explicitly")
        path = logs[-1]
    runs = load(path)
    print(f"parsed {len(runs)} runs from {path}")
    fig1_broadcast(runs)
    fig2_bandwidth(runs)
    fig3_lx_free(runs)
    fig4_coarse_tile(runs)


if __name__ == "__main__":
    main()
