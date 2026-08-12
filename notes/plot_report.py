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

"""Regenerate the figures for notes/cost_model_report.md from sweep_records.json.

Each figure is a function ``fig_<name>`` that reads the stored measured times and writes
a PNG to notes/figures/. Reproducible -- the data is the committed sweep records, not
hand-drawn. Run one figure or all:

    python notes/plot_report.py                 # all figures
    python notes/plot_report.py pointwise_baseline
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIGDIR = os.path.join(_HERE, "figures")
_RECORDS = os.path.join(_HERE, "sweep_records.json")


def _load(current_only=True):
    recs = json.load(open(_RECORDS, encoding="utf-8"))["records"]
    recs = [r for r in recs if not r.get("failed") and r.get("kernel_us")]
    if current_only:
        recs = [r for r in recs if r.get("is_current")]
    return recs


def _save(fig, name):
    os.makedirs(_FIGDIR, exist_ok=True)
    path = os.path.join(_FIGDIR, f"{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, _HERE)}")


# ============================================================================
# §1 -- pointwise is memory-I/O bound: neg time is linear in bytes through the
# origin (no fixed per-kernel cost).
# ============================================================================
def fig_pointwise_baseline(recs):
    pts = [
        (int(r["io_hbm_bytes"]) / 1e6, r["kernel_us"])
        for r in recs
        if r["op"] == "neg" and r.get("io_hbm_bytes")
    ]
    allx = np.array([x for x, _ in pts])
    ally = np.array([y for _, y in pts])
    # linear fit T[us] = a*MB + b (b -> fixed cost; a -> 1/BW)
    a, b = np.polyfit(allx, ally, 1)
    r2 = 1 - np.sum((ally - (a * allx + b)) ** 2) / np.sum((ally - ally.mean()) ** 2)
    bw = 1e3 / a  # MB/us = GB/s

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    xline = np.array([0, allx.max() * 1.03])
    ax.plot(
        xline,
        a * xline + b,
        "-",
        color="0.5",
        lw=1.2,
        zorder=1,
        label=f"linear fit: {bw:.0f} GB/s, $R^2$={r2:.4f}",
    )
    ax.scatter(
        allx,
        ally,
        s=42,
        color="#1f77b4",
        label="neg",
        zorder=3,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("HBM traffic  (MB, device / stick-padded)")
    ax.set_ylabel("kernel time  (µs)")
    ax.set_title("§1  Pointwise is memory-I/O bound (neg, 1R:1W)")
    ax.annotate(
        f"intercept b = {b:+.1f} µs  ≈ 0\n(no fixed per-kernel cost)",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=9,
        color="0.25",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.8"),
    )
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.margins(x=0.02)
    _save(fig, "fig1_pointwise_baseline")


# ============================================================================
# §2 -- the read/write mix sets the effective BW: a symmetric turnaround valley.
# effBW vs write-fraction w=W/(R+W): high at w=0 (read-only) and w=1 (write-only),
# lowest at w=0.5 (balanced). Model: effBW = 1/(1/BW_peak + alpha*min(w,1-w)).
# ============================================================================
def _pw_ratio_write_points():
    """(w, effBW) for the write-only anchor from the pw_ratio decoupler log (not yet in
    sweep_records). ROWS=2048 only (small operand resident, pre-spill)."""
    import glob

    import regex as re

    logs = sorted(
        glob.glob(os.path.join(_HERE, "..", "haoyang_logs", "pw_ratio_*.log"))
    )
    if not logs:
        return []
    R = W = None
    pts = []
    for ln in open(logs[-1], encoding="utf-8"):
        m = re.search(r"^MODEL\s+R=(\d+) B .*?W=(\d+) B", ln)
        if m:
            R, W = int(m[1]), int(m[2])
        s = re.search(r"SUMMARY op=write rows=2048 .*kernel_us=([0-9.]+)", ln)
        if s and R is not None:
            tot = R + W
            pts.append((W / tot, tot / 1e3 / float(s[1])))
            R = W = None
    return pts


def fig_pointwise_vcurve(recs):
    bw_peak, alpha = 150.0, 0.00574  # current model
    reds = ("sumrow", "read", "amax", "mean")  # read-only anchor (ROWS=2048, clean)
    # label -> (points, color, marker). Each op is swept at several sizes -> several
    # points per class (small vertical spread = mild size drift, ~2-3%).
    groups = {
        "read-only": ([], "#2ca02c", "o"),
        "2R:1W (add)": ([], "#9467bd", "o"),
        "1R:1W (neg)": ([], "#1f77b4", "o"),
        "write-only": ([], "#d62728", "o"),
        "n-ary (→§3)": ([], "#ff7f0e", "x"),
    }
    for r in recs:
        m = r.get("model") or {}
        R, W = m.get("R"), m.get("W")
        if not R or not W:
            continue
        eff = (R + W) / 1e3 / r["kernel_us"]
        w = W / (R + W)
        if r["op"] in reds and r.get("rows") == 2048:
            groups["read-only"][0].append((w, eff))
        elif r["op"] == "add":
            groups["2R:1W (add)"][0].append((w, eff))
        elif r["op"] == "neg":
            groups["1R:1W (neg)"][0].append((w, eff))
        elif r["op"] in ("add3", "add4"):
            groups["n-ary (→§3)"][0].append((w, eff))
    groups["write-only"][0].extend(_pw_ratio_write_points())

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ww = np.linspace(0, 1, 200)
    model = 1.0 / (1.0 / bw_peak + alpha * np.minimum(ww, 1 - ww))
    ax.plot(
        ww,
        model,
        "-",
        color="0.55",
        lw=1.3,
        zorder=1,
        label=f"model (BW_peak={bw_peak:.0f}, α={alpha})",
    )
    for lab, (pts, c, mk) in groups.items():
        if pts:
            xs, ys = zip(*pts)
            kw = dict(s=34, color=c, label=lab, marker=mk, zorder=3)
            if mk != "x":  # filled markers get a white edge; 'x' is a stroke marker
                kw.update(edgecolors="white", linewidths=0.4)
            ax.scatter(xs, ys, **kw)
    ax.set_xlabel("write fraction  w = W / (R + W)")
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title("§2  R/W mix sets effective BW (turnaround valley)")
    ax.set_ylim(90, 158)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=8,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    _save(fig, "fig2_pointwise_vcurve")


# ============================================================================
# §3 -- n-ary adds: the byte model predicts the SAME effBW across arity (all sit
# at w=1/3), but measured effBW DECLINES with each chained op -> a per-op derate.
# ============================================================================
def fig_pointwise_arity(recs):
    bw_peak, alpha, derate = 150.0, 0.00574, 0.075
    ops = [("add", 1), ("add3", 2), ("add4", 3)]  # n = number of chained binary adds
    meas = {n: [] for _, n in ops}
    for r in recs:
        for op, n in ops:
            if r["op"] == op:
                m = r.get("model") or {}
                R, W = m.get("R"), m.get("W")
                if R and W:
                    meas[n].append((R + W) / 1e3 / r["kernel_us"])
    # byte model: same f=1/3 for all arities -> flat effBW
    eff_byte = 1.0 / (1.0 / bw_peak + alpha * (1.0 / 3.0))
    ns = [n for _, n in ops]

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.axhline(
        eff_byte,
        ls="--",
        color="0.6",
        lw=1.2,
        label=f"byte model, no derate (flat = {eff_byte:.0f})",
    )
    ax.plot(
        ns,
        [eff_byte / (1 + derate * (n - 1)) for n in ns],
        "-",
        color="#d62728",
        lw=1.3,
        marker="s",
        ms=5,
        zorder=2,
        label=f"× (1 + {derate}·(n−1)) derate",
    )
    for n in ns:
        ax.scatter(
            [n] * len(meas[n]),
            meas[n],
            s=40,
            color="#1f77b4",
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
            label="measured" if n == 1 else None,
        )
    ax.set_xticks(ns)
    ax.set_xticklabels(["add\n(n=1)", "add3\n(n=2)", "add4\n(n=3)"])
    ax.set_xlabel("chained binary adds")
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title("§3  n-ary derate: effBW falls per chained op")
    ax.set_ylim(94, 120)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    _save(fig, "fig3_pointwise_arity")


# ============================================================================
# §4 -- a broadcast operand raises the effective BW: broadcast-operand ops
# (copy/bcast/bcastcol/mulbcast) run ~118 GB/s, above plain 1R:1W neg (~105).
# Small-ROWS points are underfilled (2-8 rows/core) and unreliable -> flagged.
# ============================================================================
def fig_broadcast_effbw(_recs):
    # measured times are version-independent -> use all records (not just is_current,
    # which the newest-log SHA can shrink). neg (1R:1W) baseline + the broadcast ops.
    recs = _load(current_only=False)
    neg = [
        int(r["io_hbm_bytes"]) / 1e3 / r["kernel_us"]
        for r in recs
        if r["op"] == "neg" and r.get("io_hbm_bytes")
    ]
    # all broadcast points at ROWS=2048 from the records (includes the small-COLS sweep)
    from collections import defaultdict

    brec = defaultdict(lambda: defaultdict(list))
    for r in _load(current_only=False):
        if (
            r.get("op") in ("copy", "bcast", "bcastcol", "mulbcast")
            and r.get("rows") == 2048
            and r.get("io_hbm_bytes")
        ):
            brec[r["op"]][r["cols"]].append(
                int(r["io_hbm_bytes"]) / 1e3 / r["kernel_us"]
            )
    colors = {
        "copy": "#d62728",
        "bcast": "#2ca02c",
        "bcastcol": "#9467bd",
        "mulbcast": "#ff7f0e",
    }

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.axhspan(
        min(neg),
        max(neg),
        color="#1f77b4",
        alpha=0.12,
        zorder=0,
        label=f"neg (1R:1W) range, all sizes: {min(neg):.0f}–{max(neg):.0f}",
    )
    ax.axhline(
        np.mean(neg),
        ls="--",
        color="#1f77b4",
        lw=1.1,
        label=f"neg mean ≈ {np.mean(neg):.0f}",
    )
    ax.axhline(
        118, ls="--", color="0.5", lw=1.0, label="broadcast rate = 118 (large-size)"
    )
    for op, c in colors.items():
        pts = sorted((C, sum(v) / len(v)) for C, v in brec[op].items())
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=c, ms=4.5, label=op)
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384])
    ax.set_xticklabels(["256", "512", "1k", "2k", "4k", "8k", "16k"])
    ax.set_xlabel("COLS  (ROWS = 2048)")
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title("§4  Broadcast ops: ~118 at large size, rising to ~130 when small")
    ax.set_ylim(100, 140)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, ncol=2)
    _save(fig, "fig4_broadcast_effbw")


# ============================================================================
# §5 -- reductions are read-dominated -> read-only rate (~150), EXCEPT the output
# [R] is stick-inflated to R×64 and written scattered; at large R it drags effBW down.
# ============================================================================
def fig_reduction(_recs):
    # The read rate FALLS with ROWS, op-independently. Use all records (version-
    # independent measured times) for the full ROWS sweep; overlay the model.
    recs = _load(current_only=False)
    ops = ("read", "sumrow", "amax", "mean", "sumall")
    colors = {
        "read": "#1f77b4",
        "sumrow": "#2ca02c",
        "amax": "#9467bd",
        "mean": "#ff7f0e",
        "sumall": "#8c564b",
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    # model: effBW = min(150, 114 + 61*exp(-ROWS/3700))
    xs = np.linspace(2048, 16384, 100)
    ax.plot(
        xs,
        np.minimum(150, 114 + 61 * np.exp(-xs / 3700)),
        "-",
        color="0.4",
        lw=1.6,
        zorder=2,
        label="model 114+61·e^(−ROWS/3700)",
    )
    for op in ops:
        pts = {}
        for r in recs:
            if r["op"] == op and r.get("io_hbm_bytes") and (r.get("cols") == 2048):
                pts.setdefault(r.get("rows") or 0, []).append(
                    int(r["io_hbm_bytes"]) / 1e3 / r["kernel_us"]
                )
        pts = sorted((x, sum(v) / len(v)) for x, v in pts.items())
        if pts:
            xx, yy = zip(*pts)
            ax.scatter(
                xx,
                yy,
                s=36,
                color=colors[op],
                label=op,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks([2048, 4096, 8192, 16384])
    ax.set_xticklabels(["2048", "4096", "8192", "16384"])
    ax.set_xlabel("ROWS  (input height;  COLS = 2048)")
    ax.set_ylabel("read rate  (R+W)/time  (GB/s)")
    ax.set_title("§5  Reduction read rate falls with ROWS (op-independent)")
    ax.set_ylim(105, 158)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    _save(fig, "fig5_reduction")


def _broadcast_log_rows():
    """Parse the standalone broadcast sweep log (not folded into sweep_records)."""
    import glob

    import regex as re

    logs = sorted(
        glob.glob(os.path.join(_HERE, "..", "haoyang_logs", "broadcast_*.log"))
    )
    if not logs:
        return []
    rows = []
    for ln in open(logs[-1], encoding="utf-8"):
        s = re.search(
            r"SUMMARY op=(\w+) rows=(\d+) cols=(\d+).*io_hbm_bytes=(\d+) kernel_us=([0-9.]+)",
            ln,
        )
        if s:
            rows.append((s[1], int(s[2]), int(s[3]), int(s[4]), float(s[5])))
    return rows


# ============================================================================
# §4b -- write spill: per-output-byte cost vs COLS, one line per ROWS. Rises with
# C (the b[1,C] row operand) AND with R -> C-dominant but not a clean single spill.
# ============================================================================
def fig_write_spill(_recs):
    # effective BW = HBM bytes moved / time -- collapses as the op does far more work
    # (the outer-product write) than its counted bytes suggest. Use ALL write records
    # (version-independent measured times), averaging repeats, for denser coverage.
    from collections import defaultdict

    recs = _load(current_only=False)
    agg = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r.get("op") == "write" and r.get("io_hbm_bytes") and r.get("cols"):
            agg[r.get("rows")][r["cols"]].append(
                int(r["io_hbm_bytes"]) / 1e3 / r["kernel_us"]
            )
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    colors = {512: "#1f77b4", 2048: "#ff7f0e", 8192: "#d62728"}
    for R in (512, 2048, 8192):
        pts = sorted((c, sum(v) / len(v)) for c, v in agg.get(R, {}).items())
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=colors[R], ms=5, label=f"ROWS={R}")
    ax.axhline(
        118, ls="--", color="0.5", lw=1.0, label="broadcast rate 118 (small-op start)"
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 2048, 4096, 8192, 16384])
    ax.set_xticklabels(["1k", "2k", "4k", "8k", "16k"])
    ax.set_xlabel("COLS")
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title("§4  write: effective BW falls with ROWS and COLS")
    ax.annotate(
        "starts near ~118 at small size;\ncollapses as ROWS and COLS grow",
        xy=(0.03, 0.05),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=8,
        color="0.3",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.8"),
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    _save(fig, "fig4b_write_spill")


# ============================================================================
# §6 -- transport ops (transpose / cat) lower to a byte-copy (`clone`); the only
# difference from a plain copy is the access pattern, which sets an effective BW.
# transpose is flat-fast (116); cat0 and transpose_outer fall with size.
# ============================================================================
def fig_transport(_recs):
    # transport times are version-independent -> use ALL records. Plot effBW vs COLS
    # (the driver for cat0/transpose_outer), showing EVERY distinct shape (not averaged
    # by total bytes, which would collapse aspect ratios into one dot). transpose and
    # cat1 are flat across C; cat0 and transpose_outer fall with C.
    recs = _load(current_only=False)

    def points(op):  # all distinct (C, R, effBW) shape points, averaging exact repeats
        agg = {}
        for r in recs:
            if r["op"] == op and r.get("io_hbm_bytes") and r.get("cols"):
                key = (r["cols"], r.get("rows"))
                agg.setdefault(key, []).append(
                    int(r["io_hbm_bytes"]) / 1e3 / r["kernel_us"]
                )
        return [(c, rw, sum(v) / len(v)) for (c, rw), v in agg.items()]

    styles = {
        "transpose": ("#2ca02c", "o", "transpose (flat ~116)"),
        "cat1": ("#1f77b4", "s", "cat1 (flat ~108)"),
        "cat0": ("#d62728", "^", "cat0 (falls with C)"),
        "transpose_outer": ("#9467bd", "D", "transpose_outer (falls with C)"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for op, (c, m, lab) in styles.items():
        pts = points(op)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[2] for p in pts]
        ax.scatter(
            xs,
            ys,
            color=c,
            marker=m,
            s=36,
            label=f"{lab}  (n={len(pts)})",
            zorder=3,
            edgecolors="white",
            linewidths=0.4,
        )
        # label each point with its ROWS config -> explains the several points per COLS
        for cc_, rw, y in pts:
            ax.annotate(
                f"R{rw}",
                (cc_, y),
                textcoords="offset points",
                xytext=(4, 2),
                fontsize=5.6,
                color=c,
            )
    # cat0 model curve at a representative ROWS (2048): 252 - 4*log2(R) - 12.3*log2(C)
    cc = np.array([256, 512, 1024, 2048, 4096, 8192, 16384], dtype=float)
    ax.plot(
        cc,
        np.clip(252 - 4 * np.log2(2048) - 12.3 * np.log2(cc), 45, 150),
        "--",
        color="#d62728",
        lw=1.1,
        alpha=0.8,
        label="cat0 model (R=2048)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["512", "1k", "2k", "4k", "8k"])
    ax.set_xlabel("COLS  (row width C)")
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title("§6  Transport: transpose/cat1 flat; cat0/transpose_outer fall with C")
    ax.set_ylim(40, 130)
    ax.legend(loc="lower left", fontsize=7.0, framealpha=0.9, ncol=1)
    _save(fig, "fig6_transport")


# ============================================================================
# §8-§11 -- matmul. These need model predictions, so load cost_model the same
# hardware-free way eval_model does (importlib; it imports only math/dataclasses).
# ============================================================================
def _cost_model():
    import importlib.util

    path = os.path.join(
        os.path.dirname(_HERE), "torch_spyre", "_inductor", "cost_model.py"
    )
    spec = importlib.util.spec_from_file_location("cost_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mm_rows(section_prefix):
    recs = _load(current_only=False)
    out = []
    for r in recs:
        if r.get("op") not in ("mm", "mmwd") or not r.get("feats"):
            continue
        if not (r.get("section") or "").startswith(section_prefix):
            continue
        out.append(r)
    return out


def fig_matmul_hbm(_recs):
    # Accuracy of the single-rate baseline memory model (R+W)/150 + a*min(R,W) on the
    # COMPUTE-FREE matmuls (thin-K write-heavy, thin-M read-heavy). Measured vs predicted,
    # labeled by shape -> shows coverage AND where the baseline strays (read-heavy corner).
    cm = _cost_model()
    p = cm.CostParams()
    rows = _mm_rows("M1")
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    lim = 300
    ax.plot(
        [0, lim], [0, lim], "-", color="0.6", lw=1.0, zorder=1, label="perfect (y = x)"
    )
    ax.plot([0, lim], [0, lim * 1.1], ":", color="0.75", lw=0.8, zorder=1)
    ax.plot(
        [0, lim], [0, lim * 0.9], ":", color="0.75", lw=0.8, zorder=1, label="±10 %"
    )
    for r in rows:
        feats = r["feats"]
        feats = feats if isinstance(feats, list) else json.loads(feats)
        ops = cm.ops_from_json(json.dumps(feats))
        R, W = cm._fused_hbm_bytes(ops)
        base = (
            (R + W) / p.bw_peak_gbps + p.rw_turnaround_ns_per_byte * min(R, W)
        ) / 1e3
        meas, M, N, K = r["kernel_us"], r.get("M"), r.get("N"), r.get("K")
        wheavy = (K or 0) <= 64
        col = "#d62728" if wheavy else "#1f77b4"
        ax.scatter(
            base, meas, color=col, s=44, zorder=3, edgecolors="white", linewidths=0.5
        )
        ax.annotate(
            f"{M}×{K}×{N}",
            (base, meas),
            textcoords="offset points",
            xytext=(5, -1),
            fontsize=6.3,
            color="0.25",
        )
    ax.scatter(
        [], [], color="#d62728", s=44, label="write-heavy  (thin K ∈ {16,32,64})"
    )
    ax.scatter([], [], color="#1f77b4", s=44, label="read-heavy  (thin M ∈ {32,64})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("predicted µs   —   baseline  (R+W)/150 + α·min(R,W)")
    ax.set_ylabel("measured µs")
    ax.set_title("§8  Baseline memory model vs measured (compute-free matmuls)")
    ax.annotate(
        "within ~4 % on write-heavy;\nread-heavy (large N, thin M)\nunder-predicted ~7–15 %",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=7.5,
        color="0.3",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.8"),
    )
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    _save(fig, "fig8_matmul_hbm")


def fig_matmul_spill(_recs):
    # THE OBSERVATION: with spill OFF, the base model leaves a residual (measured - base)
    # that grows with the per-core OUTPUT-tile AREA (M/m)*(N/n). Two balanced (4x8) decouple
    # sweeps grow the tile: DC2 varies M/m (N/n=256 fixed), DC1 varies N/n (M/m=512 fixed) --
    # both trace the SAME area axis. Residual is ~0/negative below the on-chip capacity knee
    # (~64K elems) and climbs positive above it (the re-read). The spill term (dashed) is one
    # area-driven curve. Every point labeled with its full 2-D per-core tile M/m x N/n.
    cm = _cost_model()
    p0 = cm.CostParams(mm_spill_slope=0.0)  # base model, spill OFF

    def collect(section):
        out = []
        for r in _mm_rows(section):
            feats = r["feats"]
            feats = feats if isinstance(feats, list) else json.loads(feats)
            ops = cm.ops_from_json(json.dumps(feats))
            mm = next(o for o in ops if getattr(o, "is_matmul", False))
            rpc, cpc = mm.matmul_rows_per_core, mm.matmul_cols_per_core
            area = rpc * cpc * 2  # per-core output-tile size in BYTES (fp16) -> x-axis
            base = cm.predict_ops(ops, p0) / 1e3
            resid = r["kernel_us"] - base
            spill = cm.predict_ops(ops) / 1e3 - base  # modeled spill effect
            out.append((area, resid, spill, rpc, cpc))
        return sorted(out)

    dc2 = collect("DC2")  # vary M/m (N/n held at 256)
    dc1 = collect("DC1")  # vary N/n (M/m held at 512)
    knee = cm.CostParams().mm_spill_area0 * 2  # elems -> bytes (fp16)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.axhline(0, color="0.6", lw=1.0, zorder=1)
    ax.axvline(
        knee,
        ls="--",
        color="0.5",
        lw=1.1,
        zorder=1,
        label="on-chip capacity ≈ 128 KB",
    )
    for data, col, lab in [
        (dc2, "#8c564b", "vary M/m  (N/n = 256 fixed, split 4×8)"),
        (dc1, "#1f77b4", "vary N/n  (M/m = 512 fixed, split 4×8)"),
    ]:
        xs = [d[0] for d in data]
        ax.scatter(
            xs,
            [d[1] for d in data],
            color=col,
            s=48,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
            label=f"residual: {lab}",
        )
        ax.plot(xs, [d[2] for d in data], "--", color=col, lw=1.1, alpha=0.7, zorder=2)
        for area, resid, spill, rpc, cpc in data:
            # label EVERY point with the full 2-D per-core tile  M/m × N/n
            ax.annotate(
                f"{rpc:.0f}×{cpc:.0f}",
                (area, resid),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=6.2,
                color=col,
            )
    ax.plot([], [], "--", color="0.5", lw=1.1, label="modeled spill (dashed)")
    ax.set_xscale("log", base=2)
    # Label ticks in plain element counts (32K, 64K, ...) not 2^n -- more legible, and the
    # 64K tick lines up with the "on-chip capacity" knee.
    import matplotlib.ticker as _mt

    ticks = [65536, 131072, 262144, 524288, 1048576]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(
        _mt.FuncFormatter(
            lambda v, _: f"{v / 1048576:.0f} MB"
            if v >= 1048576
            else f"{v / 1024:.0f} KB"
        )
    )
    ax.xaxis.set_minor_formatter(_mt.NullFormatter())
    ax.set_xlabel("per-core output-tile size   2·(M/m)·(N/n)   [bytes, fp16]")
    ax.set_ylabel("residual:  measured − base model (no spill)   (µs)")
    ax.set_title(
        "§11  Base-model residual grows once the output tile overflows on-chip"
    )
    ax.annotate(
        "tile fits → residual ≈ 0\ntile overflows → under-predict (re-read)",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=7.2,
        color="0.3",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.8"),
    )
    ax.legend(loc="lower right", fontsize=7.2, framealpha=0.9)
    _save(fig, "fig11_matmul_spill")


def _mm_balanced_points(K_set=(2048, 4096), M=2048, N=2048):
    """All (cores, m, n, k, t) for M×N×K matmuls, deduped, from every sweep."""
    recs = _load(current_only=False)
    out = {K: [] for K in K_set}
    for r in recs:
        if r.get("op") not in ("mm", "mmwd") or r.get("M") != M or r.get("N") != N:
            continue
        K, sa, c = r.get("K"), r.get("split_actual") or {}, r.get("cores")
        if K not in out or not c:
            continue
        out[K].append((c, sa.get("m"), sa.get("n"), sa.get("k", 1), r["kernel_us"]))
    return out


def fig_matmul_peak(_recs):
    # Compute-dominant: kernel time is linear in 1/cores at a fixed matmul; slope = 1/peak.
    # Two problem sizes (K=2048, K=4096 at M=N=2048). BALANCED (k=1) splits at equal cores
    # COLLAPSE (time tracks cores, not the m*n factoring). Every point labeled with m*n.
    pts = _mm_balanced_points()
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    colors = {2048: "#1f77b4", 4096: "#2ca02c"}
    for K in (4096, 2048):
        k1 = sorted({(c, m, n, 1, round(t, 1)) for c, m, n, k, t in pts[K] if k == 1})
        xs = [1.0 / c for c, *_ in k1]
        ys = [t for *_, t in k1]
        a, b = np.polyfit(xs, ys, 1)
        ax.plot(
            np.linspace(0, 0.27, 20),
            a * np.linspace(0, 0.27, 20) + b,
            "-",
            color=colors[K],
            lw=1.0,
            alpha=0.6,
            zorder=1,
        )
        ax.scatter(
            xs,
            ys,
            color=colors[K],
            s=42,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
            label=f"M=N=2048, K={K}  (balanced k=1)",
        )
        for c in sorted({p[0] for p in k1}):
            spl = sorted({(m, n) for cc, m, n, k, t in k1 if cc == c})
            ym = np.mean([t for cc, m, n, k, t in k1 if cc == c])
            ax.annotate(
                "/".join(f"{m}×{n}" for m, n in spl),
                (1.0 / c, ym),
                textcoords="offset points",
                xytext=(6, -3),
                fontsize=6.2,
                color=colors[K],
            )
    ax.set_xlim(0, 0.27)
    ax.set_ylim(bottom=0)
    ax.set_xticks([1 / 4, 1 / 8, 1 / 16, 1 / 32])
    ax.set_xticklabels(["4", "8", "16", "32"])
    ax.set_xlabel(
        "cores used   (axis positioned at 1/cores → straight line = time ∝ 1/cores)"
    )
    ax.set_ylabel("kernel time  (µs)")
    ax.set_title(
        "§9  Time halves when cores double; balanced splits at equal cores collapse"
    )
    ax.annotate(
        "each point is labelled with its m×n core split (cores = m·n).\n"
        "at 32 cores, K=2048 the splits 4×8 / 8×4 / 2×16 all ≈ 385 µs\n"
        "→ time tracks the core COUNT, not how m×n is factored.",
        xy=(0.28, 0.04),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=7.0,
        color="0.3",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.8"),
    )
    ax.legend(loc="upper left", fontsize=7.2, framealpha=0.9)
    _save(fig, "fig9_matmul_peak")


def fig_matmul_overlap(_recs):
    # Does gamma generalize across shapes/splits? For EVERY balanced (k=1) matmul, plot
    # prediction error vs the MEMORY FRACTION (memory/(compute+memory)). Overlap matters
    # most when memory is a big fraction: the ADDITIVE model (gamma=0) over-predicts more
    # and more as memory grows (open red, climbing), while the OVERLAP model (gamma=0.46)
    # stays flat near 0 across the whole range AND across aspect types. Marker = aspect.
    cm = _cost_model()
    p_add = cm.CostParams(overlap_gamma=0.0)
    recs = _load(current_only=False)
    seen, rows = set(), []
    for r in recs:
        if r.get("op") not in ("mm", "mmwd") or not r.get("feats"):
            continue
        sa = r.get("split_actual") or {}
        M, N, K = r.get("M"), r.get("N"), r.get("K")
        if sa.get("k", 1) != 1 or (sa.get("m") or 9) > 8 or (sa.get("n") or 9) > 8:
            continue
        if not M or not N or not K or M < 512 or N < 512 or K < 512 or M * N * K < 5e8:
            continue
        key = (M, N, K, sa.get("m"), sa.get("n"))
        if key in seen:
            continue
        seen.add(key)
        feats = r["feats"]
        feats = feats if isinstance(feats, list) else json.loads(feats)
        ops = cm.ops_from_json(json.dumps(feats))
        mm = next(o for o in ops if getattr(o, "is_matmul", False))
        t_add = cm.predict_ops(ops, p_add) / 1e3
        t_ov = cm.predict_ops(ops) / 1e3
        meas = r["kernel_us"]
        compute = mm.matmul_macs / mm.cores / 1140 / 1e3  # µs
        frac = max(0.0, min(1.0, (t_add - compute) / t_add))  # memory fraction
        if K >= 2 * max(M, N):
            asp = "fat-K"
        elif min(M, N) * 2 <= max(M, N):
            asp = "thin M/N"
        elif M == N == K:
            asp = "square"
        else:
            asp = "rectangular"
        rows.append(
            (
                frac,
                100 * (t_add - meas) / meas,
                100 * (t_ov - meas) / meas,
                asp,
                M,
                N,
                K,
            )
        )

    markers = {"square": "o", "fat-K": "^", "thin M/N": "s", "rectangular": "D"}
    neg_out = 0  # stagger labels of the (colliding) negative outliers
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.axhline(0, color="0.6", lw=1.0, zorder=1)
    ax.axhspan(-10, 10, color="0.9", zorder=0, label="±10 %")
    for frac, ea, eo, asp, M, N, K in rows:
        ax.plot([frac, frac], [ea, eo], "-", color="0.85", lw=0.6, zorder=1)
        mk = markers[asp]
        ax.scatter(
            frac,
            ea,
            facecolors="none",
            edgecolors="#d62728",
            s=34,
            marker=mk,
            lw=1.0,
            zorder=2,
        )
        ax.scatter(
            frac,
            eo,
            color="#2ca02c",
            s=34,
            marker=mk,
            zorder=3,
            edgecolors="white",
            linewidths=0.3,
        )
        # Ring + label the residual outliers (|overlap err| > 12 %): these are all
        # thin-M/N shapes at high memory fraction -- flagged here, explained in §12.
        if abs(eo) > 12:
            ax.scatter(
                frac,
                eo,
                facecolors="none",
                edgecolors="#7f00ff",
                s=150,
                marker="o",
                lw=1.6,
                zorder=4,
            )
            if eo > 0:
                off = (6, 4)
            else:
                off = (8, -16 - 12 * neg_out)  # stagger colliding negative labels
                neg_out += 1
            ax.annotate(
                f"{M}×{K}×{N}",
                xy=(frac, eo),
                xytext=off,
                textcoords="offset points",
                fontsize=6.5,
                color="#5000a0",
                zorder=5,
                arrowprops=dict(arrowstyle="-", color="#7f00ff", lw=0.5)
                if eo < 0
                else None,
            )
    # aspect legend (marker shape) + model legend (color)
    for asp, mk in markers.items():
        ax.scatter([], [], color="0.4", marker=mk, s=34, label=asp)
    ax.scatter([], [], color="#2ca02c", s=40, label="overlap γ=0.46 (filled)")
    ax.scatter(
        [],
        [],
        facecolors="none",
        edgecolors="#d62728",
        s=40,
        label="additive γ=0 (open)",
    )
    ax.scatter(
        [],
        [],
        facecolors="none",
        edgecolors="#7f00ff",
        s=60,
        lw=1.6,
        label="thin-M/N outlier (|err|>12%, see §12)",
    )
    ax.set_xlabel("memory fraction   memory / (compute + memory)")
    ax.set_ylabel("prediction error  (%)")
    ax.set_title(
        "§10  γ=0.46 holds to ±10 % for the compute-leaning bulk;\n"
        "thin-M/N shapes (ringed) scatter at high memory fraction — deferred to §12"
    )
    ax.set_ylim(-25, 55)
    ax.annotate(
        f"{len(rows)} balanced configs (k=1, splits ≤8×8, K≥512)",
        xy=(0.03, 0.04),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=7.5,
        color="0.3",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.8"),
    )
    ax.legend(loc="upper left", fontsize=7.0, framealpha=0.9, ncol=2)
    _save(fig, "fig10_matmul_overlap")


# ============================================================================
# §14 -- coarse tiling LX-spill: the spilled bytes ARE counted (HBM), but the
# EFFECTIVE BANDWIDTH falls once the per-core working set overflows LX (~512 KB).
# Measured effBW = (R+W)/time per config; the model derates BW past the knee.
# ============================================================================
def fig_coarse_spill(_recs):
    cm = _cost_model()
    recs = _load(current_only=False)
    palette = {
        (16384, 4096): "#d62728",
        (16384, 2048): "#1f77b4",
        (8192, 2048): "#2ca02c",
        (4096, 4096): "#ff7f0e",
        (4096, 2048): "#8c564b",
        (2048, 2048): "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.axvspan(0.5, 100, color="#f2dede", alpha=0.5, zorder=0)
    ax.axvline(
        0.5, ls="--", color="0.4", lw=1.2, zorder=1, label="usable LX ≈ 512 KB/core"
    )
    seen = set()
    for r in recs:
        if r.get("op") != "softmax_row_tiling" or not r.get("feats"):
            continue
        R, C, t = r.get("rows"), r.get("cols"), r.get("tiles")
        if not (R and C and t) or t < 2:  # tiles=1 untiled = HBM by design, not a spill
            continue
        f = r["feats"]
        f = f if isinstance(f, list) else json.loads(f)
        ops = cm.ops_from_json(json.dumps(f))
        Rb, Wb = cm._fused_hbm_bytes(ops)
        effbw = (Rb + Wb) / 1e3 / r["kernel_us"]  # GB/s
        ws = 2 * (R / t / 32) * C * 2 / 1e6  # MB/core
        col = palette.get((R, C), "0.4")
        ax.scatter(
            ws, effbw, s=42, color=col, zorder=3, edgecolors="white", linewidths=0.4
        )
        if (R, C) not in seen:
            ax.scatter([], [], color=col, s=42, label=f"softmax {R}×{C}")
            seen.add((R, C))
    # model: effBW = balanced-softmax peak (~100) * spill derate past the knee
    wsx = np.logspace(np.log2(0.06), np.log2(9), 60, base=2)
    peak = 100.0
    model = [peak * min(1.0, (0.5 / w) ** 0.15) if w > 0.5 else peak for w in wsx]
    ax.plot(
        wsx,
        model,
        "-",
        color="0.35",
        lw=1.5,
        label="model: 100·min(1,(0.5/ws)$^{0.15}$)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("per-core working set  (MB;  ≈ 2 · rows/core · COLS · 2 B)")
    ax.set_ylabel("effective BW  (R+W)/time   (GB/s)")
    ax.set_title(
        "§14  Spilled bytes are counted (HBM); the effective BW just falls past LX"
    )
    ax.set_ylim(40, 115)
    ax.legend(loc="lower left", fontsize=7.2, framealpha=0.9)
    _save(fig, "fig12_coarse_spill")


# ============================================================================
# §15 -- coarse underfill `eff`: a short per-core tile (rpc rows) never fills the
# streaming pipeline. Softmax effective BW climbs with rpc to a plateau; the model
# eff = min(0.95, (rpc/13)^0.68) (calibrated) captures the rise. Above rpc~32 a
# mild decline is unmodeled.
# ============================================================================
def fig_coarse_eff(_recs):
    # One marker per CONFIG (not averaged): color = ROWS×COLS shape, label = tile count.
    # x = per-core tile height h = ROWS/(cores·tiles); LX-fitting points only.
    from collections import defaultdict

    recs = _load(current_only=False)
    pts = defaultdict(list)  # (R,C) -> [(h, effBW, tiles)]
    for r in recs:
        if (
            r.get("op") != "softmax_row_tiling"
            or not r.get("io_hbm_bytes")
            or not r.get("tiles")
        ):
            continue
        R, C, t = r.get("rows"), r.get("cols"), r.get("tiles")
        if not (R and C and t) or t < 2:
            continue
        h = R / t / 32
        ws = 2 * h * C * 2 / 1e6
        if ws > 1.2:  # LX-fitting only (isolate underfill from §14 spill)
            continue
        pts[(R, C)].append((h, int(r["io_hbm_bytes"]) / 1e3 / r["kernel_us"], t))
    plateau = max(e for v in pts.values() for _, e, _ in v)  # filled-pipeline effBW
    palette = {
        (16384, 4096): "#d62728",
        (16384, 2048): "#1f77b4",
        (8192, 2048): "#2ca02c",
        (8192, 4096): "#9467bd",
        (4096, 4096): "#ff7f0e",
        (4096, 2048): "#8c564b",
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for (R, C), v in sorted(pts.items()):
        col = palette.get((R, C), "0.4")
        for h, eff, t in sorted(v):
            ax.scatter(
                h, eff, s=46, color=col, zorder=3, edgecolors="white", linewidths=0.5
            )
            ax.annotate(
                f"{t}t",  # tile count identifies the config within a shape
                (h, eff),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6.2,
                color=col,
            )
        ax.scatter([], [], color=col, s=46, label=f"{R}×{C}  (ROWS×COLS)")
    rr = np.logspace(np.log2(1.5), np.log2(160), 60, base=2)
    model = plateau * np.minimum(0.95, (rr / 13) ** 0.68)
    ax.plot(
        rr,
        model,
        "-",
        color="0.4",
        lw=1.5,
        label="model: BW·min(0.95, (height/13)$^{0.68}$)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel(
        "per-core tile height  =  ROWS / (cores × tiles)   [rows]   (label = tile count)"
    )
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title(
        "§15  Underfill `eff`: BW climbs with the per-core tile, then plateaus"
    )
    ax.legend(loc="lower right", fontsize=7.0, framealpha=0.9)
    _save(fig, "fig13_coarse_eff")


_FIGS = {
    "matmul_hbm": fig_matmul_hbm,
    "matmul_spill": fig_matmul_spill,
    "coarse_spill": fig_coarse_spill,
    "coarse_eff": fig_coarse_eff,
    "matmul_peak": fig_matmul_peak,
    "matmul_overlap": fig_matmul_overlap,
    "pointwise_baseline": fig_pointwise_baseline,
    "pointwise_vcurve": fig_pointwise_vcurve,
    "pointwise_arity": fig_pointwise_arity,
    "broadcast_effbw": fig_broadcast_effbw,
    "write_spill": fig_write_spill,
    "reduction": fig_reduction,
    "transport": fig_transport,
}


def main():
    import sys

    recs = _load()
    want = sys.argv[1:] or list(_FIGS)
    for name in want:
        if name not in _FIGS:
            raise SystemExit(f"unknown figure {name!r} (have: {', '.join(_FIGS)})")
        print(f"figure {name}:")
        _FIGS[name](recs)


if __name__ == "__main__":
    main()
