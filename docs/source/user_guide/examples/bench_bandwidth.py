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

"""Pure-I/O DRAM bandwidth probe for Spyre (Exp B in the cost-model plan).

Why: our pointwise cost model fits an effective DRAM rate of ~111 GB/s, but the
compiler's own matmul model uses the LPDDR5 *peak* of 204.8 GB/s
(``_HBM_BW_GBS`` in ``torch_spyre/_inductor/work_division.py``). This probe pushes
large, low-compute memory traffic across all cores to find what bandwidth is
actually achievable and explain the ~1.85x gap (204.8 / 111).

Workloads, contrasted:
    neg   : y = -x                genuine 1R+1W, NO constant -> constant-free check
    copy  : y = x + 1.0           1R+1W (the scalar 1.0 is a broadcast immediate,
            costs ~no HBM -- copy's latency ~= gelu's, a true unary)
    read  : y = x.sum(dim=-1)     read-only (~1R, tiny write)  -> isolates read BW
    write : y = b[1,N] + c[N,1]   write-only (inputs broadcast -> cached ~free per
            rung 6, output is the full grid) -> isolates write BW

copy IS genuinely 1R+1W: the scalar 1.0 is the maximally-broadcast input, cached on-chip
like the rung-6 broadcasts (empirically copy ~= gelu, a true unary, in latency). An old
dump counted it as a full read (3 passes) -- a cost-model bug, now fixed (0-loop-var
indices are flagged broadcast/free). So copy ~98 GB/s IS the genuine balanced read+write
rate, and the penalty (98 << read ~172 / write ~146) stands. `neg` re-confirms at large
size. The OPEN question is the penalty's MECHANISM (turnaround? half-duplex? shared-bus
saturation?) -- see rung 9 + notes/bandwidth_turnaround_experiment.md.

w2/w3 (multi-output) are CONFOUNDED -- each output re-reads x; do not trust their BW.

The effective BW here is bytes / device-min-latency (== GB/s). The fixed ~20 us
per-kernel term depresses it at small sizes, so SWEEP size: the asymptote (large
size, or the slope d(bytes)/d(latency) across the sweep) is the real ceiling.

Pair with ``aiu-smi`` for the ACTUAL DDR bandwidth (latency-independent): set
``BENCH_BW_SUSTAIN_S=20`` to keep the device saturated for a sampling window, and
run ``aiu-smi`` in a second shell (see docs/source/user_guide/profiling/
device_monitoring.md). If aiu-smi reads ~200 while our latency BW reads ~111, the
gap is host/ramp overhead; if aiu-smi also reads ~111, that IS the ceiling.

Knobs:
    BENCH_BW_OP        neg | copy | read | write | w2 | w3   (default copy)
    BENCH_ROWS, BENCH_COLS                      shape (default 512 x 8192)
    BENCH_RUNS, BENCH_WARMUP
    BENCH_BW_SUSTAIN_S N   after measuring, loop the op for ~N seconds so aiu-smi
                           can sample DDR bandwidth (default 0 = off)
    SENCORES           cores (runtime env; default 32)

Examples:
    # size sweep: does effective BW plateau at ~111 or keep climbing toward 204.8?
    for n in 1024 4096 16384 65536; do \
      BENCH_BW_OP=copy BENCH_COLS=$n python examples/bench_bandwidth.py; done
    # read-only vs read+write at one big size
    BENCH_BW_OP=read BENCH_COLS=16384 python examples/bench_bandwidth.py
    # write-only: confirm the write bottleneck (both inputs broadcast, output full)
    BENCH_BW_OP=write BENCH_COLS=16384 python examples/bench_bandwidth.py
    # write-fraction sweep (turnaround V-curve): copy(0.5) -> w2(0.67) -> w3(0.75)
    for op in copy w2 w3; do \
      BENCH_BW_OP=$op BENCH_COLS=65536 python examples/bench_bandwidth.py; done
    # tile-size test: copy at large size, sweep cores (bigger tile = bigger bursts)
    for c in 1 2 4 8 16 32; do SENCORES=$c BENCH_BW_OP=copy \
      BENCH_COLS=16384 python examples/bench_bandwidth.py; done
    # aiu-smi window (start aiu-smi in another shell during the sustained phase)
    BENCH_BW_OP=copy BENCH_COLS=65536 BENCH_BW_SUSTAIN_S=20 \
      python examples/bench_bandwidth.py
"""

import os
import time

os.environ.setdefault("SPYRE_PROFILE", "1")
os.environ.setdefault("SPYRE_PROFILE_SYNC", "1")

import torch  # noqa: E402

from torch_spyre.execution import profiling  # noqa: E402
from torch_spyre.execution.bench import measure_device  # noqa: E402

DEVICE = torch.device("spyre")
OP = os.environ.get("BENCH_BW_OP", "copy")
ROWS = int(os.environ.get("BENCH_ROWS", "512"))
COLS = int(os.environ.get("BENCH_COLS", "8192"))
RUNS = int(os.environ.get("BENCH_RUNS", "100"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "20"))
SUSTAIN_S = float(os.environ.get("BENCH_BW_SUSTAIN_S", "0"))

DT = 2  # fp16 bytes
# Overhead subtracted from the SPYRE_PROFILE_SYNC-min measurement this probe uses: the
# non-deterministic Memset/host-setup bucket, NOT the cost-model fill (now 0). The
# golden kernel time has no fixed term; only this min-based probe still sees the bucket.
FILL_NS = 20_000.0
PEAK_GBPS = 204.8  # _HBM_BW_GBS: LPDDR5 aggregate peak (work_division.py)

# op -> (fn, hbm_passes). passes = the dominant HBM memory passes for the BW est:
#   copy  = read x + write y                 -> 2  (read + write)
#   read  = read x + write tiny reduction    -> 1  (read-only; the [ROWS] write is
#           negligible next to the [ROWS,COLS] read)
#   write = write full grid from broadcasts  -> 1  (write-only; both inputs [1,N]/[N,1]
#           load once and cache ~free per rung 6, so ~no read traffic)
#   neg   = genuine 1 read + 1 write, NO constant operand -> a constant-free check
#           on copy. copy (x+1.0) is ALSO 1R+1W: the scalar 1.0 is a broadcast
#           immediate that costs ~no HBM (same caching as rung 6; copy's latency
#           ~= gelu's, a true unary). The dump's old "3 passes" was an extraction
#           bug (scalars have 0 loop vars, now flagged broadcast/free). Expect
#           neg ~= copy; both give the genuine balanced read+write rate.
#   w2/w3 = multi-output: CONFOUNDED -- each output re-reads x (not a shared load),
#           so they are NOT 1R:NW. Kept only for the record; do not trust their BW.
_OPS = {
    "neg": (lambda x: -x, 2),
    "copy": (lambda x: x + 1.0, 2),
    "read": (lambda x: x.sum(dim=-1), 1),
    "write": (lambda b, c: b + c, 1),
    "w2": (lambda x: (x + 1.0, x - 1.0), 3),
    "w3": (lambda x: (x + 1.0, x - 1.0, x * 2.0), 4),
}


def _make_inputs(op):
    """Device input tensors for OP (all produce a [ROWS,COLS] working set).

    ``write`` adds two broadcast operands (b[1,COLS] + c[ROWS,1]) -> a full
    [ROWS,COLS] output; both inputs load once and cache (~free per rung 6), so the
    kernel is write-dominated. Confirm via the broadcast flags in SPYRE_DUMP_COST.
    """
    if op == "write":
        b = torch.rand(1, COLS, dtype=torch.float16).to(DEVICE)
        c = torch.rand(ROWS, 1, dtype=torch.float16).to(DEVICE)
        return (b, c)
    return (torch.rand(ROWS, COLS, dtype=torch.float16).to(DEVICE),)

torch.manual_seed(0xAFFE)


def _sync():
    """Drain the device (no D2H copy); no-op if the runtime sync is absent."""
    try:
        import torch_spyre._C as _C

        _C.synchronize()
    except Exception:  # noqa: BLE001 - tolerate a runtime without sync
        pass


def main():
    if OP not in _OPS:
        raise SystemExit(f"unknown BENCH_BW_OP={OP!r} (use {list(_OPS)})")
    fn, passes = _OPS[OP]
    inputs = _make_inputs(OP)
    compiled = torch.compile(fn)

    elems = ROWS * COLS
    bytes_moved = passes * elems * DT
    cores = os.environ.get("SENCORES", "default(32)")
    profiling.set_report_at_exit(False)
    print(
        f"bandwidth probe: op={OP} shape=[{ROWS}x{COLS}] elems={elems} "
        f"passes={passes} bytes={bytes_moved}  SENCORES={cores}"
    )

    snap = measure_device(lambda: compiled(*inputs), runs=RUNS, warmup=WARMUP)
    print(profiling.format_report())

    # Total device time for one workload = sum of its kernels' deterministic mins
    # (kernels run sequentially within the call; each min is jitter-stripped).
    n_kernels = len(snap)
    total_min_ns = sum(rec["min_ns"] for rec in snap.values()) if snap else 0.0
    if total_min_ns > 0:
        bw = bytes_moved / total_min_ns  # bytes/ns == GB/s
        net_ns = total_min_ns - FILL_NS * n_kernels  # drop per-kernel fixed term
        line = (
            f"  effective BW: {bw:6.1f} GB/s  "
            f"({bytes_moved} B / {total_min_ns / 1000:.2f} us, "
            f"{n_kernels} kernel(s))  = {bw / PEAK_GBPS * 100:.0f}% of {PEAK_GBPS} peak"
        )
        if net_ns > 0:
            bw_net = bytes_moved / net_ns
            line += (
                f"\n  BW excl. fill: {bw_net:6.1f} GB/s "
                f"(minus {FILL_NS / 1000:.0f}us x {n_kernels}; "
                f"true asymptote = sweep slope)"
            )
        print(line)

    if SUSTAIN_S > 0:
        # Keep the device saturated so aiu-smi can read steady-state DDR BW.
        # Back-to-back async launches with a periodic sync bound the queue.
        print(
            f"=== SUSTAINED I/O START: start aiu-smi NOW; "
            f"saturating ~{SUSTAIN_S:.0f}s ===",
            flush=True,
        )
        t0 = time.perf_counter()
        launches = 0
        while time.perf_counter() - t0 < SUSTAIN_S:
            for _ in range(50):
                compiled(*inputs)
            launches += 50
            _sync()
        wall = time.perf_counter() - t0
        host_bw = launches * bytes_moved / (wall * 1e9)
        print(
            f"=== SUSTAINED I/O END: {launches} launches in {wall:.1f}s, "
            f"host-side BW {host_bw:.1f} GB/s (compare to aiu-smi DDR reading) ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
