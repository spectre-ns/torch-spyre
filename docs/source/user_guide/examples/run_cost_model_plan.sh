#!/usr/bin/env bash
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
#
# Runs the cost-model calibration plan (see notes/cost_model_design.md sec 8) and
# collects every step's output into ONE log file you can forward back.
#
#   bash examples/run_cost_model_plan.sh
#
# Output: stdout (live) AND haoyang_logs/cost_model_results_<timestamp>.log (forward this file).
# Each run prints: cost-model PREDICTION (SPYRE_DUMP_COST, stderr) + measured
# per-kernel DEVICE latency table (SPYRE_PROFILE_SYNC).

set -u
cd "$(dirname "$0")/.." || exit 1

mkdir -p haoyang_logs
LOG="haoyang_logs/cost_model_results_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

# Common environment for every run.
export SPYRE_PROFILE=1            # record per-kernel times
export SPYRE_PROFILE_SYNC=1       # -> per-kernel DEVICE latency (deterministic min)
export SPYRE_DUMP_COST=1          # print cost-model prediction at compile
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1   # recompile each run so dumps fire
export BENCH_RUNS="${BENCH_RUNS:-100}"
export BENCH_WARMUP="${BENCH_WARMUP:-20}"

echo "================ cost-model calibration run ================"
echo "date:    $(date)"
echo "host:    $(hostname)"
echo "git:     $(git rev-parse --short HEAD 2>/dev/null) $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "python:  $(python -c 'import sys,torch; print(sys.version.split()[0], "torch", torch.__version__)' 2>/dev/null)"
echo "runs=$BENCH_RUNS warmup=$BENCH_WARMUP  SENCORES=${SENCORES:-default(32)}"
echo "log:     $LOG"
echo "==========================================================="

step() {
  local label="$1"; shift
  echo ""
  echo "########################################################################"
  echo "## $label"
  echo "## + $*"
  echo "########################################################################"
  ( "$@" ) || echo "[STEP FAILED: $label]"
}

# ---------------------------------------------------------------------------
# RUNG 1 — fit fill + BW_HBM: single gelu, size sweep (ROWS=512, COLS varies).
#   Expect: device min linear in elements. slope -> BW_HBM, intercept -> fill.
# ---------------------------------------------------------------------------
for n in 512 1024 2048 4096 8192; do
  step "RUNG1 size sweep: gelu[512x${n}]" \
    env BENCH_OP=gelu BENCH_ROWS=512 BENCH_COLS="$n" python examples/bench_ops.py
done

# ---------------------------------------------------------------------------
# RUNG 2 — arithmetic free? relu vs gelu, same size.
#   Expect: ~equal if memory-bound; gelu slower => activations add a compute term.
# ---------------------------------------------------------------------------
for op in relu gelu sigmoid exp; do
  step "RUNG2 arithmetic: ${op}[512x1024]" \
    env BENCH_OP="$op" BENCH_ROWS=512 BENCH_COLS=1024 python examples/bench_ops.py
done

# ---------------------------------------------------------------------------
# RUNG 3 — traffic counting: 1-input (gelu) vs 2-input (mul/add).
#   Expect: mul/add ~1.5x gelu (3 memory passes vs 2).
# ---------------------------------------------------------------------------
for op in gelu mul add; do
  step "RUNG3 traffic: ${op}[512x1024]" \
    env BENCH_OP="$op" BENCH_ROWS=512 BENCH_COLS=1024 python examples/bench_ops.py
done

# ---------------------------------------------------------------------------
# RUNG 3b — attribute the 2-input gap to fill vs BW (controlled).
#   (a) mul size sweep at fixed 3-stream pattern: slope->BW_3stream, intercept->fill.
#       If intercept ~= gelu's 20us, the 2-input gap is a BW (slope) effect, not fill.
#   (b) EQUAL total HBM bytes, different stream count (2 vs 3). Model predicts each
#       PAIR equal; any measured gap is the pure stream-count effect with fill held
#       equal. Two budgets: gap constant 3MB->6MB => fixed per-stream cost; gap
#       doubles => per-byte BW effect.
#         3MB: mul[512x1024] (3 streams) == gelu[512x1536] (2 streams)  [3,145,728 B]
#         6MB: mul[512x2048] (3 streams) == gelu[512x3072] (2 streams)  [6,291,456 B]
# ---------------------------------------------------------------------------
for n in 512 1024 2048 4096 8192; do
  step "RUNG3b mul size: mul[512x${n}]" \
    env BENCH_OP=mul BENCH_ROWS=512 BENCH_COLS="$n" python examples/bench_ops.py
done
step "RUNG3b equal-bytes 3MB / 3-stream: mul[512x1024]" \
  env BENCH_OP=mul BENCH_ROWS=512 BENCH_COLS=1024 python examples/bench_ops.py
step "RUNG3b equal-bytes 3MB / 2-stream: gelu[512x1536]" \
  env BENCH_OP=gelu BENCH_ROWS=512 BENCH_COLS=1536 python examples/bench_ops.py
step "RUNG3b equal-bytes 6MB / 3-stream: mul[512x2048]" \
  env BENCH_OP=mul BENCH_ROWS=512 BENCH_COLS=2048 python examples/bench_ops.py
step "RUNG3b equal-bytes 6MB / 2-stream: gelu[512x3072]" \
  env BENCH_OP=gelu BENCH_ROWS=512 BENCH_COLS=3072 python examples/bench_ops.py

# ---------------------------------------------------------------------------
# RUNG 4 — BW_LX: unary gelu chain, ALL intermediates forced into LX.
#   Expect: HBM traffic fixed (read x + write y); latency slope vs depth =
#   2*|x_tile|/BW_LX. First confirm intermediates actually got 'lx'.
# ---------------------------------------------------------------------------
echo ""
echo "## RUNG4 LX-placement check (depth=4): expect 'lx' allocations on intermediates"
env SPYRE_DUMP_IR=1 SPYRE_DUMP_IR_FILE=/tmp/lxchk_ir.txt \
    BENCH_LX_ALL=1 BENCH_DEPTH=4 BENCH_ROWS=512 BENCH_COLS=1024 \
    python examples/bench_ops.py || echo "[depth-4 dump run failed]"
echo "-- intermediate allocations in the depth-4 chain IR --"
grep -cE "'lx'" /tmp/lxchk_ir.txt 2>/dev/null && echo "(count of lx allocations above)"
grep -E "allocation=" /tmp/lxchk_ir.txt 2>/dev/null | head -20

for d in 1 2 4 8 16; do
  step "RUNG4 LX chain: gelu depth=${d} (all-LX)" \
    env BENCH_OP=gelu BENCH_LX_ALL=1 BENCH_DEPTH="$d" BENCH_ROWS=512 BENCH_COLS=1024 \
    python examples/bench_ops.py
done

# ---------------------------------------------------------------------------
# RUNG 5 — shared vs per-core HBM BW: gelu[512x1024], sweep SENCORES.
#   Expect: latency ∝ 1/cores => per-core BW; flat => shared HBM BW.
# ---------------------------------------------------------------------------
for c in 1 2 4 8 16 32; do
  step "RUNG5 SENCORES=${c}: gelu[512x1024]" \
    env SENCORES="$c" BENCH_OP=gelu BENCH_ROWS=512 BENCH_COLS=1024 \
    python examples/bench_ops.py
done

# ---------------------------------------------------------------------------
# RUNG 6 — broadcast reuse: cached vs re-fetched.
#   bcast = a[512xN] + b[1xN]  vs  add = a[512xN] + b[512xN] (full), same size.
#   bcast ~ add (3-pass) => RE-FETCH (count broadcast at full).
#   bcast ~ gelu (2-pass) => CACHED  (count broadcast at ~one row).
#   The SPYRE_DUMP_COST line for bcast also flags whether b is a true broadcast
#   load in the IR (index uses only the column var), not expanded to full.
# ---------------------------------------------------------------------------
for n in 1024 4096; do
  step "RUNG6 broadcast: bcast[512x${n}] (b is [1x${n}])" \
    env BENCH_OP=bcast BENCH_ROWS=512 BENCH_COLS="$n" python examples/bench_ops.py
  step "RUNG6 full ref:  add[512x${n}]" \
    env BENCH_OP=add BENCH_ROWS=512 BENCH_COLS="$n" python examples/bench_ops.py
  step "RUNG6 mul-broadcast: mulbcast[512x${n}] (b is [1x${n}])" \
    env BENCH_OP=mulbcast BENCH_ROWS=512 BENCH_COLS="$n" python examples/bench_ops.py
  step "RUNG6 mul full ref: mul[512x${n}]" \
    env BENCH_OP=mul BENCH_ROWS=512 BENCH_COLS="$n" python examples/bench_ops.py
done

# ---------------------------------------------------------------------------
# RUNG 7 — effective per-byte rate vs # concurrent streams.
#   ops: gelu (2-stream) | mul (3) | add3 (4) | add4 (5). Each a size sweep.
#   Per op, fit slope -> rate(n_streams) and intercept -> fill. Tells us whether
#   the rate keeps dropping with each added stream (contention curve) or steps
#   once at 2->3 then plateaus, and whether fill stays op/stream-independent.
#   NOTE: this measures effective per-byte RATE, not proven DRAM bandwidth --
#   from latency alone we can't separate DRAM contention from on-chip issue rate.
#   Each n-ary add must compile to ONE fused kernel (n input loads + 1 store);
#   the per-kernel table below confirms it (a single sdsc_fused_* row).
# ---------------------------------------------------------------------------
for op in gelu mul add3 add4; do
  for n in 512 1024 2048 4096; do
    step "RUNG7 streams: ${op}[512x${n}]" \
      env BENCH_OP="$op" BENCH_ROWS=512 BENCH_COLS="$n" python examples/bench_ops.py
  done
done

# ---------------------------------------------------------------------------
# RUNG 8 — peak-bandwidth probe: why is effective BW ~111 vs the 204.8 GB/s
#   LPDDR5 peak (_HBM_BW_GBS in work_division.py)?
#   copy (2-stream r/w) and read (read-only reduction), size sweep to large.
#   Watch: does effective BW PLATEAU (~111 = a real ceiling) or keep climbing
#   toward 204.8 (so 111 was a small-op/ramp artifact)? Does read-only (1 pass)
#   come in at ~half copy's latency (shared rate) or faster (bidirectional cap)?
#   NB: this is the latency-inferred BW only. The ACTUAL DDR bandwidth needs
#   aiu-smi paired with BENCH_BW_SUSTAIN_S -- run that separately (two-terminal;
#   see examples/bench_bandwidth.py header + docs .../profiling/device_monitoring.md).
# ---------------------------------------------------------------------------
# neg is a constant-free 1R+1W check on copy (scalar should be free -> neg ~= copy).
for op in neg copy read write; do
  for n in 1024 4096 16384 65536; do
    step "RUNG8 bandwidth: ${op}[512x${n}]" \
      env BENCH_BW_OP="$op" BENCH_ROWS=512 BENCH_COLS="$n" python examples/bench_bandwidth.py
  done
done

# ---------------------------------------------------------------------------
# RUNG 9 — tile-footprint / burst-size test: copy at a large FIXED size, sweep
#   SENCORES. Fewer cores => bigger per-core tile => bigger contiguous DMA bursts.
#   If copy's read+write BW RISES as tiles grow, the ~98 cap is burst/scratchpad-
#   limited (the "memory requests are limited" effect). If FLAT, the cap is
#   intrinsic to mixing reads+writes (turnaround / half-duplex), not request size.
#   This is the control that separates the burst-size mechanism from turnaround.
# ---------------------------------------------------------------------------
for c in 1 2 4 8 16 32; do
  step "RUNG9 tile-size: copy[512x16384] SENCORES=${c}" \
    env SENCORES="$c" BENCH_BW_OP=copy BENCH_ROWS=512 BENCH_COLS=16384 \
    python examples/bench_bandwidth.py
done

# ---------------------------------------------------------------------------
# RUNG 10 — read/write turnaround V-curve (latency side): hold the read pattern
#   fixed (one contiguous read of x) and add writes, so write-fraction goes
#   copy 1R1W=0.50 -> w2 1R2W=0.67 -> w3 1R3W=0.75. With read-only (0.0 ~172) and
#   write-only (1.0 ~146) from rung 8, this maps effective BW vs write-fraction.
#   A V-shaped dip (minimum near balanced 1:1) is the turnaround signature; a flat
#   or monotonic curve is not. REQUIRES w2/w3 to FUSE to one kernel (1 read + N
#   writes) -- the printed kernel count + SPYRE_DUMP_COST (1 input, N outputs) confirm
#   it; if they split, the read count is wrong and the point is invalid (note it).
# ---------------------------------------------------------------------------
for op in copy w2 w3; do
  for n in 16384 65536; do
    step "RUNG10 write-frac: ${op}[512x${n}]" \
      env BENCH_BW_OP="$op" BENCH_ROWS=512 BENCH_COLS="$n" python examples/bench_bandwidth.py
  done
done

# ---------------------------------------------------------------------------
# RUNG 11 — reductions (INITIAL model: read the full input @ the read rate ~176,
#   plus a cross-core ring combine when the reduced axis is split across cores).
#   (a) arithmetic-free? sumrow vs amax vs mean, same size (equal => free, like rung 2).
#   (b) read-dominated traffic: sumrow size sweep -> latency ~ full-input read at the
#       read rate; the [ROWS] output is negligible. Fits bw_read for reductions.
#   (c) cross-core ring combine: sumall (scalar out -> reduced axis split across ALL
#       cores) vs sumrow (output=ROWS -> reduced axis NOT split). At equal input,
#       sumall ~= sumrow if combine << HBM I/O (expected); a gap fits psum_per_elem.
#   Pair with SPYRE_DUMP_COST to check the reduction input is now sized as the FULL
#   input (out_elems x reduction_size), not the output size.
# ---------------------------------------------------------------------------
for op in sumrow amax mean; do
  step "RUNG11a arithmetic: ${op}[512x4096]" \
    env BENCH_OP="$op" BENCH_ROWS=512 BENCH_COLS=4096 python examples/bench_ops.py
done
for n in 1024 4096 16384; do
  step "RUNG11b read-BW: sumrow[512x${n}]" \
    env BENCH_OP=sumrow BENCH_ROWS=512 BENCH_COLS="$n" python examples/bench_ops.py
  step "RUNG11c combine: sumall[512x${n}]" \
    env BENCH_OP=sumall BENCH_ROWS=512 BENCH_COLS="$n" python examples/bench_ops.py
done

echo ""
echo "================ DONE. forward this file: $LOG ================"
