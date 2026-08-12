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
# ============================================================================
# COMPREHENSIVE ~5-HOUR GOLDEN SWEEP (torch.profiler "Self SPYRE" kernel time).
# ~43 profiled runs x ~6.5 min wall (60s sync hang + forced recompile) ~= 4.7 h.
#
# PART 1 -- re-prove the core findings (data for 3 figures):
#   FIG 1  broadcast operand is a ONE-TIME load, not a per-core reload   (P1a)
#   FIG 2  read/write bandwidth V-shape -> the turnaround model          (P1b,P1c,P1d)
#   FIG 3  is treating LX as ~free reasonable?  (fused-add arity + P2 LX on/off)
# PART 2 -- verify/refine the COARSE-TILING reduction model (sum/amax/amin),
#   each in a NORMAL (LX_PLANNING=0, partials in HBM) and an LX-ON variant.
#
# *** I/O CAVEAT (read before trusting bw_gbps) ***
# bw_gbps = model_io_hbm_bytes / kernel_us. For BROADCAST / FUSED-BROADCAST ops
# (bcast, bcastcol, write) the model has historically MIS-SIZED the broadcast
# operand. Every run here logs the per-tensor IO breakdown (the "IO ..." lines)
# so you can confirm each [1,C]/[R,1] operand is counted at its SMALL device
# size, NOT expanded to the full output. For the write-only point in FIG 2,
# derive the write BW from the OUTPUT bytes (unambiguous), not the total io.
#
#   bash docs/source/user_guide/examples/run_5h_sweep.sh            # full ~5h
#   SECTIONS="P1a P2a" bash .../run_5h_sweep.sh                     # subset
# Output: <repo-root>/haoyang_logs/grand_sweep_<timestamp>.log (forward this).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/grand_sweep_$(date +%Y%m%d_%H%M%S).log"
# Default = only PART 3 (the chain LX-residency win) -- P1/P2 numbers already collected.
# For a fresh full sweep: SECTIONS="P1a P1b P1c P1d P2a P2b P2c P2d P3" bash ...
SECTIONS="${SECTIONS:-P3}"
ROWS="${ROWS:-2048}"   # default rows for pointwise ops (>=64/core at 32 cores)
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1   # recompile each run

echo "==== grand 5h sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  rows: $ROWS  sections: $SECTIONS" \
  | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {  # read profile_ops stdout on stdin; keep IO + MODEL estimate + SUMMARY; tee
  local out; out=$(grep -E '^(IO |MODEL |SUMMARY)')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
run() {  # run <op> <cols>          pointwise, rows=$ROWS
  BENCH_OP="$1" BENCH_ROWS="$ROWS" BENCH_COLS="$2" \
    python "$PROFILE_OPS" 2>/dev/null | _emit "op=$1 cols=$2"
}
runR() {  # runR <rows> <op> <cols>  cores=32, rows override (broadcast reload probe)
  SENCORES=32 BENCH_OP="$2" BENCH_ROWS="$1" BENCH_COLS="$3" \
    python "$PROFILE_OPS" 2>/dev/null | _emit "op=$2 rows=$1 cols=$3"
}
runct() {  # runct <lx> <tiles> <op> <B> <D>   coarse-tiled reduction
  LX_PLANNING="$1" BENCH_TILES="$2" BENCH_OP="$3" BENCH_ROWS="$4" BENCH_COLS="$5" \
    python "$PROFILE_OPS" 2>/dev/null | _emit "op=$3 tiles=$2 lx=$1 B=$4 D=$5"
}

# ======================= PART 1: core findings ==============================
# P1a) FIG 1 -- broadcast = ONE-TIME load (not per core). cores=32, small ROWS so the
#   per-core reload (cores*C) would be a visible fraction of the 2*R*C output traffic.
#   gap = kernel(bcast) - kernel(bcastcol): ~0 => loaded once; ~cores*C => per core.
#   bcast/bcastcol are BROADCAST -> CHECK the IO lines size the [1,C]/[R,1] operand small.
has P1a && { echo "## P1a broadcast one-time-load (FIG1; cores=32, vary ROWS, C=16384)" \
  | tee -a "$LOG"
  for r in 64 128 256; do
    runR "$r" bcast 16384; runR "$r" bcastcol 16384; runR "$r" add 16384
  done; }
# P1b) FIG 2 anchor -- kernel time linear in I/O, fill~0, BW_PEAK slope. Clean 1R+1W neg
#   (no broadcast -> exact I/O) across a size sweep.
has P1b && { echo "## P1b BW size-sweep (FIG2 anchor; neg, rows=2048)" | tee -a "$LOG"
  for n in 1024 2048 4096 8192; do run neg "$n"; done; }
# P1c) FIG 2 main -- read/write V-shape -> turnaround. Read fraction spans 1.0 (read), 0.8
#   (add4), 0.75 (add3), 0.67 (mul), 0.5 (neg), 0.0 (write). read/neg/mul/add3/add4 have
#   EXACT non-broadcast I/O. `write` is the only FUSED-BROADCAST op -> its bw_gbps is the
#   least reliable point; verify its IO breakdown and use OUTPUT bytes for the write BW.
has P1c && { echo "## P1c read/write V-shape (FIG2; turnaround)" | tee -a "$LOG"
  for op in read neg mul add3 add4 write; do run "$op" 4096; done
  for op in read neg write; do run "$op" 8192; done; }   # 3 corners at a 2nd size
# P1d) FIG 2 -- reduction read-only cluster (~150) + the sumcol access-pattern outlier.
has P1d && { echo "## P1d reductions (FIG2 read-only + sumcol)" | tee -a "$LOG"
  for op in sumrow sumcol amax; do run "$op" 4096; done; }

# ===================== PART 2: coarse-tiling model ==========================
# B=2048, D=512 unless noted. Each run logs the loop-aware IO (loop_factor/loop_trip) so
# the predicted R/W can be checked against bw_gbps. NORMAL = LX_PLANNING=0 (per-tile
# partial in HBM -> combine round-trips counted); LX-ON = LX_PLANNING=1 (partial in LX,
# ~free). The LX on/off delta is the controlled "is LX free?" test (FIG 3).
# P2a) K-SWEEP -- calibrate c_loop + the combine overhead. Total input read is constant;
#   combine traffic ~ 2*K*D grows with K. kernel time vs K (after subtracting combine)
#   gives the per-iteration loop overhead c_loop. Both LX modes.
has P2a && { echo "## P2a coarse-tile K-sweep (ctsum, B=2048 D=512; LX off+on)" \
  | tee -a "$LOG"
  for k in 2 4 8 16; do runct 0 "$k" ctsum 2048 512; runct 1 "$k" ctsum 2048 512; done; }
# P2b) OP coverage -- amax/amin behave like sum (combine = maximum/minimum). K=8, both LX.
has P2b && { echo "## P2b coarse-tile op coverage (ctamax, ctamin; K=8)" | tee -a "$LOG"
  for op in ctamax ctamin; do runct 0 8 "$op" 2048 512; runct 1 8 "$op" 2048 512; done; }
# P2c) D-SWEEP -- the accumulator round-trip is ~2*K*D; sweeping D (K=8, LX off so the
#   accum stays in HBM) exposes the combine traffic as a K*D-linear term.
has P2c && { echo "## P2c coarse-tile D-sweep (ctsum K=8, LX off, B=2048)" | tee -a "$LOG"
  for d in 64 256 1024; do runct 0 8 ctsum 2048 "$d"; done; }
# P2d) UNTILED baselines -- tiles=1 (no hint) -> plain reduction, the tiled-vs-untiled
#   reference (full input read once, tiny output, NO combine). LX off.
has P2d && { echo "## P2d untiled reduction baselines (B=2048 D=512)" | tee -a "$LOG"
  for op in ctsum ctamax ctamin; do runct 0 1 "$op" 2048 512; done; }

# ============ PART 3: pointwise-chain tiling overhead (c_loop_chain) ============
# The first 2 runs (2026-06-25, [2048,4096]) overturned the premise: the compiler ALREADY
# places the intermediate y=a+b in LX UNTILED (it fits per-core), so tiling wins nothing
# here -- it only ADDS overhead, and a lot: K=1 -> 584us, K=8 -> 780us (~28us/tile, ~30x
# the reduction c_loop=860ns). The model (single c_loop) under-predicts this badly
# (tiled err -29.5%). The [4096,8192] untiled run (y EXCEEDS LX) HUNG -- skip it.
# So: pin the POINTWISE-chain c_loop with a clean K-sweep at [2048,4096] (y in LX every
# run, only the loop count varies). We already have K=1 and K=8; fill in K=2,4,16.
has P3 && { echo "## P3 chain c_loop K-sweep ([2048,4096], LX on; have K=1,8)" \
  | tee -a "$LOG"
  for k in 2 4 16; do runct 1 "$k" chain 2048 4096; done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
