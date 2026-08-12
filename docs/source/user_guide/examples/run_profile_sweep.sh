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
# GOLDEN re-sweep: collect torch.profiler per-kernel DEVICE times (profile_ops
# SUMMARY lines) across the op ladder, to REBUILD the cost model from kernel time.
# We forget the old SPYRE_PROFILE_SYNC fit (its ~20us "fixed" was non-deterministic
# Memset/host overhead, not kernel cost) and refit fill + BW on `kernel_us`.
#
# Needs the kineto-spyre wheel (see docs/.../profiling/pytorch_profiler.md).
# Output: haoyang_logs/profile_sweep_<timestamp>.log (forward this file).
#
#   bash docs/source/user_guide/examples/run_profile_sweep.sh         # full sweep
#   SECTIONS="A D" bash docs/source/user_guide/examples/run_profile_sweep.sh   # A and D
# (run from anywhere; logs always land in <repo-root>/haoyang_logs.)
#
# WARNING: the profiled call has hit a ~60s sync hang ("possible lost completion").
# RUN SECTION A FIRST (12 runs) to gauge per-run wall time before the full sweep.

set -u
# Resolve paths from the script's own location, not a fixed depth below the repo root
# (the examples/ dir has moved to docs/source/user_guide/examples/). Logs always land in
# <repo-root>/haoyang_logs; profile_ops.py is found next to this script.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/profile_sweep_$(date +%Y%m%d_%H%M%S).log"
SECTIONS="${SECTIONS:-A B C D E F G}"
ROWS="${ROWS:-2048}"   # bigger than 512 so each of 32 cores gets many rows (>=64/core)
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1   # recompile each run

echo "==== golden profiler re-sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  rows: $ROWS  sections: $SECTIONS" \
  | tee -a "$LOG"

run() {  # run <op> <cols> -> append the IO device-layout breakdown + SUMMARY line
  local out
  out=$(BENCH_OP="$1" BENCH_ROWS="$ROWS" BENCH_COLS="$2" \
        python "$PROFILE_OPS" 2>/dev/null | grep -E '^(IO |MODEL |SUMMARY)')
  echo "${out:-SUMMARY op=$1 cols=$2 FAILED}" | tee -a "$LOG"
}
rung() {  # rung <rows> <op> <cols> -> pin cores=32, override ROWS (broadcast reload probe)
  local out
  out=$(SENCORES=32 BENCH_OP="$2" BENCH_ROWS="$1" BENCH_COLS="$3" \
        python "$PROFILE_OPS" 2>/dev/null | grep -E '^(IO |MODEL |SUMMARY)')
  echo "${out:-SUMMARY op=$2 rows=$1 cols=$3 FAILED}" | tee -a "$LOG"
}
has() { [[ " $SECTIONS " == *" $1 "* ]]; }

# A) size-sweep fit on a clean 1R+1W op -> kernel = fill + bytes/BW. Linearity + BW
#    already confirmed (R^2~1, ~102 GB/s); 3 neg sizes pin the slope (arithmetic-free is
#    covered by B's exp vs neg@4096). (Per run ~7 min wall: 60s sync hang + forced
#    recompile dominate, NOT tensor size -> the lever is fewer run() calls.)
has A && { echo "## A size-sweep fit (neg)" | tee -a "$LOG"
  for n in 512 2048 8192; do run neg "$n"; done; }
# B) arithmetic-free: neg==gelu already shown equal in A, so ONE strongest check is
#    enough -- exp (transcendental, priciest activation) must still match the neg@4096
#    1R+1W baseline that D produces at the same size.
has B && { echo "## B arithmetic-free" | tee -a "$LOG"
  run exp 4096; }
# C) traffic / stream count at ONE size vs the neg@4096 (2-pass) baseline from D:
#    mul=3-pass, add3=4, add4=5. Kernel should scale with PASSES (bytes/BW), NOT operand
#    count -- rung 7 already falsified a per-stream law; this re-checks on kernel time and
#    watches the unexplained mul/add ~15-25% anomaly. One size shows the progression.
has C && { echo "## C traffic / stream count" | tee -a "$LOG"
  for op in mul add3 add4; do run "$op" 4096; done; }
# D) bandwidth re-anchor (KEY: bw_read/bw_write/the R+W penalty are still MIN-based and
#    flagged). read=1R, write=1W, neg=1R+1W balanced, at 2 sizes so the slope (= BW) is
#    clean and same-size read/write/balanced give the R+W penalty on GOLDEN kernel time.
has D && { echo "## D bandwidth read/write/balanced" | tee -a "$LOG"
  for op in read write neg; do
    for n in 1024 4096; do run "$op" "$n"; done; done; }
# E) broadcast caching is STRUCTURAL (size-independent) -> ONE size each. Compare
#    row-vec bcast + col-vec bcastcol (b[R,1] -> stick-padded device size) against full
#    add: the cached operand must show "hbm counted: 0 B" and land on the 2-pass BW
#    (~102). mul/mulbcast dropped (same pattern as add/bcast; mul anomaly is section C).
has E && { echo "## E broadcast (row + col)" | tee -a "$LOG"
  for op in bcast bcastcol add; do run "$op" 2048; done; }
# F) reductions: sumrow at 2 sizes -> read-dominated BW slope; the OTHER reduced axis
#    (sumcol = dim 0), arith-free (amax/mean), and the ring combine (sumall -> scalar)
#    at one size each. Structural checks need only one size; only the BW slope needs two.
has F && { echo "## F reductions" | tee -a "$LOG"
  for n in 1024 4096; do run sumrow "$n"; done
  for op in sumcol amax mean sumall; do run "$op" 2048; done; }
# G) broadcast-operand RELOAD probe -- PURE POINTWISE (one kernel, no materialized
#    intermediate; reductions are unusable here: Spyre spills the pre-reduction op to
#    HBM, an 8MB round-trip that drowns the signal). Fix cores=32 and SHRINK ROWS so the
#    per-core reload cores*C of b[1,C] is a visible fraction of the 2*R*C output traffic.
#    gap = kernel(bcast) - kernel(bcastcol) isolates b's residency (both are 2-pass; only
#    which axis b spans differs):
#      gap ~CONSTANT across R (= cores*C) -> loaded ONCE PER CORE  (model needs xcores)
#      gap grows ~R          (= R*C)      -> loaded once PER ROW   (no caching)
#      gap ~0                (= C)        -> loaded once GLOBALLY   (count-once is right)
#    add (full 2R+1W = 3-pass) is the reference; whichever broadcast op approaches add has
#    its operand reloaded (and reveals which axis work-division split). Small R (64/128)
#    carry the signal; R=256 mainly confirms the constant-vs-growing trend.
has G && { echo "## G broadcast reload (cores=32, vary ROWS, cols=16384)" | tee -a "$LOG"
  for r in 64 128 256; do
    rung "$r" bcast 16384
    rung "$r" bcastcol 16384
    rung "$r" add 16384
  done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
