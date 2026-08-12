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
# REDUCTION ROWS SWEEP -- fill the large-ROWS gap in the reduction data.
#
# Motivation: the report's §5 large-ROWS read-slowdown (~150 GB/s at ROWS=2048
# falling to ~123 GB/s at ROWS=8192) is currently backed by only two ops
# (`read`, `sumrow`) at two ROWS values (2048, 8192). To characterize the
# falloff and confirm it is op-independent we need:
#   - the OTHER reductions (amax, mean, sumcol, sumall) at large ROWS, and
#   - the intermediate ROWS=4096 and the larger ROWS=16384 points,
# all at a couple of COLS so the input-size dependence is separable from ROWS.
#
#   RR1  read/sumrow/amax/mean/sumall x ROWS{2048,4096,8192,16384} @ COLS 2048
#          -> is the drop op-independent? where is the knee in ROWS (rows/core)?
#   RR2  same ops x COLS{1024,4096} @ ROWS 8192
#          -> confirm the deficit tracks the INPUT read (COLS), not the output.
#   RR3  sumcol at the same ROWS grid @ COLS 2048  (col-reduction control)
#
# cores=32, LX_PLANNING=0 (pure-HBM reductions, no scratchpad). ROWS=16384 @
# cores=32 is 512 rows/core; well within range. Measured via the AIU profiler.
#
#   bash docs/source/user_guide/examples/run_reduction_rows_sweep.sh
# Output: <repo-root>/haoyang_logs/reduction_rows_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/reduction_rows_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-RR1 RR2 RR3}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== reduction rows sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runp() {  # runp <op> <rows> <cols>   reduction, cores=32, LX planning OFF
  echo "-- $1 [$2,$3]" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=0 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 [$2,$3]"
}

# ===== RR1: op-independence + ROWS knee, at a fixed COLS ====================
has RR1 && { echo "## RR1 reductions x ROWS{2048,4096,8192,16384} @ COLS 2048" \
    | tee -a "$LOG"
  for r in 2048 4096 8192 16384; do
    for op in read sumrow amax mean sumall; do runp "$op" "$r" 2048; done
  done; }

# ===== RR2: input-size (COLS) dependence at fixed large ROWS ================
has RR2 && { echo "## RR2 reductions x COLS{1024,4096} @ ROWS 8192" | tee -a "$LOG"
  for c in 1024 4096; do
    for op in read sumrow amax mean sumall; do runp "$op" 8192 "$c"; done
  done; }

# ===== RR3: col-reduction control across the ROWS grid =====================
has RR3 && { echo "## RR3 sumcol x ROWS{2048,4096,8192,16384} @ COLS 2048" \
    | tee -a "$LOG"
  for r in 2048 4096 8192 16384; do runp sumcol "$r" 2048; done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
