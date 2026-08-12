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
# SOFTMAX FLOOR: double-count fix vs compute-bound-by-exp. The fused softmax "floor"
# (per-core tile ~32 rows, no underfill) runs at ~100 GB/s -- the balanced-copy rate --
# which the model reproduces by counting arg0's HBM read ONCE (a fused kernel serves the
# 2nd read on-chip). But ~100 GB/s could instead be a COMPUTE-bound floor set by exp. To
# settle it, isolate exp.
#
# DESIGN-REVIEW FIX: an untiled copy is the WRONG control -- its lack of tiling/pipeline
# overhead confounds "exp" with "tiling". The control must MATCH the tiling and fusion
# depth, differing ONLY in exp. So we compare, at matched [ROWS,COLS,TILES] and by
# WALL-CLOCK time (not effBW, which presupposes the byte-count answer):
#   FL  softmax_row_tiling         : amax, sub, EXP, sum, div  (the real op)
#       softmax_noexp_row_tiling   : amax, sub, MUL, sum, div  (matched control, no exp)
#   If kernel_us(softmax) ~= kernel_us(noexp): exp is free/overlapped -> the floor is
#   BW-bound at the 1-read rate -> DOUBLE-COUNT FIX CONFIRMED. If softmax is slower, the
#   gap IS exp compute. Read the gap's COLS-scaling (2048 vs 4096) to split exp-COMPUTE
#   (fixed fraction of time) from an exp-result SPILL (threshold in COLS).
#   RF  neg/copy untiled at the same shapes -- a raw 1R+1W BW *ceiling* reference only
#       (NOT the exp control; untiled, so confounded -- for context, flag as such).
#
# Shapes at rpc~=32 (the underfill-free floor): ROWS/(32*TILES)=32.
# cores=32, LX on. Measured via the AIU profiler.
#
#   bash docs/source/user_guide/examples/run_softmax_floor_sweep.sh   # FL RF
# Output: <repo-root>/haoyang_logs/softmax_floor_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/softmax_floor_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-FL RF}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== softmax floor sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runtile() {  # runtile <op> <rows> <cols> <tiles>   tiled coarse op, cores=32, LX on
  echo "-- $1 [$2,$3] tiles=$4 (rpc=$(($2 / 32 / $4)))" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=1 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" BENCH_TILES="$4" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 [$2,$3] t=$4"
}
runp() {  # runp <op> <rows> <cols>   untiled pure-HBM reference, cores=32, LX OFF
  echo "-- $1 [$2,$3] (untiled ref)" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=0 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 [$2,$3]"
}

# ===== FL: softmax vs matched no-exp control at the floor (rpc~=32) ==========
has FL && { echo "## FL softmax vs softmax_noexp @ rpc~32: [8192,2048]t8 [16384,2048]t16 [16384,4096]t16" \
    | tee -a "$LOG"
  for shape in "8192 2048 8" "16384 2048 16" "16384 4096 16"; do
    # shellcheck disable=SC2086
    set -- $shape
    runtile softmax_row_tiling "$1" "$2" "$3"
    runtile softmax_noexp_row_tiling "$1" "$2" "$3"
  done; }

# ===== RF: raw 1R+1W BW ceiling reference (untiled -- context only) ==========
has RF && { echo "## RF neg/copy untiled at floor shapes (raw-BW ceiling ref, confounded)" \
    | tee -a "$LOG"
  for shape in "8192 2048" "16384 2048" "16384 4096"; do
    # shellcheck disable=SC2086
    set -- $shape
    runp neg "$1" "$2"; runp copy "$1" "$2"
  done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
