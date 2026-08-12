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
# COARSE-TILING TERM SWEEP -- isolate the tile-dependent terms. The db_sweep showed
# the three coarse ops move in OPPOSITE directions as tiles increase, which a flat
# model cannot capture and the thin coverage confounds:
#   softmax_row_tiling : kernel DECREASES (smaller tile -> arg0's 2nd read stays in
#                        LX instead of re-reading HBM) -- LX REUSE saving.
#   chain              : kernel flat then CLIFFS (~t=8) -- UNDERFILL derate too weak.
#   matmul_row_tiling  : kernel INCREASES -- per-tile systolic fill / underfill not
#                        applied to the coarse M-tile.
#
# Each section sweeps TILES finely (untiled t=1 as the baseline) so the per-term
# curve can be fit:
#   SM  softmax LX reuse: 2 COLS (tile bytes = ROWS/T x COLS) -> find the tile-bytes
#         KNEE where the 2nd arg0 read stops re-reading HBM (=> LX capacity).
#   CH  chain underfill: a TALL shape ([16384,512], rows/core stays large -> underfill
#         delayed) vs the wide [2048,4096] -> separate underfill from any per-tile cost.
#   MR  matmul_row per-tile: TILES sweep -> per-tile fill / effective rows/core.
#
# cores=32, LX on. Measured via the AIU profiler. TILES=1 = untiled baseline.
#
#   bash docs/source/user_guide/examples/run_coarse_terms_sweep.sh   # SM CH MR
# Output: <repo-root>/haoyang_logs/coarse_terms_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/coarse_terms_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-SM CH MR}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== coarse-tiling term sweep $(date) ====" | tee -a "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runtile() {  # runtile <op> <rows> <cols> <tiles>   softmax/chain, cores=32, LX on
  echo "-- $1 [$2,$3] tiles=$4" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=1 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" BENCH_TILES="$4" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "$1 [$2,$3] tiles=$4"
}
runmt() {  # runmt <M> <K> <N> <tiles>   matmul_row_tiling, cores=32
  echo "-- matmul_row_tiling M=$1 K=$2 N=$3 tiles=$4" | tee -a "$LOG"
  SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=matmul_row_tiling BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
    BENCH_TILES="$4" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "matmul_row_tiling M=$1 K=$2 N=$3 tiles=$4"
}

# ===== SM: softmax LX reuse -- tile-bytes knee at 2 COLS ====================
has SM && { echo "## SM softmax_row_tiling: TILES{1..32} x COLS{4096,2048} (LX-reuse knee)" \
    | tee -a "$LOG"
  for t in 1 2 4 8 16 32; do runtile softmax_row_tiling 16384 4096 "$t"; done
  for t in 1 2 4 8 16 32; do runtile softmax_row_tiling 16384 2048 "$t"; done; }

# ===== CH: chain underfill -- tall (underfill delayed) vs wide ==============
has CH && { echo "## CH chain: TILES{1..64} tall [16384,512] vs wide [2048,4096]" \
    | tee -a "$LOG"
  for t in 1 2 4 8 16 32 64; do runtile chain 16384 512 "$t"; done
  for t in 1 2 4 8 16 32;    do runtile chain 2048 4096 "$t"; done; }

# ===== MR: matmul_row_tiling per-tile -- TILES sweep at 2 shapes ============
has MR && { echo "## MR matmul_row_tiling: TILES{1..16} [4096,2048]x2048 + [2048,2048]x2048" \
    | tee -a "$LOG"
  for t in 1 2 4 8 16; do runmt 4096 2048 2048 "$t"; done
  for t in 1 2 4 8;    do runmt 2048 2048 2048 "$t"; done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
