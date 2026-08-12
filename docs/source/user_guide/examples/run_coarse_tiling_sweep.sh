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
# COARSE-TILING SWEEP -- characterize the coarse-tiling model over TILE COUNT and
# shape. The golden-profiler spot checks showed softmax_row_tiling +11.8% (fused
# turnaround a bit high) and matmul_row_tiling -9.9% (small per-tile overhead);
# this sweeps BENCH_TILES + shapes so those residuals can be fit as a function of
# tile size instead of a single point. Coarse tiling is ONLY softmax and matmul
# row-tiling for now (per the user); chain / ct-reductions round out the LX and
# c_loop terms.
#
#   CT1  softmax_row_tiling: (ROWS,COLS) x BENCH_TILES {2,4,8,16}.
#   CT2  matmul_row_tiling : (M,K,N)     x BENCH_TILES {2,4,8}.
#   CT3  chain c_loop (TILES {1,2,4,8,16}) + ctsum LX on/off (TILES {1,2,4,8}).
#
# cores=32. Measured via the AIU profiler. TILES=1 is the untiled baseline.
#
#   bash docs/source/user_guide/examples/run_coarse_tiling_sweep.sh   # CT1 CT2 CT3
# Output: <repo-root>/haoyang_logs/coarse_tiling_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/coarse_tiling_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-CT1 CT2 CT3}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== coarse-tiling sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runtile() {  # runtile <op> <rows> <cols> <tiles> [lx=1]  softmax/chain/ct, cores=32
  echo "-- $1 [$2,$3] tiles=$4 lx=${5:-1}" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING="${5:-1}" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" BENCH_TILES="$4" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 [$2,$3] tiles=$4 lx=${5:-1}"
}
runmt() {  # runmt <M> <K> <N> <tiles>   matmul_row_tiling (needs BENCH_N), cores=32
  echo "-- matmul_row_tiling M=$1 K=$2 N=$3 tiles=$4" | tee -a "$LOG"
  SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=matmul_row_tiling BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
    BENCH_TILES="$4" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "matmul_row_tiling M=$1 K=$2 N=$3 tiles=$4"
}

# ============ CT1: softmax_row_tiling, tile-count x shape ===================
# LX-SAFE: keep tile_rows = ROWS/tiles <= 4096 so the per-tile intermediate
# (tile_rows x COLS x 2B) stays <= 33.5MB -- the proven-OK point ([4096,4096] at
# tiles=4). tiles=2 on [16384,4096] (67MB tile) would overflow the LX planner and
# hang, like write[2048,16384], so it is dropped. Still spans tiles {2..16} and
# tile_rows {512..4096} across shapes.
has CT1 && { echo "## CT1 softmax_row_tiling: tile_rows<=4096 (<=33.5MB LX; no hang)" \
    | tee -a "$LOG"
  for t in 4 8 16;   do runtile softmax_row_tiling 16384 4096 "$t"; done  # tr 4096..1024
  for t in 4 8 16;   do runtile softmax_row_tiling 16384 2048 "$t"; done  # tr 4096..1024
  for t in 2 4 8 16; do runtile softmax_row_tiling 8192  2048 "$t"; done  # tr 4096..512
  for t in 2 4 8;    do runtile softmax_row_tiling 4096  4096 "$t"; done; }  # tr 2048..512

# ============ CT2: matmul_row_tiling, tile-count x shape ====================
has CT2 && { echo "## CT2 matmul_row_tiling: 3 shapes x TILES {2,4,8}" | tee -a "$LOG"
  for mkn in "2048 2048 2048" "4096 2048 2048" "2048 2048 4096"; do
    for t in 2 4 8; do runmt $mkn "$t"; done
  done; }

# ============ CT3: chain c_loop + ctsum LX on/off ==========================
has CT3 && { echo "## CT3 chain [2048,4096] TILES{1,2,4,8,16} + ctsum LX on/off" \
    | tee -a "$LOG"
  for t in 1 2 4 8 16; do runtile chain 2048 4096 "$t"; done
  for lx in 0 1; do
    for t in 1 2 4 8; do runtile ctsum 2048 512 "$t" "$lx"; done
  done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
