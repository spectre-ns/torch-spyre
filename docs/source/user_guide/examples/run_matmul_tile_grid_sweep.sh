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
# MATMUL TILE-GRID SWEEP -- identify the correct 2-D form of the spill term (§11).
#
# The current spill is SEPARABLE and per-edge: |A|*f(M/m) + |B|*f(N/n) with a
# per-edge knee at 448. But on-chip capacity bounds the per-core TILE AS A WHOLE
# (M/m x N/n area, the output accumulator), not each edge independently -- so the
# form is suspect. The two decouple sweeps we have (DC1, DC2) are only 1-D slices
# (each fixes one edge), which CANNOT distinguish "per-edge" from "area" from
# "elongation-dependent". Early evidence: at equal area 262144, M/m x N/n =
# 1024x256 costs +136 us but 512x512 only +90 us -> NOT area-only, the shape
# matters. To fit the true 2-D form we need a full grid over (M/m, N/n).
#
# Method: FORCED split 4x8x1 (cores=32), K=2048 fixed, vary M and N so
#   M/m = M/4 in {128,256,512,1024,2048}   and   N/n = N/8 in {64,128,256,512}.
# The residual (measured - base model WITHOUT spill) then maps the (M/m, N/n)
# plane. MNK kept <= 3.4e10 (>=6.9e10 hangs), so the wide-M x wide-N corner is
# dropped (flagged below), not run.
#
#   TG1  M/m sweep at each N/n column (the full grid, staying under the MNK cap)
#
#   bash docs/source/user_guide/examples/run_matmul_tile_grid_sweep.sh
# Output: <repo-root>/haoyang_logs/matmul_tile_grid_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/matmul_tile_grid_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-TG1 TG2}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== matmul tile-grid sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runmmwd() {  # runmmwd <M> <K> <N> <m> <n> <k>   FORCED split, cores=m*n*k
  echo "-- mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6 (cores=$(($4 * $5 * $6)), " \
    "M/m=$(($1 / $4)), N/n=$(($3 / $5)))" | tee -a "$LOG"
  SENCORES=32 WD_M="$4" WD_N="$5" WD_K="$6" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=mmwd BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6"
}

# ===== TG1: (M/m x N/n) grid at split 4x8x1, K=2048 =========================
# For each N (-> N/n) column, sweep M (-> M/m), skipping M*N > 1.6e7 (MNK>3.4e10).
has TG1 && { echo "## TG1 tile grid: split 4x8x1, K=2048, M/m x N/n plane" | tee -a "$LOG"
  for N in 512 1024 2048 4096; do          # N/n = N/8 = 64 128 256 512
    for M in 512 1024 2048 4096 8192; do   # M/m = M/4 = 128 256 512 1024 2048
      if (( M * N > 16777216 )); then       # MNK > 3.4e10 -> hangs; skip corner
        echo "-- SKIP M=$M N=$N (M*N*K=$((M * N * 2048)) > 3.4e10)" | tee -a "$LOG"
        continue
      fi
      runmmwd "$M" 2048 "$N" 4 8 1
    done
  done; }

# ===== TG2: SPLIT sweep at a fixed shape -> the extreme-split residual (§12a) =
# Fix M=4096 N=2048 K=2048 (cores=32) and walk the m x n factoring from balanced
# to extreme. This isolates the operand re-read / fanout effect at a FIXED total
# work, so the residual maps directly to (M/m, N/n) and the fanout (m, n). Covers
# the asymmetry (huge N/n breaks; huge M/m is fine) that the log-spill misses.
has TG2 && { echo "## TG2 split sweep: M=4096 N=2048 K=2048, m x n from balanced to extreme" \
    | tee -a "$LOG"
  #        M    K    N   m  n  k     (cores=32 throughout; M/m , N/n vary)
  runmmwd 4096 2048 2048  4  8  1   # M/m=1024 N/n=256   balanced (ref)
  runmmwd 4096 2048 2048  8  4  1   # M/m=512  N/n=512
  runmmwd 4096 2048 2048  2 16  1   # M/m=2048 N/n=128
  runmmwd 4096 2048 2048 16  2  1   # M/m=256  N/n=1024  wide-N (breaks now)
  runmmwd 4096 2048 2048  1 32  1   # M/m=4096 N/n=64    tall-M (fine now)
  runmmwd 4096 2048 2048 32  1  1   # M/m=128  N/n=2048  wide-N extreme
  # a second shape so the fit is not tied to one M
  runmmwd 8192 2048 2048  4  8  1   # M/m=2048 N/n=256   balanced (ref)
  runmmwd 8192 2048 2048  2 16  1   # M/m=4096 N/n=128
  runmmwd 8192 2048 2048 16  2  1; }  # M/m=512 N/n=1024 wide-N

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
