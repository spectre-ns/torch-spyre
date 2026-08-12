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
# MATMUL RE-READ SWEEP -- step (3): pin the operand re-read (the split penalty)
# on the CORRECTED compute+HBM baseline (peak~1150, overlap gamma~0.46). The DB
# showed the penalty is (a) PER-CORE TILE SPILL (re-read grows with M/m for A,
# N/n for B; ~0 below ~256) and (b) EXTREME FANOUT (m>8 / n>16). This sweep
# isolates BOTH cleanly by holding the *other* operand's total size FIXED, so a
# re-read multiplier reads off directly:
#
#   RA  A tile-spill: fixed 4x8 (fanout 4,8 minimal), K=N=2048, sweep M ->
#         M/m in {128,256,384,512,768,1024,1536,2048}. |A| grows; R_A(M/m).
#   RB  B tile-spill: fixed 4x8, K=M=2048, sweep N ->
#         N/n in {64,128,192,256,384,512,768,1024}. |B| grows; R_B(N/n).
#   FB  B-fanout: per-core tile FIXED small 128x128 (no spill), |B| FIXED (n=1,
#         N=128, K=2048 -> |B|=0.5MB), vary B-fanout m in {1,2,4,8,16,32}
#         (M=128*m, cores=m). |B| const -> reread_B/|B| = R_B(fanout) directly.
#   FA  A-fanout: mirror -- tile 128x128, |A| FIXED (m=1, M=128 -> |A|=0.5MB),
#         vary A-fanout n in {1,2,4,8,16,32} (N=128*n, cores=n). R_A(fanout).
#
# Why this decouples what the DB could not: in FB/FA the FANNED operand's size is
# held constant AND the per-core tile is below the spill knee, so only fanout
# moves; in RA/RB fanout is fixed minimal so only the tile moves. All forced
# (mmwd), k=1, MNK <= 3.44e10 (no hang). Measured via the AIU profiler.
#
#   bash docs/source/user_guide/examples/run_reread_sweep.sh   # RA RB FB FA
# Output: <repo-root>/haoyang_logs/reread_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/reread_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-RA RB FB FA}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== matmul re-read sweep $(date) ====" | tee -a "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runmmwd() {  # runmmwd <M> <K> <N> <m> <n> <k>   FORCED split, cores=m*n*k
  echo "-- mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6 (cores=$(($4 * $5 * $6)), " \
    "M/m=$(($1 / $4)), N/n=$(($3 / $5)), A-fanout=$5, B-fanout=$4)" | tee -a "$LOG"
  SENCORES=32 WD_M="$4" WD_N="$5" WD_K="$6" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=mmwd BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6"
}

# ===== RA: A tile-spill -- fixed 4x8, K=N=2048, sweep M (M/m 128..2048) =====
has RA && { echo "## RA A-spill: 4x8 K=N=2048, M/m in {128..2048} (|A| grows)" \
    | tee -a "$LOG"
  for M in 512 1024 1536 2048 3072 4096 6144 8192; do runmmwd "$M" 2048 2048 4 8 1; done; }

# ===== RB: B tile-spill -- fixed 4x8, K=M=2048, sweep N (N/n 64..1024) ======
has RB && { echo "## RB B-spill: 4x8 K=M=2048, N/n in {64..1024} (|B| grows)" \
    | tee -a "$LOG"
  for N in 512 1024 1536 2048 3072 4096 6144 8192; do runmmwd 2048 2048 "$N" 4 8 1; done; }

# ===== FB: B-fanout at FIXED tile 128x128, |B| fixed (n=1,N=128,K=2048) =====
has FB && { echo "## FB B-fanout: tile 128x128, |B|=0.5MB fixed, m in {1..32}" \
    | tee -a "$LOG"
  for m in 1 2 4 8 16 32; do runmmwd $((128 * m)) 2048 128 "$m" 1 1; done; }

# ===== FA: A-fanout at FIXED tile 128x128, |A| fixed (m=1,M=128,K=2048) =====
has FA && { echo "## FA A-fanout: tile 128x128, |A|=0.5MB fixed, n in {1..32}" \
    | tee -a "$LOG"
  for n in 1 2 4 8 16 32; do runmmwd 128 2048 $((128 * n)) 1 "$n" 1; done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
