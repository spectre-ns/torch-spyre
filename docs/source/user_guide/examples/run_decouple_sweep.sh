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
# MATMUL DECOUPLE SWEEP -- separate PER-CORE-TILE-SIZE from FANOUT.
#
# The split sweep (run_split_sweep.sh) found a 1.5-1.9x penalty at extreme
# splits, but could not say whether the cause is large fanout or a starved
# per-core output tile: with m*n=32 fixed, n=32 means BOTH A-fanout=32 AND
# N/n=64 (1 stick wide). They onset together and are confounded.
#
# Here we HOLD the split at 4x8 (a "good"-zone split: B-fanout=4<=8,
# A-fanout=8<=8, both safe) and vary the MATRIX dimension. So the per-core
# tile (M/m, N/n) changes across the same range that was "bad" in the split
# sweep, while BOTH fanouts stay in the safe zone:
#   DC1  M=2048, n=8 fixed, vary N in {512,1024,2048,4096}
#          -> N/n in {64,128,256,512}   (A-fanout=8 fixed, B-fanout=4 fixed)
#   DC2  N=2048, m=4 fixed, vary M in {512,1024,2048,4096,8192}
#          -> M/m in {128,256,512,1024,2048} (fanouts fixed)
# Verdict:
#   * If a penalty appears at small N/n (DC1) or small M/m (DC2) with fanout
#     held safe -> the split-sweep penalty was PER-CORE-TILE underutilization.
#   * If DC1/DC2 stay flat (~ the split sweep's 4x8 accuracy) down to the
#     smallest tile -> the penalty was FANOUT (m>8 / n>16), and n=32's 1.91
#     was A-fanout, not the 1-stick width.
# Either way the two effects are separated and a split-penalty derate can be
# committed on the right variable.
#
# The tile=64 (N/n=64) point is the smallest stick-aligned width; N=256 would
# be half a stick and is skipped. All forced (mmwd), planner-independent,
# cores=32, k=1; measured via the AIU profiler. MNK <= 3.44e10 (M=8192 case,
# already run OK in SP2; hang threshold is ~6.9e10 for a 4096^3).
#
#   bash docs/source/user_guide/examples/run_decouple_sweep.sh   # DC1 DC2
# Output: <repo-root>/haoyang_logs/decouple_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/decouple_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-DC1 DC2}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== matmul decouple sweep $(date) ====" | tee "$LOG"
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
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6"
}

# ===== DC1: vary N (per-core width N/n) at FIXED split 4x8, M=2048 ==========
# fanouts held safe (A=8, B=4); N/n sweeps 64..512 -- the range that was bad
# in the split sweep only when it coincided with big fanout.
has DC1 && { echo "## DC1 M=2048 K=2048 split 4x8: N in {512,1024,2048,4096} (N/n 64..512)" \
    | tee -a "$LOG"
  runmmwd 2048 2048 512  4 8 1   # N/n=64  (1 stick)
  runmmwd 2048 2048 1024 4 8 1   # N/n=128
  runmmwd 2048 2048 2048 4 8 1   # N/n=256 (== split-sweep 4x8 baseline)
  runmmwd 2048 2048 4096 4 8 1; }  # N/n=512

# ===== DC2: vary M (per-core height M/m) at FIXED split 4x8, N=2048 =========
# fanouts held safe; M/m sweeps 128..2048 -- incl. the 128/256 that were bad
# in the split sweep only under big B-fanout (m=16,32).
has DC2 && { echo "## DC2 N=2048 K=2048 split 4x8: M in {512,1024,2048,4096,8192} (M/m 128..2048)" \
    | tee -a "$LOG"
  runmmwd 512  2048 2048 4 8 1   # M/m=128
  runmmwd 1024 2048 2048 4 8 1   # M/m=256
  runmmwd 2048 2048 2048 4 8 1   # M/m=512  (== DC1 N/n=256 baseline)
  runmmwd 4096 2048 2048 4 8 1   # M/m=1024
  runmmwd 8192 2048 2048 4 8 1; }  # M/m=2048

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
