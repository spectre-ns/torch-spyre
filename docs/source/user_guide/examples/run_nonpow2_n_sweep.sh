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
# MATMUL NON-POW2-N STICK-PADDING SWEEP -- characterize the sawtooth the model misses
# (it currently INVERTS the 6144-vs-8192 ranking). The effect is per-core-tile rounding:
# a core's N-tile (N/n) rounds UP to a multiple of the 64-elem stick, wasting work when
# N/n is not a multiple of 64. The model counts FULL-N device bytes (N itself is stick-
# aligned here) so it never sees the per-core-tile padding.
#
# DESIGN-REVIEW FIXES (the naive "N in {2048..8192} step 1024" was BROKEN -- every N was
# stick-aligned so the sawtooth was invisible, and N=8192 broke the MNK<=3.4e10 cap):
#   - FORCE the split (mmwd 4x8x1) so N is not confounded with a planner split-shape
#     change across the sweep.
#   - Step N so the PER-CORE tile N/n (n=8) sweeps THROUGH a 64-stick boundary: N in
#     {4096..4608} step 64 -> N/8 in {512..576}. N/8=512 and 576 are exact (multiples of
#     64, no waste); 520..568 pad up to 576 (up to ~11% waste) -> one full sawtooth tooth.
#   - MNK max = 2048*2048*4608 = 1.9e10 < 3.4e10 cap. (Full N stays a multiple of 64, so
#     the MODEL's byte count is exact -> any measured bump is the tile padding it omits.)
#
# M=2048 K=2048, cores=32 (4x8x1). Measured via the AIU profiler.
#
#   bash docs/source/user_guide/examples/run_nonpow2_n_sweep.sh   # NP
# Output: <repo-root>/haoyang_logs/nonpow2_n_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/nonpow2_n_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-NP}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== nonpow2-N sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runmmwd() {  # runmmwd <M> <K> <N> <m> <n> <k>   FORCED split, cores=m*n*k
  echo "-- mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6 (N/n=$(($3 / $5)), " \
    "tile_stick_pad=$(( ( ($3 / $5 + 63) / 64 * 64 ) - $3 / $5 )))" | tee -a "$LOG"
  SENCORES=32 WD_M="$4" WD_N="$5" WD_K="$6" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=mmwd BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6"
}

# ===== NP: N stick-granular across the N/8 = 512->576 boundary (one tooth) ===
has NP && { echo "## NP M=2048 K=2048 4x8x1, N{4096..4608 step 64} (per-core tile 512..576)" \
    | tee -a "$LOG"
  for N in 4096 4160 4224 4288 4352 4416 4480 4544 4608; do
    runmmwd 2048 2048 "$N" 4 8 1
  done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
