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
# MATMUL COMPUTE SWEEP -- step (2): pin the compute term (mac_peak_per_core_ns,
# currently 1536) with compute DOMINANT and the accurate two-rate HBM subtracted.
#
# Why LOW cores (not big K): compute ~ MNK/cores, reads ~ (MK+KN); increasing K
# scales BOTH, so the compute FRACTION is flat in K. Compute only dominates at
# LOW core counts (compute ~ 1/cores; HBM bytes are fixed). Our HBM model is
# byte-based (two-rate 143/156, core-count-INDEPENDENT), so measured - hbm_model
# is valid at any core count; at cores=4 compute is ~85-90%, so even a 10% HBM
# error moves compute by ~1%. That is exactly step (2)'s "keep compute dominant".
#
# ALWAYS a BALANCED split (both fanouts <= 8) so the split penalty (m>8 / n>16,
# run_split_sweep) stays out and compute is the only moving part.
#
#   MC1  mac_peak: cores=4 (2x2) and 8 (2x4), vary shape/K (compute-dominant).
#          mac_peak_eff = MNK/cores/(kernel_us - hbm_model_us); const & ~1536?
#   MC2  cores-scaling (per-core-peak; resolves the cores=16 anomaly cleanly):
#          fixed M=N=2048 K=4096, cores {4,8,16,32} = 2x2/2x4/4x4/4x8.
#          With hbm subtracted, does compute*cores stay CONSTANT?
#   MC3  peak across shapes at cores=4 (2x2), MNK held ~1.7e10: move the big dim
#          (M / N / K) -> is peak invariant to which dim is large?
#
# All forced (mmwd), planner-independent, k=1; MNK <= 1.7e10 (no hang). Measured
# via the AIU profiler (harness torch.profiler path).
#
#   bash docs/source/user_guide/examples/run_matmul_compute_sweep.sh   # MC1 MC2 MC3
# Output: <repo-root>/haoyang_logs/matmul_compute_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/matmul_compute_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-MC1 MC2 MC3}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== matmul compute sweep $(date) ====" | tee "$LOG"
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

# ============ MC1: mac_peak, compute-dominant (cores 4 and 8) ================
has MC1 && { echo "## MC1 mac_peak compute-dominant: cores 4 (2x2) + 8 (2x4)" \
    | tee -a "$LOG"
  runmmwd 2048 2048 2048 2 2 1   # cores4  MNK=8.6e9   compute ~85%
  runmmwd 2048 4096 2048 2 2 1   # cores4  MNK=1.7e10  compute ~90% (big K)
  runmmwd 1024 4096 1024 2 2 1   # cores4  MNK=4.3e9   compute-heavy small tile
  runmmwd 2048 2048 2048 2 4 1   # cores8  MNK=8.6e9
  runmmwd 2048 4096 2048 2 4 1; }  # cores8  MNK=1.7e10

# ============ MC2: cores-scaling (per-core-peak; cores=16 anomaly) ==========
has MC2 && { echo "## MC2 cores-scaling M=N=2048 K=4096: {4,8,16,32}=2x2/2x4/4x4/4x8" \
    | tee -a "$LOG"
  runmmwd 2048 4096 2048 2 2 1   # cores4
  runmmwd 2048 4096 2048 2 4 1   # cores8
  runmmwd 2048 4096 2048 4 4 1   # cores16
  runmmwd 2048 4096 2048 4 8 1; }  # cores32

# ============ MC3: peak across shapes at cores=4 (MNK ~ 1.7e10) =============
has MC3 && { echo "## MC3 peak across shapes cores=4 (2x2), MNK~1.7e10: big M / N / K" \
    | tee -a "$LOG"
  runmmwd 4096 2048 2048 2 2 1   # big M
  runmmwd 2048 2048 4096 2 2 1   # big N
  runmmwd 2048 4096 2048 2 2 1; }  # big K

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
