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
# MATMUL COMPUTE ISOLATION -- pin the compute term now that HBM is a validated
# two-rate model (R/143 + W/156 + turnaround, +/-4% on M1). The M2 sweep showed
# compute efficiency depends on the per-core M-TILE (M/m), asymmetrically (heavy
# M-split = short tile = slow; heavy N-split = fine), NOT on max(m,n). This sweep
# isolates pt_eff'(M-tile) with the two confounders removed:
#   - RE-READ: keep the split at m=4,n=4 (fanout 4 < the 8-core cohort) -> operand
#     broadcast is free, so no un-counted re-read leaks into the residual.
#   - HBM: subtract the (now-accurate) two-rate hbm from the measured kernel ->
#     compute_measured = kernel - hbm; pt_eff'(M/m) = ideal_compute / compute_meas.
#
#   CM1  vary M at fixed split m4 n4 (N=K=2048) -> M-tile = M/4 from 64 to 2048.
#        Fit pt_eff'(M-tile): does efficiency keep improving past 64 rows, and to
#        what saturation point? (current model saturates at 64 -- M2 says higher.)
#   CM2  confirm the driver is M/m (not M or m alone): hold M=4096, n=4, vary m ->
#        same M-tiles as CM1 reached via different M, at different cores. If both
#        collapse onto one pt_eff'(M-tile) curve, M/m is confirmed.
#
# All forced (mmwd) -> planner-independent. New runtime is fast, so the sweep is
# dense. Keep MNK <= ~3.4e10 (>= ~6.9e10 hangs); largest here is 8192x2048x2048.
#
#   bash docs/source/user_guide/examples/run_compute_isolate_sweep.sh   # CM1 CM2
# Output: <repo-root>/haoyang_logs/compute_isolate_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/compute_isolate_$(date +%Y%m%d_%H%M%S).log"
SECTIONS="${SECTIONS:-CM1 CM2}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== matmul compute-isolation sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runmmwd() {  # runmmwd <M> <K> <N> <m> <n> <k>   FORCED split, cores = m*n*k
  echo "-- mmwd M=$1 K=$2 N=$3 split m=$4 n=$5 k=$6 (cores=$(($4 * $5 * $6)), " \
    "M-tile=$(($1 / $4)))" | tee -a "$LOG"
  SENCORES=32 WD_M="$4" WD_N="$5" WD_K="$6" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=mmwd BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
    python "$PROFILE_OPS" 2>&1 | _emit "mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6"
}

# ============ CM1: pt_eff'(M-tile) -- vary M at fixed m4 n4 (re-read free) ====
# M-tile = M/4 = 64,128,192,256,384,512,768,1024,1536,2048. N=K=2048, cores=16.
has CM1 && { echo "## CM1 pt_eff'(M-tile): m4 n4 k1, N=K=2048, vary M (M-tile=M/4)" \
  | tee -a "$LOG"
  for M in 256 512 768 1024 1536 2048 3072 4096 6144 8192; do
    runmmwd "$M" 2048 2048 4 4 1
  done; }

# ============ CM2: confirm the driver is M/m (not M or m alone) =============
# Fix M=4096, n=4; vary m -> M-tile 2048/1024/512 at cores 8/16/32. Cross-check
# vs CM1's M-tile 2048 (M=8192), 1024 (M=4096), 512 (M=2048): should MATCH.
has CM2 && { echo "## CM2 confirm M/m: M=4096 N=K=2048 n4, m in {2,4,8}" | tee -a "$LOG"
  for m in 2 4 8; do runmmwd 4096 2048 2048 "$m" 4 1; done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
