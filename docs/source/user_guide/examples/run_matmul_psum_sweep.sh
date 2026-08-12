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
# MATMUL PSUM SWEEP -- step (4): the K-split combine term (k-1)*out*psum_per_elem
# (currently psum_per_elem_ns=0.14, model 3-10x over). ONLY meaningful once
# compute (step 2) + split (step 3) are pinned, so run this LAST.
#
# Isolation: HOLD (m,n) fixed so the per-core output tile M/m x N/n is CONSTANT,
# and vary WD_K only. Then cores = m*n*k grows with k, compute = MNK/cores drops
# (modeled), the counted HBM bytes are ~constant (operands read once, one output
# write), and the ONLY new cost is the psum ring across k partials. So
#   residual(k) = kernel(k) - compute_model(k) - hbm_model   ->  psum(k),
# and psum(k) should be ~ (k-1) * out_elems * psum_per_elem. Because (m,n) is
# fixed (NOT m*n=const), out-tile per core doesn't move -> no re-read/split
# confound, unlike run_matmul_validate M3.
#
#   PS1  k at fixed (m,n): m2n2 k{1,2,4,8} (cores 4..32); m2n4 k{1,2,4} (8..32).
#   PS2  out-size (M*N) dependence at fixed (m,n)=(2,2): M=N in {1024,4096},
#          k{1,2,4} -> psum ~ out_elems, K-independent?
#
# All forced (mmwd), planner-independent; MNK <= 3.4e10 (no hang). Measured via
# the AIU profiler.
#
#   bash docs/source/user_guide/examples/run_matmul_psum_sweep.sh   # PS1 PS2
# Output: <repo-root>/haoyang_logs/matmul_psum_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/matmul_psum_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-PS1 PS2}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== matmul psum sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runmmwd() {  # runmmwd <M> <K> <N> <m> <n> <k>   FORCED split, cores=m*n*k
  echo "-- mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6 (cores=$(($4 * $5 * $6)), " \
    "out=$(($1 * $3)) elems, K/k=$(($2 / $6)))" | tee -a "$LOG"
  SENCORES=32 WD_M="$4" WD_N="$5" WD_K="$6" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=mmwd BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6"
}

# ============ PS1: k-sweep at FIXED (m,n) -- isolate the psum ring ==========
has PS1 && { echo "## PS1 M=N=2048 K=2048: m2n2 k{1,2,4,8} + m2n4 k{1,2,4}" \
    | tee -a "$LOG"
  echo "# PS1a (m,n)=(2,2), k in {1,2,4,8}  (cores 4,8,16,32)" | tee -a "$LOG"
  runmmwd 2048 2048 2048 2 2 1
  runmmwd 2048 2048 2048 2 2 2
  runmmwd 2048 2048 2048 2 2 4
  runmmwd 2048 2048 2048 2 2 8
  echo "# PS1b (m,n)=(2,4), k in {1,2,4}  (cores 8,16,32)" | tee -a "$LOG"
  runmmwd 2048 2048 2048 2 4 1
  runmmwd 2048 2048 2048 2 4 2
  runmmwd 2048 2048 2048 2 4 4; }

# ============ PS2: out-size (M*N) dependence at fixed (m,n)=(2,2) ===========
has PS2 && { echo "## PS2 (m,n)=(2,2) K=2048, k{1,2,4}: M=N in {1024,4096}" \
    | tee -a "$LOG"
  echo "# PS2a M=N=1024 (out 1.0M elems)" | tee -a "$LOG"
  runmmwd 1024 2048 1024 2 2 1
  runmmwd 1024 2048 1024 2 2 2
  runmmwd 1024 2048 1024 2 2 4
  echo "# PS2b M=N=4096 (out 16.8M elems -- 16x PS2a)" | tee -a "$LOG"
  runmmwd 4096 2048 4096 2 2 1
  runmmwd 4096 2048 4096 2 2 2
  runmmwd 4096 2048 4096 2 2 4; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
