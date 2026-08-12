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
# MATMUL SPLIT SWEEP -- isolate the compute/re-read error at cores=32 (the value
# the HBM two-rate and the additive+datasheet conclusion were calibrated at, so
# it is comparable, unlike the cores=16 compute-isolate sweep). k=1, so compute
# (MNK/cores) and the counted HBM bytes are CONSTANT within a shape -- only the
# split (m,n) changes. Two effects vary with the split and are coupled:
#   B-fanout = m  (B broadcast to m M-cores; re-read past the 8-core cohort)
#   M-tile   = M/m (per-core M rows; the systolic-tile length)
# We DE-COUPLE them by sweeping the split at several M, so a given B-fanout m is
# seen at several M-tiles and vice-versa:
#   SP1  M=4096, m in {1,2,4,8,16,32}  (n=32/m)  -> M-tile 4096..128
#   SP2  M=8192, m in {2,4,8,16}                 -> same m at 2x the M-tile
#   SP3  M=2048, m in {2,4,8}                    -> same m at 0.5x the M-tile
# Then: does kernel/ideal depend on m (fanout) at fixed M-tile, or on M-tile at
# fixed m? And does the SYMMETRIC (fanout/8)^0.75 form (max(m,n)) fit, or is the
# effect ASYMMETRIC (on m only)? -- the (16,2) vs (2,16) pair answers that.
#
# All forced (mmwd) -> planner-independent; measured via the AIU profiler (the
# harness torch.profiler path). N=K=2048; MNK <= 3.4e10 (no hang).
#
#   bash docs/source/user_guide/examples/run_split_sweep.sh        # SP1 SP2 SP3
# Output: <repo-root>/haoyang_logs/split_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/split_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-SP1 SP2 SP3}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== matmul split sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runmmwd() {  # runmmwd <M> <K> <N> <m> <n> <k>   FORCED split, cores=m*n*k
  echo "-- mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6 (cores=$(($4 * $5 * $6)), " \
    "B-fanout=$4, M-tile=$(($1 / $4)))" | tee -a "$LOG"
  SENCORES=32 WD_M="$4" WD_N="$5" WD_K="$6" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=mmwd BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6"
}

# ============ SP1: M=4096, full m-sweep (B-fanout m vs M-tile M/m) ==========
has SP1 && { echo "## SP1 M=4096 N=K=2048 cores=32: m in {1,2,4,8,16,32}" | tee -a "$LOG"
  runmmwd 4096 2048 2048 1 32 1   # M-tile 4096, fanout 1
  runmmwd 4096 2048 2048 2 16 1   # M-tile 2048, fanout 2
  runmmwd 4096 2048 2048 4 8 1    # M-tile 1024, fanout 4
  runmmwd 4096 2048 2048 8 4 1    # M-tile 512,  fanout 8
  runmmwd 4096 2048 2048 16 2 1   # M-tile 256,  fanout 16
  runmmwd 4096 2048 2048 32 1 1; }  # M-tile 128, fanout 32

# ============ SP2: M=8192 -- same m at 2x the M-tile (de-couple) ============
has SP2 && { echo "## SP2 M=8192 N=K=2048 cores=32: m in {2,4,8,16}" | tee -a "$LOG"
  for m in 2 4 8 16; do runmmwd 8192 2048 2048 "$m" $((32 / m)) 1; done; }

# ============ SP3: M=2048 -- same m at 0.5x the M-tile (de-couple) ==========
has SP3 && { echo "## SP3 M=2048 N=K=2048 cores=32: m in {2,4,8}" | tee -a "$LOG"
  for m in 2 4 8; do runmmwd 2048 2048 2048 "$m" $((32 / m)) 1; done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
