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
# HBM-BOUND OPS SWEEP -- per-op-type effective bandwidth on the CURRENT runtime.
# The plan is to record a DIFFERENT effective HBM bandwidth / turnaround per op
# family (pointwise / reduction / broadcast; transport lives in its own sweep),
# so the global model can key BW on the op category. Every run reports bw_gbps =
# io_hbm_bytes / kernel_us in the SUMMARY -- this sweep gives broad, clean
# coverage of that number across arity, reduction axis, and broadcast reuse.
#
#   HB1  pointwise arity: 1R/2R/3R/4R + arithmetic-heavy (gelu/exp) x sizes.
#          -> BW vs stream count (turnaround a*min(R,W)) and vs compute content.
#   HB2  reduction: row/col/all reductions x sizes -> reduction BW + axis effect.
#   HB3  broadcast one-time-load: bcast/bcastcol/write, SHRINK rows -> confirms
#          the broadcast operand is loaded ONCE (BW rises as rows shrink).
#
# cores=32, LX_PLANNING=0 for all (these are pure-HBM ops -- no scratchpad, so LX
# planning is irrelevant to their bandwidth AND it is what hangs the planner on a
# large broadcast output like write[2048,16384]). Measured via the AIU profiler.
#
#   bash docs/source/user_guide/examples/run_hbm_ops_sweep.sh   # HB1 HB2 HB3
# Output: <repo-root>/haoyang_logs/hbm_ops_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/hbm_ops_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-HB1 HB2 HB3}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== hbm-bound ops sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runp() {  # runp <op> <rows> <cols>   HBM-bound op, cores=32, LX planning OFF
  echo "-- $1 [$2,$3]" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=0 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 [$2,$3]"
}

# ============ HB1: pointwise arity x size (stream count -> turnaround) =======
has HB1 && { echo "## HB1 pointwise: {neg,copy,gelu,exp,mul,add,add3,add4} x COLS" \
    | tee -a "$LOG"
  for c in 1024 4096 16384; do
    for op in neg copy gelu exp mul add add3 add4; do runp "$op" 2048 "$c"; done
  done; }

# ============ HB2: reduction axis x size ===================================
has HB2 && { echo "## HB2 reduction: {sumrow,sumcol,amax,mean,sumall,read} x COLS" \
    | tee -a "$LOG"
  for c in 2048 8192; do
    for op in sumrow sumcol amax mean sumall read; do runp "$op" 2048 "$c"; done
  done; }

# ============ HB3: broadcast one-time-load (shrink rows) ====================
has HB3 && { echo "## HB3 broadcast: {bcast,mulbcast,bcastcol,write} x ROWS @ C=16384" \
    | tee -a "$LOG"
  for r in 64 256 2048; do
    for op in bcast mulbcast bcastcol write; do runp "$op" "$r" 16384; done
  done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
