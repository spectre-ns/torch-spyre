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
# BROADCAST SMALL-COLS SWEEP -- is the small-COLS effective-BW rise a real trend?
#
# The broadcast rate sweep showed the three VECTOR-broadcast ops (bcast/bcastcol/
# mulbcast) run above the ~118 GB/s large-COLS plateau at small COLS: ~130/124/132
# at COLS=1024, easing to ~118 by COLS>=4096 (copy, a scalar broadcast, stays flat).
# That is exactly the +9..+13% error at COLS=1024 in the §4 table. Before deciding
# whether to model a C-dependent broadcast rate, extend the sweep BELOW 1024 to see
# whether the rise keeps growing (a real trend) or flattens/reverses (a small-tensor
# artifact to leave alone).
#
#   SC1  {copy,bcast,bcastcol,mulbcast} x COLS{256,512,1024,2048,4096} @ ROWS=2048
#          -> does effBW keep rising below 1024? does copy stay flat?
#   SC2  ROWS control: {bcast,mulbcast} x ROWS{512,2048,8192} @ COLS=512
#          -> is the rise COLS-only, or does it also move with ROWS (small-tensor)?
#
# cores=32, LX_PLANNING=0 (pure-HBM). Measured via the AIU profiler.
#
#   bash docs/source/user_guide/examples/run_broadcast_smallcols_sweep.sh
# Output: <repo-root>/haoyang_logs/broadcast_smallcols_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/broadcast_smallcols_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-SC1 SC2 SC3}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== broadcast small-cols sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runp() {  # runp <op> <rows> <cols>   pure-HBM op, cores=32, LX OFF
  echo "-- $1 [$2,$3]" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=0 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 [$2,$3]"
}

# ===== SC1: small-COLS trend at ROWS=2048 ==================================
has SC1 && { echo "## SC1 broadcast small-COLS: {copy,bcast,bcastcol,mulbcast} x COLS{256..4096} @ ROWS=2048" \
    | tee -a "$LOG"
  for c in 256 512 1024 2048 4096; do
    for op in copy bcast bcastcol mulbcast; do runp "$op" 2048 "$c"; done
  done; }

# ===== SC2: is the rise COLS-only, or ROWS-driven (small tensor)? ==========
has SC2 && { echo "## SC2 ROWS control: {bcast,mulbcast} x ROWS{512,2048,8192} @ COLS=512" \
    | tee -a "$LOG"
  for r in 512 2048 8192; do
    for op in bcast mulbcast; do runp "$op" "$r" 512; done
  done; }

# ===== SC3: re-measure the bad small-ROWS x large-COLS corner ==============
# In the current data ROWS=256, COLS=16384 is -20% for bcast/mulbcast (~95 GB/s) but
# fine for bcastcol -- likely single-run scatter. Re-measure with repeats to confirm
# whether it is a real small-ROWS slowdown at large COLS or just noise.
has SC3 && { echo "## SC3 small-ROWS @ COLS=16384: {bcast,bcastcol,mulbcast} x ROWS{64,128,256,512}, x2" \
    | tee -a "$LOG"
  for rep in 1 2; do
    for r in 64 128 256 512; do
      for op in bcast bcastcol mulbcast; do runp "$op" "$r" 16384; done
    done
  done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
