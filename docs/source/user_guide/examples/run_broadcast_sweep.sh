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
# BROADCAST SWEEP -- pin the broadcast effective-BW rate and CONFIRM the write spill.
# The existing broadcast data is thin (each of bcast/bcastcol/mulbcast has ONE clean size,
# ROWS=2048 COLS=16384), so the ~118 GB/s "broadcast rate" and the "write is super-linear
# in C because the row operand b[1,C] spills" claims both need real data.
#
#   BR  broadcast RATE across size: {copy,bcast,bcastcol,mulbcast} x COLS{1024..16384} at
#         ROWS=2048 (64 rows/core -> well-filled). Pins effBW vs size; ALSO reveals whether
#         the fast broadcast ops themselves spill at large C (effBW should drop if they do).
#   WS  write SPILL confirm: write (b[1,C] + c[R,1], both operands broadcast) over a
#         ROWS x COLS grid. If the cost per output byte rises with COLS (the row operand
#         b[1,C] grows past on-chip capacity -> re-streamed per row) but stays FLAT in ROWS
#         (c[R,1] does not), the spilling operand is b[1,C] (the C-sized one), confirming the
#         hypothesis. A ROWS-only effect would instead implicate c[R,1].
#
# cores=32, LX planning OFF (pure HBM). Measured via the AIU profiler. COLS are multiples of
# 64 (no stick-padding tail). Output is [ROWS,COLS] fp16 (write [8192,16384] = 256 MB).
#
#   bash docs/source/user_guide/examples/run_broadcast_sweep.sh   # BR WS
# Output: <repo-root>/haoyang_logs/broadcast_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/broadcast_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-BR WS}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== broadcast sweep $(date) ====" | tee "$LOG"
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

# ===== BR: broadcast rate vs size (ROWS=2048, well-filled) ==================
has BR && { echo "## BR broadcast rate: {copy,bcast,bcastcol,mulbcast} x COLS{1024..16384} @ ROWS=2048" \
    | tee -a "$LOG"
  for c in 1024 2048 4096 8192 16384; do
    for op in copy bcast bcastcol mulbcast; do runp "$op" 2048 "$c"; done
  done; }

# ===== WS: write spill -- ROWS x COLS grid (is it C-driven or R-driven?) =====
has WS && { echo "## WS write spill grid: ROWS{512,2048,8192} x COLS{1024,4096,16384}" \
    | tee -a "$LOG"
  for r in 512 2048 8192; do
    for c in 1024 4096 16384; do runp write "$r" "$c"; done
  done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
