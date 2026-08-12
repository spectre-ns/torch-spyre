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
# TRANSPORT (data-movement) SWEEP -- the 4th op category (after pointwise,
# reduction, matmul). A transpose / cat / restickify moves the SAME bytes as a
# plain copy (neg), but reorganizes the STICK layout, so its access pattern is
# scattered. The cost model sees a restickify as a Pointwise copy and predicts
# (R+W)/BW_PEAK + turnaround -- this sweep measures whether the scattered stick
# access makes it SLOWER than that (an access-pattern penalty), by comparing the
# effective BW (= io_hbm_bytes / kernel_us, in the SUMMARY) of each transport op
# against `neg` (a contiguous 1R+1W copy) at matched shapes.
#
# Transport ops (BENCH_OP): transpose ([R,C]->[C,R], stick dim MOVES -- heavy),
# transpose_outer (3D [R,8,C] outer swap, stick PRESERVED -- light), cat0/cat1
# (concat 2 copies along rows / cols). Unsupported ops self-report SUMMARY ...
# FAILED, so a fallback does not abort the sweep.
#
#   T1  op-variety at fixed shapes -> which access patterns are expensive?
#   T2  transpose vs neg across aspect ratios -> does cost depend on R:C?
#
#   bash docs/source/user_guide/examples/run_transport_sweep.sh        # T1 T2
#   SECTIONS=T1 bash docs/source/user_guide/examples/run_transport_sweep.sh
# Output: <repo-root>/haoyang_logs/transport_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/transport_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-T1 T2}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1   # force recompile so the dumps fire

echo "==== transport (data-movement) sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {  # stdin: profile_ops output; keep splits/IO/MODEL/SUMMARY; FAILED if empty
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runt() {  # runt <op> <rows> <cols>   transport/pointwise copy, cores=32, LX OFF
  echo "-- $1 [$2,$3]" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=0 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 [$2,$3]"
}

# ============ T1: op variety at fixed shapes (access-pattern cost) ==========
# neg = contiguous-copy baseline; the rest are transport. Same shape -> compare
# effective BW. (transpose_outer is 3D [R,8,C], so 8x the bytes -> use bw, not us.)
has T1 && { echo "## T1 op variety (neg vs transport) at [2048,2048] and [4096,4096]" \
  | tee -a "$LOG"
  for sz in "2048 2048" "4096 4096"; do
    for op in neg transpose transpose_outer cat0 cat1; do runt "$op" $sz; done
  done; }

# ============ T2: transpose vs neg across aspect ratios =====================
# Does the stick-moving transpose cost depend on R:C (square / tall / wide)? The
# stick dim is the LAST (cols); transpose moves it to rows -> the reshuffle volume
# may scale with min(R,C) sticks or the aspect ratio. neg at each shape = baseline.
has T2 && { echo "## T2 transpose vs neg, aspect-ratio sweep (bytes ~ const)" \
  | tee -a "$LOG"
  for shape in "2048 2048" "4096 1024" "1024 4096" "8192 512" "512 8192"; do
    runt neg $shape; runt transpose $shape
  done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
