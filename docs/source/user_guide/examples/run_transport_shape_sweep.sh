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
# TRANSPORT SHAPE SWEEP -- resolve the two size-dependent copies (§6).
#
# Motivation: transpose runs at a flat 116 GB/s across all shapes (well modeled),
# but cat0 (concat on the 64-elem block axis) and transpose_outer (3-D outer
# swap) have effective bandwidths that FALL with operand size (cat0 96->59,
# transpose_outer 100->89) and were each measured at only 2-3 near-square shapes.
# To decide whether a size-dependent rate is needed (vs the current fixed 60 /
# default copy model) we need each swept over BOTH size and aspect ratio.
#
#   TS1  cat0 x size @ square:        R=C in {512,1024,2048,4096,8192}
#   TS2  cat0 x aspect @ fixed bytes: (R,C) in {(512,8192),(1024,4096),
#          (2048,2048),(4096,1024),(8192,512)}  (all ~= 16 M elems)
#   TS3  transpose_outer x size @ square: R=C in {1024,2048,4096,8192}
#   TS4  transpose_outer x aspect @ fixed bytes: same aspect grid as TS2
#   TS5  transpose + cat1 controls x the same aspect grid (confirm flat)
#
# cores=32, LX_PLANNING=0 (pure-HBM copies). Measured via the AIU profiler.
#
#   bash docs/source/user_guide/examples/run_transport_shape_sweep.sh
# Output: <repo-root>/haoyang_logs/transport_shape_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/transport_shape_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-TS1 TS2 TS3 TS4 TS5}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== transport shape sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runp() {  # runp <op> <rows> <cols>   copy/transport, cores=32, LX planning OFF
  echo "-- $1 [$2,$3]" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=0 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 [$2,$3]"
}

# aspect grid at a fixed ~16 M-element footprint (R*C constant)
ASPECT=("512 8192" "1024 4096" "2048 2048" "4096 1024" "8192 512")

has TS1 && { echo "## TS1 cat0 x size (square)" | tee -a "$LOG"
  for s in 512 1024 2048 4096 8192; do runp cat0 "$s" "$s"; done; }

has TS2 && { echo "## TS2 cat0 x aspect (fixed bytes)" | tee -a "$LOG"
  for rc in "${ASPECT[@]}"; do runp cat0 $rc; done; }

has TS3 && { echo "## TS3 transpose_outer x size (square)" | tee -a "$LOG"
  for s in 1024 2048 4096 8192; do runp transpose_outer "$s" "$s"; done; }

has TS4 && { echo "## TS4 transpose_outer x aspect (fixed bytes)" | tee -a "$LOG"
  for rc in "${ASPECT[@]}"; do runp transpose_outer $rc; done; }

has TS5 && { echo "## TS5 transpose + cat1 controls x aspect" | tee -a "$LOG"
  for rc in "${ASPECT[@]}"; do runp transpose $rc; runp cat1 $rc; done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
