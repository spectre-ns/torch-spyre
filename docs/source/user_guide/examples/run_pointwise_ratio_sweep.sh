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
# POINTWISE R:W RATIO SWEEP -- separate BW_PEAK from the turnaround alpha, which a
# BALANCED (1R:1W) op cannot (only 1/BW_PEAK + alpha/2 is constrained). Measures the
# EFFECTIVE BW at several distinct R:W ratios so both fall out of a joint fit.
#
# IMPORTANT (from the design review): this is a TEST OF READ/WRITE ASYMMETRY, not a fit
# that assumes symmetry. Fit the 3-param model  T = R/BW_read + W/BW_write + a*min(R,W)
# and CHECK whether BW_read == BW_write. The unresolved tension (read-only lands ~150,
# balanced ~105, 2:1 prefers ~138-147) is itself a symptom that a single symmetric
# BW_PEAK may be misspecified -- a symmetric 2-param fit can never surface that.
#
# Ratios (op -> R:W):
#   sumrow  x.sum(-1)   ~1:0  read-dominated (reduction -- see caveat)
#   read    x.sum(-1)   ~1:0  read-dominated STREAMING probe (compare vs sumrow: if they
#                             differ, the reduction rate != streaming read rate, so the
#                             150 anchor -- itself reduction-derived -- does NOT transfer)
#   neg     -x          1:1   balanced
#   add     a+b         2:1   read-heavy
#   write   b[1,C]+c[R,1] 0:1 write-dominated -- ONLY at SMALL COLS (operand resident,
#                             BEFORE the large-C operand spill). LOW-confidence anchor
#                             (double-broadcast is an atypical write pattern); flag it.
#
# ROWS is swept {2048, 8192} to confirm effBW has PLATEAUED (large-ROWS asymptote) so a
# per-op fixed startup cost does not masquerade as alpha. Inspect per-op residuals: the
# ops differ in operand COUNT (add has 2 inputs), a confound with the R:W ratio.
#
# cores=32, LX planning OFF (pure HBM). Measured via the AIU profiler.
#
#   bash docs/source/user_guide/examples/run_pointwise_ratio_sweep.sh   # RD WR
# Output: <repo-root>/haoyang_logs/pw_ratio_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/pw_ratio_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-RD WR}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== pointwise ratio sweep $(date) ====" | tee "$LOG"
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

# ===== RD: read-dominated + balanced + 2:1 (anchors BW_read, alpha) ==========
# ROWS {2048,8192} x COLS {1024,2048,4096}; sumrow vs read cross-checks reduction-vs-
# streaming; neg/add give the 1:1 and 2:1 ratios at matched sizes.
has RD && { echo "## RD ratio {sumrow,read,neg,add} x ROWS{2048,8192} x COLS{1024,2048,4096}" \
    | tee -a "$LOG"
  for r in 2048 8192; do
    for c in 1024 2048 4096; do
      for op in sumrow read neg add; do runp "$op" "$r" "$c"; done
    done
  done; }

# ===== WR: write-dominated anchor (BW_write) -- SMALL COLS only (pre-spill) ==
# write = b[1,C]+c[R,1] -> near pure output. Keep COLS small so the broadcast operand
# stays resident (the large-C operand spill is a DIFFERENT, super-linear effect). Sweep
# ROWS to grow the output without growing the operand.
has WR && { echo "## WR write-only anchor: write x ROWS{2048,8192} x COLS{1024,2048}" \
    | tee -a "$LOG"
  for r in 2048 8192; do
    for c in 1024 2048; do runp write "$r" "$c"; done
  done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
