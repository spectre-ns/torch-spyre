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
# SOFTMAX COARSE-TILING TERM SWEEP -- decouple tile-COUNT (L) from ROWS/tile for
# softmax_row_tiling (torch.softmax(x,dim=-1) on [ROWS,COLS], row-tiled into L
# tiles; a fused kernel). The db_sweep trend (kernel decreases as L increases) is
# CONFOUNDED there: both shapes used ROWS=16384, so L and ROWS/tile were locked,
# and per-tile-BYTES was ruled out (the two COLS shapes agree at matched L, not at
# matched bytes). This sweep breaks the lock -- and holds COLS FIXED so the softmax
# op (its reduction-axis length) does NOT change (a COLS sweep would change the op).
#
#   GR  ROWS x TILES grid at COLS=2048 (spill-free: max tile 16.8MB):
#         ROWS in {2048,4096,8192,16384} x TILES in {4,8,16,32}.
#         - fix TILES, read across ROWS  -> varies ROWS/tile at CONSTANT L.
#         - fix ROWS/tile (the diagonal), read across TILES -> varies L at const ROWS/tile.
#       Compare the NORMALIZED re-read fraction (÷arg0). Whichever axis it collapses
#       onto is the driver (L=pipeline overlap vs ROWS/tile).
#   VAR variance: repeat ONE point 5x -> is the ~19% kernel swing real or noise?
#   SP  LX-spill knee: COLS=4096, TILES=2, ROWS in {6144..16384} -> per-tile
#         intermediate 25..67MB, crossing the ~33-67MB knee where sub/exp spill to
#         HBM (the categorical [16384,4096] t=2 jump) -> model the spill separately.
#
# cores=32, LX on. Measured via the AIU profiler. SPYRE_DUMP_COST so R/W are logged.
#
#   bash docs/source/user_guide/examples/run_softmax_terms_sweep.sh   # GR VAR SP
# Output: <repo-root>/haoyang_logs/softmax_terms_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/softmax_terms_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-GR VAR SP}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== softmax term sweep $(date) ====" | tee -a "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runsm() {  # runsm <rows> <cols> <tiles>   softmax_row_tiling, cores=32, LX on
  echo "-- softmax [$1,$2] tiles=$3 (ROWS/tile=$(($1 / $3)))" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=1 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=softmax_row_tiling BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_TILES="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "softmax [$1,$2] tiles=$3"
}

# ===== GR: ROWS x TILES grid at COLS=2048 (decouple L from ROWS/tile) =======
has GR && { echo "## GR softmax COLS=2048: ROWS{2048,4096,8192,16384} x TILES{4,8,16,32}" \
    | tee -a "$LOG"
  for R in 2048 4096 8192 16384; do
    for T in 4 8 16 32; do runsm "$R" 2048 "$T"; done
  done; }

# ===== VAR: variance -- same point 5x (is the swing real?) ==================
has VAR && { echo "## VAR softmax [8192,2048] tiles=8 x5 (variance)" | tee -a "$LOG"
  for i in 1 2 3 4 5; do runsm 8192 2048 8; done; }

# ===== SP: LX-spill knee -- COLS=4096, tiles=2, vary ROWS/tile 3072..8192 ===
has SP && { echo "## SP softmax COLS=4096 tiles=2: ROWS{6144,8192,10240,12288,16384}" \
    | tee -a "$LOG"
  for R in 6144 8192 10240 12288 16384; do runsm "$R" 4096 2; done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
