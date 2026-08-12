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
# COST-MODEL PROFILING DATABASE -- run EVERY sweep back-to-back into ONE unified
# log, then fold it into notes/sweep_records.{json,csv}. One command rebuilds the
# measurement DB so any model change is validated by diffing model vs the recorded
# kernel_us (never re-derived). ~160 runs, ~30-45 min (runs are CPU-time-bound).
#
# Robustness: no set -e (a failing sweep does not abort the rest), and every run
# has a RUN_TIMEOUT (default 180s) guard -- a wedged compile/run is killed and
# logged `... FAILED` instead of stalling the whole chain (that was the write[
# 2048,16384] LX-planner hang). Lower it to fail faster: RUN_TIMEOUT=60.
#
# ONE log: DB_LOG is exported; each child sends its output here and suppresses its
# own per-child log (LOG=/dev/null), so the whole run is a single file that is
# overwritten each time -- easy to forward. Chained in the matmul isolation order
# (HBM -> compute -> split -> psum) + global op coverage:
#   run_hbm_ops / run_transport / run_matmul_validate(M1) / run_matmul_compute /
#   run_split / run_decouple / run_matmul_psum / run_coarse_tiling
# Final: parse_sweep_logs.py folds DB_LOG into the DB.
#
#   bash docs/source/user_guide/examples/run_db_sweep.sh
#   RUN_TIMEOUT=60 bash docs/source/user_guide/examples/run_db_sweep.sh
# Output: ONE haoyang_logs/db_sweep.log + updated notes/sweep_records.{json,csv}.
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
mkdir -p "$ROOT/haoyang_logs"
unset SECTIONS 2>/dev/null || true               # every child runs its full default
export RUN_TIMEOUT="${RUN_TIMEOUT:-180}"         # per-run hang guard (see children)
export DB_LOG="$ROOT/haoyang_logs/db_sweep.log"  # children tee here; per-child logs off

echo "==== run_db_sweep $(date) ===="
echo "unified log: $DB_LOG   (RUN_TIMEOUT=${RUN_TIMEOUT}s per run)"

run_one() {  # run_one <script> [SECTIONS override for this child only]
  local s="$1" sec="${2:-}"
  echo
  echo "################################################################"
  echo "## >>> $s ${sec:+(SECTIONS=$sec)}"
  echo "################################################################"
  if [[ -n "$sec" ]]; then
    SECTIONS="$sec" bash "$SCRIPT_DIR/$s" || echo "## !!! $s exited non-zero (continuing)"
  else
    bash "$SCRIPT_DIR/$s" || echo "## !!! $s exited non-zero (continuing)"
  fi
}

DB_SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)"
{
  echo "==== cost-model database rebuild $(date) ===="
  echo "git: $DB_SHA"
  run_one run_hbm_ops_sweep.sh
  run_one run_transport_sweep.sh
  run_one run_matmul_validate_sweep.sh "M1"   # HBM only; compute/psum have own scripts
  run_one run_matmul_compute_sweep.sh
  run_one run_split_sweep.sh
  run_one run_decouple_sweep.sh
  run_one run_matmul_psum_sweep.sh
  run_one run_coarse_tiling_sweep.sh
  # Softmax coarse-tiling validation (fused-input-once + coarse_underfill re-fit, 2026-07-09):
  # the rpc grid + LX-spill knee, and the cross-COLS control (coarse_terms SM only -- skip the
  # dropped `chain` CH and the deferred matmul_row MR).
  run_one run_softmax_terms_sweep.sh
  run_one run_coarse_terms_sweep.sh "SM"
  # Decoupler sweeps (2026-07-09, design-review-vetted) -- upgrade the HYPOTHESIS terms:
  run_one run_pointwise_ratio_sweep.sh   # BW_peak vs alpha (read/write asymmetry test)
  run_one run_broadcast_sweep.sh         # broadcast rate vs size + write-spill confirm
  run_one run_matmul_gamma_sweep.sh      # peak/gamma off the ridge + BW_r/BW_w rank-2 grid
  run_one run_nonpow2_n_sweep.sh         # per-core-tile stick-padding sawtooth
  run_one run_softmax_floor_sweep.sh     # double-count vs exp (matched no-exp control)
} 2>&1 | tee "$DB_LOG"

echo
echo "==== parsing $DB_LOG into notes/sweep_records.{json,csv} ===="
# Drop retired ops (chain) and stamp THIS run's sha as the authoritative current model,
# so every row from this rebuild is flagged is_current (old mixed-version rows are not).
python "$ROOT/notes/parse_sweep_logs.py" "$DB_LOG" \
  --drop-ops chain --current-sha "$DB_SHA" \
  || echo "## !!! parse_sweep_logs.py failed (run it by hand)"

echo
echo "==== run_db_sweep DONE -- forward $DB_LOG + notes/sweep_records.* ===="
