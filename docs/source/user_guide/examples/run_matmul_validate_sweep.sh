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
# MATMUL COST-MODEL VALIDATION SWEEP. The matmul model is T = compute + hbm
# (+ psum), but compute and hbm are NEVER separable in one matmul (a matmul
# always carries >=20% hbm; a purely compute-bound shape needs huge dims that
# hang). So we CONFIRM the terms in the right ORDER, each against MEASUREMENTS
# (no model-minus-model subtraction):
#
#   M1  HBM, compute-free (<10% compute). Pin hbm = f(R,W) from BYTES alone:
#         write-dominated = thin K (output M*N >> operand reads)
#         read-dominated  = thin M (operand reads >> tiny output)
#       Vary bytes + the R/W mix. NO operand re-read here (thin K/M -> operands
#       are tiny), so this isolates the bandwidth-vs-bytes law cleanly.
#
#   M2  COMPUTE + re-read, compute-heavy (>=80% compute via LOW cores). With hbm
#       pinned from M1, (measured - hbm) should be MNK/cores/peak:
#         - cores-scaling: peak constant in 1/cores?
#         - m<->n swap at fixed cores: compute is split-INVARIANT, so any delta
#           is operand RE-READ (lives here, where operands are big).
#         - shape variation at fixed cores: is peak constant across M/N/K?
#
#   M3  PSUM, only trustworthy once M1+M2 hold. measured(k) - measured(k=1) at
#       fixed shape+cores; plus K- and M*N-dependence (model says M*N only).
#   E2E coarse-tile e2e examples (softmax row, matmul row, matmul K tiling) -- a
#       SMOKE TEST that they RUN + FINISH on the new runtime (compare_with_cpu ->
#       PASSED). Not a measurement; just pass/fail.
#
# Forced (m,n,k) splits via spyre_hint(work_div) -> cores = m*n*k (FINAL). Each
# run logs op_it_space_splits (actual split), per-tensor IO, the model estimate,
# and the measured SUMMARY. SENCORES=32. Keep MNK <= ~3.4e10 (>= ~6.9e10 hangs).
#
#   bash docs/source/user_guide/examples/run_matmul_validate_sweep.sh        # M1 M2 M3
#   SECTIONS=M1 bash docs/source/user_guide/examples/run_matmul_validate_sweep.sh
# Output: <repo-root>/haoyang_logs/matmul_validate_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/matmul_validate_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-M1 M2 M3 E2E}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1   # force recompile so the dumps fire

echo "==== matmul model-validation sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {  # stdin: profile_ops output; keep splits/IO/MODEL/SUMMARY; FAILED if empty
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runmm() {  # runmm <M> <K> <N>   plain matmul, planner-chosen split, all 32 cores
  echo "-- mm M=$1 K=$2 N=$3" | tee -a "$LOG"
  SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=mm BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "mm M=$1 K=$2 N=$3"
}
runmmwd() {  # runmmwd <M> <K> <N> <m> <n> <k>   FORCED split, cores = m*n*k
  echo "-- mmwd M=$1 K=$2 N=$3 split m=$4 n=$5 k=$6 (cores=$(($4 * $5 * $6)))" \
    | tee -a "$LOG"
  SENCORES=32 WD_M="$4" WD_N="$5" WD_K="$6" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=mmwd BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6"
}

# ============== M1: HBM, compute-free (pin hbm = f(R,W) from bytes) ==========
# WRITE-dominated: thin K (K<=64 -> compute <10%; output M*N dominates traffic).
# READ-dominated : thin M (M<=64 -> tiny output; operand reads dominate).
# FORCED split (operands tiny -> no re-read to confound; planner-independent).
has M1 && { echo "## M1 HBM compute-free (write-dom thin-K + read-dom thin-M)" \
  | tee -a "$LOG"
  # FORCED splits (mmwd) -> data is planner-INDEPENDENT: the in-tree matmul cost model
  # changed in the merge, so we fix (m,n,k) and trust the measurement, not its choice.
  echo "# M1a write-dom: M=N=2048, vary K  (split m4 n8; ~2-7% compute)" | tee -a "$LOG"
  for kk in 16 32 64; do runmmwd 2048 "$kk" 2048 4 8 1; done
  echo "# M1b write-dom: K=32, vary M=N  (size sweep; hbm ~ bytes?)" | tee -a "$LOG"
  runmmwd 4096 32 2048 4 8 1; runmmwd 4096 32 4096 4 8 1
  echo "# M1c read-dom: K=N=4096, vary M (<=64 -> tiny output, big reads; split N)" \
    | tee -a "$LOG"
  runmmwd 32 4096 4096 1 32 1; runmmwd 64 4096 4096 1 32 1
  echo "# M1d read-dom: M=64, K=4096, vary N (R/W mix)" | tee -a "$LOG"
  runmmwd 64 4096 2048 1 32 1; }

# ============== M2: COMPUTE + re-read (compute-heavy, hbm pinned) ============
# (measured - hbm_pinned) should be MNK/cores/peak. Low cores -> compute >=80%.
has M2 && { echo "## M2 compute + re-read (compute-heavy)" | tee -a "$LOG"
  echo "# M2a cores-scaling, M=4096 N=2048 K=2048 (peak constant in 1/cores?)" \
    | tee -a "$LOG"
  runmmwd 4096 2048 2048 8 4 1   # cores 32  (~48% compute)
  runmmwd 4096 2048 2048 8 2 1   # cores 16  (~65%)
  runmmwd 4096 2048 2048 4 2 1   # cores 8   (~79%)
  runmmwd 4096 2048 2048 2 2 1   # cores 4   (~88%)
  echo "# M2b m<->n swap at cores=32 (compute invariant -> delta = re-read)" \
    | tee -a "$LOG"
  runmmwd 4096 2048 2048 4 8 1   # m4n8 vs m8n4 above
  runmmwd 4096 2048 2048 16 2 1  # heavy M-split
  runmmwd 4096 2048 2048 2 16 1  # heavy N-split
  echo "# M2c peak across shapes, cores=4 (compute-dominant >=88%)" | tee -a "$LOG"
  runmmwd 2048 2048 2048 2 2 1   # MNK=8.6e9
  runmmwd 2048 4096 2048 2 2 1   # MNK=1.7e10 (vary K)
  runmmwd 2048 2048 4096 2 2 1; }  # MNK=1.7e10 (vary N)

# ============== M3: PSUM (only after M1 + M2 confirmed) =====================
# measured(k) - measured(k=1) at FIXED shape+cores=32. k forces (m,n) down, so
# subtract the M2b re-read delta to deconfound. Then K- and M*N-dependence.
has M3 && { echo "## M3 psum (k-sweep; K- and M*N-dependence)" | tee -a "$LOG"
  echo "# M3a M=N=2048 K=2048 cores=32, vary k (1,2,4,8)" | tee -a "$LOG"
  runmmwd 2048 2048 2048 8 4 1   # k=1
  runmmwd 2048 2048 2048 4 4 2   # k=2
  runmmwd 2048 2048 2048 4 2 4   # k=4
  runmmwd 2048 2048 2048 2 2 8   # k=8
  echo "# M3b K-dependence: K=4096, k=1 vs 4 (model says psum ~ M*N, K-indep)" \
    | tee -a "$LOG"
  runmmwd 2048 4096 2048 8 4 1; runmmwd 2048 4096 2048 4 2 4
  echo "# M3c M*N-dependence: M=N=4096 K=2048, k=1 vs 4" | tee -a "$LOG"
  runmmwd 4096 2048 4096 8 4 1; runmmwd 4096 2048 4096 4 2 4; }

# ============== E2E: coarse-tile examples run + finish on the new runtime =====
# Manager check: these three coarse-tile e2e examples should now RUN + FINISH. Each
# runs compare_with_cpu and prints PASSED (or raises). `python coarse_tile/run_X.py`
# puts coarse_tile/ on sys.path so its `from utils import ...` resolves. timeout
# guards against a hang (reported as FAILED, not a stuck sweep).
has E2E && { echo "## E2E coarse-tile examples (run + finish on new runtime)" \
  | tee -a "$LOG"
  for ex in run_softmax_row_tiling run_matmul_row_tiling run_matmul_k_tiled; do
    echo "-- e2e: $ex" | tee -a "$LOG"
    o="haoyang_logs/e2e_${ex}.out"
    if timeout 900 python "$ROOT/coarse_tile/$ex.py" > "$o" 2>&1 \
        && grep -q '^PASSED' "$o"; then
      echo "E2E $ex PASSED" | tee -a "$LOG"
    else
      echo "E2E $ex FAILED (exit $?, see $o)" | tee -a "$LOG"
      tail -8 "$o" | sed 's/^/    /' | tee -a "$LOG"
    fi
  done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
