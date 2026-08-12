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
# COST-MODEL TERM-ISOLATION SWEEP. Each section varies ONE thing so a single
# model term can be fit/challenged in isolation. ~47 runs, ~2-3 h (well < 12 h).
#
# COARSE-TILING (pointwise chain z=(a+b)*c) -- separate the two tiling terms:
#   A  c_loop      tall-thin [8192,128], vary K, rows/core>=16 (eff=1) -> slope=c_loop
#   B  underfill   [2048,4096], vary K -> rows/core 64..1 -> challenge eff form
#
# MATMUL (a@b) -- the split (m,n,k) is FORCED via spyre_hint(work_div=...) so each
# term is isolated; T_matmul = compute + H_turnaround + psum, all ADDITIVE.
#   C  compute     fix shape, vary cores (m*n, k=1, M/m>=64 so pt_eff=1):
#                  compute ~ MNK/cores. Also m<->n swap at fixed cores (invariance).
#   D  hbm         thin K (compute small), vary K down -> write-dominated; tests the
#                  turnaround H on the matmul R/W mix. Planner-chosen split (plain mm).
#   E  psum        FIX cores=32 and shape, vary k (1,2,4,8): compute+hbm constant, so
#                  delta = psum = (k-1)*M*N*psum_per_elem. The cleanest psum probe.
#   F  pt_eff      small M, compute-bound (N=K=4096), force m so M/m = 64,32,16:
#                  deviation from the 1/m compute scaling = pt_eff derate.
#   G  asym        large M!=N (both >=2k), planner split -> coverage on real shapes.
#   H  lx          plain mm x LX_PLANNING in {0,1}: does scratchpad planning move a
#                  standalone matmul, and does the model still hold?
#
# Every run logs op_it_space_splits (the ACTUAL m/n/k), the per-tensor IO, the
# model's MODEL estimate, and the measured SUMMARY. SENCORES=32 throughout; for
# forced splits cores = m*n*k (the hint is FINAL, not floor-filled).
#
#   bash docs/source/user_guide/examples/run_tiling_terms_sweep.sh          # A..H
#   SECTIONS="C E" bash docs/source/user_guide/examples/run_tiling_terms_sweep.sh
# Output: <repo-root>/haoyang_logs/tiling_terms_<timestamp>.log (forward this).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/tiling_terms_$(date +%Y%m%d_%H%M%S).log"
SECTIONS="${SECTIONS:-A B C D E F G H}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1   # force recompile so the dumps fire

echo "==== cost-model term-isolation sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {  # stdin: profile_ops output; keep splits/IO/MODEL/SUMMARY; FAILED if empty
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runchain() {  # runchain <rows> <cols> <tiles>   pointwise chain, all 32 cores
  SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=chain BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_TILES="$3" \
    python "$PROFILE_OPS" 2>&1 | _emit "chain rows=$1 cols=$2 tiles=$3"
}
runmm() {  # runmm <M> <K> <N> [lx=1]   plain matmul, planner-chosen split
  echo "-- mm M=$1 K=$2 N=$3 lx=${4:-1}" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING="${4:-1}" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=mm BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
    python "$PROFILE_OPS" 2>&1 | _emit "mm M=$1 K=$2 N=$3 lx=${4:-1}"
}
runmmwd() {  # runmmwd <M> <K> <N> <m> <n> <k>   FORCED split, cores = m*n*k
  echo "-- mmwd M=$1 K=$2 N=$3 split m=$4 n=$5 k=$6 (cores=$(($4 * $5 * $6)))" \
    | tee -a "$LOG"
  SENCORES=32 WD_M="$4" WD_N="$5" WD_K="$6" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=mmwd BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
    python "$PROFILE_OPS" 2>&1 | _emit "mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6"
}

# ===================== A: coarse-tiling c_loop (eff=1) ======================
has A && { echo "## A c_loop [8192,128] (rows/core=256/K>=16, eff=1; slope=c_loop)" \
  | tee -a "$LOG"
  for k in 1 2 4 8 16; do
    echo "-- A: K=$k  rows/core=$((256 / k))" | tee -a "$LOG"; runchain 8192 128 "$k"
  done; }

# ================= B: coarse-tiling underfill form =========================
has B && { echo "## B underfill [2048,4096] (rows/core=64/K, 64..1)" | tee -a "$LOG"
  for k in 1 2 4 8 16 32 64; do
    echo "-- B: K=$k  rows/core=$((64 / k))" | tee -a "$LOG"; runchain 2048 4096 "$k"
  done; }

# ===================== C: matmul compute / mac_peak ========================
# Fix shape, vary cores (k=1, M/m>=64 so pt_eff=1): compute ~ MNK/cores. The first
# two (m<->n swap) hold cores=32 -> compute should be IDENTICAL (m/n-invariant). The
# cores-scaling (32..4) spans compute 350..2800 us, so it pins mac_peak on its own --
# no large square anchor needed (M=N=K=4096 forced-split HANGS at compile, ~6.9e10
# MACs; keep forced (mmwd) shapes <= ~3.4e10, the largest that runs).
has C && { echo "## C compute: M=4096 N=2048 K=2048, vary cores (k=1)" | tee -a "$LOG"
  runmmwd 4096 2048 2048 8 4 1   # cores 32
  runmmwd 4096 2048 2048 4 8 1   # cores 32 (m<->n swap)
  runmmwd 4096 2048 2048 8 2 1   # cores 16
  runmmwd 4096 2048 2048 4 2 1   # cores 8
  runmmwd 4096 2048 2048 2 2 1; }  # cores 4

# ===================== D: matmul hbm (thin K) ==============================
# Small K -> compute small, output M*N dominates -> WRITE-dominated; tests turnaround.
has D && { echo "## D hbm thin-K: M=N=2048, K=64..512 (plain mm)" | tee -a "$LOG"
  for kk in 64 128 256 384 512; do runmm 2048 "$kk" 2048; done; }

# ===================== E: matmul psum (force k-split) ======================
# Fix cores=32 and shape -> compute + hbm constant; delta vs k=1 = psum=(k-1)*M*N*c.
has E && { echo "## E psum: M=N=2048 K=2048, cores=32, vary k" | tee -a "$LOG"
  runmmwd 2048 2048 2048 8 4 1   # k=1
  runmmwd 2048 2048 2048 4 4 2   # k=2
  runmmwd 2048 2048 2048 4 2 4   # k=4
  runmmwd 2048 2048 2048 2 2 8   # k=8
  echo "## E psum vs K (psum ~ M*N, K-independent): K=4096, k=1 vs k=4" | tee -a "$LOG"
  runmmwd 2048 4096 2048 8 4 1   # k=1
  runmmwd 2048 4096 2048 4 2 4; }  # k=4

# ===================== F: matmul pt_eff (small M) ==========================
# Compute-bound (N=K=4096), force m so M/m = 128,64,32,16: below 64 pt_eff<1, so the
# compute time deviates ABOVE the ideal 1/m scaling -> reads off the pt_eff derate.
has F && { echo "## F pt_eff: M=512 N=4096 K=4096, force m (M/m=128,64,32,16)" \
  | tee -a "$LOG"
  for mm in 4 8 16 32; do runmmwd 512 4096 4096 "$mm" 1 1; done; }

# ===================== G: matmul asymmetric large ==========================
has G && { echo "## G asym large (M!=N, both >=2k; K=2048; planner split)" \
  | tee -a "$LOG"
  runmm 4096 2048 2048; runmm 2048 2048 4096   # 1.7e10
  runmm 8192 2048 2048; runmm 2048 2048 8192   # 3.4e10
  runmm 4096 2048 4096; }                       # 3.4e10 (8192x4096 = 6.9e10 HANGS, drop)

# ===================== H: matmul LX planning on/off ========================
has H && { echo "## H LX on/off (plain mm, LX_PLANNING 0 vs 1)" | tee -a "$LOG"
  for lx in 0 1; do
    runmm 2048 2048 2048 "$lx"; runmm 4096 2048 2048 "$lx"; runmm 2048 4096 2048 "$lx"
  done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
