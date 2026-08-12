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
# MATMUL gamma / peak / BW_r / BW_w ISOLATION. The current peak=1140 & gamma=0.46 sit on
# a NON-IDENTIFIABLE RIDGE (fit jointly at the intercept), and BW_w=156 > the 150 peak
# (the "compute-free" fit did not subtract compute-overlap). This sweep decouples them.
# Model: T = compute + HBM - gamma*min(compute,HBM); compute = MACs/(cores*peak);
# HBM = R/BW_r + W/BW_w + a*min(R,W) + spill;  spill grows once a per-core tile > ~448.
#
#   GD  peak, COMPUTE-DOMINANT cores-scan: M=N=2048 K=4096, balanced splits at cores
#         {4,8,16,32}. compute >> HBM at every core count -> min=HBM -> T = compute +
#         (1-gamma)*HBM; the SLOPE of T vs 1/cores = MACs/peak is gamma-INDEPENDENT, so
#         it recovers peak and genuinely escapes the ridge (design review: SOUND).
#   GH  gamma, HBM-DOMINANT cores-scan at a SPILL-FREE small shape: M=N=512 K=64 (and a
#         512->768 cross-check), balanced splits at cores {8,16,32}. compute < HBM ->
#         min=compute -> T = HBM + (1-gamma)*compute; the coeff of 1/cores = (1-gamma)/
#         peak. SMALL M/N keeps every per-core tile < 448 so SPILL STAYS 0 across the
#         scan (fix: else spill drifts with 1/cores and aliases into the gamma slope).
#         gamma = 1 - (GH coeff * MACs_GD)/(GD slope * MACs_GH); the two GH shapes must
#         agree. (pt_eff = 1 throughout: every M/m >= 64.)
#   BW  BW_r / BW_w rank-2 grid at cores=32 (4x8x1), K>=64, per-core tile in [64,448]
#         (pt_eff=1, spill~0). Shapes span read-dom AND write-dom (min(R,W) on BOTH
#         sides -> frees alpha) with a rank-2 (R,W) design matrix -> BW_r, BW_w, alpha
#         are separately identifiable. Fit AFTER subtracting the modeled compute (a true
#         minority here) using the peak/gamma pinned by GD/GH.
#
# cores set via forced WD split. Measured via the AIU profiler.
#
#   bash docs/source/user_guide/examples/run_matmul_gamma_sweep.sh   # GD GH BW
# Output: <repo-root>/haoyang_logs/mm_gamma_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/mm_gamma_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-GD GH BW}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== matmul gamma sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runmmwd() {  # runmmwd <M> <K> <N> <m> <n> <k>   FORCED split, cores=m*n*k
  echo "-- mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6 (cores=$(($4 * $5 * $6)), " \
    "M/m=$(($1 / $4)), N/n=$(($3 / $5)))" | tee -a "$LOG"
  SENCORES=32 WD_M="$4" WD_N="$5" WD_K="$6" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=mmwd BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "mmwd M=$1 K=$2 N=$3 m=$4 n=$5 k=$6"
}

# ===== GD: peak from compute-dominant cores-scan (slope is gamma-free) =======
has GD && { echo "## GD peak (compute-dom) M=N=2048 K=4096 cores{4,8,16,32}" | tee -a "$LOG"
  runmmwd 2048 4096 2048 2 2 1   # cores 4
  runmmwd 2048 4096 2048 2 4 1   # cores 8
  runmmwd 2048 4096 2048 4 4 1   # cores 16
  runmmwd 2048 4096 2048 4 8 1;  }  # cores 32

# ===== GH: gamma from HBM-dominant cores-scan, SPILL-FREE (M/m,N/n < 448) ====
has GH && { echo "## GH gamma (HBM-dom, spill-free) M=N=512 & 768, K=64, cores{8,16,32}" \
    | tee -a "$LOG"
  for MN in 512 768; do
    runmmwd "$MN" 64 "$MN" 2 4 1   # cores 8   (tile MN/2, MN/4)
    runmmwd "$MN" 64 "$MN" 4 4 1   # cores 16  (tile MN/4, MN/4)
    runmmwd "$MN" 64 "$MN" 4 8 1   # cores 32  (tile MN/4, MN/8)
  done; }

# ===== BW: BW_r/BW_w rank-2 grid, cores=32 (4x8x1), tiles in [64,448] ========
# R:W ratios span 1:19 .. 12:1 (min on both sides) so BW_r, BW_w, alpha separate.
has BW && { echo "## BW BW_r/BW_w rank-2 grid: cores=32 (4x8x1), K>=64, tile in [64,448]" \
    | tee -a "$LOG"
  runmmwd 1792 64   3584 4 8 1   # write-dom  R:W ~1:19   (tile 448,448)
  runmmwd 1024 128  2048 4 8 1   # write-dom  R:W ~1:5.3  (tile 256,256)
  runmmwd  512 512  1024 4 8 1   # balanced   R:W ~1.5:1  (tile 128,128)
  runmmwd 1792 2048  512 4 8 1   # read-dom   R:W ~5:1    (tile 448,64)
  runmmwd  256 2048  512 4 8 1;  }  # read-dom R:W ~12:1   (tile 64,64)

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
