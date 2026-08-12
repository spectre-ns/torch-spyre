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
# Run several cost-model sweeps back-to-back in ONE invocation. Each child writes
# its OWN timestamped haoyang_logs/*.log (forward all of them). A child that fails
# does not abort the rest (no set -e), so one bad sweep won't lose the others.
#
# Chains, in order:
#   1. run_matmul_validate_sweep.sh  (matmul M1/M2/M3 + E2E coarse-tile smoke test)
#   2. run_transport_sweep.sh        (transport / data-movement T1/T2)
#
# SECTIONS (if set) passes through to BOTH children; each ignores section names it
# doesn't own (has() just won't match), so e.g. SECTIONS="M1 T1" runs only those:
#   bash docs/source/user_guide/examples/run_all_sweeps.sh                 # everything
#   SECTIONS="M1 T1" bash docs/source/user_guide/examples/run_all_sweeps.sh
# To add a sweep, append its script to SWEEPS below.
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SWEEPS=(
  "run_matmul_validate_sweep.sh"
  "run_transport_sweep.sh"
)

echo "==== run_all_sweeps $(date) -- chaining ${#SWEEPS[@]} sweeps ===="
for s in "${SWEEPS[@]}"; do
  echo
  echo "################################################################"
  echo "## >>> $s"
  echo "################################################################"
  bash "$SCRIPT_DIR/$s" || echo "## !!! $s exited non-zero (continuing)"
done

echo
echo "==== all sweeps DONE -- forward every haoyang_logs/*.log printed above ===="
