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

"""Run all standalone coarse-tile e2e scripts and report pass/fail.

Each script corresponds to one compare_with_cpu test case from
test_coarse_tile_e2e.py and can also be run independently.

Usage:
    python3 tests/inductor/standalone/run_all.py
    python3 tests/inductor/standalone/run_all.py --stop-on-first-failure
"""

import argparse
import importlib
import sys
import traceback
from pathlib import Path

SCRIPTS = [
    "run_softmax_unrolled",
    "run_softmax_row_tiling",
    "run_matmul_row_tiling",
    "run_matmul_k_tiled",
    "run_sum_dim0_tiled",
    "run_amax_dim0_tiled",
    "run_amin_dim0_tiled",
    "run_mm_k_tiled",
    "run_bmm_k_tiled",
    "run_bmm_3d2d_k_tiled",
    "run_bmm_nested_outer_B_inner_K",
    "run_mm_nested_outer_M_inner_K",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stop-on-first-failure",
        action="store_true",
        help="Abort after the first failing script",
    )
    args = parser.parse_args()

    standalone_dir = Path(__file__).parent
    sys.path.insert(0, str(standalone_dir))

    passed = []
    failed = []

    for name in SCRIPTS:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print("=" * 60)
        try:
            mod = importlib.import_module(name)
            mod.main()
            passed.append(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            if args.stop_on_first_failure:
                break

    print(f"\n{'=' * 60}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"  {name}")
        sys.exit(1)
    else:
        print("ALL PASSED")


if __name__ == "__main__":
    main()
