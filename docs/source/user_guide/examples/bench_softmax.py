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

"""Per-kernel DEVICE-latency measurement for a compiled Spyre softmax.

Measures the softmax kernel's device latency directly via the sync'd profiling
registry (``SPYRE_PROFILE_SYNC``), with no host-side end-to-end timing or
identity baseline -- the device number is read straight off the kernel. The min
strips host/sync jitter; the static-dataflow device is deterministic.

The header prints ``LX_PLANNING`` so runs are self-documenting. To check whether
LX scratchpad planning helps this example, compare two runs:

    LX_PLANNING=1 python examples/bench_softmax.py
    LX_PLANNING=0 python examples/bench_softmax.py

Other knobs: BENCH_ROWS, BENCH_COLS, BENCH_RUNS, BENCH_WARMUP.
"""

import os

# Device-side measurement needs the sync'd per-kernel profiling path; enable by
# default (read at launch time, so setting them here is sufficient).
os.environ.setdefault("SPYRE_PROFILE", "1")
os.environ.setdefault("SPYRE_PROFILE_SYNC", "1")

import torch  # noqa: E402

from torch_spyre._inductor import config  # noqa: E402
from torch_spyre.execution import profiling  # noqa: E402
from torch_spyre.execution.bench import measure_device  # noqa: E402

DEVICE = torch.device("spyre")
RUNS = int(os.environ.get("BENCH_RUNS", "100"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "20"))
ROWS = int(os.environ.get("BENCH_ROWS", "512"))
COLS = int(os.environ.get("BENCH_COLS", "1024"))

torch.manual_seed(0xAFFE)
x = torch.rand(ROWS, COLS, dtype=torch.float16).to(DEVICE)
compiled_sm = torch.compile(lambda a: torch.softmax(a, dim=0))

profiling.set_report_at_exit(False)  # we print the report ourselves below
print(
    f"softmax[{ROWS}x{COLS}]  runs={RUNS}  "
    f"LX_PLANNING={config.lx_planning}  SYNC={profiling.profile_sync_enabled()}"
)
measure_device(lambda: compiled_sm(x), runs=RUNS, warmup=WARMUP)
print(profiling.format_report())
