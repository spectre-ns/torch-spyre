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

"""Per-kernel DEVICE-latency bench for the pointwise cost-model op ladder.

Measures one pointwise workload's device latency (sync'd profiling registry).
Pair with ``SPYRE_DUMP_COST=1`` to print the cost-model prediction at compile
time, so predicted vs measured can be compared in one run.

Knobs:
    BENCH_OP    gelu | relu | sigmoid | exp | mul | add | add3 | add4 |
                bcast | mulbcast | sumrow | sumall | amax | mean   (default gelu)
    BENCH_DEPTH N   chain a unary op N times (for the LX-bandwidth sweep)
    BENCH_LX_ALL 1  force ALL ops LX-eligible (config.allow_all_ops_in_lx_planning)
    BENCH_ROWS, BENCH_COLS, BENCH_RUNS, BENCH_WARMUP

Cost-model rungs:
    # 1) single op, size sweep  -> fit fill + BW_HBM
    for n in 512 1024 2048 4096; do BENCH_OP=gelu BENCH_COLS=$n python examples/bench_ops.py; done
    # 2) arithmetic-free check
    BENCH_OP=relu python examples/bench_ops.py ; BENCH_OP=gelu python examples/bench_ops.py
    # 3) traffic counting: 1-input vs 2-input
    BENCH_OP=gelu python examples/bench_ops.py ; BENCH_OP=mul python examples/bench_ops.py
    # 4) LX bandwidth: chain depth sweep, all intermediates in LX
    for d in 1 2 4 8; do BENCH_LX_ALL=1 BENCH_DEPTH=$d python examples/bench_ops.py; done
"""

import os

os.environ.setdefault("SPYRE_PROFILE", "1")
os.environ.setdefault("SPYRE_PROFILE_SYNC", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from torch_spyre._inductor import config  # noqa: E402
from torch_spyre.execution import profiling  # noqa: E402
from torch_spyre.execution.bench import measure_device  # noqa: E402

DEVICE = torch.device("spyre")
OP = os.environ.get("BENCH_OP", "gelu")
DEPTH = int(os.environ.get("BENCH_DEPTH", "1"))
ROWS = int(os.environ.get("BENCH_ROWS", "512"))
COLS = int(os.environ.get("BENCH_COLS", "1024"))
RUNS = int(os.environ.get("BENCH_RUNS", "100"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "20"))

if os.environ.get("BENCH_LX_ALL") == "1":
    config.allow_all_ops_in_lx_planning = True

_UNARY = {
    "gelu": F.gelu,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "exp": torch.exp,
}
_BINARY = {"mul": lambda a, b: a * b, "add": lambda a, b: a + b}
# n full inputs summed in ONE fused kernel: stream count = n_inputs + 1
# (n reads + 1 write). gelu=2, mul/add=3, add3=4, add4=5 -> a stream-count sweep
# to map effective per-byte rate vs # concurrent streams.
_NARY_ADD = {"add3": 3, "add4": 4}
# Broadcast second operand b[1,COLS]: counted at ~one row iff the device caches it.
# mulbcast mirrors bcast to check the caching result is not add-specific.
_BCAST = {"bcast": lambda a, b: a + b, "mulbcast": lambda a, b: a * b}
# Reductions (1 input -> reduced output). sumrow/amax/mean keep a big [ROWS] output
# (the reduced axis is NOT split across cores -> no cross-core combine); sumall
# reduces to a scalar (reduced axis split across ALL cores -> exercises the ring
# combine). sum vs amax vs mean at one size = the arithmetic-free check for reductions.
_REDUCE = {
    "sumrow": lambda x: x.sum(dim=-1),
    "sumall": lambda x: x.sum(),
    "amax": lambda x: x.amax(dim=-1),
    "mean": lambda x: x.mean(dim=-1),
}

torch.manual_seed(0xAFFE)


def _sum_all(*ts):
    acc = ts[0]
    for t in ts[1:]:
        acc = acc + t
    return acc


def make_workload():
    if OP in _UNARY:
        f = _UNARY[OP]

        def chain(a):
            for _ in range(DEPTH):
                a = f(a)
            return a

        x = torch.rand(ROWS, COLS, dtype=torch.float16).to(DEVICE)
        return torch.compile(chain), (x,)
    if OP in _BINARY:
        f = _BINARY[OP]
        x = torch.rand(ROWS, COLS, dtype=torch.float16).to(DEVICE)
        y = torch.rand(ROWS, COLS, dtype=torch.float16).to(DEVICE)
        return torch.compile(f), (x, y)
    if OP in _NARY_ADD:
        # n full [ROWS,COLS] inputs summed in ONE fused kernel: n reads + 1 write
        # => n+1 streams. Sweeping add3 (4-stream) / add4 (5-stream) alongside gelu
        # (2) and mul/add (3) maps effective per-byte rate vs stream count, so we
        # can tell a smooth contention curve from a one-time 2->3 step.
        n = _NARY_ADD[OP]
        xs = [torch.rand(ROWS, COLS, dtype=torch.float16).to(DEVICE) for _ in range(n)]
        return torch.compile(_sum_all), tuple(xs)
    if OP in _BCAST:
        # a[ROWS,COLS] (op) b[1,COLS]: b is read with a BROADCAST index (only the
        # column var). Compare to the full binary (a (op) [ROWS,COLS]) at the same
        # size: bcast ~ full => hardware re-fetches b each row (count it full);
        # bcast ~ a unary 2-pass op => it loads b once and reuses (cached).
        f = _BCAST[OP]
        x = torch.rand(ROWS, COLS, dtype=torch.float16).to(DEVICE)
        b = torch.rand(1, COLS, dtype=torch.float16).to(DEVICE)
        return torch.compile(f), (x, b)
    if OP in _REDUCE:
        # 1 input reduced. The model reads the FULL input (out x reduction_size) at
        # the read rate; sumall (scalar out) also exercises the cross-core ring combine.
        x = torch.rand(ROWS, COLS, dtype=torch.float16).to(DEVICE)
        return torch.compile(_REDUCE[OP]), (x,)
    known = (
        list(_UNARY) + list(_BINARY) + list(_NARY_ADD) + list(_BCAST) + list(_REDUCE)
    )
    raise SystemExit(f"unknown BENCH_OP={OP!r} (use {known})")


compiled, args = make_workload()

profiling.set_report_at_exit(False)
depth_str = f" depth={DEPTH}" if (OP in _UNARY and DEPTH > 1) else ""
print(
    f"{OP}[{ROWS}x{COLS}]{depth_str}  runs={RUNS}  "
    f"LX_PLANNING={config.lx_planning}  ALL_LX={config.allow_all_ops_in_lx_planning}"
)
measure_device(lambda: compiled(*args), runs=RUNS, warmup=WARMUP)
print(profiling.format_report())
