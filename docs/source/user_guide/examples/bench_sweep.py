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

"""Op/size sweep mapping Spyre device latency vs problem size.

For each size ``N`` it times the chosen op at ``NxN`` (amortized min-of-N
harness) alongside an identity baseline, and prints per-call device latency vs
size. Use it to find the host-floor crossover and to gather
(size -> device latency) calibration points for the cost model. A *same-op*
sweep keeps the host wrapper constant, so the growth across rows is device work.

Example:
    BENCH_OP=softmax BENCH_SIZES=512,1024,2048,4096,8192 python examples/bench_sweep.py
    BENCH_OP=matmul  BENCH_SIZES=512,1024,2048,4096      python examples/bench_sweep.py
"""

import os

import torch

from torch_spyre.execution.bench import measure_latency

DEVICE = torch.device("spyre")
OP = os.environ.get("BENCH_OP", "softmax")
SIZES = [int(s) for s in os.environ.get("BENCH_SIZES", "512,1024,2048,4096").split(",")]
RUNS = int(os.environ.get("BENCH_RUNS", "50"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))
INNER = int(os.environ.get("BENCH_INNER", "400"))

torch.manual_seed(0xAFFE)


def make_workload(op: str, n: int):
    """Return (compiled_fn, args) for op at size NxN."""
    if op == "softmax":
        x = torch.rand(n, n, dtype=torch.float16).to(DEVICE)
        return torch.compile(lambda a: torch.softmax(a, dim=0)), (x,)
    if op == "matmul":
        a = torch.rand(n, n, dtype=torch.float16).to(DEVICE)
        b = torch.rand(n, n, dtype=torch.float16).to(DEVICE)
        return torch.compile(lambda u, v: u @ v), (a, b)
    raise SystemExit(f"unknown BENCH_OP={op!r} (use softmax|matmul)")


print(f"op={OP}  runs={RUNS} inner={INNER}  (per-call min, min-of-N)")
header = (
    f"{'N':>6}{'elements':>12}{'op_us':>11}{'identity_us':>13}"
    f"{'net_us':>11}{'us/Melem':>11}{'spread%':>9}"
)
print(header)

for n in SIZES:
    fn, args = make_workload(OP, n)
    op_stats = measure_latency(
        lambda: fn(*args), runs=RUNS, warmup=WARMUP, inner=INNER, label=OP
    )
    base_in = args[0]
    cid = torch.compile(lambda a: a + 0.0)
    id_stats = measure_latency(
        lambda: cid(base_in), runs=RUNS, warmup=WARMUP, inner=INNER, label="identity"
    )
    elements = n * n
    op_us = op_stats.min_ns / 1000.0
    id_us = id_stats.min_ns / 1000.0
    net_us = op_us - id_us
    us_per_melem = op_us / (elements / 1e6)
    print(
        f"{n:>6}{elements:>12}{op_us:>11.2f}{id_us:>13.2f}"
        f"{net_us:>11.2f}{us_per_melem:>11.3f}{op_stats.spread * 100:>9.1f}"
    )
