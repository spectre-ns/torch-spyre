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

"""Minimal profiler demo: the GOLDEN per-kernel device time vs the cost model.

Runs a compiled op under ``torch.profiler`` (PrivateUse1, needs the kineto-spyre wheel)
and prints the per-kernel ``sdsc_fused_*`` "Self SPYRE" DEVICE time -- the golden
measurement -- next to the cost model's prediction (SPYRE_DUMP_COST also dumps the
device-layout I/O + the step-by-step calc during compile). The separate "Memset
(Device)" event is the non-deterministic host/setup overhead, reported but NOT the
kernel time. For the full op ladder + parseable SUMMARY lines see
docs/source/user_guide/examples/profile_ops.py.

    python profile_test.py
"""

import os

os.environ.setdefault("SPYRE_DUMP_COST", "1")  # dump cost-model I/O + prediction
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")  # force fresh compile

import torch  # noqa: E402
import torch_spyre  # noqa: E402, F401
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from torch_spyre._inductor import cost_model, dump_cost_model  # noqa: E402


def main():
    device = torch.device("spyre")
    compiled = torch.compile(lambda x, y: x + y)
    x = torch.randn(512, 1024, dtype=torch.float16, device=device)
    y = torch.randn(512, 1024, dtype=torch.float16, device=device)

    for _ in range(5):  # compile (-> SPYRE_DUMP_COST fires) + warm the kernel
        compiled(x, y).cpu()
    feats = list(dump_cost_model.LAST_FEATS)  # cost-model features for the prediction

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        compiled(x, y).cpu()

    ka = prof.key_averages()
    print(ka.table(sort_by="cuda_time_total", row_limit=20).replace("CUDA", "AIU"))

    # GOLDEN kernel time = the "Self SPYRE" DEVICE time that is neither Memset nor
    # Memcpy. Classify by EXCLUSION, not by name: older runtimes named the fused kernel
    # sdsc_fused_*/inductor-spyre-*, but the new image leaves its event name BLANK, so a
    # name match silently reports 0.
    kernel = memset = other = 0.0
    for ev in ka:
        us = getattr(ev, "self_device_time_total", 0) or getattr(
            ev, "self_cuda_time_total", 0
        )
        if not us or us <= 0:
            continue
        key = ev.key or ""
        if "Memset" in key:
            memset += us
        elif "Memcpy" in key:
            other += us
        else:
            kernel += us
    pred = cost_model.predict_ops(feats) / 1000 if feats else 0.0
    print(f"\nKERNEL device time              = {kernel:8.3f} us  <- golden")
    print(f"Memset / host-setup overhead    = {memset:8.3f} us  (non-deterministic)")
    print(f"Memcpy (HtoD+DtoH) overhead     = {other:8.3f} us")
    print(f"cost-model prediction           = {pred:8.3f} us")


if __name__ == "__main__":
    main()
