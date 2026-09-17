#!/usr/bin/env python3
# Copyright 2026 The Torch-Spyre Authors.
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

"""A work_div hint splitting K in x.view(B*M, K).mm(w) returns wrong rows.

Output rows (b0, m1) and (b1, m0) come back swapped, so assert_close fails.
"""

import torch
import torch_spyre  # noqa: F401
from torch_spyre._inductor import config, spyre_hint
from torch_spyre._inductor.wsr import propagate_named_dims as pnd

B, M, K, N = 2, 3, 192, 128
torch._inductor.config.force_disable_caches = True  # cache ignores the hint
for dim, size in (("B", B), ("M", M), ("K", K), ("N", N)):
    pnd.declare_tensor_dim(dim, size)
x = torch.randn(B, M, K, dtype=torch.float16)
w = torch.randn(K, N, dtype=torch.float16)


def fn(x, w):
    with spyre_hint(work_div={"K": 2}):
        return x.view(B * M, K).mm(w)


with config.patch({"sencores": 8, "lx_planning": False}):
    out = torch.compile(fn, dynamic=False)(
        pnd.name_tensor_dims(x.to("spyre"), ["B", "M", "K"]),
        pnd.name_tensor_dims(w.to("spyre"), ["K", "N"]),
    )
expected = (x.float().view(B * M, K) @ w.float()).half()
torch.testing.assert_close(out.cpu(), expected, rtol=0.1, atol=0.1)
