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

"""End to end: the default CP-SAT encodings (the derived view relation,
``Lazy`` work split and relayouts as constraints) plan graphs that compile and
compute the right numbers on device."""

import pytest

pytest.importorskip("ortools")

import torch  # noqa: E402

import torch_spyre  # noqa: E402,F401
from torch_spyre._inductor import config  # noqa: E402

_BASE = {
    "lx_planning": True,
    "co_optimizing_lx_planning": True,
    "layout_solver": "cpsat",
}


def _mlp(x, g, u, d):
    return (x @ u * torch.nn.functional.silu(x @ g)) @ d


def _norm_chain(x, w):
    y = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w
    return torch.relu(y) + y.softmax(-1)


@pytest.mark.parametrize("model", ["mlp", "norm_chain"])
def test_default_encodings_compute_the_reference(model):
    torch.manual_seed(0)
    if model == "mlp":
        fn = _mlp
        args = [
            torch.randn(64, 128, dtype=torch.float16),
            torch.randn(128, 256, dtype=torch.float16) / 16,
            torch.randn(128, 256, dtype=torch.float16) / 16,
            torch.randn(256, 128, dtype=torch.float16) / 16,
        ]
        # fp16 accumulation over K = 128 and 256.
        tolerance = {"rtol": 2e-2, "atol": 5e-2}
    else:
        fn = _norm_chain
        args = [
            torch.randn(128, 256, dtype=torch.float16),
            torch.randn(256, dtype=torch.float16),
        ]
        tolerance = {"rtol": 1e-2, "atol": 1e-2}
    expected = fn(*[a.float() for a in args]).half()
    torch._inductor.codecache.FxGraphCache.clear()
    with config.patch(_BASE):
        got = torch.compile(fn, dynamic=False)(*[a.to("spyre") for a in args])
    torch.testing.assert_close(got.cpu(), expected, **tolerance)
