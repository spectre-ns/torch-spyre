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

"""Standalone: bmm with nested outer-B (output) + inner-K (reduction) tiling.

Mirrors test_nested_bmm_outer_Batch_inner_K_correct in test_coarse_tile_e2e.py.

torch.bmm([B,M,K] @ [B,K,N]) with two nested spyre_hints:
  outer: num_tiles_per_dim={"B": 2}  (tiles the output batch dimension)
  inner: num_tiles_per_dim={"K": 4}  (tiles the reduction dimension)

The fill op runs once per outer B-tile; the accum buffer is per-tile sized.
Shapes: B=4, M=64, K=512, N=32.

Usage:
    python3 tests/inductor/standalone/run_bmm_nested_outer_B_inner_K.py
"""

import torch
import torch_spyre  # noqa: F401

from utils import compare_with_cpu

import torch_spyre._inductor.propagate_named_dims as _pnd
from torch_spyre._inductor import config, spyre_hint

_declare_tensor_dim = _pnd.declare_tensor_dim
_name_tensor_dims = _pnd.name_tensor_dims


def main():
    B, M, K, N = 4, 64, 512, 32
    torch.manual_seed(0xCAFE)
    a = torch.randn(B, M, K, dtype=torch.float16) * 0.01
    b = torch.randn(B, K, N, dtype=torch.float16) * 0.01

    _declare_tensor_dim("B", B)
    _declare_tensor_dim("M", M)
    _declare_tensor_dim("K", K)
    _declare_tensor_dim("N", N)

    def fn(a, b):
        _name_tensor_dims(a, ["B", "M", "K"])
        _name_tensor_dims(b, ["B", "K", "N"])
        with spyre_hint(num_tiles_per_dim={"B": 2}):
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return torch.bmm(a, b)

    compare_with_cpu(fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05)

    print("PASSED")


if __name__ == "__main__":
    main()
