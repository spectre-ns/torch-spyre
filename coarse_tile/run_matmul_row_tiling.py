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

"""Standalone: matmul row-tiling over M.

Mirrors test_hint_matmul_row_tiling in test_coarse_tile_e2e.py.

spyre_hint(num_tiles_per_dim={"M": 4}) tiles [M,K] @ [K,N] over the M
(row) dimension.  Shapes: M=256, K=128, N=64.

Usage:
    python3 tests/inductor/standalone/run_matmul_row_tiling.py
"""

import torch
import torch_spyre  # noqa: F401

from utils import compare_with_cpu

import torch_spyre._inductor.propagate_named_dims as _pnd
from torch_spyre._inductor import spyre_hint

_declare_tensor_dim = _pnd.declare_tensor_dim
_name_tensor_dims = _pnd.name_tensor_dims


def main():
    M, K, N = 256, 128, 64
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.float16) * 0.01
    y = torch.randn(K, N, dtype=torch.float16) * 0.01

    _declare_tensor_dim("M", M)
    _declare_tensor_dim("K", K)
    _declare_tensor_dim("N", N)

    def fn(x, y):
        _name_tensor_dims(x, ["M", "K"])
        _name_tensor_dims(y, ["K", "N"])
        with spyre_hint(num_tiles_per_dim={"M": 4}):
            return x @ y

    compare_with_cpu(fn, x, y, run_compile=True, run_eager=False, atol=0.01, rtol=0.01)

    print("PASSED")


if __name__ == "__main__":
    main()
