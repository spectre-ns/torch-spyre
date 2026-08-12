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

"""Standalone: x.amax(dim=0) tiled over B.

Mirrors test_hint_tiled_reduction_dim0_max_correct in test_coarse_tile_e2e.py.

Reduces a [B, D] tensor to the element-wise max over B (dim=0).
spyre_hint(num_tiles_per_dim={"B": 4}).

Usage:
    python3 tests/inductor/standalone/run_amax_dim0_tiled.py
"""

import torch
import torch_spyre  # noqa: F401

from utils import compare_with_cpu

import torch_spyre._inductor.propagate_named_dims as _pnd
from torch_spyre._inductor import config, spyre_hint

_declare_tensor_dim = _pnd.declare_tensor_dim
_name_tensor_dims = _pnd.name_tensor_dims


def main():
    B, D = 512, 64
    torch.manual_seed(0xAFFE)
    x = torch.randn(B, D, dtype=torch.float16)

    _declare_tensor_dim("B", B)
    _declare_tensor_dim("D", D)

    def fn(x):
        _name_tensor_dims(x, ["B", "D"])
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            return x.amax(dim=0)

    compare_with_cpu(fn, x, run_compile=True, run_eager=False, atol=1e-3, rtol=1e-3)

    print("PASSED")


if __name__ == "__main__":
    main()
