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

"""Standalone: softmax row-tiling over NROW.

Mirrors test_hint_softmax_row_tiling in test_coarse_tile_e2e.py.

spyre_hint(num_tiles_per_dim={"NROW": 4}) tiles a [16384, 4096] softmax
over the row dimension with lx_planning enabled.

Usage:
    python3 tests/inductor/standalone/run_softmax_row_tiling.py
"""

import torch
import torch_spyre  # noqa: F401

from utils import compare_with_cpu

import torch_spyre._inductor.propagate_named_dims as _pnd
from torch_spyre._inductor import config, spyre_hint

_declare_tensor_dim = _pnd.declare_tensor_dim
_name_tensor_dims = _pnd.name_tensor_dims


def main():
    NROW, NCOL = 16384, 4096
    torch.manual_seed(0)
    x = torch.rand(NROW, NCOL, dtype=torch.float16)

    _declare_tensor_dim("NROW", NROW)
    _declare_tensor_dim("NCOL", NCOL)

    def fn(x, dim=-1):
        _name_tensor_dims(x, ["NROW", "NCOL"])
        with spyre_hint(num_tiles_per_dim={"NROW": 4}):
            return torch.softmax(x, dim)

    compare_with_cpu(fn, x, run_compile=True, run_eager=False, atol=0.1, rtol=0.1)

    print("PASSED")


if __name__ == "__main__":
    main()
