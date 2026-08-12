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

"""Standalone: unrolled K=4 hint-tiled softmax-shaped execution.

Mirrors test_hint_unrolled_softmax_shaped_execution in test_coarse_tile_e2e.py.

Tiles the batch dimension (dim 0) of a softmax-like chain using
spyre_hint(num_tiles_per_dim={"B": 4}).  sencores=1 avoids core-division
issues.  Reductions collapse dim 1 (D); the loop tiles dim 0 (B).

Usage:
    python3 tests/inductor/standalone/run_softmax_unrolled.py
"""

import torch
import torch_spyre  # noqa: F401

from utils import compare_with_cpu

import torch_spyre._inductor.propagate_named_dims as _pnd
from torch_spyre._inductor import config, spyre_hint

_declare_tensor_dim = _pnd.declare_tensor_dim
_name_tensor_dims = _pnd.name_tensor_dims


def main():
    B, D = 256, 64
    torch.manual_seed(0)
    x = torch.randn(B, D, dtype=torch.float16)

    _declare_tensor_dim("B", B)
    _declare_tensor_dim("D", D)
    _name_tensor_dims(x, ["B", "D"])

    def softmax_fn(x):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            max_val = x.amax(dim=-1, keepdim=True)
            x_shifted = x - max_val
            exp_x = x_shifted.exp()
            sum_exp = exp_x.sum(dim=-1, keepdim=True)
            return exp_x / sum_exp

    with config.patch(
        {
            "unroll_loops": True,
            "sencores": 1,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    ):
        compare_with_cpu(
            softmax_fn,
            x,
            run_compile=True,
            run_eager=False,
            atol=0.1,
            rtol=0.1,
        )

    print("PASSED")


if __name__ == "__main__":
    main()
