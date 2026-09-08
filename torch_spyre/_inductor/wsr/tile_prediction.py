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

"""Predict the frame a tiling *would* produce, without applying it.

A coarse tiling rewrites an op's iteration ranges, its output layout, and the
indices its deps are written in. A per-core view taken on the committed
(untiled) layout therefore describes the wrong op, and residency is gated on
those views -- so a solver weighing a tiling it has not applied needs the
*tiled* frame as a pure prediction over the un-applied graph.

Every mutation coarse tiling performs already exists in ``coarse_tile.py``. What
is missing, and what this module supplies, is the **inverse**: a pure reading of
the frame ``_divide_ranges`` / ``_post_tile_layout_for_splits`` / ``_rescale_index``
would leave behind, composed from those same helpers rather than restated, so
prediction and application cannot drift.

Scope is deliberately the frame alone. Predicting the *buffers* a candidate
materializes (per-tile scratch, boundary ``full_buf``, reduction accumulator)
belongs with an objective that prices them; coarse tiling carries no objective
term of its own, so that predictor would be API with nothing to consume it and
is not built here.

Dependencies stay one-way (``tile_prediction -> coarse_tile``). Nothing here
mutates IR; the solver must not import this module -- the allocator calls the
predictor and hands results across, which is what keeps the solver IR-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy
from sympy import Expr

from torch._inductor.ir import ComputedBuffer, FlexibleLayout

from ..errors import Unsupported
from ..ir import FixedTiledLayout, _resize_device_layout
from ..pass_utils import (
    iteration_space_from_op,
    op_out_coords,
)
from ..scratchpad.plan_solver import TileSpec
from .coarse_tile import (
    _rescale_index,
    _stick_host_dim,
    reduction_loop_vars,
)


def _exact_div(value, count: int):
    """Divide an extent by a tile count, requiring exact division."""
    if isinstance(value, (int, sympy.Integer)):
        iv = int(value)
        if iv % count != 0:
            raise Unsupported(
                f"tile prediction: extent {iv} is not divisible by tile count "
                f"{count} (coarse tiling emits equal-sized tiles)."
            )
        return sympy.Integer(iv // count)
    return sympy.sympify(value) / count


def _output_and_reduction_counts(tiling: TileSpec):
    """Split a TileSpec into total per-dim counts, output vs reduction."""
    output_counts: dict[int, int] = {}
    reduction_counts: dict[int, int] = {}
    for axis in tiling.axes:
        target = reduction_counts if axis.is_reduction else output_counts
        target[axis.host_dim] = target.get(axis.host_dim, 1) * axis.count
    return output_counts, reduction_counts


@dataclass
class PredictedFrame:
    """The tiled frame a candidate produces for one op -- measured, not applied.

    ``ranges`` / ``reduction_ranges`` are the per-tile extents; ``layout`` is the
    per-tile output ``FixedTiledLayout`` (the op's own layout when untiled);
    ``write_index`` / ``read_index`` are rescaled to the tile strides; and
    ``iter_space`` maps each loop symbol to its per-tile extent. These are
    exactly the pieces ``_prepare_per_core_view`` consumes via ``view_parts``.
    """

    op_name: str
    tiling: TileSpec
    ranges: list
    reduction_ranges: list
    layout: FixedTiledLayout
    write_index: Expr
    read_index: Expr
    iter_space: dict

    def view_parts(self) -> tuple[dict, Expr, Expr]:
        """The ``(iter_space, write_index, read_index)`` tuple
        ``_prepare_per_core_view`` accepts as its ``parts`` argument."""
        return (self.iter_space, self.write_index, self.read_index)


def predict_frame(op: ComputedBuffer, tiling: TileSpec) -> PredictedFrame:
    """Predict the per-tile frame ``op`` would take under ``tiling`` -- no IR
    mutation.

    Output axes shrink ``op.data.ranges`` and the physical output layout (via
    ``_post_tile_layout_for_splits``, the same resize real tiling uses); reduction
    axes shrink ``op.data.reduction_ranges`` only, since the op's own output
    buffer is the accumulator and keeps its full output extent.
    """
    output_counts, reduction_counts = _output_and_reduction_counts(tiling)

    ranges = list(op.data.ranges)
    for d, c in output_counts.items():
        ranges[d] = _exact_div(ranges[d], c)

    reduction_ranges = list(getattr(op.data, "reduction_ranges", []))
    for d, c in reduction_counts.items():
        reduction_ranges[d] = _exact_div(reduction_ranges[d], c)

    if output_counts:
        layout = _predict_output_layout(op, ranges)
    else:
        layout = op.layout

    rw = op.get_read_writes()
    write_index = next(iter(rw.writes)).index
    read_index = next((d.index for d in rw.reads if hasattr(d, "index")), write_index)
    if output_counts:
        # _rescale_index matches by coefficient value and needs sympy strides
        # (it reads ``.is_number``); the mock/host layouts can carry plain ints.
        full_strides = [sympy.sympify(s) for s in op.layout.stride]
        tile_strides = [sympy.sympify(s) for s in layout.stride]
        write_index = _rescale_index(write_index, full_strides, tile_strides)
        read_index = _rescale_index(read_index, full_strides, tile_strides)

    iter_space = _predict_iter_space(op, output_counts, reduction_counts)
    return PredictedFrame(
        op_name=op.get_name(),
        tiling=tiling,
        ranges=ranges,
        reduction_ranges=reduction_ranges,
        layout=layout,
        write_index=write_index,
        read_index=read_index,
        iter_space=iter_space,
    )


def _predict_output_layout(op: ComputedBuffer, tiled_ranges: list) -> FixedTiledLayout:
    """The per-tile output ``FixedTiledLayout``, built exactly as
    ``_divide_ranges`` builds it: contiguous host strides over the tiled size,
    and ``_resize_device_layout`` from the *authoritative* stick host dim
    (recovered by coordinate identity, so transposed same-size dims resolve).
    """
    new_size = [int(r) for r in tiled_ranges]
    new_stride = list(FlexibleLayout.contiguous_strides(new_size))
    dev = op.layout.device_layout
    stick_hd = _stick_host_dim(op, dev)
    new_dev = _resize_device_layout(
        dev,
        [int(s) for s in op.layout.size],
        new_size,
        stick_host_dim=stick_hd,
    )
    return FixedTiledLayout(
        op.layout.device, op.layout.dtype, new_size, new_stride, new_dev
    )


def _predict_iter_space(
    op: ComputedBuffer,
    output_counts: dict[int, int],
    reduction_counts: dict[int, int],
) -> dict:
    """The op's iteration space with each tiled symbol's extent divided down.

    An output axis's loop symbol is the sole free symbol of
    ``op_out_coords(op)[host_dim]``; a reduction axis's is
    ``reduction_loop_vars(op)[host_dim]`` -- the same resolution
    ``tile_spec_to_dim_hints`` uses.
    """
    iter_space = dict(iteration_space_from_op(op))
    out_coords = op_out_coords(op)
    for host_dim, count in output_counts.items():
        if host_dim < len(out_coords):
            syms = out_coords[host_dim].free_symbols
            if len(syms) == 1:
                sym = next(iter(syms))
                if sym in iter_space:
                    iter_space[sym] = _exact_div(iter_space[sym], count)
    if reduction_counts:
        try:
            red_vars = reduction_loop_vars(op)
        except (AssertionError, StopIteration):
            red_vars = []
        for host_dim, count in reduction_counts.items():
            if host_dim < len(red_vars):
                sym = red_vars[host_dim]
                if sym in iter_space:
                    iter_space[sym] = _exact_div(iter_space[sym], count)
    return iter_space
