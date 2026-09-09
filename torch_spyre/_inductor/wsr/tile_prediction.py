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

The frame is stated in the op's *pre-tiling* iteration-symbol namespace, not
the one the tiled op will carry. Applying a tiling re-runs Inductor's
``extract_read_writes -> index_vars_squeeze``, which drops every dim whose
per-tile extent is 1 and renumbers the survivors from a fresh ``d0`` -- names
that do not exist yet at prediction time, and that a prediction could not use
anyway because it is paired with the op's still-committed deps. So the
symbol-carrying fields (``iter_space``, ``write_index``, ``read_index``) are
valid only against *untiled* deps, while the symbol-free ones (``ranges``,
``layout``) are exact against the applied op. See ``_predict_iter_space``.

Dependencies stay one-way (``tile_prediction -> coarse_tile``). Nothing here
mutates IR; the solver must not import this module -- the allocator calls the
predictor and hands results across, which is what keeps the solver IR-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy
from sympy import Expr

from torch._inductor.ir import ComputedBuffer

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
    resolve_tile_axis_loop_vars,
)
from .tile import compute_tile_stride


@dataclass
class PredictedFrame:
    """The tiled frame a candidate produces for one op -- measured, not applied.

    ``ranges`` / ``reduction_ranges`` are the per-tile extents; ``layout`` is the
    per-tile output ``FixedTiledLayout`` (the op's own layout when untiled);
    ``write_index`` is rescaled to the tile strides while ``read_index`` is the
    op's committed read index unchanged (see ``predict_frame`` -- coarse tiling
    resizes the op's own output buffer, never the buffers it reads); and
    ``iter_space`` maps each loop symbol to its per-tile extent. These are
    exactly the pieces ``_prepare_per_core_view`` consumes via ``view_parts``.

    ``iter_space``, ``write_index`` and ``read_index`` are keyed by the op's
    *pre-tiling* loop symbols (see ``_predict_iter_space``), so they pair only
    with the committed, untiled ``MemoryDep`` that ``_prepare_per_core_view``
    reads. ``ranges``, ``reduction_ranges`` and ``layout`` carry no symbols and
    match the applied op exactly, including when a tiled dim divides to a
    per-tile extent of 1.
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


def _predict_output_layout(op: ComputedBuffer, tiling: TileSpec) -> FixedTiledLayout:
    """The per-tile output ``FixedTiledLayout``, built exactly as
    ``_divide_ranges`` builds it.

    One resize **per tile level**, in ``TileSpec.axes`` order, chaining size,
    stride and device layout -- mirroring ``_divide_ranges``, which runs once
    per level and feeds each result to the next (coarse_tile.py:2211-2220).
    The composition is not associative, so a single full->tile resize is not
    equivalent: ``_resize_device_layout`` matches size-1 device dims to a size-1
    host dim by size alone (ir.py:236, no stride tiebreak and no one-to-one
    constraint), so once an earlier level drives a host dim to extent 1, a
    later resize can re-match a one-stick tile-count dim onto it and collapse
    its stride to the ``-1`` singleton sentinel. A single resize never sees
    that intermediate state and leaves the real stride in place. Measured: 4
    of 104 multi-level combinations diverge, all of that shape.

    Host strides come from ``compute_tile_stride``, not from
    ``contiguous_strides(new_size)``: the latter agrees only when the committed
    layout is contiguous, and silently reorders a transposed or channels-last
    layout (e.g. size [4, 128, 128] stride [128, 1, 16384] tiled to
    [4, 64, 128] yields [64, 1, 8192] applied vs [8192, 128, 1] contiguous).
    ``predict_frame`` feeds these straight to ``_rescale_index`` as the tile
    strides, so a reordered stride mismatches the applied per-core view.

    ``_stick_host_dim`` is re-resolved per level against the running device
    layout, as the applier does -- it recovers the *authoritative* stick host
    dim by coordinate identity, so transposed same-size dims resolve.
    """
    layout = op.layout
    cur_size = [int(s) for s in layout.size]
    cur_stride = [int(s) for s in layout.stride]
    cur_dev = layout.device_layout
    for axis in tiling.axes:
        if axis.is_reduction:
            continue
        new_size = list(cur_size)
        new_size[axis.host_dim] = int(_exact_div(cur_size[axis.host_dim], axis.count))
        cur_stride = [
            int(s) for s in compute_tile_stride(cur_size, cur_stride, new_size)
        ]
        cur_dev = _resize_device_layout(
            cur_dev,
            cur_size,
            new_size,
            stick_host_dim=_stick_host_dim(op, cur_dev),
        )
        cur_size = new_size
    return FixedTiledLayout(layout.device, layout.dtype, cur_size, cur_stride, cur_dev)


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

    Resolves both unguarded: ``_validate_tiling`` has already established that
    every ``host_dim`` here indexes in range and lands on exactly one loop symbol
    present in the iteration space. Any caller other than ``predict_frame`` must
    validate first -- these counts are positions in two different frames, and an
    unvalidated one reads the wrong dim or raises ``IndexError``/``KeyError``
    rather than being rejected.

    Keys stay the op's *pre-tiling* symbols; only extents move. That is
    deliberate. The applied op's symbols do not exist yet, and the caller pairs
    this dict with the op's committed (untiled) ``MemoryDep`` --
    ``_prepare_per_core_view`` builds ``dep_coeff`` as
    ``{sym: dep.index.coeff(sym) for sym in iter_space}``. Renaming the keys to
    what the tiled op will carry would break that pairing outright.

    The two namespaces are not interchangeable, and they overlap, so a mismatch
    reads the wrong dim rather than raising. Applying a tiling re-runs
    ``extract_read_writes -> index_vars_squeeze``, whose ``SqueezeView.squeezer``
    drops every dim of size 1 and mints ``d0, d1, ...`` from a fresh counter over
    the survivors; a dim tiled to per-tile extent 1 therefore loses its symbol
    and everything after it renumbers. Ranges ``[4, 128, 256]`` tiled on dim 1 by
    128 predicts ``{d0: 4, d1: 1, d2: 256}`` while the applied op carries
    ``{d0: 4, d1: 256}`` -- ``d1`` in both, meaning different dims. Never match a
    predicted frame against a post-apply dep by symbol. This is confined to the
    dep view: ``_divide_ranges`` keeps the unit dim at full rank, so ``ranges``,
    ``stride`` and ``device_size`` are unaffected.

    The surviving ``sym -> 1`` entry is inert in every consumer -- nothing splits
    a unit dim, and ``_per_core_view_from_prep`` skips ``split <= 1`` before
    device placement. Its one order-sensitive site is that function's
    ``contiguous_dim = len(dim_splits) - 1`` k-fast matmul reorder, which would
    select the phantom instead of the real trailing dim. That is unreachable
    rather than handled: it needs a ``Reduction`` (for ``is_matmul``), and both
    ``TileSpec`` producers reject Reduction unit tiles
    (``enumerate_tilings._reduction_split_counts`` and
    ``span_overflow_hint_analysis._split_candidates_for_host_dim``). If either
    filter is relaxed to admit them, drop the unit entry here instead.
    """
    iter_space = dict(iteration_space_from_op(op))
    out_coords = op_out_coords(op)
    for host_dim, count in output_counts.items():
        sym = next(iter(out_coords[host_dim].free_symbols))
        iter_space[sym] = _exact_div(iter_space[sym], count)
    if reduction_counts:
        red_vars = reduction_loop_vars(op)
        for host_dim, count in reduction_counts.items():
            sym = red_vars[host_dim]
            iter_space[sym] = _exact_div(iter_space[sym], count)
    return iter_space


def _validate_tiling(op: ComputedBuffer, tiling: TileSpec) -> None:
    """Reject a ``TileSpec`` that could not be lowered onto ``op``.

    ``predict_frame``'s single gate, and the reason the private predictors it
    calls resolve each axis unguarded. Axis legality itself is not restated here:
    :func:`resolve_tile_axis_loop_vars` is the shared authority, so this raises
    on exactly what ``tile_spec_to_dim_hints`` raises on when it lowers the same
    spec. What is added is the extra reach *prediction* has -- the two positional
    lists and the iteration space it divides, which lowering never touches.

    The symmetry with lowering is the point. ``predict_frame`` divides
    ``ranges``, ``reduction_ranges`` and the output layout for *every* axis
    unconditionally, so an axis that quietly failed to resolve downstream would
    not drop out of the prediction -- it would return a frame whose ranges and
    layout say "tiled" while its ``iter_space`` still says "untiled", priced by
    the solver as though consistent and only refused much later, at apply time.

    ``reduction_ranges`` is checked separately from the reduction loop variables
    the resolver bounds against: ``reduction_loop_vars`` is squeezed (a size-1
    dim carries no symbol) and can be the shorter list, so neither bound implies
    the other.

    Divisibility is deliberately not checked here: ``_exact_div`` already raises
    ``Unsupported`` at the point of division, which is loud rather than silent.
    """
    if tiling.is_untiled:
        return
    loop_vars = resolve_tile_axis_loop_vars(op, tiling)
    iter_space = iteration_space_from_op(op)
    ranges = list(op.data.ranges)
    reduction_ranges = list(getattr(op.data, "reduction_ranges", []))
    for axis, sym in zip(tiling.axes, loop_vars):
        if axis.is_reduction:
            if axis.host_dim >= len(reduction_ranges):
                raise Unsupported(
                    f"tile prediction: reduction host_dim={axis.host_dim} is out "
                    f"of bounds for reduction ranges {reduction_ranges} on "
                    f"{op.get_name()}."
                )
        elif axis.host_dim >= len(ranges):
            raise Unsupported(
                f"tile prediction: host_dim={axis.host_dim} is out of bounds for "
                f"data ranges {ranges} on {op.get_name()}."
            )
        if sym not in iter_space:
            raise Unsupported(
                f"tile prediction: host_dim={axis.host_dim} on {op.get_name()} "
                f"resolves to loop var {sym}, which is absent from its iteration "
                f"space {dict(iter_space)}."
            )


def predict_frame(op: ComputedBuffer, tiling: TileSpec) -> PredictedFrame:
    """Predict the per-tile frame ``op`` would take under ``tiling`` -- no IR
    mutation.

    Output axes shrink ``op.data.ranges`` and the physical output layout (via
    ``_post_tile_layout_for_splits``, the same resize real tiling uses); reduction
    axes shrink ``op.data.reduction_ranges`` only, since the op's own output
    buffer is the accumulator and keeps its full output extent.

    Raises ``Unsupported`` for a tiling ``tile_spec_to_dim_hints`` could not
    lower onto ``op`` -- see ``_validate_tiling``, which gates everything below
    so no partially-divided frame can be returned.
    """
    _validate_tiling(op, tiling)
    output_counts, reduction_counts = _output_and_reduction_counts(tiling)

    ranges = list(op.data.ranges)
    for d, c in output_counts.items():
        ranges[d] = _exact_div(ranges[d], c)

    reduction_ranges = list(getattr(op.data, "reduction_ranges", []))
    for d, c in reduction_counts.items():
        reduction_ranges[d] = _exact_div(reduction_ranges[d], c)

    if output_counts:
        layout = _predict_output_layout(op, tiling)
    else:
        layout = op.layout

    rw = op.get_read_writes()
    write_index = next(iter(rw.writes)).index
    read_index = next((d.index for d in rw.reads if hasattr(d, "index")), write_index)
    if output_counts:
        full_strides = [sympy.sympify(s) for s in op.layout.stride]
        tile_strides = [sympy.sympify(s) for s in layout.stride]
        write_index = _rescale_index(write_index, full_strides, tile_strides)

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
