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

"""Automated coarse tiling: explicit ``for_each_tile`` loops and tile discovery."""

import dataclasses
import functools
import pytest
import os
import sys
import torch
import unittest

from collections.abc import Callable, Sequence
from typing import Optional

from unittest.mock import patch

from torch._inductor import config as t_inductor_config
from torch._inductor.graph import GraphLowering

from torch_spyre.constants import DEVICE_NAME
from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor import passes as ts_passes
from torch_spyre._inductor.propagate_hints import DimHint
from torch_spyre._inductor.passes import CustomPreSchedulingPasses
from torch_spyre._inductor.wsr import for_each_tile

sys.path.insert(0, os.path.dirname(__file__))
from test_scratchpad_use import _ParameterizedScratchpadMeta  # noqa: E402

try:
    from ortools.sat.python import cp_model  # noqa: F401

    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False


def expected_unimplemented(fn):
    """Expect a test to fail *only* by reaching an unbuilt part of the feature.

    ``unittest.expectedFailure`` absorbs any exception, so a test written
    against a gate that does not exist yet would be satisfied by the resulting
    ``AttributeError`` -- and would stay satisfied after the feature landed
    wrong.  This narrows the expectation to one declared cause and fails the
    test on anything else, including a clean pass (the signal to delete the
    marker).

    Because it is imperative rather than a pytest mark, ``-m 'not xfail'`` does
    not deselect these; they still run and still xfail at runtime.

    Nothing here is specific to coarse tiling; it belongs in
    ``utils_inductor.py`` once a second suite wants it.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            fn(self, *args, **kwargs)
        except NotImplementedError as exc:
            pytest.xfail(f"not built yet: {exc}")
        else:
            self.fail(f"{fn.__name__} passed -- remove @expected_unimplemented")

    return wrapper


# One buffer's coarse-tile fingerprint: the trip counts of the loop nest it
# sits in, outermost level first.  An op at an outer level of a deeper nest
# carries a prefix of its group's counts -- a drain left outside a two-level
# nest reads (4,) where the interior ops read (4, 2).
_Counts = tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _Level:
    """One level of a loop nest, and what asked for it.

    The label is what makes a pinned level distinguishable from a discovered
    one by *identity* rather than by position, so a discovered level may land
    *outside* a pinned one without failing the test, while a pin that was
    dropped or re-tiled is still caught.

    A pinned level is one the caller wrote as a ``for_each_tile`` call:
    ``pin`` is its index into the case's pins, outermost first.  A discovered
    level is one the compiler added through a hint scope of its own:
    ``hint_id`` identifies that scope and ``dim`` is the name it tiled
    (``"_span_overflow"`` for span overflow).  All three are ``None`` on a
    level that could not be attributed (see ``_label_nest``).
    """

    count: int
    pin: Optional[int] = None
    hint_id: Optional[int] = None
    dim: Optional[str] = None

    def __repr__(self) -> str:
        if self.pin is not None:
            label = f"pin{self.pin}"
        elif self.hint_id is not None:
            label = self.dim or f"hint{self.hint_id}"
        else:
            label = "?"
        return f"{label}:{self.count}"


_Nest = tuple[_Level, ...]


def _trip_counts(nest: _Nest) -> _Counts:
    """Drop the labels: the plain outermost-first trip counts of ``nest``."""
    return tuple(level.count for level in nest)


def _is_subsequence(counts: _Counts, nest: _Counts) -> bool:
    """True if ``counts`` is ``nest`` with zero or more levels left out.

    Subsequence rather than prefix: an op at an outer level of a deeper nest
    drops the *inner* levels and so does read as a prefix, but a reduction's
    fill op keeps only the output levels outer to the reduction (see
    ``_compute_fill_loop_info_planned``), which can leave out a level in the
    middle.  Prefix would call that legitimate nest a violation.
    """
    remaining = iter(nest)
    return all(count in remaining for count in counts)


def _group_hints(ops: Sequence) -> tuple[DimHint, ...]:
    """One hint per compiler-added level of a loop group, outermost first.

    Only hint scopes count here; ``for_each_tile`` levels are labelled by
    ``_pin_order`` instead, and the filters below already drop
    them (their hints carry ``split_count=1``).

    The group, not the op, is the unit here.  ``loop_count`` is a group-level
    fact -- every member carries the whole nest, including the levels it is
    invariant at -- so a single op's ``dim_hints`` can be *shorter* than the
    nest and is not a list the counts can be zipped against.  This unions
    across the group the way ``_hints_levels`` does, keeping a scope as soon
    as *some* member is tiled by it, which is exactly the rule that decided
    the group's levels.

    Two filters mirror that function: a hint the op is broadcast against
    (``loop_var is None``) and a split of 1 both produce no loop level, so
    neither can label one.
    """
    best: dict[int, DimHint] = {}
    for op in ops:
        for h in getattr(op, "dim_hints", []):
            prev = best.get(h.hint_id)
            if (
                prev is None
                or prev.loop_var is None
                or (prev.split_count == 1 and h.split_count > 1)
            ):
                best[h.hint_id] = h
    return tuple(
        sorted(
            (h for h in best.values() if h.loop_var is not None and h.split_count != 1),
            key=lambda h: h.hint_id,
        )
    )


def _splice_hints(op) -> list[DimHint]:
    """``op``'s ``for_each_tile`` hints, outermost level first.

    ``_stamp_direct_loop_info`` appends one ``DimHint`` with ``loop_var_range``
    set to every op of each level it stamps, outermost level first, and no hint
    scope sets that field.
    """
    return [
        h
        for h in getattr(op, "dim_hints", None) or []
        if h.loop_var is not None and getattr(h, "loop_var_range", None) is not None
    ]


def _pin_order(operations: Sequence) -> dict:
    """Each ``for_each_tile`` loop variable, mapped to its pin index.

    Every level has its own loop variable, so the variable is what identifies
    a pin.  Pins nest outermost first and every op lists its levels outermost
    first, so the order in which the variables first appear over the operation
    list is the nesting order.  ``hint_id`` cannot key a pin: the splice leaves
    it at its default of 0 on every level.
    """
    order: dict = {}
    for op in operations:
        for h in _splice_hints(op):
            order.setdefault(h.loop_var, len(order))
    return order


def _label_nest(op, group_hints: tuple[DimHint, ...], pin_of: dict) -> _Nest:
    """Pair ``op``'s trip counts with the loops and hints that produced them.

    The labels are the op's ``for_each_tile`` levels, outermost first, each
    named by its loop variable's pin index, followed by the group's
    compiler-added hint levels, and the pairing is positional.  Putting the
    pinned levels outermost is an assumption about where the tile search nests
    its own levels; equal lengths are what make the pairing unambiguous, and a
    mismatch leaves the whole nest unlabelled rather than guessed at.

    The group's hints, not the op's, label the compiler-added levels -- the
    op's own being a subset, they can only agree on length by being the same
    list, and where they *would* differ (below) the op has none at all.

    The lengths disagree for a *trimmed* nest: a reduction's fill op keeps
    only the output levels outer to the reduction
    (``_compute_fill_loop_info_planned``), as does the ``reduce_copy`` built
    from it.  Neither is constructed through ``copy_op_metadata``, so neither
    carries ``dim_hints`` to fall back on, and their levels come back
    unlabelled rather than guessed at from a subset that merely fits.  That
    is safe as long as nothing keys on them: a pin still shows up labelled on
    the ops that carry the untrimmed nest, and the count-only checks in
    ``_check_loops_preserved`` cover the trimmed op.
    """
    counts = tuple(int(count) for count in op.loop_info.loop_count)
    labels = [_Level(count=0, pin=pin_of[h.loop_var]) for h in _splice_hints(op)]
    labels += [
        _Level(count=0, hint_id=h.hint_id, dim=h.dim_names[0] if h.dim_names else None)
        for h in group_hints
    ]
    if len(labels) != len(counts):
        return tuple(_Level(count=count) for count in counts)
    return tuple(
        dataclasses.replace(label, count=count) for label, count in zip(labels, counts)
    )


def _label_tiling(operations: Sequence) -> dict[str, _Nest]:
    """Every coarse-tiled op in ``operations``, mapped to its labelled nest."""
    tiled = [op for op in operations if getattr(op, "loop_info", None) is not None]

    def group_key(op) -> tuple[int, ...]:
        # Group index is loop_group_id[0]; the rest of the tuple is nesting
        # depth, which a trimmed nest truncates (_compute_fill_loop_info_planned
        # keeps the prefix), so keying on the whole tuple would split a group.
        return tuple(op.loop_info.loop_group_id[:1])

    by_group: dict[tuple[int, ...], list] = {}
    for op in tiled:
        by_group.setdefault(group_key(op), []).append(op)
    group_hints = {key: _group_hints(ops) for key, ops in by_group.items()}
    pin_of = _pin_order(operations)
    return {
        op.get_name(): _label_nest(op, group_hints[group_key(op)], pin_of)
        for op in tiled
    }


@dataclasses.dataclass(frozen=True)
class _TilingCase:
    """One model plus the tiling contract asserted against it.

    inner:
        The part of the model the pins wrap in ``for_each_tile`` loops, untiled
        as written.  It takes the first ``len(named_dims)`` arguments.
    outer:
        The rest of the model, run on ``inner``'s result and the remaining
        arguments, outside every loop, or ``None`` when the loops cover the
        whole model.  It is what automatic tiling may add loops to: a loop the
        user wrote is never re-tiled.
    args:
        Device tensors passed to the compiled model, ``inner``'s first.
    named_dims:
        Per-argument axis labels for ``inner``'s arguments, and ``out_dims``
        the same for its result.  They are local to the test: a pin names an
        axis, and these say which axis of each operand (``None`` where it has
        none) and of the result that is.  Nothing is declared to the compiler
        -- ``for_each_tile`` states its tiling in the program itself.
    pins:
        The ``for_each_tile`` loops the *explicit* mode wraps around ``inner``,
        outermost first, and the whole of that mode's expectation: a pin is a
        ``(dim, count)`` and the nest it prescribes is those counts in that
        order, so a separate ``expected`` beside it could only restate them or
        contradict them.
    explicit_auto_pins:
        The loops for the *explicit_auto* mode, where automatic tiling is on
        as well.  What must survive is each loop exactly as written.
    """

    inner: Callable[..., torch.Tensor]
    outer: Optional[Callable[..., torch.Tensor]]
    args: tuple[torch.Tensor, ...]
    named_dims: tuple[Sequence[str], ...]
    out_dims: Sequence[str]
    pins: tuple[tuple[str, int], ...]
    explicit_auto_pins: tuple[tuple[str, int], ...]
    atol: float
    rtol: float

    @property
    def explicit_nest(self) -> _Counts:
        """The loop nest ``pins`` prescribes: their counts, outermost first."""
        return tuple(count for _, count in self.pins)

    def model(self, pins: tuple[tuple[str, int], ...]) -> Callable[..., torch.Tensor]:
        """The whole model, with ``pins`` wrapped around ``inner``."""
        n_inner = len(self.named_dims)

        def run(*args: torch.Tensor) -> torch.Tensor:
            result = _apply_pins(self, pins, *args[:n_inner])
            if self.outer is None:
                return result
            return self.outer(result, *args[n_inner:])

        return run


def _apply_pins(
    case: _TilingCase, pins: tuple[tuple[str, int], ...], *args: torch.Tensor
) -> torch.Tensor:
    """Run ``case.inner`` inside one ``for_each_tile`` per pin, outermost first.

    Each pin slices every operand that has its axis into ``count`` tiles along
    it, passes the rest whole, and lays the result tiles back along the same
    axis of the output.  Tiles are rank-preserving, so an inner loop finds its
    axis at the same position in the tiles the outer one handed it, and the
    nesting order is the loop-nest order.
    """
    if not pins:
        return case.inner(*args)
    (dim, count), rest = pins[0], pins[1:]
    axes = tuple(
        list(names).index(dim) if dim in names else None for names in case.named_dims
    )
    extent = next(arg.shape[axis] for arg, axis in zip(args, axes) if axis is not None)

    def body(_, tiles):
        return None, _apply_pins(case, rest, *tiles)

    _, out = for_each_tile(
        body,
        args,
        dims=axes,
        tile_size=extent // count,
        out_dim=list(case.out_dims).index(dim),
    )
    return out


class CollectTilingPasses(CustomPreSchedulingPasses):
    """Pre-scheduling pipeline that records the applied tiling once it is done.

    ``torch_spyre._inductor.patches.enable_spyre_context`` installs
    ``CustomPreSchedulingPasses`` itself, so observing its result means
    substituting this subclass for it.  ``coarse_tile`` stamps ``loop_info``
    well before the scheduler is built, so reading it here sees the final plan.

    ``dim_hints`` is never cleared, so it is still on the ops here: the
    ``for_each_tile`` levels and the compiler's own hint scopes each leave
    theirs, and that is the only thing that distinguishes a caller's pin from
    a level the compiler found on its own.
    """

    tiling: dict[str, _Nest] = {}

    def __call__(self, graph: GraphLowering) -> None:
        super().__call__(graph)
        type(self).tiling = _label_tiling(graph.operations)


class AutomatedCoarseTilingTests(
    unittest.TestCase, metaclass=_ParameterizedScratchpadMeta
):
    """model x tiling_mode x solver, one generated method per combination.

    The metaclass expands ``parameter_models`` against ``parameter_axes`` and
    routes each generated method through ``run_case``; ``case_decorators``
    marks the combos that cannot pass until the tile search exists.
    """

    def setUp(self):
        torch.manual_seed(0xAFFE)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    # ------------------------------------------------------------------
    # Compile and observe
    # ------------------------------------------------------------------
    def _compile_and_collect(
        self,
        case: "_TilingCase",
        pins: tuple[tuple[str, int], ...],
        *,
        layout_solver: str,
        auto_tiling: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, _Nest]]:
        """Compile ``case`` and return (cpu_result, device_result, tiling)."""
        # Raises the "gate is missing" NotImplementedError before compiling.
        if auto_tiling:
            # TODO: Implement coarse tiling configuration
            raise NotImplementedError("unified-tiling: config.auto_coarse_tiling")

        cpu_result = case.model(())(*(arg.to("cpu") for arg in case.args))

        CollectTilingPasses.tiling = {}
        # TODO: Patch coarse tiling config here
        # force_disable_caches belongs to torch's inductor config, not Spyre's;
        # CustomPreSchedulingPasses is a plain module attribute that
        # enable_spyre_context re-imports per compile, so it is swapped with
        # patch.object rather than a config knob.  no_grad because the Linear
        # weights require grad, and scan cannot trace an autograd graph for a
        # nested for_each_tile.
        with (
            torch.no_grad(),
            t_inductor_config.patch(force_disable_caches=True),
            ts_inductor_config.patch(
                allow_all_ops_in_lx_planning=True,
                layout_solver=layout_solver,
            ),
            patch.object(ts_passes, "CustomPreSchedulingPasses", CollectTilingPasses),
        ):
            compiled = torch.compile(case.model(pins), fullgraph=True)
            device_result = compiled(*case.args).to("cpu")

        return cpu_result, device_result, CollectTilingPasses.tiling

    def _assert_matches_cpu(self, case: "_TilingCase", device, cpu) -> None:
        torch.testing.assert_close(
            device,
            cpu,
            atol=case.atol,
            rtol=case.rtol,
            msg=lambda m: f"coarse-tiled result diverged from CPU\n\n{m}\n",
        )

    # ------------------------------------------------------------------
    # Reading the labels
    # ------------------------------------------------------------------
    def _classify_levels(
        self, tiling: dict[str, _Nest], pins: tuple[tuple[str, int], ...]
    ) -> tuple[dict[int, _Level], dict[int, _Level]]:
        """Split the applied levels into the caller's pins and the rest.

        ``pins[i]`` is the ``(dim, count)`` of the caller's *i*-th
        ``for_each_tile``, outermost first, and a level labelled ``pin=i`` is
        the loop that call became.  Every level labelled with a ``hint_id``
        instead was added by the compiler.

        Asserts each pinned level still divides by the count it named.  The
        axis needs no check: a ``for_each_tile`` names it in the program, so
        unlike a hint scope it has nothing to bind to the wrong one.  Returns
        the ``(pinned, discovered)`` levels, keyed by pin index and hint id
        respectively, so the caller can say which of the two it expected.
        """
        seen_pinned: dict[int, _Level] = {}
        discovered: dict[int, _Level] = {}
        for name, nest in sorted(tiling.items()):
            for level in nest:
                if level.hint_id is not None:
                    discovered[level.hint_id] = level
                if level.pin is None:
                    continue
                dim, count = pins[level.pin]
                self.assertEqual(
                    level.count,
                    count,
                    f"{name} tiles the level pinned on '{dim}' {level.count} "
                    f"ways, not the pinned {count} (its nest is {list(nest)})",
                )
                seen_pinned[level.pin] = level
        return seen_pinned, discovered

    # ------------------------------------------------------------------
    # The three contracts
    # ------------------------------------------------------------------
    def _check_loops_preserved(self, case: _TilingCase, solver: str) -> None:
        """The loops are applied exactly: every loop written, no level invented."""
        cpu, device, tiling = self._compile_and_collect(
            case, case.pins, layout_solver=solver, auto_tiling=False
        )
        expected = case.explicit_nest
        self.assertTrue(
            tiling,
            "no op was coarse-tiled: the for_each_tile loops were not spliced "
            f"(expected the nest {list(case.pins)})",
        )
        # Two count-only claims, kept beside the keyed ones below as the single
        # witness here that does not depend on the labelling: a level the
        # labeller could not attribute is invisible to every keyed check.
        nests = {name: _trip_counts(nest) for name, nest in tiling.items()}
        for name, counts in sorted(nests.items()):
            self.assertTrue(
                _is_subsequence(counts, expected),
                f"{name} is tiled {counts}, which is not the explicit nest "
                f"{expected} with levels left out",
            )
        self.assertIn(
            expected,
            set(nests.values()),
            f"no op carries the full explicit nest {expected}; "
            f"the applied tiling was {nests}",
        )
        # The rest is keyed on the loops themselves: with the tile search off
        # they are the only thing that may tile anything, so every level is
        # accounted for by a pin, dividing by the count that pin named.
        seen_pinned, discovered = self._classify_levels(tiling, case.pins)
        self.assertEqual(
            sorted(seen_pinned),
            list(range(len(case.pins))),
            f"the applied tiling {tiling} does not carry one level per "
            f"pin: {len(case.pins)} for_each_tile loops wrap the model",
        )
        self.assertFalse(
            discovered,
            f"levels {list(discovered.values())} were invented by the "
            f"compiler, but only the {len(case.pins)} explicit ones were "
            f"written (the applied tiling was {tiling})",
        )
        self._assert_matches_cpu(case, device, cpu)

    def _check_tiling_discovered(self, case: "_TilingCase", solver: str) -> None:
        """With no loops at all, the compiler picks a tiling by itself."""
        cpu, device, tiling = self._compile_and_collect(
            case, (), layout_solver=solver, auto_tiling=True
        )
        self.assertTrue(
            tiling,
            "Auto tiling is on and no loops were written, but no op was "
            "coarse-tiled -- the tile search found nothing to do",
        )
        self._assert_matches_cpu(case, device, cpu)

    def _check_loops_preserved_with_auto(
        self, case: "_TilingCase", solver: str
    ) -> None:
        """The written loops survive verbatim; the compiler tiles around them.

        A ``for_each_tile`` loop is authoritative, so automatic tiling may only
        add loops over ops outside it, never re-tile an op inside it.  Checked
        by label, not by position: where the compiler puts its own loops is
        its choice, and only the numerics (``_assert_matches_cpu``) can call
        that choice wrong.
        """
        cpu, device, tiling = self._compile_and_collect(
            case,
            case.explicit_auto_pins,
            layout_solver=solver,
            auto_tiling=True,
        )
        self.assertTrue(tiling, "no op was coarse-tiled: the loops were dropped")
        seen_pinned, discovered = self._classify_levels(tiling, case.explicit_auto_pins)
        self.assertEqual(
            sorted(seen_pinned),
            list(range(len(case.explicit_auto_pins))),
            f"the applied tiling {tiling} lost a written loop: the loops "
            f"{list(case.explicit_auto_pins)} should all still be there",
        )
        self.assertTrue(
            discovered,
            f"the loops {list(case.explicit_auto_pins)} survived but nothing "
            f"was added: the tile search left every op outside them untiled "
            f"({tiling})",
        )
        self._assert_matches_cpu(case, device, cpu)

    # ------------------------------------------------------------------
    # Models.  Each returns the model, its axis labels and the tiling contract,
    # defined once and reused across every tiling_mode and solver.
    # ------------------------------------------------------------------
    def _softmax_case(self) -> "_TilingCase":
        """softmax(dim=0) over (512, 1024), dims R (reduced) x C.

        One level: C divided 4 ways, each tile a whole-column softmax.  The
        other axis, R, is the reduced one: a map loop over it would softmax
        each row block on its own and change the result, and a reduction loop
        is a different model, so C is the whole prescribed plan.  The loop
        covers the whole model, so the explicit_auto mode leaves the compiler
        nothing outside it to tile.
        """
        return _TilingCase(
            inner=functools.partial(torch.softmax, dim=0),
            outer=None,
            args=(torch.rand((512, 1024), dtype=torch.float16, device=DEVICE_NAME),),
            named_dims=(["R", "C"],),
            out_dims=["R", "C"],
            pins=(("C", 4),),  # Reduction axis is not tiled for now
            explicit_auto_pins=(("C", 4),),
            # A good run lands at 2e-5 on outputs of order 1/512; the
            # reduction-tiled one lands at 3e-3, and this has to separate them.
            atol=5e-4,
            rtol=0.02,
        )

    def _mlp_case(self) -> "_TilingCase":
        """Two-layer MLP (Linear -> silu -> Linear), dims S x Din x Dh x Dout.

        The loops cover the first Linear and silu: S divided 2 ways outside
        Dh divided 4 ways.  Both are free (output) axes there -- Din is the
        first GEMM's reduction -- and the second Linear, which reduces over
        Dh, runs outside every loop on the assembled activation.  That second
        Linear is what automatic tiling may add a loop to; the explicit_auto
        mode writes only the S loop.
        """
        seq_len, in_dim, hidden_dim, out_dim = 128, 256, 1024, 256
        fc1 = torch.nn.Linear(in_dim, hidden_dim).half()
        fc2 = torch.nn.Linear(hidden_dim, out_dim).half()

        def up_proj(x, w1, b1):
            return torch.nn.functional.silu(torch.nn.functional.linear(x, w1, b1))

        def down_proj(h, w2, b2):
            return torch.nn.functional.linear(h, w2, b2)

        args = (
            torch.randn(seq_len, in_dim, dtype=torch.float16).to(DEVICE_NAME),
            fc1.weight.to(DEVICE_NAME),
            fc1.bias.to(DEVICE_NAME),
            fc2.weight.to(DEVICE_NAME),
            fc2.bias.to(DEVICE_NAME),
        )
        return _TilingCase(
            inner=up_proj,
            outer=down_proj,
            args=args,
            named_dims=(["S", "Din"], ["Dh", "Din"], ["Dh"]),
            out_dims=["S", "Dh"],
            pins=(("S", 2), ("Dh", 4)),
            explicit_auto_pins=(("S", 2),),
            atol=0.02,
            rtol=0.05,
        )

    def _swiglu_case(self) -> "_TilingCase":
        """SwiGLU FFN (silu(gate) * up, then a down projection), dims S x Din x Dh.

        The loops cover the gated half: S divided 2 ways outside Dh divided 4
        ways.  Dh is a free axis the whole way through it -- the N dimension of
        both GEMMs and the layout of every activation -- and both weights
        carry the ``Dh`` label, so the inner loop slices the gate and up
        branches together.  The down projection reduces over Dh and runs
        outside every loop; the explicit_auto mode writes only the S loop.
        """
        seq_len, in_dim, hidden_dim = 128, 256, 1024
        fc_gate = torch.nn.Linear(in_dim, hidden_dim).half()
        fc_up = torch.nn.Linear(in_dim, hidden_dim).half()
        fc_down = torch.nn.Linear(hidden_dim, in_dim, bias=False).half()

        def gated(x, w_gate, b_gate, w_up, b_up):
            gate = torch.nn.functional.linear(x, w_gate, b_gate)
            up = torch.nn.functional.linear(x, w_up, b_up)
            return torch.nn.functional.silu(gate) * up

        def down_proj(h, w_down):
            return torch.nn.functional.linear(h, w_down)

        args = (
            torch.randn(seq_len, in_dim, dtype=torch.float16).to(DEVICE_NAME),
            fc_gate.weight.to(DEVICE_NAME),
            fc_gate.bias.to(DEVICE_NAME),
            fc_up.weight.to(DEVICE_NAME),
            fc_up.bias.to(DEVICE_NAME),
            fc_down.weight.to(DEVICE_NAME),
        )
        return _TilingCase(
            inner=gated,
            outer=down_proj,
            args=args,
            named_dims=(
                ["S", "Din"],
                ["Dh", "Din"],
                ["Dh"],
                ["Dh", "Din"],
                ["Dh"],
            ),
            out_dims=["S", "Dh"],
            pins=(("S", 2), ("Dh", 4)),
            explicit_auto_pins=(("S", 2),),
            atol=0.02,
            rtol=0.05,
        )

    # ------------------------------------------------------------------
    # Matrix
    # ------------------------------------------------------------------
    _CHECKS = {
        "explicit": _check_loops_preserved,
        "auto": _check_tiling_discovered,
        "explicit_auto": _check_loops_preserved_with_auto,
    }

    parameter_axes = {"tiling_mode": tuple(_CHECKS), "solver_method": ("cpsat",)}

    # SDPA is omitted: its Spyre decomposition emits for_each_tile loops of its
    # own, which _pin_order would count as pins, and using SDPA in this test
    # suite requires resolution of
    # https://github.com/torch-spyre/torch-spyre/issues/3198

    parameter_models = (
        ("softmax_tiling", _softmax_case),
        ("mlp_tiling", _mlp_case),
        ("swiglu_tiling", _swiglu_case),
    )

    @staticmethod
    def case_decorators(params):
        """Mark the combos that cannot pass until the tile search is built.

        These entries are never edited again: each combo stops xfailing on its
        own, the moment the last unbuilt piece on its path lands, because
        ``expected_unimplemented`` keys on the exception rather than on a list
        maintained by hand.  A combo turning green is the signal to delete its
        row here.
        """
        decorators = []
        if params["solver_method"] == "cpsat":
            decorators.append(
                unittest.skipUnless(_HAS_ORTOOLS, "the cpsat solver needs ortools")
            )
        if params["tiling_mode"] in ("auto", "explicit_auto"):
            decorators.append(expected_unimplemented)
        return decorators

    def run_case(self, params: dict, factory: Callable) -> None:
        """Body of one generated method: build the model, check its contract."""
        self._CHECKS[params["tiling_mode"]](
            self, factory(self), params["solver_method"]
        )
