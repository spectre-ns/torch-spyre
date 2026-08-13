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

"""Automated coarse tiling: hint preservation and hint-free tile discovery.

Coarse tiling today is *hint-driven*: ``spyre_hint(num_tiles_per_dim=...)``
scopes (from user code, or from a Spyre decomposition -- see
``decompositions.py``'s SDPA) are turned into ``DimHint``s by
``assign_dim_hints``, grouped by ``hints_to_coarse_tile_groups``, and stamped
onto each op as ``loop_info`` (a ``CoarseTileInfo``) by ``coarse_tile``.  Nothing
picks a tiling when the caller supplies none.

Phase 1 of the compiler-optimization roadmap ("Coarse Tiling Optimization")
makes the tiling a *solved* quantity: the allocator enumerates tile options and
chooses them jointly with core division and LX residency.  This file pins the
three behaviours that transition demands, over three of the four models the
scratchpad suite uses (softmax, MLP, SwiGLU; SDPA is excluded -- see
``parameter_models``):

===========  =============================================  ===============
hint_mode    Contract                                       Status today
===========  =============================================  ===============
hinted       Every level the hints ask for is applied, and   passes
             nothing else is tiled.
unhinted     With no hints at all the compiler finds a       xfail
             tiling on its own.
partial      Hinted levels survive verbatim; the compiler    xfail
             fills in the levels the caller left open.
===========  =============================================  ===============

The two xfail rows are marked with ``expected_unimplemented``, not
``unittest.expectedFailure``: they must fail *only* by reaching an unbuilt part
of the feature, never by an unrelated typo or backend break.

``solver_method`` is a parameter axis so a new solver is one string in
``_SOLVERS`` -- adding ``"simulated_annealing"`` there generates the whole
model x hint_mode matrix against ``SimulatedAnnealingLayoutSolver``.  The axis
has no effect on the *hinted* rows today (hint-driven tiling is decided in
``_maybe_coarse_tile_hints``, pre-stickification, long before any layout solver
runs); it exists because Phase 1 moves the tiling decision into the solver, and
these tests should start distinguishing solvers the day it does.

Two notes on scope, each measured on device rather than assumed:

* ``co_optimizing_lx_planning`` is deliberately left at its default (off).  A
  coarse-tiled graph put through the co-optimizing allocator raises
  ``AttributeError: 'MutationLayoutSHOULDREMOVE' object has no attribute
  'device_layout'`` from ``_output_stride_to_device_size``
  (``scratchpad/allocator.py:903``, reached via ``_split_fits_sticks``): the
  tile drain op ``coarse_tile_copy_*`` carries a mutation layout the sizing
  path does not expect.  Phase 1 has to fix that before tiling can be a solver
  variable; until then, turning co-optimization on here would fail every row
  for a reason that has nothing to do with hints.
* The tolerances below are far tighter than the scratchpad suite's 0.1/0.1.
  They have to be: reduction-dim coarse tiling currently returns *wrong
  numbers* (softmax tiled over its reduced axis is off by more than the output
  magnitude itself), and 0.1/0.1 hides that completely.  Every tolerance here
  is set from a measured good run with room to spare, and the tiling each model
  asks for is the one that is numerically correct today.
"""

import dataclasses
import functools
import os
import sys
from collections.abc import Callable, Sequence
from typing import Optional

import pytest
import torch
import unittest
from unittest.mock import patch

from torch._inductor import config as t_inductor_config
from torch._inductor.graph import GraphLowering

from torch_spyre.constants import DEVICE_NAME
from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor import passes as ts_passes
from torch_spyre._inductor import spyre_hint
from torch_spyre._inductor.passes import CustomPreSchedulingPasses
import torch_spyre._inductor.wsr.propagate_named_dims as _pnd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
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

    Move to ``utils_inductor.py`` when Step 0.2 of the implementation plan
    lands its shared copy.
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
class _TilingCase:
    """One model plus the tiling contract asserted against it.

    body:
        The unhinted model.  Pins are wrapped around it at compile time, so the
        same callable serves all three hint modes.
    args:
        Device tensors passed to the compiled model.
    named_dims:
        Per-argument named-dim labels, positionally aligned with ``args``.
        ``spyre_hint`` addresses dimensions by these names, so they are
        declared and attached for the hinted and partial modes and omitted
        entirely for the unhinted one -- a graph with no named dims is what
        "no hints" actually looks like to the compiler.
    pins / expected:
        The hint scopes the *hinted* mode wraps around ``body`` (outermost
        first), and the loop-nest trip counts they must produce.  The two are
        stated separately because they diverge for a model hinted by its own
        Spyre decomposition rather than by the caller: there ``pins`` is empty
        and ``expected`` is whatever the decomposition prescribes.
    partial_pins / partial_expected / partial_named_dims:
        The same for the *partial* mode, where the caller pins a strict subset
        of the tiling and leaves the rest to the compiler.  ``partial_expected``
        is the prefix that must survive verbatim.  ``partial_named_dims``
        defaults to ``named_dims``, and is overridden only when withholding a
        *name* is the only way to withhold a hint -- again, the
        decomposition-hinted case, where the caller cannot delete a scope the
        compiler emitted.
    blocked:
        hint_mode -> substring of the error a *known* backend gap raises today.
        A matching failure xfails with that reason; anything else fails red,
        and so does a clean pass.
    """

    body: Callable[..., torch.Tensor]
    args: tuple[torch.Tensor, ...]
    named_dims: tuple[Sequence[str], ...]
    pins: tuple[tuple[str, int], ...]
    expected: _Counts
    partial_pins: tuple[tuple[str, int], ...]
    partial_expected: _Counts
    atol: float
    rtol: float
    partial_named_dims: Optional[tuple[Sequence[str], ...]] = None
    blocked: dict[str, str] = dataclasses.field(default_factory=dict)

    def dims_for(self, hint_mode: str) -> Optional[tuple[Sequence[str], ...]]:
        if hint_mode == "unhinted":
            return None
        if hint_mode == "partial" and self.partial_named_dims is not None:
            return self.partial_named_dims
        return self.named_dims


def _apply_pins(pins: tuple[tuple[str, int], ...], body: Callable, *args):
    """Run ``body`` inside one ``spyre_hint`` scope per pin, outermost first.

    One scope per dimension: ``assign_dim_hints`` raises ``NotImplementedError``
    on a ``spyre_hint`` naming more than one, and the nesting order is what
    fixes the loop-nest order (hint ids increase inwards).
    """
    if not pins:
        return body(*args)
    (dim, tiles), rest = pins[0], pins[1:]
    with spyre_hint(num_tiles_per_dim={dim: tiles}):
        return _apply_pins(rest, body, *args)


class CollectTilingPasses(CustomPreSchedulingPasses):
    """Pre-scheduling pipeline that records the applied tiling once it is done.

    ``torch_spyre._inductor.patches.enable_spyre_context`` installs
    ``CustomPreSchedulingPasses`` itself, so observing its result means
    substituting this subclass for it.  ``coarse_tile`` stamps ``loop_info``
    well before the scheduler is built, so reading it here sees the final plan.
    """

    tiling: dict[str, _Counts] = {}

    def __call__(self, graph: GraphLowering) -> None:
        super().__call__(graph)
        type(self).tiling = {
            op.get_name(): tuple(int(count) for count in op.loop_info.loop_count)
            for op in graph.operations
            if getattr(op, "loop_info", None) is not None
        }


class AutomatedCoarseTilingTests(
    unittest.TestCase, metaclass=_ParameterizedScratchpadMeta
):
    """model x hint_mode x solver, one generated method per combination.

    The metaclass expands ``parameter_models`` against ``parameter_axes`` and
    routes each generated method through ``run_case``; ``case_decorators``
    marks the combos that cannot pass until the tile search exists.
    """

    def setUp(self):
        torch.manual_seed(0xAFFE)
        self.patchers = [
            t_inductor_config.patch("force_disable_caches", True),
            ts_inductor_config.patch("allow_all_ops_in_lx_planning", True),
            patch.object(ts_passes, "CustomPreSchedulingPasses", CollectTilingPasses),
        ]
        for p in self.patchers:
            p.__enter__()
        # Named dims live in module state that outlives a compile, so a stale
        # name left by another test would silently bind a hint here.
        _pnd.reset()
        self.addCleanup(_pnd.reset)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def tearDown(self):
        for p in reversed(self.patchers):
            p.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Compile and observe
    # ------------------------------------------------------------------
    def _compile_and_collect(
        self,
        case: "_TilingCase",
        hint_mode: str,
        pins: tuple[tuple[str, int], ...],
        *,
        layout_solver: str,
        auto_tiling: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, _Counts]]:
        """Compile ``case`` and return (cpu_result, device_result, tiling)."""
        # Raises the "gate is missing" NotImplementedError before compiling.
        if auto_tiling:
            # TODO: Implement coarse tiling configuration
            raise NotImplementedError("unified-tiling: config.auto_coarse_tiling")

        named_dims = case.dims_for(hint_mode)
        if named_dims is not None:
            for arg, dims in zip(case.args, named_dims):
                for dim, size in zip(dims, arg.shape):
                    _pnd.declare_tensor_dim(dim, int(size))
            for arg, dims in zip(case.args, named_dims):
                _pnd.name_tensor_dims(arg, list(dims))

        cpu_result = case.body(*(arg.to("cpu") for arg in case.args))

        model = functools.partial(_apply_pins, pins, case.body) if pins else case.body
        CollectTilingPasses.tiling = {}
        # TODO: Patch course tiling config here
        with ts_inductor_config.patch(layout_solver=layout_solver):
            device_result = torch.compile(model, fullgraph=True)(*case.args).to("cpu")

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
    # The three contracts
    # ------------------------------------------------------------------
    def _check_hints_preserved(self, case: _TilingCase, solver: str) -> None:
        """Hints are applied exactly: every level asked for, no level invented."""
        cpu, device, tiling = self._compile_and_collect(
            case, "hinted", case.pins, layout_solver=solver, auto_tiling=False
        )
        self.assertTrue(
            tiling,
            "no op was coarse-tiled: the hints were dropped before coarse_tile "
            f"(expected the nest {case.expected})",
        )
        for name, counts in sorted(tiling.items()):
            self.assertEqual(
                counts,
                case.expected[: len(counts)],
                f"{name} is tiled {counts}, which is not a prefix of the hinted "
                f"nest {case.expected}",
            )
        self.assertIn(
            case.expected,
            set(tiling.values()),
            f"no op carries the full hinted nest {case.expected}; "
            f"the applied tiling was {tiling}",
        )
        self._assert_matches_cpu(case, device, cpu)

    def _check_tiling_discovered(self, case: "_TilingCase", solver: str) -> None:
        """With no hints at all, the compiler picks a tiling by itself."""
        cpu, device, tiling = self._compile_and_collect(
            case, "unhinted", (), layout_solver=solver, auto_tiling=True
        )
        self.assertTrue(
            tiling,
            "Auto tiling is on and no hints were given, but no op was "
            "coarse-tiled -- the tile search found nothing to do",
        )
        self._assert_matches_cpu(case, device, cpu)

    def _check_partial_hints_preserved(self, case: "_TilingCase", solver: str) -> None:
        """Pinned levels survive verbatim; the compiler fills in the rest."""
        cpu, device, tiling = self._compile_and_collect(
            case,
            "partial",
            case.partial_pins,
            layout_solver=solver,
            auto_tiling=True,
        )
        pinned = case.partial_expected
        self.assertTrue(tiling, "no op was coarse-tiled: the pins were dropped")
        for name, counts in sorted(tiling.items()):
            # A pin keeps the low hint id spyre_hint handed it, and the solver
            # mints its own from a reserved base above those, so a pinned level
            # always stays outside a discovered one.
            self.assertEqual(
                counts[: len(pinned)],
                pinned[: len(counts)],
                f"{name} is tiled {counts}, which does not preserve the pinned "
                f"levels {pinned}",
            )
        self.assertTrue(
            any(len(counts) > len(pinned) for counts in tiling.values()),
            f"the pins {pinned} survived but nothing was added: the tile search "
            f"left every unpinned dimension untiled ({tiling})",
        )
        self._assert_matches_cpu(case, device, cpu)

    # ------------------------------------------------------------------
    # Models.  Each returns the model, its named dims and the tiling contract,
    # defined once and reused across every hint_mode and solver.
    # ------------------------------------------------------------------
    def _softmax_case(self) -> "_TilingCase":
        """softmax(dim=0) over (512, 1024), dims R (reduced) x C.

        One level: C divided 4 ways, tiling the eight ops of the lowered
        softmax into a single group and leaving the graph output untiled.  The
        other axis, R, is the reduced one; hinting it as a second level does
        compile and is *numerically wrong* today -- the tiled max and sum drain
        through coarse_tile_combine/reduce_copy and land further from CPU than
        the output magnitude itself -- so C is the whole prescribed plan.  The
        partial mode pins that same single level; what it leaves to the
        compiler is R, plus any finer division of C.
        """
        return _TilingCase(
            body=functools.partial(torch.softmax, dim=0),
            args=(torch.rand((512, 1024), dtype=torch.float16, device=DEVICE_NAME),),
            named_dims=(["R", "C"],),
            pins=(("C", 4),),  # Reduction axis is not tiled for now
            expected=(4,),
            partial_pins=(("C", 4),),
            partial_expected=(4,),
            # A good run lands at 2e-5 on outputs of order 1/512; the
            # reduction-tiled one lands at 3e-3, and this has to separate them.
            atol=5e-4,
            rtol=0.02,
        )

    def _mlp_case(self) -> "_TilingCase":
        """Two-layer MLP (Linear -> silu -> Linear), dims S x Din x Dh x Dout.

        Two levels: S divided 2 ways outside Dout divided 2 ways.  Both are
        free (output) axes -- Din is the first GEMM's reduction and Dh the
        second's, and pinning Dh instead of Dout compiles into a
        coarse_tile_combine/reduce_copy pair whose result is two orders of
        magnitude off CPU.  The partial mode pins only S, leaving Dout for the
        compiler to find.
        """
        seq_len, in_dim, hidden_dim, out_dim = 128, 256, 1024, 256
        fc1 = torch.nn.Linear(in_dim, hidden_dim).half()
        fc2 = torch.nn.Linear(hidden_dim, out_dim).half()

        def mlp(x, w1, b1, w2, b2):
            return torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(x, w1, b1)), w2, b2
            )

        args = (
            torch.randn(seq_len, in_dim, dtype=torch.float16).to(DEVICE_NAME),
            fc1.weight.to(DEVICE_NAME),
            fc1.bias.to(DEVICE_NAME),
            fc2.weight.to(DEVICE_NAME),
            fc2.bias.to(DEVICE_NAME),
        )
        return _TilingCase(
            body=mlp,
            args=args,
            named_dims=(
                ["S", "Din"],
                ["Dh", "Din"],
                ["Dh"],
                ["Dout", "Dh"],
                ["Dout"],
            ),
            pins=(("S", 2), ("Dout", 2)),
            expected=(2, 2),
            partial_pins=(("S", 2),),
            partial_expected=(2,),
            atol=0.02,
            rtol=0.05,
        )

    def _swiglu_case(self) -> "_TilingCase":
        """SwiGLU (two parallel Linears -> silu(gate) * up), dims S x Din x Dh.

        Two levels: S divided 2 ways outside Dh divided 4 ways.  Unlike the
        MLP's, this Dh is a free axis the whole way through -- it is the N
        dimension of both GEMMs and the layout of every activation -- so the
        entire chain, both restickified weights included, lands in one
        two-level nest.  Both weights must carry the *same* label for it: name
        them apart and the inner level binds only to the gate branch, which
        still compiles and is wrong by 300x the correctly-tiled error.  The
        partial mode pins only S.
        """
        seq_len, in_dim, hidden_dim = 128, 256, 1024
        fc_gate = torch.nn.Linear(in_dim, hidden_dim).half()
        fc_up = torch.nn.Linear(in_dim, hidden_dim).half()

        def swiglu(x, w_gate, b_gate, w_up, b_up):
            gate = torch.nn.functional.linear(x, w_gate, b_gate)
            up = torch.nn.functional.linear(x, w_up, b_up)
            return torch.nn.functional.silu(gate) * up

        args = (
            torch.randn(seq_len, in_dim, dtype=torch.float16).to(DEVICE_NAME),
            fc_gate.weight.to(DEVICE_NAME),
            fc_gate.bias.to(DEVICE_NAME),
            fc_up.weight.to(DEVICE_NAME),
            fc_up.bias.to(DEVICE_NAME),
        )
        return _TilingCase(
            body=swiglu,
            args=args,
            named_dims=(
                ["S", "Din"],
                ["Dh", "Din"],
                ["Dh"],
                ["Dh", "Din"],
                ["Dh"],
            ),
            pins=(("S", 2), ("Dh", 4)),
            expected=(2, 4),
            partial_pins=(("S", 2),),
            partial_expected=(2,),
            atol=0.02,
            rtol=0.05,
        )

    # ------------------------------------------------------------------
    # Matrix
    # ------------------------------------------------------------------
    _CHECKS = {
        "hinted": _check_hints_preserved,
        "unhinted": _check_tiling_discovered,
        "partial": _check_partial_hints_preserved,
    }

    parameter_axes = {"hint_mode": tuple(_CHECKS), "solver_method": ("cpsat",)}

    # SDPA is omitted: it is the one model whose hints come from the compiler
    # using SDPA in this test suit requires resolution of
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
        if params["hint_mode"] in ("unhinted", "partial"):
            decorators.append(expected_unimplemented)
        return decorators

    def run_case(self, params: dict, factory: Callable) -> None:
        """Body of one generated method: build the model, check its contract."""
        self._CHECKS[params["hint_mode"]](self, factory(self), params["solver_method"])
