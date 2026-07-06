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

import math
from collections.abc import Sequence
from contextlib import contextmanager
import functools
import itertools
from typing import Callable, TypeVarTuple, Unpack, Optional, override

import unittest
from unittest.mock import patch
import torch

from torch._inductor import config as t_inductor_config
from torch._inductor.graph import GraphLowering

from torch_spyre._inductor.passes import CustomPreSchedulingPasses
from torch_spyre._inductor import passes
from torch_spyre._inductor import config as ts_inductor_config

try:
    from ortools.sat.python import cp_model  # noqa: F401

    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False


Ts = TypeVarTuple("Ts")

# One buffer's entry in an allocation fingerprint (keyed by buffer name):
#   (location, size_bytes, (output_splits, reduction_splits))
# where each split list is a sorted tuple of (iteration_space_stride, factor).
_Splits = tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
_AllocEntry = tuple[str, int, _Splits]


class CustomPreSchedulingPassesWithOurPasses(CustomPreSchedulingPasses):
    """torch_spyre._inductor.patches.enable_spyre_context sets
    torch._inductor.config._post_fusion_custom_pass to
    torch_spyre._inductor.passes.CustomPostFusionPasses(), so we have to monkey patch that class
    to add the ability to add custom passes."""

    test_instance: Optional["BaseTestScratchpadUsage"] = None

    @classmethod
    def initialize(cls, test_instance: "BaseTestScratchpadUsage"):
        cls.test_instance = test_instance

    @override
    def __call__(self, graph: GraphLowering) -> None:
        assert self.test_instance is not None, (
            "CustomPreSchedulingPassesWithOurPasses.test_instance must be set to an instance of "
            "BaseTestScratchpadUsage before get_passes is called"
        )
        super().__call__(graph)
        for f in self.test_instance.our_pre_scheduling_passes:
            f(graph)


class BaseTestScratchpadUsage(unittest.TestCase):
    our_pre_scheduling_passes: list[Callable[[GraphLowering], None]] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patchers = []

    def setUp(self):
        torch.manual_seed(0xAFFE)

        self.patchers.append(t_inductor_config.patch("force_disable_caches", True))
        self.patchers.append(
            ts_inductor_config.patch("allow_all_ops_in_lx_planning", True)
        )
        CustomPreSchedulingPassesWithOurPasses.initialize(self)
        self.patchers.append(
            patch.object(
                passes,
                "CustomPreSchedulingPasses",
                CustomPreSchedulingPassesWithOurPasses,
            )
        )

        for p in self.patchers:
            p.__enter__()

        torch.compiler.reset()

    def tearDown(self):
        for p in self.patchers:
            p.__exit__(None, None, None)

        torch.compiler.reset()

    def rand_device(self, shape: Sequence[int]):
        result = torch.rand(shape, dtype=torch.float16, device="spyre")
        return result

    @contextmanager
    def pre_scheduling_iterating_pass(
        self,
        f: Callable[[GraphLowering], None],
    ):
        """Context manager to add a post fusion custom pass that processes each node independently
        using `f`."""

        def new_pass(graph: GraphLowering) -> None:
            f(graph)

        self.our_pre_scheduling_passes.append(new_pass)
        yield
        self.our_pre_scheduling_passes.remove(new_pass)

    def compile_and_collect_mem_usage(
        self, f: Callable[[Unpack[Ts]], torch.Tensor], args: tuple[Unpack[Ts]]
    ) -> tuple[torch.Tensor, dict[str, str]]:
        mem_usages = {}

        def visitor(graph: GraphLowering) -> None:
            nonlocal mem_usages
            operations = graph.operations
            for op in operations:
                buf_name = op.name
                buffer = graph.get_buffer(buf_name)
                layout = buffer.get_layout()
                device_layout = layout.device_layout
                allocation = getattr(layout, "allocation", {})
                mem_usages[buf_name] = {
                    "location": "LX" if "lx" in allocation else "HBM",
                    "size": math.prod(device_layout.device_size[:-1]) * 128,
                }

        with self.pre_scheduling_iterating_pass(visitor):
            compiled_kernel = torch.compile(f, fullgraph=True)
            result = compiled_kernel(*args).to("cpu")

        return (result, mem_usages)

    def measure_hbm_transfers(
        self, model: Callable[[Unpack[Ts]], torch.Tensor], args: tuple[Unpack[Ts]]
    ) -> tuple[torch.Tensor | None, int]:
        """Compile ``model`` and return ``(result, hbm_bytes)``, where
        ``hbm_bytes`` is the total size of all HBM-resident buffers. LX-resident
        buffers are treated as free."""
        result, mem_usages = self.compile_and_collect_mem_usage(model, args)
        hbm_transfers = sum(
            mem_usage["size"]
            for mem_usage in mem_usages.values()
            if mem_usage["location"] == "HBM"
        )
        return (result, hbm_transfers)

    def assert_uses_lx(self, mem_usages: dict[str, dict]) -> None:
        """Assert the allocator placed at least one buffer in LX."""
        self.assertTrue(
            any(mem_usage["location"] == "LX" for mem_usage in mem_usages.values()),
            "Expected at least one buffer to be allocated in LX, but none were",
        )

    def run_case(self, params: dict, factory: Callable) -> None:
        """Body for one metaclass-generated parameterized case. Overridden by
        classes using ``_ParameterizedScratchpadMeta``: ``params`` is the config
        combo (empty when the class has no ``parameter_axes``) and
        ``factory(self) -> (model, args, kwargs)``."""
        raise NotImplementedError

    def run_test(
        self,
        model: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
        **kwargs,
    ):
        """Run the current class's test procedure on the given model and arguments. Override this
        in each subclass."""
        cpu_result = model(*(t.to("cpu") for t in args))

        with ts_inductor_config.patch(lx_planning=True):
            device_result, mem_usages = self.compile_and_collect_mem_usage(model, args)

        self.assert_uses_lx(mem_usages)

        atol = kwargs.get("atol", 1e-4)
        rtol = kwargs.get("rtol", 1e-5)
        self.assertTrue(
            torch.allclose(cpu_result, device_result, atol=atol, rtol=rtol),
            "Results do not match",
        )

    def _simple_mlp(
        self,
    ) -> tuple[Callable[..., torch.Tensor], tuple[torch.Tensor, ...]]:
        """Two-layer linear MLP matching ``SimpleMLP`` from the provenance
        example: ``nn.Linear -> silu -> nn.Linear``.
        """
        seq_len, in_dim, hidden_dim, out_dim = 128, 256, 1024, 256
        fc1 = torch.nn.Linear(in_dim, hidden_dim).half()
        fc2 = torch.nn.Linear(hidden_dim, out_dim).half()

        def mlp(x, w1, b1, w2, b2):
            return torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(x, w1, b1)), w2, b2
            )

        x = torch.randn(seq_len, in_dim, dtype=torch.float16).to("spyre")
        args = (
            x,
            fc1.weight.to("spyre"),
            fc1.bias.to("spyre"),
            fc2.weight.to("spyre"),
            fc2.bias.to("spyre"),
        )
        return mlp, args


class _ParameterizedScratchpadMeta(type):
    """Data-driven metaclass that expands a model list (and, optionally, a
    cartesian product of config axes) into one test *method* per case on a
    single collected class.

    A class carrying ``parameter_models`` (the ``(label, factory)`` list) gets a
    ``test_<label>`` method per model. If it also carries ``parameter_axes``
    (axis name -> values), it gets a ``test_<label>__<combo>`` method per
    ``(model, config-combo)`` instead. Each generated method delegates to the
    class's ``run_case(self, params, factory)`` — so the per-case body (apply
    the combo and check correctness, compare HBM off vs on, ...) is defined by
    the class, not baked into the metaclass.

    A class may also define a static ``case_decorators(params) -> list`` hook.
    Each decorator it returns is applied to the generated method for that combo
    (e.g. mark the ``cpsat`` combos ``expectedFailure``). Absent hook -> no
    per-case decoration.

    Generating methods rather than sibling classes keeps everything in the
    ``attrs`` dict handed to ``__new__`` — no module-namespace or ``sys.modules``
    access, so it is immune to the OOT runner's out-of-``sys.modules``
    pre-import.
    """

    # How each axis renders into the test-id suffix. Axes not listed fall back to
    # "<name><value>"; this keeps the curated short labels while letting new axes
    # added to ``parameter_axes`` work without editing this method.
    _AXIS_LABELS = {
        "solver_method": lambda v: str(v),
        "sencores": lambda v: f"sc{v}",
        "boundary_clones": lambda v: "clones" if v else "noclones",
    }

    @staticmethod
    def _combo_suffix(params: dict) -> str:
        """Readable, test-id-safe suffix for one combo. Empty -> '' (bare name)."""
        if not params:
            return ""
        labels = _ParameterizedScratchpadMeta._AXIS_LABELS
        return "_".join(
            labels[name](value) if name in labels else f"{name}{value}"
            for name, value in params.items()
        )

    def __new__(mcs, name, bases, attrs):
        models = attrs.get("parameter_models")
        if models:
            axes = attrs.get("parameter_axes") or {}
            axis_names = list(axes)
            if axis_names:
                combos = [
                    dict(zip(axis_names, c))
                    for c in itertools.product(*(axes[a] for a in axis_names))
                ]
            else:
                combos = [{}]
            # Optional per-combo decorator hook (e.g. mark cpsat as expectedFailure).
            decorators_for = attrs.get("case_decorators")
            if isinstance(decorators_for, staticmethod):
                decorators_for = decorators_for.__func__
            for params in combos:
                suffix = mcs._combo_suffix(params)
                for label, factory in models:
                    test_name = f"test_{label}__{suffix}" if suffix else f"test_{label}"
                    test_method = mcs._make_case(params, factory)
                    if decorators_for is not None:
                        for dec in decorators_for(params):
                            test_method = dec(test_method)
                    attrs[test_name] = test_method
        return super().__new__(mcs, name, bases, attrs)

    @staticmethod
    def _make_case(params: dict, factory: Callable):
        """Build one isolated test method bound to ``params``/``factory`` via
        arguments (not loop-variable closure), so each method keeps its own
        combo. The body is the class's ``run_case``."""

        def test(self):
            self.run_case(params, factory)

        test.__doc__ = (
            f"Parameterized case under {params}." if params else ("Parameterized case.")
        )
        return test


class ParameterizedScratchpadUsage(
    BaseTestScratchpadUsage, metaclass=_ParameterizedScratchpadMeta
):
    """Full cartesian product of the scratchpad-planning configuration knobs.

    Replaces the hand-written solver-variant classes: the metaclass injects a
    ``test_<model>__<solver>_sc<n>_<clones>`` method for every model in
    ``parameter_models`` and every point in ``parameter_axes``. Edit
    ``parameter_axes`` to widen or narrow the sweep.

    The solver name is the co-optimization switch: ``greedy``/``bestfit``/
    ``firstfit`` are placement-only, while ``dfs``/``cpsat`` co-optimize core
    divisions and placement.
    """

    # Models swept by the parameterized suites, as ``(label, factory)`` where
    # ``factory(self) -> (model, args, kwargs)``. ``kwargs`` are forwarded to the
    # per-case body (e.g. relaxed tolerances for fp16 matmul). SDPA is intentionally
    # omitted — it is too slow under co-optimization.
    def _softmax_case(self):
        f = functools.partial(torch.softmax, dim=0)
        x = self.rand_device((512, 1024))
        return f, (x,), {}

    def _mlp_case(self):
        mlp, args = self._simple_mlp()
        return mlp, args, {"atol": 0.1, "rtol": 0.1}

    parameter_axes = {
        "solver_method": ("greedy", "bestfit", "firstfit", "dfs", "cpsat"),
        "sencores": (1, 32),
        "boundary_clones": (False, True),
    }

    parameter_models = (("softmax", _softmax_case), ("mlp", _mlp_case))

    def run_case(self, params: dict, factory: Callable) -> None:
        """Run ``factory``'s model for correctness under this combo, applying
        the combo's config at the test-case level (the inherited setUp only
        applies invariants)."""
        with ts_inductor_config.patch(
            layout_solver=params["solver_method"],
            sencores=params["sencores"],
            lx_boundary_clones=params["boundary_clones"],
        ):
            model, args, kwargs = factory(self)
            torch.compiler.reset()
            with ts_inductor_config.patch(lx_planning=False):
                result_without_lx, hbm_without_lx = self.measure_hbm_transfers(
                    model, args
                )
            torch.compiler.reset()
            with ts_inductor_config.patch(lx_planning=True):
                result_with_lx, hbm_with_lx = self.measure_hbm_transfers(model, args)

        self.assertLess(
            hbm_with_lx,
            hbm_without_lx,
            f"Expected LX planning to reduce HBM transfers, but it did not "
            f"({hbm_with_lx} vs {hbm_without_lx} bytes)",
        )
        # LX placement only moves buffers, so on/off should match within fp16
        # rounding (the difference is a couple of ULP). Tolerances come from the
        # model's kwargs, matching how the correctness path compares elsewhere.
        atol = kwargs.get("atol", 1e-4)
        rtol = kwargs.get("rtol", 1e-5)
        self.assertTrue(
            torch.allclose(result_without_lx, result_with_lx, atol=atol, rtol=rtol),
            "Results do not match between LX planning on and off",
        )


class TestMeasureHBMUsageCoOptimizing(BaseTestScratchpadUsage):
    """Compares HBM transfers with co-optimization off vs on.

    Co-optimization should be ≤ default on every shape, and strictly better
    where adjacent ops disagree on which iteration-space dim to split. The
    canonical case is softmax(dim=0): work_distribution picks rows for the
    pointwise ops and cols for the reductions, forcing 3 of 4 shared buffers to
    HBM by default — the DFS co-optimizing solver reconciles them and pins all 4.
    """

    @override
    def run_test(
        self,
        model: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
        strict: bool = False,
        **kwargs,
    ):
        """Compare HBM transfers with co-optimization off vs on. If
        `strict`, asserts coopt < default; otherwise coopt ≤ default. Co-opt off
        is the placement-only greedy solver; co-opt on is the DFS co-optimizing
        solver (the solver name is the co-optimization switch)."""
        # Cooptimization needs > 1 core to have anything to optimize; this class
        # applies its own config here at the test-case level.
        with ts_inductor_config.patch(sencores=4, lx_planning=True):
            with ts_inductor_config.patch(layout_solver="greedy"):
                result_default, hbm_default = self.measure_hbm_transfers(model, args)
            torch.compiler.reset()
            with ts_inductor_config.patch(layout_solver="dfs"):
                result_coopt, hbm_coopt = self.measure_hbm_transfers(model, args)

        cmp = self.assertLess if strict else self.assertLessEqual
        rel = "<" if strict else "≤"
        cmp(
            hbm_coopt,
            hbm_default,
            f"Expected cooptimization to be {rel} default HBM, got "
            f"coopt={hbm_coopt} default={hbm_default}",
        )
        self.assertTrue(
            torch.allclose(result_default, result_coopt, atol=1e-4),
            "Results do not match between cooptimization on and off",
        )

    def test_softmax_dim0_strictly_lower_hbm(self):
        """The canonical motivating case from the design doc. softmax(dim=0)
        has every adjacent op pair disagreeing on which dim to split, so
        DefaultAllocator only pins 1 of 4 shared buffers; the DFS co-optimizing
        solver should flip the pointwise ops to cols and pin all 4 → strictly
        lower HBM."""
        f = functools.partial(torch.softmax, dim=0)
        x = self.rand_device((512, 1024))
        self.run_test(f, (x,), strict=True)

    def test_softmax_dim_neg1_no_regression(self):
        """softmax(dim=-1) is the well-behaved baseline where DefaultAllocator
        already pins everything pinnable. The DFS co-optimizing solver must match
        (no regression)."""
        f = functools.partial(torch.softmax, dim=-1)
        x = self.rand_device((512, 1024))
        self.run_test(f, (x,))


class TestCloneAtGraphBoundaries(BaseTestScratchpadUsage):
    """End-to-end tests for clone insertion at graph input/output boundaries.

    The allocator now inserts clone ops on-demand inside _push_allocation rather than
    as a separate pre-scheduling pass.  These tests verify that:
    - graph inputs read by multiple ops get a clone that lands in LX
    - graph outputs that are also read inside the graph get a clone (for the HBM return
      value), while the original buffer is pinned to LX

    Enabling ``lx_boundary_clones`` flips ``clone_at_graph_boundaries()`` on and
    makes the inserted clone outputs LX-eligible, so the boundary clone path is
    exercised. This class applies that patch itself at the test-case level (in
    ``_compile_and_inspect``), so every compile here runs with it on.
    """

    def _compile_and_inspect(
        self,
        f: Callable,
        args: tuple,
    ) -> tuple:
        """Compile f, capture op count and mem_usages after the allocator runs.

        Handles both single-tensor and tuple outputs.
        Returns (result_on_cpu, n_ops, mem_usages).
        """
        n_ops_captured: list[int] = []
        mem_usages: dict[str, dict] = {}

        def visitor(graph: GraphLowering) -> None:
            n_ops_captured.append(len(graph.operations))
            for op in graph.operations:
                buf_name = op.name
                buffer = graph.get_buffer(buf_name)
                layout = buffer.get_layout()
                device_layout = layout.device_layout
                allocation = getattr(layout, "allocation", {})
                mem_usages[buf_name] = {
                    "location": "LX" if "lx" in allocation else "HBM",
                    "size": math.prod(device_layout.device_size[:-1]) * 128,
                }

        with self.pre_scheduling_iterating_pass(visitor):
            with ts_inductor_config.patch(lx_boundary_clones=True):
                compiled_kernel = torch.compile(f, fullgraph=True)
                raw = compiled_kernel(*args)
            if isinstance(raw, tuple):
                result = tuple(r.to("cpu") for r in raw)
            else:
                result = raw.to("cpu")

        n_ops = n_ops_captured[0] if n_ops_captured else 0
        return result, n_ops, mem_usages

    def test_input_clone_when_read_by_multiple_ops(self):
        """A graph input read by two different ops is cloned; the clone lands in LX."""
        x = self.rand_device((64, 1024))

        def fn(x):
            # x is consumed by both exp_op and add_op → two reads → eligible for input clone
            return torch.exp(x) + x

        with ts_inductor_config.patch(lx_planning=False):
            ref_result, n_ops_no_lx, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(lx_planning=True):
            result, n_ops_with_lx, mem_usages = self._compile_and_inspect(fn, (x,))

        self.assertGreater(
            n_ops_with_lx,
            n_ops_no_lx,
            f"Expected the input clone to add an op: {n_ops_no_lx} ops without LX, "
            f"{n_ops_with_lx} with LX",
        )
        self.assertTrue(
            any(u["location"] == "LX" for u in mem_usages.values()),
            "Expected at least one LX-allocated buffer after input cloning",
        )
        # Clone is an exact copy; LX planning must not change the numerical result.
        self.assertTrue(
            torch.equal(ref_result, result),
            "LX input clone changed the numerical result",
        )

    @unittest.skipUnless(_HAS_ORTOOLS, "joint CP-SAT boundary-clone test needs ortools")
    def test_input_clone_supported_under_joint_cpsat(self):
        """The joint CP-SAT allocator (``layout_solver="cpsat"``) inserts an
        *input* boundary clone.

        A graph input read by multiple ops is offered to the joint solver as a
        clone-eligible input (``_eligible_clone_inputs`` /
        ``_cd_input_buffers``): the solver may pin the clone into LX, growing the
        op count. This mirrors the greedy and placement-only CP-SAT paths (see
        ``test_input_clone_when_read_by_multiple_ops``). Output boundary clones
        under the joint path are still unsupported
        (``test_output_clone_unsupported_under_joint_cpsat``).

        Skipped without ortools: the allocator falls back to greedy, which also
        clones, so the assertion still holds but exercises a different path.
        """
        x = self.rand_device((64, 1024))

        def fn(x):
            # x read by both exp and add -> eligible for an input clone.
            return torch.exp(x) + x

        with ts_inductor_config.patch(lx_planning=False):
            _, n_ops_no_lx, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(
            lx_planning=True,
            layout_solver="cpsat",
        ):
            _, n_ops_with_lx, _ = self._compile_and_inspect(fn, (x,))

        # The joint CP-SAT path now inserts the input boundary clone, growing the
        # op count (see the sibling greedy / placement-only test).
        self.assertGreater(
            n_ops_with_lx,
            n_ops_no_lx,
            "joint CP-SAT did not insert the expected input boundary clone "
            f"({n_ops_no_lx} -> {n_ops_with_lx} ops)",
        )

    @unittest.skipUnless(_HAS_ORTOOLS, "joint CP-SAT boundary-clone test needs ortools")
    def test_output_clone_supported_under_joint_cpsat(self):
        """The joint CP-SAT allocator (``layout_solver="cpsat"``) inserts an
        *output* boundary clone.

        Output-side counterpart to
        ``test_input_clone_supported_under_joint_cpsat``. A buffer that is both a
        graph output and read inside the graph may now reside in LX
        (``_residency_reason`` allows graph outputs once
        ``clone_at_graph_boundaries`` is on); ``_push_allocation`` then clones it
        as the actual HBM value returned to the caller. The op count grows and the
        result must be unchanged -- a wrong HBM return is a silent miscompile.
        Mirrors the greedy / placement-only paths (see
        ``test_output_clone_when_intermediate_is_also_graph_output``).

        Skipped without ortools: the allocator falls back to greedy, which also
        clones, so the assertions still hold but exercise a different path.
        """
        x = self.rand_device((64, 1024))

        def fn(x):
            # y = exp(x) is both a graph output and read by add_op -> eligible
            # for an output clone.
            y = torch.exp(x)
            z = y + 1
            return y, z

        with ts_inductor_config.patch(lx_planning=False):
            (ref_y, ref_z), n_ops_no_lx, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(
            lx_planning=True,
            layout_solver="cpsat",
        ):
            (res_y, res_z), n_ops_with_lx, _ = self._compile_and_inspect(fn, (x,))

        # The joint CP-SAT path now inserts the output boundary clone, growing the
        # op count (see the sibling greedy / placement-only test).
        self.assertGreater(
            n_ops_with_lx,
            n_ops_no_lx,
            "joint CP-SAT did not insert the expected output boundary clone "
            f"({n_ops_no_lx} -> {n_ops_with_lx} ops)",
        )
        # The cloned HBM return must match the un-pinned result exactly.
        self.assertTrue(
            torch.equal(ref_y, res_y), "LX output clone changed the returned y"
        )
        self.assertTrue(
            torch.equal(ref_z, res_z), "LX output clone changed the returned z"
        )

    def test_output_clone_when_intermediate_is_also_graph_output(self):
        """A buffer that is both a graph output and read inside the graph is pinned to LX;
        a clone of it is inserted as the actual (HBM) graph output returned to the caller."""
        x = self.rand_device((64, 1024))

        def fn(x):
            # After CSE, y = exp(x) is produced once.
            # y is a graph output AND is read by add_op → eligible for output clone.
            y = torch.exp(x)
            z = y + 1  # add_op reads y
            return y, z

        with ts_inductor_config.patch(lx_planning=False):
            (ref_y, ref_z), n_ops_no_lx, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(lx_planning=True):
            (result_y, result_z), n_ops_with_lx, mem_usages = self._compile_and_inspect(
                fn, (x,)
            )

        self.assertGreater(
            n_ops_with_lx,
            n_ops_no_lx,
            f"Expected the output clone to add an op: {n_ops_no_lx} ops without LX, "
            f"{n_ops_with_lx} with LX",
        )
        self.assertTrue(
            any(u["location"] == "LX" for u in mem_usages.values()),
            "Expected at least one LX-allocated buffer after output cloning",
        )
        # Clone is an exact copy; LX planning must not change the numerical result.
        self.assertTrue(
            torch.equal(ref_y, result_y), "LX output clone changed result y"
        )
        self.assertTrue(
            torch.equal(ref_z, result_z), "LX output clone changed result z"
        )

    def test_input_read_at_multiple_offsets_is_correct(self):
        """A graph input read by one op at two distinct offsets must not be
        LX-pinned.

        An LX-pinned buffer is addressed by a single base (SDSC start_address
        = allocation["lx"]); per-access slice offsets are not folded into it.
        Pinning ``x`` for ``x[:, 0:512] + x[:, 512:1024]`` made both reads
        resolve to the LX base, so the op computed ``x0 + x0`` instead of
        ``x0 + x1``. The allocator now skips such inputs (they stay in HBM,
        where multi-offset reads work)."""
        x = self.rand_device((64, 1024))

        def fn(x):
            # The fused add reads x at offset 0 and offset 512 -> two distinct
            # offsets on the same buffer -> ineligible for LX pinning.
            return x[:, 0:512] + x[:, 512:1024]

        with ts_inductor_config.patch(lx_planning=False):
            ref, _, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(lx_planning=True):
            result, _, _ = self._compile_and_inspect(fn, (x,))

        self.assertTrue(
            torch.equal(ref, result),
            "Multi-offset input read produced wrong values under LX planning",
        )

    def test_input_feeding_reduction_is_cloned_and_correct(self):
        """A graph input read by a reduction is LX-cloned, with the clone's
        per-core split re-keyed correctly.

        push_allocation_with_clone re-keys the consumer's op_it_space_splits
        through the buffer's strides before assigning them to the clone. A reduction consumer's split is keyed to its
        reduced-shape output; copied verbatim it would split the wrong axis of
        the full-shape clone (wrong values / SDSC abort at multi-core). The
        numerical failure only manifests when work is split across cores; here
        (sencores=1) we assert the clone is inserted and the result is correct.
        Multi-core numerical coverage lives in
        tests/inductor/test_inductor_ops.py (max_sub_broadcast, aminmax,
        softmax)."""
        x = self.rand_device((64, 256))

        def fn(x):
            # x feeds the max reduction (and the sub) -> reduction consumer.
            return x - torch.unsqueeze(torch.max(x, dim=1).values, dim=1)

        with ts_inductor_config.patch(lx_planning=False):
            ref, n_ops_no_lx, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(lx_planning=True):
            result, n_ops_with_lx, mem_usages = self._compile_and_inspect(fn, (x,))

        self.assertGreater(
            n_ops_with_lx,
            n_ops_no_lx,
            "Expected a boundary clone for the reduction-fed input, but the op "
            f"count did not grow ({n_ops_no_lx} -> {n_ops_with_lx})",
        )
        self.assertTrue(
            any(u["location"] == "LX" for u in mem_usages.values()),
            "Expected at least one LX-allocated buffer for the reduction input",
        )
        self.assertTrue(
            torch.equal(ref, result),
            "Reduction-fed input changed result under LX planning",
        )

    def test_input_read_partially_is_correct(self):
        """A graph input read only over a sub-extent (a slice) must not be
        LX-pinned.

        Strided partial reads of a multi-dim LX buffer mis-address against the
        single LX base. Pinning ``x`` for ``add(x[:, :, 0:64].clone(),
        x[:, :, 0:64])`` produced wrong values; the allocator now leaves such
        inputs in HBM, where partial reads work."""
        x = self.rand_device((3, 3, 192))

        def fn(x):
            s = x[:, :, 0:64]  # partial inner-dim slice -> sub-extent read
            return torch.add(s.clone(), s)

        with ts_inductor_config.patch(lx_planning=False):
            ref, _, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(lx_planning=True):
            result, _, _ = self._compile_and_inspect(fn, (x,))

        self.assertTrue(
            torch.equal(ref, result),
            "Partial input read produced wrong values under LX planning",
        )


@unittest.skipUnless(_HAS_ORTOOLS, "ortools not installed")
class TestCpSatCloneAtGraphBoundaries(TestCloneAtGraphBoundaries):
    """Re-run the boundary-clone tests through the CP-SAT joint allocator."""

    @override
    def setUp(self):
        self.patchers.append(ts_inductor_config.patch("layout_solver", "cpsat"))
        super().setUp()

    def test_input_cloned_under_multicore_split(self):
        """At ``sencores=32`` the input clone's per-core split must be re-keyed
        consistently for every consumer, which the joint solver enforces by
        forcing one shared slicing (or leaving the input in HBM). Asserts the
        input is pinned and the multi-core result matches CPU -- the case the
        placement path cannot co-optimize when consumers' committed divisions
        disagree.
        """
        x = self.rand_device((512, 1024))

        def fn(x):
            # x is read by exp and by the trailing add -> two reads -> eligible.
            return torch.exp(x) + x

        cpu_result = fn(x.to("cpu"))

        with ts_inductor_config.patch(lx_planning=True):
            with ts_inductor_config.patch(sencores=32):
                result, mem_usages = self.compile_and_collect_mem_usage(fn, (x,))

        self.assertTrue(
            any(u["location"] == "LX" for u in mem_usages.values()),
            "Expected an LX-pinned buffer (the input clone) at multi-core",
        )
        torch.testing.assert_close(
            result,
            cpu_result,
            atol=0.1,
            rtol=0.1,
            msg="multi-core input clone miscompiled — is the split re-keyed?",
        )

    def test_input_feeding_reduction_multicore_is_correct(self):
        """The dangerous case the clone re-keying exists for: an input that feeds
        a *reduction* at ``sencores=32``, where the consumer's output space (the
        reduced shape) differs from its read space on the input. The reduction
        consumer's split is keyed to its reduced-shape output; the clone (a
        full-shape identity copy) must re-key that split onto the input's own axes
        or it slices the wrong axis (wrong values / SDSC abort at multi-core). The
        joint solver also has to converge the reduction and the pointwise consumer
        on one shared slicing for the clone to pin at all.
        """
        x = self.rand_device((512, 256))

        def fn(x):
            # x feeds the dim=1 max reduction (output (512,)) and the subtract
            # (output (512, 256)) -> the two consumers read x in different spaces.
            return x - torch.unsqueeze(torch.max(x, dim=1).values, dim=1)

        cpu_result = fn(x.to("cpu"))

        with ts_inductor_config.patch(lx_planning=True):
            with ts_inductor_config.patch(sencores=32):
                result, mem_usages = self.compile_and_collect_mem_usage(fn, (x,))

        torch.testing.assert_close(
            result,
            cpu_result,
            atol=0.1,
            rtol=0.1,
            msg="reduction-fed input clone miscompiled at multi-core — is the "
            "consumer's reduced-output split re-keyed onto the input axes?",
        )

    def test_output_cloned_under_multicore_split(self):
        """Output-side counterpart to ``test_input_cloned_under_multicore_split``:
        at ``sencores=32`` a buffer that is both a graph output and read inside the
        graph is pinned to LX and its HBM return is a clone, materialized against
        the solver's committed per-core split. Asserts the buffer is pinned and
        both returned values match CPU -- the clone must be re-keyed to the same
        multi-core division, or the HBM return is miscompiled.
        """
        x = self.rand_device((512, 1024))

        def fn(x):
            # y is a graph output AND read by the trailing add -> output clone.
            y = torch.exp(x)
            z = y + 1
            return y, z

        cpu_y, cpu_z = fn(x.to("cpu"))

        # setUp patches layout_solver="cpsat", which routes to the joint
        # allocator. sencores=32 forces the multi-core split the clone re-keys to.
        with ts_inductor_config.patch(
            lx_planning=True,
            sencores=32,
        ):
            (res_y, res_z), _, mem_usages = self._compile_and_inspect(fn, (x,))

        self.assertTrue(
            any(u["location"] == "LX" for u in mem_usages.values()),
            "Expected an LX-pinned buffer (the cloned graph output) at multi-core",
        )
        torch.testing.assert_close(
            res_y,
            cpu_y,
            atol=0.1,
            rtol=0.1,
            msg="multi-core output clone (y) miscompiled",
        )
        torch.testing.assert_close(
            res_z,
            cpu_z,
            atol=0.1,
            rtol=0.1,
            msg="multi-core output clone (z) miscompiled",
        )


class TestIntermediatePartialReadNotPinned(BaseTestScratchpadUsage):
    """An *intermediate* buffer read partially (sliced) must not be LX-pinned.

    Companion to ``TestCloneAtGraphBoundaries``, which guards graph
    input/output clones. ``_filter_ops`` applies the same
    ``buffer_not_read_in_full`` guard to intermediate buffers: a buffer that is
    produced in full and then read over a sub-extent (an inner-dim slice that
    feeds a chained op) would be LX-pinned and mis-addressed by the single-base
    LX path. Without the intermediate guard this regresses to a large
    numerical mismatch (~94%).
    """

    def test_sliced_intermediate_is_correct(self):
        # Both leading dims large so the chained ops divide cleanly across
        # cores (no core-division mismatch) — the case that would otherwise
        # LX-pin the sliced intermediate. allow_all_ops_in_lx_planning makes
        # the intermediate LX-eligible; sencores=32 gives the multi-core split.
        x = self.rand_device((128, 192, 256))

        def fn(x):
            t = torch.exp(x)  # full intermediate, produced once
            s = t[:, :, 32:96]  # sub-stick partial read of the intermediate
            return s.clone() + s

        cpu_result = fn(x.to("cpu"))

        with ts_inductor_config.patch(lx_planning=True):
            with ts_inductor_config.patch(allow_all_ops_in_lx_planning=True):
                with ts_inductor_config.patch(sencores=32):
                    result, mem_usages = self.compile_and_collect_mem_usage(fn, (x,))

        # The scenario must still exercise LX-pinning, else it would pass
        # trivially without covering the guard.
        self.assertTrue(
            any(u["location"] == "LX" for u in mem_usages.values()),
            "Expected at least one LX-allocated buffer in this scenario",
        )
        torch.testing.assert_close(
            result,
            cpu_result,
            atol=0.1,
            rtol=0.1,
            msg="sliced intermediate miscompiled — is the _filter_ops guard present?",
        )


@unittest.skipUnless(_HAS_ORTOOLS, "ortools not installed")
class TestCpSatAllocatorIntegration(BaseTestScratchpadUsage):
    """Real-graph coverage for CoOptimizingAllocator.
    Patching layout_solver="cpsat" routes _maybe_scratchpad_planning to
    CoOptimizingAllocator, puts a compiled graph through the
    allocator's translation layer (_division_map /
    _enumerate_core_divisions / _cd_parent_matches / _build_cd_bound_buffers /
    _residency_by_buf) and _commit_divisions.
    """

    def test_pointwise_reduction_chain_cpsat(self):
        # Pointwise producer -> reduction consumer: exercises both branches of
        # _enumerate_core_divisions and a real producer->consumer match edge.
        def model(a, b):
            return (a + b).sum(dim=0, keepdim=True)

        a = self.rand_device((512, 1024))
        b = self.rand_device((512, 1024))
        cpu_result = model(a.to("cpu"), b.to("cpu"))

        mem_usages: dict[str, str] = {}
        splits: dict[str, tuple] = {}

        def visitor(graph: GraphLowering) -> None:
            for op in graph.operations:
                layout = graph.get_buffer(op.name).get_layout()
                allocation = getattr(layout, "allocation", {})
                mem_usages[op.name] = "LX" if "lx" in allocation else "HBM"
                splits[op.name] = getattr(op, "op_it_space_splits", ({}, {}))

        with ts_inductor_config.patch(sencores=32):
            # layout_solver="cpsat" routes to CoOptimizingAllocator (see class
            # docstring); the solver name is the co-optimization switch.
            with ts_inductor_config.patch(layout_solver="cpsat"):
                with ts_inductor_config.patch(lx_planning=True):
                    with self.pre_scheduling_iterating_pass(visitor):
                        compiled = torch.compile(model, fullgraph=True)
                        device_result = compiled(a, b).to("cpu")

        # 1. Correctness end-to-end through the glue.
        torch.testing.assert_close(
            device_result,
            cpu_result,
            atol=0.1,
            rtol=0.1,
            msg="cpsat-allocated result diverged from CPU",
        )
        # 2. Residency: rules out an all-spilled degenerate plan.
        self.assertTrue(
            any(loc == "LX" for loc in mem_usages.values()),
            f"expected >=1 LX buffer under cpsat, got {mem_usages}",
        )

        # 3. _commit_divisions wrote a multi-core split onto at least one op.
        def cores(s):
            out, red = s
            return math.prod(out.values() or [1]) * math.prod(red.values() or [1])

        self.assertTrue(
            any(cores(s) > 1 for s in splits.values()),
            f"_commit_divisions never committed a multi-core split: {splits}",
        )


class TestCpSatAllocatorFallback(BaseTestScratchpadUsage):
    """When ``layout_solver="cpsat"`` is selected but ortools is unavailable, the
    CoOptimizingAllocator falls back to the pure-Python DFS co-optimizing solver
    (still co-optimizing, not placement-only). The compile must still succeed,
    still place a buffer in LX, and still match CPU -- which exercises the fallback
    through the real pipeline rather than asserting on it directly.
    """

    @contextmanager
    def _ortools_absent(self):
        """Force the missing-ortools condition: CpSatLayoutSolver.__init__ raises
        ImportError (so the allocator falls back) exactly when cp_model is None,
        which is how a real missing install presents."""
        from torch_spyre._inductor.scratchpad import ilp_solver_ortools

        saved = ilp_solver_ortools.cp_model
        ilp_solver_ortools.cp_model = None
        try:
            yield
        finally:
            ilp_solver_ortools.cp_model = saved

    @override
    def run_test(self, model, args, **kwargs):
        cpu_result = model(*(t.to("cpu") for t in args))

        with self._ortools_absent():
            with ts_inductor_config.patch(layout_solver="cpsat"):
                with ts_inductor_config.patch(lx_planning=True):
                    device_result, mem_usages = self.compile_and_collect_mem_usage(
                        model, args
                    )

        # The greedy fallback still pins to LX -- a degenerate all-HBM result
        # would mean the fallback never ran (or did nothing useful).
        self.assertTrue(
            any(mem_usage["location"] == "LX" for mem_usage in mem_usages.values()),
            f"expected the greedy fallback to still use LX, got {mem_usages}",
        )

        atol = kwargs.get("atol", 1e-4)
        self.assertTrue(
            torch.allclose(cpu_result, device_result, atol=atol), "Results do not match"
        )


class TestSelectAllocator(unittest.TestCase):
    """select_allocator maps config -> (allocator, solver) so the allocators
    never inspect config themselves. Pure dispatch, no device needed."""

    def test_dispatch_by_config(self):
        from torch_spyre._inductor.scratchpad.allocator import (
            CoOptimizingAllocator,
            DefaultAllocator,
            select_allocator,
        )
        from torch_spyre._inductor.scratchpad.plan_solver import GreedyLayoutSolver
        from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
            BestFitLayoutSolver,
        )
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            CpSatLayoutSolver,
        )
        from torch_spyre._inductor.scratchpad.dfs_solver import DfsLayoutSolver

        # Placement-only solvers -> DefaultAllocator.
        with ts_inductor_config.patch(layout_solver="greedy"):
            a = select_allocator()
            self.assertIs(type(a), DefaultAllocator)
            self.assertIsInstance(a.layout_planning, GreedyLayoutSolver)

        with ts_inductor_config.patch(layout_solver="bestfit"):
            a = select_allocator()
            self.assertIs(type(a), DefaultAllocator)
            self.assertIsInstance(a.layout_planning, BestFitLayoutSolver)

        # dfs -> co-optimizing CoOptimizingAllocator with a DFS solver.
        with ts_inductor_config.patch(layout_solver="dfs"):
            a = select_allocator()
            self.assertIs(type(a), CoOptimizingAllocator)
            self.assertIsInstance(a.layout_planning, DfsLayoutSolver)

        # cpsat -> co-optimizing CoOptimizingAllocator; its solver is CP-SAT when
        # ortools is present, else the pure-Python DFS fallback.
        with ts_inductor_config.patch(layout_solver="cpsat"):
            a = select_allocator()
            self.assertIs(type(a), CoOptimizingAllocator)
            if _HAS_ORTOOLS:
                self.assertIsInstance(a.layout_planning, CpSatLayoutSolver)
            else:
                self.assertIsInstance(a.layout_planning, DfsLayoutSolver)

        with ts_inductor_config.patch(layout_solver="bogus"):
            with self.assertRaises(ValueError):
                select_allocator()


if __name__ == "__main__":
    unittest.main()
