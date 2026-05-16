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


from unittest import TestCase
from unittest.mock import MagicMock, patch

from torch._inductor.graph import GraphLowering

from torch_spyre._inductor.scratchpad.allocator import (
    DefaultAllocator,
    ScratchpadAllocator,
)
from torch_spyre._inductor.scratchpad.passes import CloneInputNodesPass
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad.utils import (
    calculate_buffer_statistics,
    calculate_liveness,
    get_buffer_users,
    get_ncores_for_buffers,
    mem_usage_by_op,
)


def _make_buf(num_sticks: int, allocation: dict | None = None) -> MagicMock:
    """Return a mock buffer whose device_layout.device_size encodes stick count."""
    buf = MagicMock()
    buf.layout.device_layout.device_size = [num_sticks, 1]
    buf.layout.allocation = {} if allocation is None else allocation
    return buf


def _make_dep(name: str) -> MagicMock:
    dep = MagicMock()
    dep.name = name
    return dep


def _make_op(
    name: str,
    read_deps: list,
    write_deps: list,
    splits: list | None = None,
) -> MagicMock:
    _spec = ["name", "get_read_writes", "get_read_names"]
    if splits is not None:
        _spec.append("op_it_space_splits")
    op = MagicMock(spec=_spec)
    op.name = name
    rw = MagicMock()
    rw.reads = set(read_deps)
    rw.writes = set(write_deps)
    op.get_read_writes.return_value = rw
    op.get_read_names.return_value = [d.name for d in read_deps]
    if splits is not None:
        op.op_it_space_splits = splits
    return op


class TestAllocationUtilities(TestCase):
    def test_get_buffer_users(self):
        buf_a, buf_b, buf_c = _make_dep("buf_a"), _make_dep("buf_b"), _make_dep("buf_c")
        op0 = _make_op("op0", [buf_a], [buf_b])
        op1 = _make_op("op1", [buf_b], [buf_c])
        mock_graph = MagicMock(spec=GraphLowering)
        mock_graph.operations = [op0, op1]

        result = get_buffer_users(mock_graph)

        self.assertEqual(result["buf_a"], [op0])
        self.assertEqual(result["buf_b"], [op0, op1])
        self.assertEqual(result["buf_c"], [op1])

    def test_calculate_buffer_statistics(self):
        buf_a, buf_b = _make_dep("buf_a"), _make_dep("buf_b")
        op0 = _make_op("op0", [buf_a], [buf_b])
        mock_graph = MagicMock(spec=GraphLowering)
        mock_graph.operations = [op0]

        result = calculate_buffer_statistics(mock_graph)

        self.assertEqual(result["buf_a"], {"reads": 1, "writes": 0})
        self.assertEqual(result["buf_b"], {"reads": 0, "writes": 1})

    def test_calculate_buffer_statistics_shared_buffer(self):
        buf_a = _make_dep("buf_a")
        op0 = _make_op("op0", [], [buf_a])
        op1 = _make_op("op1", [buf_a], [])
        mock_graph = MagicMock(spec=GraphLowering)
        mock_graph.operations = [op0, op1]

        result = calculate_buffer_statistics(mock_graph)

        self.assertEqual(result["buf_a"], {"reads": 1, "writes": 1})


class TestDefaultAllocator(TestCase):
    def _make_allocator(
        self,
        layout_planning=None,
        pre_optimization_passes=None,
        post_optimization_passes=None,
    ):
        if layout_planning is None:
            layout_planning = MagicMock()
        kwargs = {}
        if pre_optimization_passes is not None:
            kwargs["pre_optimization_passes"] = pre_optimization_passes
        if post_optimization_passes is not None:
            kwargs["post_optimization_passes"] = post_optimization_passes
        return DefaultAllocator(layout_planning, **kwargs)

    # --- plan_allocation ---

    @patch.object(DefaultAllocator, "_push_allocation")
    @patch.object(DefaultAllocator, "_generate_buffers")
    def test_plan_allocation_call_order(self, mock_generate, mock_push):
        call_order = []
        buf0 = LifetimeBoundBuffer("buf0", 1024, 0, 2)
        buf1 = LifetimeBoundBuffer("buf1", 512, 1, 3)
        planned = [buf0, buf1]

        def _generate(g):
            call_order.append("generate")
            return [buf0, buf1]

        def _push(g, b):
            call_order.append("push")

        def _solve(b):
            call_order.append("solve")
            return planned

        mock_generate.side_effect = _generate
        mock_push.side_effect = _push

        solver = MagicMock()
        solver.plan_layout.side_effect = _solve

        graph = MagicMock(spec=GraphLowering)
        # Satisfy the ComputedBuffer / MutationLayout guards in plan_allocation.
        from torch._inductor.ir import ComputedBuffer, FixedLayout

        op = MagicMock(spec=ComputedBuffer)
        op.layout = MagicMock(spec=FixedLayout)
        graph.operations = [op]
        pass1, pass2 = MagicMock(), MagicMock()
        pass1.apply_pass.side_effect = lambda g: call_order.append("pre")
        pass2.apply_pass.side_effect = lambda g: call_order.append("post")

        DefaultAllocator(
            solver,
            pre_optimization_passes=[pass1],
            post_optimization_passes=[pass2],
        ).plan_allocation(graph)

        self.assertEqual(call_order, ["pre", "generate", "solve", "push", "post"])
        mock_generate.assert_called_once_with(graph)
        solver.plan_layout.assert_called_once_with([buf0, buf1])
        mock_push.assert_called_once_with(graph, planned)
        pass1.apply_pass.assert_called_once_with(graph)
        pass2.apply_pass.assert_called_once_with(graph)


class _ConcreteAllocator(ScratchpadAllocator):
    """Minimal concrete subclass for testing ScratchpadAllocator base methods."""

    def plan_allocation(self, graph):
        pass


class TestScratchpadAllocatorBase(TestCase):
    def setUp(self):
        self.allocator = _ConcreteAllocator()

    # --- _op_good_for_lx_inplace ---

    def test_op_good_for_lx_inplace_allowed(self):
        for name in ("exp", "sub"):
            with self.subTest(name=name):
                self.assertTrue(self.allocator._op_good_for_lx_inplace(name))

    def test_op_good_for_lx_inplace_disallowed(self):
        for name in ("max", "sum", "clone", "mul"):
            with self.subTest(name=name):
                self.assertFalse(self.allocator._op_good_for_lx_inplace(name))

    # --- _push_allocation ---

    def test_push_allocation_with_address(self):
        ADDRESS = 0x1000
        mock_graph = MagicMock(spec=GraphLowering)
        mock_layout = MagicMock()
        mock_layout.allocation = {}
        mock_graph.get_buffer.return_value.get_layout.return_value = mock_layout

        self.allocator._push_allocation(
            mock_graph, [LifetimeBoundBuffer("buf0", 1024, 0, 2, address=ADDRESS)]
        )

        mock_graph.get_buffer.assert_called_once_with("buf0")
        self.assertEqual(mock_layout.allocation["lx"], ADDRESS)

    def test_push_allocation_skips_none_address(self):
        mock_graph = MagicMock(spec=GraphLowering)
        self.allocator._push_allocation(
            mock_graph, [LifetimeBoundBuffer("buf0", 1024, 0, 2, address=None)]
        )
        mock_graph.get_buffer.assert_not_called()

    # --- calculate_liveness ---

    def test_calculate_liveness_sets_start_and_end(self):
        dep_a, dep_b, dep_c = _make_dep("a"), _make_dep("b"), _make_dep("c")
        op0 = _make_op("op0", [dep_a], [dep_b])
        op1 = _make_op("op1", [dep_b], [dep_c])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op0, op1]
        mem_usage: dict = {"a": {}, "b": {}, "c": {}}

        calculate_liveness(mem_usage, graph.operations)

        self.assertEqual(mem_usage["a"]["liveness_start"], 0)
        self.assertEqual(mem_usage["a"]["liveness_end"], 1)
        self.assertEqual(mem_usage["b"]["liveness_start"], 0)
        self.assertEqual(mem_usage["b"]["liveness_end"], 2)
        self.assertEqual(mem_usage["c"]["liveness_start"], 1)
        self.assertEqual(mem_usage["c"]["liveness_end"], 2)

    def test_calculate_liveness_single_op(self):
        dep_a, dep_b = _make_dep("a"), _make_dep("b")
        op0 = _make_op("op0", [dep_a], [dep_b])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op0]
        mem_usage: dict = {"a": {}, "b": {}}

        calculate_liveness(mem_usage, graph.operations)

        self.assertEqual(mem_usage["a"]["liveness_start"], 0)
        self.assertEqual(mem_usage["a"]["liveness_end"], 1)
        self.assertEqual(mem_usage["b"]["liveness_start"], 0)
        self.assertEqual(mem_usage["b"]["liveness_end"], 1)

    # --- _filter_buffers ---

    @patch.object(
        ScratchpadAllocator, "_op_output_good_for_lx_reuse", return_value=False
    )
    def test_filter_buffers_drops_disallowed_op_outputs(self, _mock_reuse):
        bad_dep = _make_dep("bad")
        bad_op = _make_op("bad", [], [bad_dep])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [bad_op]
        graph.get_output_names.return_value = []
        graph.graph_input_names = []
        bufs = [LifetimeBoundBuffer("bad", 100, 0, 1)]

        result = self.allocator._filter_buffers(graph, bufs, None)

        self.assertEqual(result, [])

    def test_filter_buffers_drops_graph_outputs(self):
        graph = MagicMock(spec=GraphLowering)
        graph.operations = []
        graph.get_output_names.return_value = ["out"]
        graph.graph_input_names = []
        result = self.allocator._filter_buffers(
            graph, [LifetimeBoundBuffer("out", 100, 0, 1)], None
        )
        self.assertEqual(result, [])

    def test_filter_buffers_drops_graph_inputs(self):
        graph = MagicMock(spec=GraphLowering)
        graph.operations = []
        graph.get_output_names.return_value = []
        graph.graph_input_names = ["inp"]
        result = self.allocator._filter_buffers(
            graph, [LifetimeBoundBuffer("inp", 100, 0, 1)], None
        )
        self.assertEqual(result, [])

    @patch.object(
        ScratchpadAllocator, "_op_output_good_for_lx_reuse", return_value=True
    )
    def test_filter_buffers_retains_eligible(self, _mock_reuse):
        ok_dep = _make_dep("ok")
        ok_op = _make_op("ok", [], [ok_dep])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [ok_op]
        graph.get_output_names.return_value = []
        graph.graph_input_names = []
        bufs = [LifetimeBoundBuffer("ok", 100, 0, 1)]

        result = self.allocator._filter_buffers(graph, bufs, None)

        self.assertEqual(result, bufs)

    # --- _generate_buffers ---

    @patch.object(ScratchpadAllocator, "_filter_buffers")
    @patch.object(ScratchpadAllocator, "_build_bound_buffers")
    @patch.object(ScratchpadAllocator, "_determine_in_place")
    def test_generate_buffers_chains_determine_build_filter(
        self, m_ip, m_build, m_filter
    ):
        graph = MagicMock(spec=GraphLowering)
        graph.operations = []
        in_place = {"b0": ["b1"]}
        raw = [LifetimeBoundBuffer("b0", 100, 0, 1)]
        filtered = [LifetimeBoundBuffer("b0", 100, 0, 1)]
        m_ip.return_value = in_place
        m_build.return_value = raw
        m_filter.return_value = filtered

        result = self.allocator._generate_buffers(graph)

        m_ip.assert_called_once_with(graph, [])
        m_build.assert_called_once_with(graph, in_place, [])
        m_filter.assert_called_once_with(graph, raw, [])
        self.assertEqual(result, filtered)


class TestGetNcoresForBuffers(TestCase):
    @patch("torch_spyre._inductor.scratchpad.utils.config")
    def test_single_core_returns_one_for_all_buffers(self, mock_config):
        mock_config.sencores = 1
        dep = _make_dep("buf_a")
        op0 = _make_op("op0", [], [dep])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op0]

        result = get_ncores_for_buffers(graph)

        self.assertEqual(result["buf_a"], 1)

    @patch("torch_spyre._inductor.scratchpad.utils.config")
    def test_multicore_same_splits_returns_product(self, mock_config):
        mock_config.sencores = 4
        dep = _make_dep("buf_a")
        splits = [{16: 4}]
        op0 = _make_op("op0", [], [dep], splits=splits)
        op1 = _make_op("op1", [dep], [], splits=splits)
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op0, op1]

        result = get_ncores_for_buffers(graph)

        self.assertEqual(result["buf_a"], 4)

    @patch("torch_spyre._inductor.scratchpad.utils.config")
    def test_multicore_mismatched_splits_returns_minus_one(self, mock_config):
        mock_config.sencores = 4
        dep = _make_dep("buf_a")
        op0 = _make_op("op0", [], [dep], splits=[{256: 4}])
        op1 = _make_op("op1", [dep], [], splits=[{128: 1}, {128: 1}])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op0, op1]

        result = get_ncores_for_buffers(graph)

        self.assertEqual(result["buf_a"], -1)


class TestMemUsageByOp(TestCase):
    # --- size and size_per_core ---

    def test_size_computed_from_stick_count(self):
        dep = _make_dep("buf0")
        op = _make_op("op0", [], [dep])
        graph = MagicMock(spec=GraphLowering)
        graph.get_buffer.return_value = _make_buf(4)  # 4 sticks * 128 = 512

        result, _ = mem_usage_by_op(graph, [op])

        self.assertEqual(result["op0"]["buf0"]["size"], 4 * 128)
        self.assertEqual(result["op0"]["buf0"]["size_per_core"], 4 * 128)  # num_cores=1

    def test_size_per_core_divides_by_num_cores(self):
        dep = _make_dep("buf0")
        op = _make_op("op0", [], [dep], splits=[{16: 4}])  # 4 cores
        graph = MagicMock(spec=GraphLowering)
        graph.get_buffer.return_value = _make_buf(8)  # 8 sticks * 128 = 1024

        result, _ = mem_usage_by_op(graph, [op])

        self.assertEqual(result["op0"]["buf0"]["size"], 8 * 128)
        self.assertEqual(result["op0"]["buf0"]["size_per_core"], 8 * 128 // 4)

    # --- core_div_mismatch ---

    def test_core_div_mismatch_propagated_from_argument(self):
        dep = _make_dep("buf0")
        op = _make_op("op0", [], [dep])
        graph = MagicMock(spec=GraphLowering)
        graph.get_buffer.return_value = _make_buf(4)

        result, _ = mem_usage_by_op(graph, [op], core_div_mismatch={"buf0": True})

        self.assertTrue(result["op0"]["buf0"]["core_div_mismatch"])

    # --- buffer lists ---

    def test_all_buf_used_aggregates_inputs_and_outputs(self):
        dep_in = _make_dep("inp")
        dep_out = _make_dep("out")
        op = _make_op("op0", [dep_in], [dep_out])
        graph = MagicMock(spec=GraphLowering)
        graph.get_buffer.return_value = _make_buf(4)

        result, _ = mem_usage_by_op(graph, [op])

        self.assertEqual(result["op0"]["all_inputs"], ["inp"])
        self.assertEqual(result["op0"]["all_outputs"], ["out"])
        self.assertEqual(sorted(result["op0"]["all_buf_used"]), ["inp", "out"])

    # --- liveness ---

    def test_liveness_populated_across_two_ops(self):
        # buf0 written at op0 (i=0), read at op1 (i=1) → start=0, end=2
        dep = _make_dep("buf0")
        op0 = _make_op("op0", [], [dep])
        op1 = _make_op("op1", [dep], [])
        graph = MagicMock(spec=GraphLowering)
        graph.get_buffer.return_value = _make_buf(2)

        _, lifetimes = mem_usage_by_op(graph, [op0, op1])

        self.assertEqual(lifetimes["buf0"]["liveness_start"], 0)
        self.assertEqual(lifetimes["buf0"]["liveness_end"], 2)


class TestDetermineInPlace(TestCase):
    def setUp(self):
        self.allocator = _ConcreteAllocator()

    def _make_mem_usage(self, out_size, inp_size, out_start, inp_end):
        mem_usage = {
            "some_op": {
                "all_inputs": ["inp"],
                "all_outputs": ["out"],
                "inp": {"size_per_core": inp_size},
                "out": {"size_per_core": out_size},
            }
        }
        lifetimes = {
            "inp": {"liveness_start": 0, "liveness_end": inp_end},
            "out": {"liveness_start": out_start, "liveness_end": out_start + 2},
        }
        return mem_usage, lifetimes

    def _make_graph(self, shared_layout=True):
        graph = MagicMock(spec=GraphLowering)
        if shared_layout:
            # Both "out" and "inp" resolve to the same device_layout object → equal.
            dev_layout = MagicMock()
            graph.get_buffer.return_value.layout.device_layout = dev_layout
        else:
            out_buf, inp_buf = MagicMock(), MagicMock()
            out_buf.layout.device_layout = MagicMock()
            inp_buf.layout.device_layout = MagicMock()  # distinct object → not equal
            graph.get_buffer.side_effect = (
                lambda name: out_buf if name == "out" else inp_buf
            )
        return graph

    @patch("torch_spyre._inductor.scratchpad.allocator.mem_usage_by_op")
    def test_inplace_allowed_when_all_conditions_met(self, mock_mem):
        mock_mem.return_value = self._make_mem_usage(512, 512, 2, 3)
        graph = self._make_graph(shared_layout=True)

        result = self.allocator._determine_in_place(graph, [])

        self.assertIn("inp", result["out"])

    @patch("torch_spyre._inductor.scratchpad.allocator.mem_usage_by_op")
    def test_inplace_skipped_size_mismatch(self, mock_mem):
        mock_mem.return_value = self._make_mem_usage(512, 256, 2, 3)
        graph = self._make_graph(shared_layout=True)

        result = self.allocator._determine_in_place(graph, [])

        self.assertNotIn("inp", result["out"])

    @patch("torch_spyre._inductor.scratchpad.allocator.mem_usage_by_op")
    def test_inplace_skipped_layout_mismatch(self, mock_mem):
        mock_mem.return_value = self._make_mem_usage(512, 512, 2, 3)
        graph = self._make_graph(shared_layout=False)

        result = self.allocator._determine_in_place(graph, [])

        self.assertNotIn("inp", result["out"])

    @patch("torch_spyre._inductor.scratchpad.allocator.mem_usage_by_op")
    def test_inplace_skipped_when_input_not_end_of_life(self, mock_mem):
        mock_mem.return_value = self._make_mem_usage(512, 512, 2, 4)
        graph = self._make_graph(shared_layout=True)

        result = self.allocator._determine_in_place(graph, [])

        self.assertNotIn("inp", result["out"])


class TestCloneInputNodesPass(TestCase):
    LIMIT = 10 * 128  # 1280 bytes

    def _make_graph(self, inp_names, buf_per_name=None):
        graph = MagicMock(spec=GraphLowering)
        graph.graph_input_names = inp_names
        graph.operations = []
        if buf_per_name is None:
            graph.get_buffer.return_value = _make_buf(4)
        else:
            graph.get_buffer.side_effect = lambda name: buf_per_name[name]
        return graph

    @patch(
        "torch_spyre._inductor.scratchpad.passes.OP_OUTPUT_GOOD_FOR_LX_REUSE",
        new=["clone"],
    )
    @patch("torch_spyre._inductor.scratchpad.passes.buf_analysis")
    @patch("torch_spyre._inductor.scratchpad.passes.get_buffer_users")
    @patch.object(CloneInputNodesPass, "insert_op_after")
    def test_eligible_input_triggers_insert(
        self, mock_insert, mock_users, mock_buf_analysis
    ):
        graph = self._make_graph(["inp0"])
        mock_users.return_value = {"inp0": [MagicMock(), MagicMock()]}
        mock_buf_analysis.return_value = ({}, {}, {"inp0": False})

        CloneInputNodesPass(self.LIMIT).apply_pass(graph)

        mock_insert.assert_called_once()

    @patch(
        "torch_spyre._inductor.scratchpad.passes.OP_OUTPUT_GOOD_FOR_LX_REUSE",
        new=["clone"],
    )
    @patch("torch_spyre._inductor.scratchpad.passes.buf_analysis")
    @patch("torch_spyre._inductor.scratchpad.passes.get_buffer_users")
    @patch.object(CloneInputNodesPass, "insert_op_after")
    def test_skips_input_used_only_once(
        self, mock_insert, mock_users, mock_buf_analysis
    ):
        graph = self._make_graph(["inp0"])
        mock_users.return_value = {"inp0": [MagicMock()]}  # single user
        mock_buf_analysis.return_value = ({}, {}, {"inp0": False})

        CloneInputNodesPass(self.LIMIT).apply_pass(graph)

        mock_insert.assert_not_called()

    @patch(
        "torch_spyre._inductor.scratchpad.passes.OP_OUTPUT_GOOD_FOR_LX_REUSE",
        new=["clone"],
    )
    @patch("torch_spyre._inductor.scratchpad.passes.buf_analysis")
    @patch("torch_spyre._inductor.scratchpad.passes.get_buffer_users")
    @patch.object(CloneInputNodesPass, "insert_op_after")
    def test_skips_input_too_large_for_lx(
        self, mock_insert, mock_users, mock_buf_analysis
    ):
        # 20 sticks * 128 = 2560 > LIMIT=1280
        graph = self._make_graph(["inp0"], buf_per_name={"inp0": _make_buf(20)})
        mock_users.return_value = {"inp0": [MagicMock(), MagicMock()]}
        mock_buf_analysis.return_value = ({}, {}, {"inp0": False})

        CloneInputNodesPass(self.LIMIT).apply_pass(graph)

        mock_insert.assert_not_called()

    @patch(
        "torch_spyre._inductor.scratchpad.passes.OP_OUTPUT_GOOD_FOR_LX_REUSE",
        new=["clone"],
    )
    @patch("torch_spyre._inductor.scratchpad.passes.buf_analysis")
    @patch("torch_spyre._inductor.scratchpad.passes.get_buffer_users")
    @patch.object(CloneInputNodesPass, "insert_op_after")
    def test_skips_input_with_core_div_mismatch(
        self, mock_insert, mock_users, mock_buf_analysis
    ):
        graph = self._make_graph(["inp0"])
        mock_users.return_value = {"inp0": [MagicMock(), MagicMock()]}
        mock_buf_analysis.return_value = ({}, {}, {"inp0": True})

        CloneInputNodesPass(self.LIMIT).apply_pass(graph)

        mock_insert.assert_not_called()

    @patch(
        "torch_spyre._inductor.scratchpad.passes.OP_OUTPUT_GOOD_FOR_LX_REUSE",
        new=["clone"],
    )
    @patch("torch_spyre._inductor.scratchpad.passes.buf_analysis")
    @patch("torch_spyre._inductor.scratchpad.passes.get_buffer_users")
    @patch.object(CloneInputNodesPass, "insert_op_after")
    def test_budget_decremented_between_inputs(
        self, mock_insert, mock_users, mock_buf_analysis
    ):
        # inp0: 8 sticks = 1024 bytes (fits LIMIT=1280, consumes it)
        # inp1: 4 sticks = 512 bytes (remaining budget=256, 512>256 → skipped)
        buf0 = _make_buf(8)
        buf1 = _make_buf(4)
        graph = self._make_graph(
            ["inp0", "inp1"], buf_per_name={"inp0": buf0, "inp1": buf1}
        )
        mock_users.return_value = {
            "inp0": [MagicMock(), MagicMock()],
            "inp1": [MagicMock(), MagicMock()],
        }
        mock_buf_analysis.return_value = ({}, {}, {"inp0": False, "inp1": False})

        CloneInputNodesPass(self.LIMIT).apply_pass(graph)

        self.assertEqual(mock_insert.call_count, 1)
        self.assertIs(mock_insert.call_args[0][1], buf0)


if __name__ == "__main__":
    import unittest

    unittest.main()
