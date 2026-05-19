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
    buf_analysis,
    calculate_liveness,
    get_buffer_users,
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


class TestBufAnalysis(TestCase):
    # --- bufs_to_dealloc_at_idx ---

    @patch("torch_spyre._inductor.scratchpad.utils.config")
    def test_dealloc_idx_reflects_last_use(self, mock_config):
        mock_config.sencores = 1
        dep_a, dep_b = _make_dep("a"), _make_dep("b")
        op0 = _make_op("op0", [], [dep_a])
        op1 = _make_op("op1", [dep_a], [dep_b])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op0, op1]

        dealloc, _, _ = buf_analysis(graph)

        # both 'a' and 'b' are last seen at idx=1 → freed at idx=2
        self.assertIn("a", dealloc[2])
        self.assertIn("b", dealloc[2])

    @patch("torch_spyre._inductor.scratchpad.utils.config")
    def test_dealloc_idx_separate_when_last_uses_differ(self, mock_config):
        mock_config.sencores = 1
        dep_a, dep_b = _make_dep("a"), _make_dep("b")
        op0 = _make_op("op0", [dep_a], [dep_b])
        op1 = _make_op("op1", [], [dep_a])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op0, op1]

        dealloc, _, _ = buf_analysis(graph)

        # 'b' last seen at idx=0 → freed at idx=1
        # 'a' last seen at idx=1 → freed at idx=2
        self.assertIn("b", dealloc[1])
        self.assertIn("a", dealloc[2])

    # --- buf_users (read ops only) ---

    @patch("torch_spyre._inductor.scratchpad.utils.config")
    def test_buf_users_includes_only_read_ops(self, mock_config):
        mock_config.sencores = 1
        dep_a, dep_b = _make_dep("a"), _make_dep("b")
        op0 = _make_op("op0", [], [dep_a])  # writes 'a'
        op1 = _make_op("op1", [dep_a], [dep_b])  # reads 'a'
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op0, op1]

        _, users, _ = buf_analysis(graph)

        # op0 writes 'a' → not in users; op1 reads 'a' → in users
        self.assertEqual(users.get("a"), [op1])
        # 'b' is only written, never read
        self.assertNotIn("b", users)

    # --- core_div_mismatch ---

    @patch("torch_spyre._inductor.scratchpad.utils.config")
    def test_core_div_no_mismatch_single_core(self, mock_config):
        mock_config.sencores = 1
        dep = _make_dep("buf")
        op0 = _make_op("op0", [], [dep], splits=[{256: 4}])
        op1 = _make_op("op1", [dep], [], splits=[{128: 1}, {128: 1}])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op0, op1]

        _, _, mismatch = buf_analysis(graph)

        # single core → mismatch check is skipped
        self.assertFalse(mismatch["buf"])

    @patch("torch_spyre._inductor.scratchpad.utils.config")
    def test_core_div_no_mismatch_same_splits(self, mock_config):
        mock_config.sencores = 4
        dep = _make_dep("buf")
        splits = [{16: 4}]
        op0 = _make_op("op0", [], [dep], splits=splits)
        op1 = _make_op("op1", [dep], [], splits=splits)
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op0, op1]

        _, _, mismatch = buf_analysis(graph)

        self.assertFalse(mismatch["buf"])

    @patch("torch_spyre._inductor.scratchpad.utils.config")
    def test_core_div_mismatch_different_splits(self, mock_config):
        mock_config.sencores = 4
        dep = _make_dep("buf")
        op0 = _make_op("op0", [], [dep], splits=[{256: 4}])
        op1 = _make_op("op1", [dep], [], splits=[{128: 2}])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op0, op1]

        _, _, mismatch = buf_analysis(graph)

        self.assertTrue(mismatch["buf"])


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

        liveness = calculate_liveness(graph)

        self.assertEqual(liveness["a"]["liveness_start"], 0)
        self.assertEqual(liveness["a"]["liveness_end"], 1)
        self.assertEqual(liveness["b"]["liveness_start"], 0)
        self.assertEqual(liveness["b"]["liveness_end"], 2)
        self.assertEqual(liveness["c"]["liveness_start"], 1)
        self.assertEqual(liveness["c"]["liveness_end"], 2)

    def test_calculate_liveness_single_op(self):
        dep_a, dep_b = _make_dep("a"), _make_dep("b")
        op0 = _make_op("op0", [dep_a], [dep_b])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op0]

        liveness = calculate_liveness(graph)

        self.assertEqual(liveness["a"]["liveness_start"], 0)
        self.assertEqual(liveness["a"]["liveness_end"], 1)
        self.assertEqual(liveness["b"]["liveness_start"], 0)
        self.assertEqual(liveness["b"]["liveness_end"], 1)

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

        result = self.allocator._filter_buffers(graph, bufs)

        self.assertEqual(result, [])

    def test_filter_buffers_drops_graph_outputs(self):
        graph = MagicMock(spec=GraphLowering)
        graph.operations = []
        graph.get_output_names.return_value = ["out"]
        graph.graph_input_names = []
        result = self.allocator._filter_buffers(
            graph, [LifetimeBoundBuffer("out", 100, 0, 1)]
        )
        self.assertEqual(result, [])

    def test_filter_buffers_drops_graph_inputs(self):
        graph = MagicMock(spec=GraphLowering)
        graph.operations = []
        graph.get_output_names.return_value = []
        graph.graph_input_names = ["inp"]
        result = self.allocator._filter_buffers(
            graph, [LifetimeBoundBuffer("inp", 100, 0, 1)]
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

        result = self.allocator._filter_buffers(graph, bufs)

        self.assertEqual(result, bufs)

    @patch("torch_spyre._inductor.scratchpad.allocator.buf_analysis")
    @patch.object(
        ScratchpadAllocator, "_op_output_good_for_lx_reuse", return_value=True
    )
    def test_filter_buffers_removes_dropped_buffer_from_in_place(
        self, _mock_reuse, mock_ba
    ):
        # "inp" is flagged with core_div_mismatch → dropped from the result.
        # "out" is retained, but its in_place=["inp"] must be purged because
        # handing a reference to a non-LX buffer to the solver is possibly invalid.
        mock_ba.return_value = ({}, {}, {"out": False, "inp": True})
        out_dep, inp_dep = _make_dep("out"), _make_dep("inp")
        ok_op = _make_op("op0", [inp_dep], [out_dep])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [ok_op]
        graph.get_output_names.return_value = []
        graph.graph_input_names = []
        bufs = [
            LifetimeBoundBuffer("out", 100, 1, 2, in_place=["inp"]),
            LifetimeBoundBuffer("inp", 100, 0, 1),
        ]

        result = self.allocator._filter_buffers(graph, bufs)

        self.assertEqual([b.name for b in result], ["out"])
        self.assertEqual(result[0].in_place, [])

    # --- _build_bound_buffers ---

    @patch("torch_spyre._inductor.scratchpad.allocator.mem_usage_by_op")
    @patch("torch_spyre._inductor.scratchpad.allocator.buf_analysis")
    @patch("torch_spyre._inductor.scratchpad.allocator.calculate_liveness")
    def test_build_bound_buffers_skips_non_viable(
        self, mock_liveness, mock_ba, mock_mem
    ):
        mock_ba.return_value = ({}, {}, {})
        mock_liveness.return_value = {"buf0": {"liveness_start": 0, "liveness_end": 1}}
        mock_mem.return_value = {
            "op0": {
                "all_buf_used": ["buf0"],
                "buf0": {"is_lx_viable": False},
            }
        }
        graph = MagicMock(spec=GraphLowering)

        result = self.allocator._build_bound_buffers(graph, {})

        self.assertEqual(result, [])

    @patch("torch_spyre._inductor.scratchpad.allocator.mem_usage_by_op")
    @patch("torch_spyre._inductor.scratchpad.allocator.buf_analysis")
    @patch("torch_spyre._inductor.scratchpad.allocator.calculate_liveness")
    def test_build_bound_buffers_applies_in_place(
        self, mock_liveness, mock_ba, mock_mem
    ):
        mock_ba.return_value = ({}, {}, {})
        mock_liveness.return_value = {
            "inp": {"liveness_start": 0, "liveness_end": 1},
            "out": {"liveness_start": 1, "liveness_end": 2},
        }
        mock_mem.return_value = {
            "op0": {
                "all_buf_used": ["inp", "out"],
                "inp": {
                    "is_lx_viable": True,
                    "size_per_core": 256,
                    "core_div_mismatch": False,
                },
                "out": {
                    "is_lx_viable": True,
                    "size_per_core": 256,
                    "core_div_mismatch": False,
                },
            }
        }
        graph = MagicMock(spec=GraphLowering)

        result = self.allocator._build_bound_buffers(graph, {"out": ["inp"]})

        out_buf = next(b for b in result if b.name == "out")
        self.assertEqual(out_buf.in_place, ["inp"])

    @patch("torch_spyre._inductor.scratchpad.allocator.mem_usage_by_op")
    @patch("torch_spyre._inductor.scratchpad.allocator.buf_analysis")
    @patch("torch_spyre._inductor.scratchpad.allocator.calculate_liveness")
    def test_build_bound_buffers_deduplicates_by_name(
        self, mock_liveness, mock_ba, mock_mem
    ):
        mock_ba.return_value = ({}, {}, {})
        mock_liveness.return_value = {
            "shared": {"liveness_start": 0, "liveness_end": 2}
        }
        mock_mem.return_value = {
            "op0": {
                "all_buf_used": ["shared"],
                "shared": {
                    "is_lx_viable": True,
                    "size_per_core": 128,
                    "core_div_mismatch": False,
                },
            },
            "op1": {
                "all_buf_used": ["shared"],
                "shared": {
                    "is_lx_viable": True,
                    "size_per_core": 128,
                    "core_div_mismatch": False,
                },
            },
        }
        graph = MagicMock(spec=GraphLowering)

        result = self.allocator._build_bound_buffers(graph, {})

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "shared")

    @patch("torch_spyre._inductor.scratchpad.allocator.mem_usage_by_op")
    @patch("torch_spyre._inductor.scratchpad.allocator.buf_analysis")
    @patch("torch_spyre._inductor.scratchpad.allocator.calculate_liveness")
    def test_build_bound_buffers_drops_core_div_mismatch(
        self, mock_liveness, mock_ba, mock_mem
    ):
        mock_ba.return_value = ({}, {}, {"buf0": True})
        mock_liveness.return_value = {"buf0": {"liveness_start": 0, "liveness_end": 2}}
        mock_mem.return_value = {
            "op0": {
                "all_buf_used": ["buf0"],
                "buf0": {
                    "is_lx_viable": True,
                    "size_per_core": 512,
                    "core_div_mismatch": True,
                },
            }
        }
        graph = MagicMock(spec=GraphLowering)

        result = self.allocator._build_bound_buffers(graph, {})

        self.assertEqual(result, [])


class TestMemUsageByOp(TestCase):
    # --- size and size_per_core ---

    def test_size_computed_from_stick_count(self):
        dep = _make_dep("buf0")
        op = _make_op("op0", [], [dep])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op]
        graph.get_buffer.return_value = _make_buf(4)  # 4 sticks * 128 = 512

        result = mem_usage_by_op(graph)

        self.assertEqual(result["op0"]["buf0"]["size"], 4 * 128)
        self.assertEqual(result["op0"]["buf0"]["size_per_core"], 4 * 128)  # num_cores=1

    def test_size_per_core_divides_by_num_cores(self):
        dep = _make_dep("buf0")
        op = _make_op("op0", [], [dep], splits=[{8: 4}])  # 4 cores
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op]
        graph.get_buffer.return_value = _make_buf(8)  # 8 sticks * 128 = 1024

        result = mem_usage_by_op(graph)

        self.assertEqual(result["op0"]["buf0"]["size"], 8 * 128)
        self.assertEqual(result["op0"]["buf0"]["size_per_core"], 8 * 128 // 4)

    # --- core_div_mismatch ---

    def test_core_div_mismatch_propagated_from_argument(self):
        dep = _make_dep("buf0")
        op = _make_op("op0", [], [dep])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op]
        graph.get_buffer.return_value = _make_buf(4)

        result = mem_usage_by_op(graph, {"buf0": True})

        self.assertTrue(result["op0"]["buf0"]["core_div_mismatch"])

    # --- buffer lists ---

    def test_all_buf_used_aggregates_inputs_and_outputs(self):
        dep_in = _make_dep("inp")
        dep_out = _make_dep("out")
        op = _make_op("op0", [dep_in], [dep_out])
        graph = MagicMock(spec=GraphLowering)
        graph.operations = [op]
        graph.get_buffer.return_value = _make_buf(4)

        result = mem_usage_by_op(graph)

        self.assertEqual(result["op0"]["all_inputs"], ["inp"])
        self.assertEqual(result["op0"]["all_outputs"], ["out"])
        self.assertEqual(sorted(result["op0"]["all_buf_used"]), ["inp", "out"])


class TestDetermineInPlace(TestCase):
    def setUp(self):
        self.allocator = _ConcreteAllocator()

    def _make_mem_usage(self, out_size, inp_size, out_start, inp_end):
        mem_usage = {
            "some_op": {
                "all_inputs": ["inp"],
                "all_outputs": ["out"],
                "inp": {
                    "size_per_core": inp_size,
                    "is_lx_viable": True,
                    "core_div_mismatch": False,
                },
                "out": {
                    "size_per_core": out_size,
                    "is_lx_viable": True,
                    "core_div_mismatch": False,
                },
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

    @patch("torch_spyre._inductor.scratchpad.allocator.calculate_liveness")
    @patch("torch_spyre._inductor.scratchpad.allocator.mem_usage_by_op")
    @patch("torch_spyre._inductor.scratchpad.allocator.buf_analysis")
    def test_inplace_allowed_when_all_conditions_met(
        self, mock_ba, mock_mem, mock_liveness
    ):
        mem_usage, lifetimes = self._make_mem_usage(512, 512, 2, 3)
        mock_ba.return_value = ({}, {}, {})
        mock_mem.return_value = mem_usage
        mock_liveness.return_value = lifetimes
        graph = self._make_graph(shared_layout=True)

        result = self.allocator._determine_in_place(graph)

        self.assertIn("inp", result["out"])

    @patch("torch_spyre._inductor.scratchpad.allocator.calculate_liveness")
    @patch("torch_spyre._inductor.scratchpad.allocator.mem_usage_by_op")
    @patch("torch_spyre._inductor.scratchpad.allocator.buf_analysis")
    def test_inplace_skipped_size_mismatch(self, mock_ba, mock_mem, mock_liveness):
        mem_usage, lifetimes = self._make_mem_usage(512, 256, 2, 3)
        mock_ba.return_value = ({}, {}, {})
        mock_mem.return_value = mem_usage
        mock_liveness.return_value = lifetimes
        graph = self._make_graph(shared_layout=True)

        result = self.allocator._determine_in_place(graph)

        self.assertNotIn("inp", result["out"])

    @patch("torch_spyre._inductor.scratchpad.allocator.calculate_liveness")
    @patch("torch_spyre._inductor.scratchpad.allocator.mem_usage_by_op")
    @patch("torch_spyre._inductor.scratchpad.allocator.buf_analysis")
    def test_inplace_skipped_layout_mismatch(self, mock_ba, mock_mem, mock_liveness):
        mem_usage, lifetimes = self._make_mem_usage(512, 512, 2, 3)
        mock_ba.return_value = ({}, {}, {})
        mock_mem.return_value = mem_usage
        mock_liveness.return_value = lifetimes
        graph = self._make_graph(shared_layout=False)

        result = self.allocator._determine_in_place(graph)

        self.assertNotIn("inp", result["out"])

    @patch("torch_spyre._inductor.scratchpad.allocator.calculate_liveness")
    @patch("torch_spyre._inductor.scratchpad.allocator.mem_usage_by_op")
    @patch("torch_spyre._inductor.scratchpad.allocator.buf_analysis")
    def test_inplace_skipped_when_input_not_end_of_life(
        self, mock_ba, mock_mem, mock_liveness
    ):
        mem_usage, lifetimes = self._make_mem_usage(512, 512, 2, 4)
        mock_ba.return_value = ({}, {}, {})
        mock_mem.return_value = mem_usage
        mock_liveness.return_value = lifetimes
        graph = self._make_graph(shared_layout=True)

        result = self.allocator._determine_in_place(graph)

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
    @patch.object(CloneInputNodesPass, "_insert_op_after")
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
    @patch.object(CloneInputNodesPass, "_insert_op_after")
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
    @patch.object(CloneInputNodesPass, "_insert_op_after")
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
    @patch.object(CloneInputNodesPass, "_insert_op_after")
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
    @patch.object(CloneInputNodesPass, "_insert_op_after")
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
