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
from torch._inductor.ir import ComputedBuffer
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad.allocator import (
    DefaultAllocator,
    GraphBufferConverstion,
)
from torch_spyre._inductor.scratchpad.utility import (
    op_output_good_for_lx_reuse,
    is_permissible_op,
    get_buffer_users,
    determine_core_division,
    calculate_buffer_statistics,
    mem_usage_by_buffer,
    calculate_liveness,
    push_allocation,
)


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
    def test_op_lx_reuse(self):
        allowed_ops = ["add", "sub"]
        self.assertTrue(op_output_good_for_lx_reuse("add", allowed_ops))
        self.assertFalse(op_output_good_for_lx_reuse("mul", allowed_ops))

    def test_permissible_op(self):
        allowed_ops = ["add", "sub"]

        def make_op(opname: str, buf_name: str) -> MagicMock:
            write_dep = MagicMock()
            write_dep.name = buf_name
            rw = MagicMock()
            rw.writes = [write_dep]
            op = MagicMock(spec=ComputedBuffer)
            op.get_read_writes.return_value = rw
            op.origin_node.target._opname = opname
            return op

        mock_graph = MagicMock(spec=GraphLowering)
        mock_graph.operations = [
            make_op("add", "buf0"),
            make_op("sub", "buf1"),
            make_op("mul", "buf2"),
        ]

        result = is_permissible_op(mock_graph, allowed_ops)

        self.assertTrue(result["buf0"])
        self.assertTrue(result["buf1"])
        self.assertFalse(result["buf2"])

    def test_get_buffer_users(self):
        buf_a, buf_b, buf_c = _make_dep("buf_a"), _make_dep("buf_b"), _make_dep("buf_c")
        op0 = _make_op("op0", [buf_a], [buf_b])
        op1 = _make_op("op1", [buf_b], [buf_c])
        mock_graph = MagicMock(spec=GraphLowering)
        mock_graph.operations = [op0, op1]

        result = get_buffer_users(mock_graph)

        self.assertEqual(result["buf_a"], ["op0"])
        self.assertEqual(result["buf_b"], ["op0", "op1"])
        self.assertEqual(result["buf_c"], ["op1"])

    def test_get_buffer_users_empty_graph(self):
        mock_graph = MagicMock(spec=GraphLowering)
        mock_graph.operations = []
        self.assertEqual(get_buffer_users(mock_graph), {})

    def test_determine_core_division_no_core_division(self):
        buf_a = _make_dep("buf_a")
        op0 = _make_op("op0", [], [buf_a])
        op1 = _make_op("op1", [buf_a], [])
        mock_graph = MagicMock(spec=GraphLowering)
        mock_graph.operations = [op0, op1]

        result = determine_core_division(mock_graph)

        self.assertTrue(result["buf_a"])

    def test_determine_core_division_matching_core_division(self):
        buf_a = _make_dep("buf_a")
        op0 = _make_op("op0", [], [buf_a], splits=[16, 1])
        op1 = _make_op("op1", [buf_a], [], splits=[16, 1])
        mock_graph = MagicMock(spec=GraphLowering)
        mock_graph.operations = [op0, op1]

        result = determine_core_division(mock_graph)

        self.assertTrue(result["buf_a"])

    def test_determine_core_division_mismatched_core_division(self):
        buf_a = _make_dep("buf_a")
        op0 = _make_op("op0", [], [buf_a], splits=[16, 1])
        op1 = _make_op("op1", [buf_a], [], splits=[8, 2])
        mock_graph = MagicMock(spec=GraphLowering)
        mock_graph.operations = [op0, op1]

        result = determine_core_division(mock_graph)

        self.assertFalse(result["buf_a"])

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

    def _buf_with_device_size(self, device_size: list) -> MagicMock:
        buf = MagicMock()
        buf.layout.device_layout.device_size = device_size
        return buf

    def test_mem_usage_by_buffer_one_entry_per_op(self):
        # One entry is produced per op, not per individual buffer dep.
        buf0 = self._buf_with_device_size([4, 4, 128])  # prod([4,4])*128 = 2048
        buf1 = self._buf_with_device_size([8, 4, 128])  # prod([8,4])*128 = 4096
        mock_graph = MagicMock(spec=GraphLowering)
        mock_graph.operations = [MagicMock(), MagicMock()]
        mock_graph.get_buffer.side_effect = [buf0, buf1]

        result = mem_usage_by_buffer(mock_graph)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[buf0], 2048)
        self.assertEqual(result[buf1], 4096)

    def test_mem_usage_by_buffer_1d_device_size(self):
        # device_size[:-1] is empty → prod([]) = 1 → size = 128
        mock_graph = MagicMock(spec=GraphLowering)
        mock_graph.operations = [MagicMock()]
        mock_graph.get_buffer.return_value = self._buf_with_device_size([128])

        result = mem_usage_by_buffer(mock_graph)

        self.assertEqual(list(result.values()), [128])

    def test_calculate_liveness(self):
        buf_a, buf_b, buf_c, buf_d = (
            _make_dep("buf_a"),
            _make_dep("buf_b"),
            _make_dep("buf_c"),
            _make_dep("buf_d"),
        )
        op0 = _make_op("op0", [buf_a], [buf_b])
        op1 = _make_op("op1", [buf_b], [buf_c])
        op2 = _make_op("op2", [buf_c], [buf_d])
        mock_graph = MagicMock(spec=GraphLowering)
        mock_graph.operations = [op0, op1, op2]

        start, end = calculate_liveness(mock_graph)

        self.assertEqual(start["buf_a"], 0)
        self.assertEqual(start["buf_b"], 0)
        self.assertEqual(start["buf_c"], 1)
        self.assertEqual(start["buf_d"], 2)
        self.assertEqual(end["buf_a"], 1)
        self.assertEqual(end["buf_b"], 2)
        self.assertEqual(end["buf_c"], 3)
        self.assertEqual(end["buf_d"], 3)

    def test_push_allocation_with_address(self):
        mock_graph = MagicMock(spec=GraphLowering)
        mock_layout = MagicMock()
        mock_layout.allocation = {}
        mock_graph.get_buffer.return_value.get_layout.return_value = mock_layout

        push_allocation(
            mock_graph, [LifetimeBoundBuffer("buf0", 1024, 0, 2, address=0x1000)]
        )

        mock_graph.get_buffer.assert_called_once_with("buf0")
        self.assertEqual(mock_layout.allocation["lx"], 0x1000)

    def test_push_allocation_no_address(self):
        mock_graph = MagicMock(spec=GraphLowering)

        push_allocation(
            mock_graph, [LifetimeBoundBuffer("buf0", 1024, 0, 2, address=None)]
        )

        mock_graph.get_buffer.assert_not_called()

    def test_push_allocation_empty(self):
        mock_graph = MagicMock(spec=GraphLowering)
        push_allocation(mock_graph, [])
        mock_graph.get_buffer.assert_not_called()


class TestGraphBufferConverstion(TestCase):
    # --- generate_lifetime_bound_buffers ---

    @patch("torch_spyre._inductor.scratchpad.allocator.mem_usage_by_buffer")
    @patch("torch_spyre._inductor.scratchpad.allocator.calculate_liveness")
    def test_build_bound_buffers(self, mock_liveness, mock_mem):
        s = {"buf0": 0, "buf1": 1}
        e = {"buf0": 2, "buf1": 2}
        mock_liveness.return_value = [s, e]
        mock_mem.return_value = {"buf0": 1024, "buf1": 512}

        result = GraphBufferConverstion()._build_bound_buffers(
            MagicMock(spec=GraphLowering)
        )
        buf0 = next(b for b in result if b.name == "buf0")
        buf1 = next(b for b in result if b.name == "buf1")

        self.assertEqual(len(result), 2)
        self.assertEqual(buf0.size, 1024)
        self.assertEqual(buf0.start_time, 0)
        self.assertEqual(buf0.end_time, 2)
        self.assertIsNone(buf0.heuristic)
        self.assertEqual(buf1.size, 512)
        self.assertEqual(buf1.start_time, 1)
        self.assertEqual(buf1.end_time, 2)


class TestDefaultAllocator(TestCase):
    def _make_allocator(
        self,
        layout_planning=None,
        pre_optimization_passes=None,
        post_optimization_passes=None,
        buffer_source=None,
    ):
        if layout_planning is None:
            layout_planning = MagicMock()
        kwargs = {}
        if pre_optimization_passes is not None:
            kwargs["pre_optimization_passes"] = pre_optimization_passes
        if post_optimization_passes is not None:
            kwargs["post_optimization_passes"] = post_optimization_passes
        if buffer_source is not None:
            kwargs["buffer_source"] = buffer_source
        return DefaultAllocator(layout_planning, **kwargs)

    # --- __init__ ---

    def test_init_none_layout_planning_raises(self):
        with self.assertRaises(AssertionError):
            DefaultAllocator(None)

    def test_init_default_buffer_source(self):
        allocator = self._make_allocator()
        self.assertIsInstance(allocator.buffer_source, GraphBufferConverstion)

    # --- plan_allocation ---

    @patch("torch_spyre._inductor.scratchpad.allocator.push_allocation")
    def test_plan_allocation_calls_dependencies(self, mock_push):
        call_order = []
        buf0 = LifetimeBoundBuffer("buf0", 1024, 0, 2)
        buf1 = LifetimeBoundBuffer("buf1", 512, 1, 3)
        buffer_source = MagicMock()
        buffer_source.generate_buffers.return_value = [buf0, buf1]
        solver = MagicMock()
        solver.plan_layout.return_value = []
        graph = MagicMock(spec=GraphLowering)
        pass1, pass2 = MagicMock(), MagicMock()
        pass1.apply_pass.side_effect = lambda g: call_order.append("pass1")
        pass2.apply_pass.side_effect = lambda g: call_order.append("pass2")

        DefaultAllocator(
            solver,
            pre_optimization_passes=[pass1],
            post_optimization_passes=[pass2],
            buffer_source=buffer_source,
        ).plan_allocation(graph)

        buffer_source.generate_buffers.assert_called_once_with(graph)
        solver.plan_layout.assert_called_once_with([buf0, buf1])
        mock_push.assert_called_once_with(graph, [])
        pass1.apply_pass.assert_called_once_with(graph)
        pass2.apply_pass.assert_called_once_with(graph)
        self.assertEqual(call_order, ["pass1", "pass2"])


if __name__ == "__main__":
    import unittest

    unittest.main()
