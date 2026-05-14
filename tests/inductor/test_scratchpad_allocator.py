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
    DefaultAllocator
)
from torch_spyre._inductor.scratchpad.utils import (
    get_buffer_users,
    calculate_buffer_statistics,
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

    def test_get_buffer_users_empty_graph(self):
        mock_graph = MagicMock(spec=GraphLowering)
        mock_graph.operations = []
        self.assertEqual(get_buffer_users(mock_graph), {})

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
    def test_plan_allocation_calls_dependencies(self, mock_generate, mock_push):
        call_order = []
        buf0 = LifetimeBoundBuffer("buf0", 1024, 0, 2)
        buf1 = LifetimeBoundBuffer("buf1", 512, 1, 3)
        mock_generate.return_value = [buf0, buf1]
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
        ).plan_allocation(graph)

        mock_generate.assert_called_once_with(graph)
        solver.plan_layout.assert_called_once_with([buf0, buf1])
        mock_push.assert_called_once_with(graph, [])
        pass1.apply_pass.assert_called_once_with(graph)
        pass2.apply_pass.assert_called_once_with(graph)
        self.assertEqual(call_order, ["pass1", "pass2"])


if __name__ == "__main__":
    import unittest

    unittest.main()
