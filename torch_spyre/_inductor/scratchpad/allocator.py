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

from abc import ABC, abstractmethod
import math
from typing import Any, Optional

from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
)
from torch._inductor.graph import GraphLowering
from .. import config

from torch_spyre._inductor.scratchpad.plan_solver import (
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
    MemoryPlanSolver,
)
from torch_spyre._inductor.scratchpad.passes import ScratchpadOptimizationPass
from torch_spyre._inductor.scratchpad.utils import (
    get_ncores_for_buffers,
)
from torch_spyre._inductor.scratchpad.passes import CloneInputNodesPass


OP_OUTPUT_GOOD_FOR_LX_REUSE = [
    "max",
    "sum",
    # "clone",
    "exp",
    "sub",
    # "mul",
]

OP_GOOD_FOR_LX_INPLACE = [
    "exp",
    "sub",
]


class ScratchpadAllocator(ABC):
    """
    Abstract class for all implementations of ScratchpadAllocator
    """

    @abstractmethod
    def plan_allocation(self, graph: GraphLowering):
        """
        Accepts a graph to be considerd for scratchpad memory according
        to its composition and the specific implementation used.

        Args:
            graph (GraphLowering): Graph to be considered for scratchpad planning
        """
        pass

    def op_output_good_for_lx_reuse(self, op: Any) -> bool:
        return (
            isinstance(op, ComputedBuffer)
            and not isinstance(op.layout, MutationLayoutSHOULDREMOVE)
            and (
                config.allow_all_ops_in_lx_planning
                or op._opname in OP_OUTPUT_GOOD_FOR_LX_REUSE
            )
        )

    def op_good_for_lx_inplace(self, org_op_name: str) -> bool:
        return any(op in org_op_name for op in OP_GOOD_FOR_LX_INPLACE)

    def generate_buffers(self, graph: GraphLowering) -> list[LifetimeBoundBuffer]:
        in_place = self._determine_in_place(graph)
        buffers = self._build_bound_buffers(graph, in_place)
        filtered_buffers = self._filter_buffers(graph, buffers)
        return filtered_buffers

    def _filter_buffers(
        self, graph: GraphLowering, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        """
        From the list of buffers, drop buffers that are outputs of
        unpermitted ops, graph outputs, and graph inputs
        """

        drop_list = set()
        for op in graph.operations:
            if not self.op_output_good_for_lx_reuse(op.origin_node.target):
                rw = op.get_read_writes()
                for mem_dep in rw.writes:
                    drop_list.add(mem_dep.name)

        drop_list.update(graph.get_output_names())
        drop_list.update(graph.graph_input_names)

        return [b for b in buffers if b.name not in drop_list]

    def _mem_usage_by_buffer(
        self, graph: GraphLowering
    ) -> dict[str, dict[str, bool | int]]:
        """
        Get a summary of memory usage for all operations
        Detailed info of individual buf, e.g. mem_usage[<buf_name>], which has
            "size", "core_div_mismatch", "size_per_core", "liveness_start",
            "liveness_end" fields
        """
        mem_usage = {}
        for buf_name, num_cores in get_ncores_for_buffers(graph).items():
            buf = graph.get_buffer(buf_name)
            dev_layout = buf.layout.device_layout
            dev_size = (
                math.prod(dev_layout.device_size[:-1]) * 128
            )  # num_sticks * bytes_per_stick
            mem_usage[buf_name] = {
                "size": dev_size,
                "size_per_core": dev_size // num_cores,
                "core_div_mismatch": num_cores == -1,
            }
        self._calculate_liveness(graph, mem_usage)
        return mem_usage

    def _calculate_liveness(
        self, graph: GraphLowering, mem_usage: dict[str, dict[str, bool | int]]
    ) -> None:
        for i, op in enumerate(graph.operations):
            rw = op.get_read_writes()
            for mem_dep in rw.reads | rw.writes:
                buf_name = mem_dep.name
                if "liveness_start" not in mem_usage[buf_name]:
                    mem_usage[buf_name]["liveness_start"] = i
                mem_usage[buf_name]["liveness_end"] = i + 1

    def _build_bound_buffers(
        self,
        graph: GraphLowering,
        in_place: dict[str, list[str]] = {},
    ) -> list[LifetimeBoundBuffer]:
        mem_usage = self._mem_usage_by_buffer(graph)

        return [
            LifetimeBoundBuffer(
                buffer_name,
                info["size_per_core"],
                info["liveness_start"],
                info["liveness_end"],
                in_place=in_place[buffer_name] if buffer_name in in_place else [],
            )
            for buffer_name, info in mem_usage.items()
            if not info["core_div_mismatch"]
        ]

    def _determine_in_place(
        self,
        graph: GraphLowering,
    ) -> dict[str, list[str]]:
        return {}

    def push_allocation(self, graph: GraphLowering, buffers: list[LifetimeBoundBuffer]):
        # push the allocation into the code generation
        for b in buffers:
            if b.address is not None:
                buf = graph.get_buffer(b.name)
                layout = buf.get_layout()
                layout.allocation["lx"] = b.address


class DefaultAllocator(ScratchpadAllocator):
    def __init__(
        self,
        layout_planning: MemoryPlanSolver | None = None,
        pre_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
        post_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
    ):
        if layout_planning is None:
            layout_planning = GreedyLayoutSolver()
        if pre_optimization_passes is None:
            pre_optimization_passes = [CloneInputNodesPass(layout_planning.limit)]
        if post_optimization_passes is None:
            post_optimization_passes = []

        self.pre_optimization_passes = pre_optimization_passes
        self.post_optimization_passes = post_optimization_passes
        self.layout_planning = layout_planning

    def plan_allocation(self, graph: GraphLowering):
        for p in self.pre_optimization_passes:
            p.apply_pass(graph)
        buffers = self.generate_buffers(graph)
        allocation = self.layout_planning.plan_layout(buffers)
        self.push_allocation(graph, allocation)
        for p in self.post_optimization_passes:
            p.apply_pass(graph)


def scratchpad_planning(
    graph: GraphLowering,
    allocator: Optional[ScratchpadAllocator] = None,
) -> None:
    # Operations are in topological order (guaranteed by GraphLowering).
    # Core division has already been done.
    # Stickification has already been done (therefore all ComputedBuffers have FixedTiledLayouts).
    if not allocator:
        allocator = DefaultAllocator()
    allocator.plan_allocation(graph)
