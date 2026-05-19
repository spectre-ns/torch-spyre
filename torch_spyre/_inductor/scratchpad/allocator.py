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

from abc import ABC, abstractmethod
from typing import Any, Optional

from torch._inductor.ir import ComputedBuffer, MutationLayoutSHOULDREMOVE
from torch._inductor.graph import GraphLowering

from torch_spyre._inductor.scratchpad.plan_solver import (
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
    MemoryPlanSolver,
)
from torch_spyre._inductor.scratchpad.passes import (
    CloneInputNodesPass,
    ScratchpadOptimizationPass,
)
from torch_spyre._inductor.scratchpad.utils import (
    mem_usage_by_op,
    buf_analysis,
    calculate_liveness,
)

from torch_spyre._inductor import config


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
        Accepts a graph to be considered for scratchpad memory according
        to its composition and the specific implementation used.

        Args:
            graph (GraphLowering): Graph to be considered for scratchpad planning
        """
        pass

    def _op_output_good_for_lx_reuse(self, op: Any) -> bool:
        return (
            isinstance(op, ComputedBuffer)
            and not isinstance(op.layout, MutationLayoutSHOULDREMOVE)
            and (
                config.allow_all_ops_in_lx_planning
                or (
                    op.origin_node is not None
                    and op.origin_node.target._opname in OP_OUTPUT_GOOD_FOR_LX_REUSE
                )
            )
        )

    def _op_good_for_lx_inplace(self, org_op_name: str) -> bool:
        return org_op_name in OP_GOOD_FOR_LX_INPLACE

    def _filter_buffers(
        self, graph: GraphLowering, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        """
        From the list of buffers, drop buffers that are outputs of
        unpermitted ops, graph outputs, and graph inputs
        """
        _, _, core_div_mismatch = buf_analysis(graph)
        drop_list = set()
        for op in graph.operations:
            if not self._op_output_good_for_lx_reuse(op):
                rw = op.get_read_writes()
                for mem_dep in rw.writes:
                    drop_list.add(mem_dep.name)

        drop_list.update(
            [key for key, mismatch in core_div_mismatch.items() if mismatch]
        )

        # These can be relaxed once node cloning is implemented as a post-solve optimization
        # rather than just filling the scratchpad at t = 0
        drop_list.update(graph.get_output_names())
        drop_list.update(graph.graph_input_names)

        # Clean up inplace so as to not rely on the solver to exclude inplace options which
        # are not valid scratchpad buffers.
        filtered_buffers = [b for b in buffers if b.name not in drop_list]
        valid_buffers = [buf.name for buf in filtered_buffers]
        for buf in filtered_buffers:
            buf.in_place = [name for name in buf.in_place if name in valid_buffers]

        return filtered_buffers

    def _build_bound_buffers(
        self,
        graph: GraphLowering,
        in_place: Optional[dict[str, list[str]]],
    ) -> list[LifetimeBoundBuffer]:
        lifetimes = calculate_liveness(graph)
        _, _, core_div_mismatch = buf_analysis(graph)
        mem_usage = mem_usage_by_op(graph, core_div_mismatch)
        in_place = {} if in_place is None else in_place
        buffers = []
        for _, op in mem_usage.items():
            for buffer_name in op["all_buf_used"]:
                if (
                    op[buffer_name]["is_lx_viable"]
                    and not op[buffer_name]["core_div_mismatch"]
                ):
                    buffers.append(
                        LifetimeBoundBuffer(
                            buffer_name,
                            op[buffer_name]["size_per_core"],
                            lifetimes[buffer_name]["liveness_start"],
                            lifetimes[buffer_name]["liveness_end"],
                            in_place=in_place[buffer_name]
                            if buffer_name in in_place
                            else [],
                        )
                    )

        buffers = list({obj.name: obj for obj in buffers}.values())
        return buffers

    def _determine_in_place(self, graph: GraphLowering) -> dict[str, list[str]]:
        allow_inplace: dict[str, list[str]] = {}
        _, _, core_div_mismatch = buf_analysis(graph)
        mem_usage = mem_usage_by_op(graph, core_div_mismatch)
        lifetimes = calculate_liveness(graph)
        for _, op_name in mem_usage.items():
            for input_buf in op_name["all_inputs"]:
                for output_buf in op_name["all_outputs"]:
                    if (
                        op_name[output_buf]["is_lx_viable"]
                        and op_name[input_buf]["is_lx_viable"]
                    ):
                        allow_inplace[output_buf] = allow_inplace.get(output_buf, [])
                        out_ten_layout = graph.get_buffer(
                            output_buf
                        ).layout.device_layout
                        in_ten_layout = graph.get_buffer(input_buf).layout.device_layout
                        out_start = lifetimes[output_buf]["liveness_start"]
                        in_end = lifetimes[input_buf]["liveness_end"]
                        out_size = op_name[output_buf]["size_per_core"]
                        in_size = op_name[input_buf]["size_per_core"]
                        inp_i_size_match = out_size == in_size
                        inp_i_lay_match = out_ten_layout == in_ten_layout
                        inp_i_eol = in_end == out_start + 1
                        no_core_div_mismatch = not (
                            op_name[output_buf]["core_div_mismatch"]
                            or op_name[input_buf]["core_div_mismatch"]
                        )
                        if (
                            inp_i_size_match
                            and inp_i_lay_match
                            and inp_i_eol
                            and no_core_div_mismatch
                        ):
                            allow_inplace[output_buf].append(input_buf)
        return allow_inplace

    def _generate_buffers(self, graph: GraphLowering) -> list[LifetimeBoundBuffer]:
        in_place = self._determine_in_place(graph)
        buffers = self._build_bound_buffers(graph, in_place)
        filtered_buffers = self._filter_buffers(graph, buffers)
        return filtered_buffers

    def _push_allocation(
        self, graph: GraphLowering, buffers: list[LifetimeBoundBuffer]
    ):
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
        """Configure the allocator with an optional solver and graph passes.

        Sub-components must handle types other than ComputeBuffer without
        filtering from the harness.

        Args:
            layout_planning: Solver that assigns LX addresses to lifetime-bound
                buffers. Defaults to GreedyLayoutSolver sized to available LX memory.
            pre_optimization_passes: Graph passes applied before layout planning.
                Defaults to [CloneInputNodesPass].
            post_optimization_passes: Graph passes applied after layout planning.
                Defaults to no passes.
        """
        size = int((2 << 20) * (1.0 - config.dxp_lx_frac_avail))
        if layout_planning is None:
            layout_planning = GreedyLayoutSolver(size)
        if pre_optimization_passes is None:
            pre_optimization_passes = [CloneInputNodesPass(size)]
        if post_optimization_passes is None:
            post_optimization_passes = []

        self.pre_optimization_passes = pre_optimization_passes
        self.post_optimization_passes = post_optimization_passes
        self.layout_planning = layout_planning

    def plan_allocation(self, graph: GraphLowering):
        """Run pre-passes, assign LX addresses to eligible buffers, then run post-passes.

        Args:
            graph: Lowered graph whose buffers will be assigned LX scratchpad
                addresses where viable.
        """
        for p in self.pre_optimization_passes:
            p.apply_pass(graph)
        buffers = self._generate_buffers(graph)
        allocation = self.layout_planning.plan_layout(buffers)
        self._push_allocation(graph, allocation)
        for p in self.post_optimization_passes:
            p.apply_pass(graph)


def scratchpad_planning(
    graph: GraphLowering,
    allocator: Optional[ScratchpadAllocator] = None,
) -> None:
    """Assign LX scratchpad addresses to eligible buffers in a lowered graph.

    Called after stickification and core-division are complete. Graph operations
    are expected to be in topological order as guaranteed by GraphLowering.

    Args:
        graph: Lowered graph to plan scratchpad memory for.
        allocator: Allocator strategy to use. Defaults to DefaultAllocator.
    """
    if allocator is None:
        allocator = DefaultAllocator()
    allocator.plan_allocation(graph)
