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
from typing import Optional
from torch._inductor.graph import GraphLowering
from torch_spyre._inductor.scratchpad.plan_solver import (
    MemoryPlanSolver,
    LifetimeBoundBuffer,
)
from torch_spyre._inductor.scratchpad.passes import ScratchpadOptimizationPass
from torch_spyre._inductor.scratchpad.utility import (
    is_permissible_op,
    is_core_division_equal,
    is_graph_edge,
    calculate_liveness,
    push_allocation,
    mem_usage_by_buffer,
)


class GraphBufferConverstion:
    def __init__(
        self,
        permitted_ops: list[str] = [
            "max",
            "sum",
            "exp",
            "sub",
        ],
        in_place_ops: list[str] = ["exp", "sub"],
    ):
        self.permitted_ops = permitted_ops
        self.in_place_ops = in_place_ops

    def _filter_buffers(
        self, graph: GraphLowering, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        permissible_ops = is_permissible_op(graph, self.permitted_ops)
        core_division_match = is_core_division_equal(graph)
        input_output = is_graph_edge(graph)

        drop_list = (
            [key for key, permissible in permissible_ops.items() if not permissible] + 
            [key for key, permissible in core_division_match.items() if not permissible] +
            [key for key, edge in input_output.items() if edge]
        )

        return [b for b in buffers if b.name not in drop_list]

    def _build_bound_buffers(
        self,
        graph: GraphLowering,
        heuristics: dict[str, float] = {},
        in_place: dict[str, list[str]] = {},
    ) -> list[LifetimeBoundBuffer]:
        lifetime_starts, lifetime_ends = calculate_liveness(graph)
        sizes = mem_usage_by_buffer(graph)

        assert lifetime_starts.keys() == sizes.keys(), (
            "The keys in lifetimes and sizes must match"
        )

        assert lifetime_starts.keys() == lifetime_ends.keys(), (
            "The keys in lifetimes must match"
        )

        return [
            LifetimeBoundBuffer(
                buffer_name,
                sizes[buffer_name],
                lifetime_starts[buffer_name],
                lifetime_ends[buffer_name],
                heuristics[buffer_name] if buffer_name in heuristics else None,
                in_place=in_place[buffer_name] if buffer_name in in_place else [],
            )
            for buffer_name in lifetime_starts.keys()
        ]

    def _determine_in_place(
        self, graph: GraphLowering, in_place_ops: list[str]
    ) -> dict[str, list[str]]:
        return {}

    def generate_buffers(self, graph: GraphLowering) -> list[LifetimeBoundBuffer]:
        heuristics: dict[
            str, float
        ] = {}  # TODO: implement a concrete way of calculating this from a graph
        in_place = self._determine_in_place(graph, self.in_place_ops)
        buffers = self._build_bound_buffers(graph, heuristics, in_place)
        filtered_buffers = self._filter_buffers(graph, buffers)
        return filtered_buffers


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


class DefaultAllocator(ScratchpadAllocator):
    def __init__(
        self,
        layout_planning: MemoryPlanSolver,
        pre_optimization_passes: list[ScratchpadOptimizationPass] = [],
        post_optimization_passes: list[ScratchpadOptimizationPass] = [],
        buffer_source: Optional[GraphBufferConverstion] = None,
    ):
        assert layout_planning is not None
        self.pre_optimization_passes = pre_optimization_passes
        self.post_optimization_passes = post_optimization_passes
        self.layout_planning = layout_planning
        self.buffer_source = (
            buffer_source if buffer_source is not None else GraphBufferConverstion()
        )

    def plan_allocation(self, graph: GraphLowering):
        for p in self.pre_optimization_passes:
            p.apply_pass(graph)
        buffers = self.buffer_source.generate_buffers(graph)
        allocation = self.layout_planning.plan_layout(buffers)
        push_allocation(graph, allocation)
        for p in self.post_optimization_passes:
            p.apply_pass(graph)
