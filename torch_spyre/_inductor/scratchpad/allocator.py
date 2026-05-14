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
from typing import Callable, Any, Optional

from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
)
from torch._inductor.lowering import lowerings, clone as clone_lowering
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.virtualized import V
from torch._inductor.graph import GraphLowering
from .ir import FixedTiledLayout, TensorBox
from . import config

from torch_spyre._inductor.scratchpad.plan_solver import (
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
    MemoryPlanSolver,
)
from torch_spyre._inductor.scratchpad.passes import ScratchpadOptimizationPass
from torch_spyre._inductor.scratchpad.utility import (
    get_buffer_users,
    get_ncores_for_buffers,
)


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

    class NameSwapHandler(WrapperHandler):
        def __init__(self, inner, name_map: dict[str, str]):
            super().__init__(inner)
            self._name_map = name_map

        def load(self, name, index):
            return super().load(self._name_map.get(name, name), index)

    def create_Loop_hack_inner_fn(self, old_Loop, name_map):
        """Use ops_handler to swap the name of buffers"""

        def new_inner_fn(*args):
            # Pointwise has 1 pos arg index while Reduction has 2, i.e. (index, rindex)
            with V.set_ops_handler(self.NameSwapHandler(V.ops, name_map)):
                return old_Loop.inner_fn(*args)

        # old_Loop could be a Pointwise or Reduction.
        kwargs = {k: getattr(old_Loop, k) for k in old_Loop.__dataclass_fields__.keys()}
        kwargs["inner_fn"] = new_inner_fn
        new_Loop = old_Loop.__class__(**kwargs)
        # Additional attr that are not included in dataclass_fields. NOTE it relies on a
        # special method to force reset attrs of a frozen dataclass, see ir.Loops.create()
        new_Loop._post_init_setattr("origins", old_Loop.origins)
        new_Loop._post_init_setattr("origin_node", old_Loop.origin_node)
        new_Loop._post_init_setattr("traceback", old_Loop.traceback)
        # .get_stack_traces() get info from "origins", no need to manually set anything
        # LoopBody will be created later when we call CompBuf.recompute()

        return new_Loop

    def try_insert_clone_op_for_inputs(
        self,
        graph: GraphLowering,
        lx_free_total: int,
    ) -> None:
        """
        Check if any input tensors can fit onto scratchpad and needed more than once =>
        Add corresponding "clone operation" to copy it to scratchpad and reduce HBM read.
        """
        buf_users = get_buffer_users(graph)
        ncores = get_ncores_for_buffers(graph)
        for inp_name in graph.graph_input_names:
            buf = graph.get_buffer(inp_name)  # this is a TensorBox
            dev_layout = buf.layout.device_layout
            dev_size = math.prod(dev_layout.device_size[:-1]) * 128
            is_on_lx = buf.layout.allocation != {}
            used_only_once = len(buf_users[inp_name]) == 1
            core_div_mismatch = ncores[inp_name] == -1
            if (
                used_only_once
                or dev_size > lx_free_total
                or is_on_lx
                or core_div_mismatch
            ):
                continue

            self.insert_op_after(buf, clone_lowering, buf_users, graph)
            lx_free_total -= dev_size

    def insert_op_after(
        self,
        buf: TensorBox,
        lowering_func: Callable,
        buf_users: dict,
        graph: GraphLowering,
    ) -> None:
        """
        Insert an operation using the provided lowering function (e.g. clone_lowering) in
        GraphLowering.operations list after the given op (buf, a TensorBox representing a
        ComputedBuffer). Will update GraphLowering FX graph and the operations list.
        For example, original ops list looks like:
            buf0 -> buf1 -> buf2
        insert a clone of buf0 and let buf1 read from it, will become
            buf0 ->(clone) buf3 -> buf1 ->buf2

        NOTE:
        - Simplified flow, everything is done before Scheduler. Only take care of FX and
          GraphLowering. list operations will be updated inplace, no need to return.
        - Even though it is not a necessary condition, we assume FX graph and Operations are
          fully consistent and we will try to maintain it that way.
        - To update existing users of the old buffer -> hack the inner_fn then refresh LoopIR
        """
        fx_graph = graph.graph

        # Step 1: Add a new FX node for clone and update dependencies
        buf_name = buf.data.data.name  # buf is a TensorBox
        buf_fx = list(buf.origins)[0]  # .origin_node may not exist
        old_users = list(buf_fx.users.keys())
        # make sure the user-provided lowering_func is legit
        assert lowering_func in lowerings.values(), (
            f"The provided lowering function {lowering_func} is not properly registered."
        )
        LUTlower_func_to_op = {func: aten_op for aten_op, func in lowerings.items()}
        user_aten_op = LUTlower_func_to_op[lowering_func]
        # TODO this is a large dict, move it to upper scope so we only need to do it once
        # aten_op is the overloaded version, e.g. ops.aten.clone.*out* instead of .default
        fx_graph.inserting_after(buf_fx)
        new_fx_node = fx_graph.create_node("call_function", user_aten_op, (buf_fx,))
        for user in old_users:
            user.args = tuple(new_fx_node if ar is buf_fx else ar for ar in user.args)
        graph.orig_gm.recompile()

        # Step 2: Create a new ComBuf of a Pointwise IR (need to support Reduction?)
        pw_ir_tb = lowering_func(buf)  # a TensorBox wrapping a PointwiseIR
        new_com_buf = ComputedBuffer(
            name=None,
            layout=FixedTiledLayout(
                buf.layout.device,
                buf.layout.dtype,
                buf.layout.size,
                buf.layout.stride,
                buf.layout.device_layout,
            ),  # create a new copy of FixedTiledLayout from buf's layout
            data=pw_ir_tb.data.data,
        )
        new_com_buf.origins.add(new_fx_node)
        new_com_buf.origin_node = new_fx_node
        # TODO why arg0 ComputedBuffer doesn't have this attr?
        new_com_buf.name = graph.register_buffer(new_com_buf)
        graph.register_operation(new_com_buf)
        new_buf_name = new_com_buf.name

        # Step 3: Update graph_lowering.name_to_users (a list of TensorBox), eg, existing
        # users of arg0, other than InpBuf and new_buf, should become users of new_buf.
        users_of_inp, users_of_new_buf = [], []
        for tb in graph.name_to_users[buf_name]:
            if tb.data.data.name in [buf_name, new_buf_name]:
                users_of_inp.append(tb)
            else:
                users_of_new_buf.append(tb)
        graph.name_to_users[buf_name] = users_of_inp
        graph.name_to_users[new_buf_name] = users_of_new_buf

        # Step 4: Hack user nodes' inner_fn
        for old_com_buf in buf_users[buf_name]:
            # hack inner_fn with a nameSwapper ops handler and make a new LoopIR
            new_Loop = self.create_Loop_hack_inner_fn(
                old_com_buf.data, name_map={buf_name: new_buf_name}
            )
            old_com_buf.data = new_Loop

        operations = graph.operations
        # NOTE: operations is a reference to graph_lowering.operations, which is already
        # updated when we call graph_lowering.register_operation() earlier. But the new Op
        # was appended at the end of the list, need to insert at the correct position.
        first_user = buf_users[buf_name][0]
        idx_to_first_user = operations.index(first_user)
        operations.remove(new_com_buf)
        operations.insert(idx_to_first_user, new_com_buf)

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
        layout_planning: Optional[MemoryPlanSolver] = None,
        pre_optimization_passes: list[ScratchpadOptimizationPass] = [],
        post_optimization_passes: list[ScratchpadOptimizationPass] = [],
    ):
        if layout_planning is None:
            layout_planning = GreedyLayoutSolver()
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
    strategy: Optional[ScratchpadAllocator] = None,
) -> None:
    # Operations are in topological order (guaranteed by GraphLowering).
    # Core division has already been done.
    # Stickification has already been done (therefore all ComputedBuffers have FixedTiledLayouts).
    if not strategy:
        strategy = DefaultAllocator()
    strategy.plan_allocation(graph)
