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

import math
from typing import Callable, Optional, Tuple
from abc import abstractmethod
from collections.abc import Sequence
import functools
from torch_spyre._inductor.layout_backend import (
    AllocationResult,
    Component,
    LayoutSolver,
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
)
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
)
from torch._inductor.lowering import lowerings, clone as clone_lowering
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.virtualized import V
from torch._inductor.graph import GraphLowering
from .logging_utils import get_inductor_logger
from .ir import FixedTiledLayout, TensorBox
from . import config

OP_OUTPUT_GOOD_FOR_LX_REUSE = [
    "max",
    "sum",
    "clone",
    # "exp",
    # "mul",
]

logger = get_inductor_logger("LX_PLANNING")
__LX_CAPACITY__ = int((2 << 20) * (1.0 - config.dxp_lx_frac_avail))


def calculate_liveness(ops: list[Operation]) -> Tuple[dict[str, int], dict[str, int]]:
    # Verify that the actual run's allocation is valid. We assume that any allocation is "live"
    # during the entire liveness of the corresponding buffer.
    liveness_start = {}
    liveness_end = {}
    for i, op in enumerate(ops):
        rw = op.get_read_writes()
        for mem_dep in rw.reads | rw.writes:
            buffer_name = mem_dep.name
            if buffer_name not in liveness_start:
                liveness_start[buffer_name] = i
            liveness_end[buffer_name] = i
    return liveness_start, liveness_end

# TODO: Update this to either be obsolete or broken up into distinct checks
def buf_analysis(operations: list[Operation]):
    """
    First, find out the last time each buffer was used. {buf1: idx_last_used, ...}
    Turn it into {idx_last_used+1:[buf1, ], ...}, ie. buffers to be deleted at given idx
    Then check core division -> If any of the operations on a given buffer has different
    core division => should not pin this buffer to LX
    NOTE Because each core can only write to its own scratchpad. For example, if a
            buffer is sliced 8 ways (stored on 8 LX) but next Op is 4-cores -> each core
            in next op has to read from 2 different scratchpads...
    TODO looking for options to broadcast to or all_reduce from multiple scratchpad
    """
    last_used: dict = {}
    buf_read_counts: dict[str, int] = {}
    buf_write_counts: dict[str, int] = {}
    buf_users: dict[str, Operation] = {}
    buf_users_read_and_write: dict[str, list[Operation]] = {}
    core_div_mismatch: dict[str, bool] = {}

    for idx, op in enumerate(operations):
        rw = op.get_read_writes()
        read_names = op.get_read_names()
        for dep in rw.reads | rw.writes:  # union of the OrderedSets
            buf = dep.name  # buffer name, i.e. a str
            last_used[buf] = idx
            if buf in read_names:
                buf_read_counts[buf] = buf_read_counts.get(buf, 0) + 1
                buf_users[buf] = buf_users.get(buf, []) + [op]
            else:
                buf_write_counts[buf] = buf_write_counts.get(buf, 0) + 1
            buf_users_read_and_write[buf] = buf_users_read_and_write.get(buf, []) + [op]

    bufs_to_dealloc_at_idx: dict = {}
    for buf, idx in last_used.items():
        # if last used at idx => del at idx+1
        if idx + 1 in bufs_to_dealloc_at_idx:
            bufs_to_dealloc_at_idx[idx + 1].append(buf)
        else:
            bufs_to_dealloc_at_idx[idx + 1] = [buf]

    using_multicore = config.sencores > 1
    for buf_name, users_rw in buf_users_read_and_write.items():
        # this dict includes graph input and output
        same_core_div = True
        if using_multicore and len(users_rw) > 1:
            # graph input and output can have only 1 read or 1 write user.
            u0_split = users_rw[0].op_it_space_splits  # a list like [16, 1]
            same_core_div = all(u0_split == u.op_it_space_splits for u in users_rw[1:])
        core_div_mismatch[buf_name] = not same_core_div

    return bufs_to_dealloc_at_idx, buf_users, core_div_mismatch


class AbstractAllocator:
    def __init__(self, graph):
        self.graph = graph

    @abstractmethod
    def plan_allocation(self, operations: list[Operation]):
        pass

    def push_allocation(self, allocation: AllocationResult):
        # push the allocation into the code generation
        for b in allocation:
            buf = self.graph.get_buffer(b.buffer)
            layout = buf.get_layout()
            layout.allocation["lx"] = b.address

    def op_output_good_for_lx_reuse(self, org_op_name: str) -> bool:
        return any(op in org_op_name for op in OP_OUTPUT_GOOD_FOR_LX_REUSE)

    def get_output_names(self) -> list[str]:
        return self.graph.get_output_names()

    def is_graph_input(self, buffer: str) -> bool:
        return buffer not in self.graph.name_to_buffer

    def mem_usage_by_op(self, op: ComputedBuffer) -> dict[str, dict[str, bool | int]]:
        """Get a summary of memory usage of the input operation."""
        rw = op.get_read_writes()
        mem_usage = {}

        for is_input, deps in [(True, rw.reads), (False, rw.writes)]:
            for dep in deps:
                buf = self.graph.get_buffer(dep.name)
                dev_layout = buf.layout.device_layout
                dev_size = (
                    math.prod(dev_layout.device_size[:-1]) * 128
                )  # num_sticks * bytes_per_stick
                mem_usage[dep.name] = {
                    "is_input": is_input,
                    "size": dev_size,
                }

        return mem_usage


# Potential other optimizations... Output cloning, buffer lifetime splitting, rematerialization (advanced)
class SpyreLxOptimizationPass:
    @abstractmethod
    def apply_pass(self, operations: list[Operation]) -> list[Operation]:
        pass


class InputBufferOptimization(SpyreLxOptimizationPass):
    def __init__(
        self,
        graph_lowering: Optional[GraphLowering] = None,
    ):
        self.graph_lowering = graph_lowering if graph_lowering else V.graph

    def mem_usage_by_op(self, op: ComputedBuffer) -> dict[str, dict[str, bool | int]]:
        """Get a summary of memory usage of the input operation."""
        rw = op.get_read_writes()
        mem_usage = {}

        for is_input, deps in [(True, rw.reads), (False, rw.writes)]:
            for dep in deps:
                buf = V.graph.get_buffer(dep.name)
                dev_layout = buf.layout.device_layout
                dev_size = (
                    math.prod(dev_layout.device_size[:-1]) * 128
                )  # num_sticks * bytes_per_stick
                mem_usage[dep.name] = {
                    "is_input": is_input,
                    "size": dev_size,
                }

        return mem_usage

    def should_consider_op(self, op: Operation) -> bool:
        return isinstance(op, ComputedBuffer) and not isinstance(
            op.layout, MutationLayoutSHOULDREMOVE
        )

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

    def insert_op_after(
        self,
        buf: TensorBox,
        lowering_func: Callable,
        buf_users: dict,
        operations: list[Operation],
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
        fx_graph = self.graph_lowering.graph

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
        self.graph_lowering.orig_gm.recompile()

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
        new_com_buf.name = self.graph_lowering.register_buffer(new_com_buf)
        self.graph_lowering.register_operation(new_com_buf)
        new_buf_name = new_com_buf.name

        # Step 3: Update graph_lowering.name_to_users (a list of TensorBox), eg, existing
        # users of arg0, other than InpBuf and new_buf, should become users of new_buf.
        users_of_inp, users_of_new_buf = [], []
        for tb in self.graph_lowering.name_to_users[buf_name]:
            if tb.data.data.name in [buf_name, new_buf_name]:
                users_of_inp.append(tb)
            else:
                users_of_new_buf.append(tb)
        self.graph_lowering.name_to_users[buf_name] = users_of_inp
        self.graph_lowering.name_to_users[new_buf_name] = users_of_new_buf

        # Step 4: Hack user nodes' inner_fn
        for old_com_buf in buf_users[buf_name]:
            # hack inner_fn with a nameSwapper ops handler and make a new LoopIR
            new_Loop = self.create_Loop_hack_inner_fn(
                old_com_buf.data, name_map={buf_name: new_buf_name}
            )
            old_com_buf.data = new_Loop

        # NOTE: operations is a reference to graph_lowering.operations, which is already
        # updated when we call graph_lowering.register_operation() earlier. But the new Op
        # was appended at the end of the list, need to insert at the correct position.
        first_user = buf_users[buf_name][0]
        idx_to_first_user = operations.index(first_user)
        operations.remove(new_com_buf)
        operations.insert(idx_to_first_user, new_com_buf)

    def try_insert_clone_op_for_inputs(
        self,
        operations: list[Operation],
        lx_free_total: int,
        buf_users: dict[str, Operation],
        core_div_mismatch: dict[str, bool],
    ) -> None:
        """
        Check if any input tensors can fit onto scratchpad and needed more than once =>
        Add corresponding "clone operation" to copy it to scratchpad and reduce HBM read.
        """
        for inp_name in self.graph_lowering.graph_input_names:
            buf = self.graph_lowering.get_buffer(inp_name)  # this is a TensorBox
            dev_layout = buf.layout.device_layout
            dev_size = math.prod(dev_layout.device_size[:-1]) * 128
            is_on_lx = buf.layout.allocation != {}
            used_only_once = len(buf_users[inp_name]) == 1
            if (
                used_only_once
                or dev_size > lx_free_total
                or is_on_lx
                or core_div_mismatch[inp_name]
            ):
                continue

            self.insert_op_after(buf, clone_lowering, buf_users, operations)

            lx_free_total -= dev_size

    def apply_pass(self, operations: list[Operation]) -> list[Operation]:
        idx_to_dealloc_bufs, buf_users, core_div_mismatch = buf_analysis(operations)

        if "clone" in OP_OUTPUT_GOOD_FOR_LX_REUSE:
            num_ops_before = len(operations)
            self.try_insert_clone_op_for_inputs(
                operations,
                __LX_CAPACITY__,
                buf_users,
                core_div_mismatch,
            )

            # refresh LUTs -- insertion may not happen, e.g. input tensor is used only once
            if len(operations) > num_ops_before:
                idx_to_dealloc_bufs, buf_users, core_div_mismatch = buf_analysis(
                    operations
                )

        return operations


class DefaultAllocator(AbstractAllocator):
    def __init__(
        self,
        optimization_passes: list[SpyreLxOptimizationPass] | None = None,
        layout_planning: list[LayoutSolver] | None = None,
        graph: GraphLowering | None = None,
    ):
        if graph:
            super().__init__(graph)
        else:
            super().__init__(V.graph)
        
        # allow no optimizations passes and guard where needed
        self.optimization_passes = optimization_passes

        # ensure a layout solver is always available. The greedy
        # solver is likely not optimal moving forward but it is the
        # default for now until others are verified.
        if layout_planning:
            self.layout_planning = layout_planning
        else:
            self.layout_planning = [GreedyLayoutSolver(__LX_CAPACITY__)]

    def plan_allocation(self, operations: list[Operation]):
        # compute the optimized graph with the optimized operation list
        # ideally not apply changes to the FX graph until the end but
        # currently FX graph updates are done stepwise.
        if self.optimization_passes:
            optimized_ops = functools.reduce(
                lambda intermediate_ops, optimization_pass: optimization_pass.apply_pass(
                    intermediate_ops
                ),
                self.optimization_passes,
                operations,
            )
        else:
            optimized_ops = operations

        mem_usage = {}
        for op in operations:
            mem_usage[op.name] = self.mem_usage_by_op(op)

        start_times, end_times = calculate_liveness(optimized_ops)

        # TODO: Get rid of this heinous looping structure
        buffer_list = {}
        for op in optimized_ops:
            rw = op.get_read_writes()
            for mem_dep in rw.reads | rw.writes:
                buffer_name = mem_dep.name
                if buffer_name not in buffer_list:
                    for _, buffer in mem_usage.items():
                        if buffer_name in buffer:
                            org_op_name = op.origin_node.target._opname
                            allowed_output_op = self.op_output_good_for_lx_reuse(
                                org_op_name
                            )
                            buffer_list[buffer_name] = {
                                "lx_compatible": allowed_output_op,
                                "size": buffer[buffer_name]["size"],
                                "start_time": start_times[buffer_name],
                                "end_time": end_times[buffer_name],
                            }

        # TODO: This is also pretty terrible
        _, _, core_div_mismatch = buf_analysis(optimized_ops)
        graph_output_buf_name = self.get_output_names()
        for tensor_name, meta_data in buffer_list.items():
            is_graph_input = self.is_graph_input(tensor_name)
            is_graph_output = tensor_name in graph_output_buf_name
            # core_div_mismatch = (not needed["is_input"]) and core_div_mismatch[tensor_name]
            is_core_div_mismatch = core_div_mismatch[tensor_name]
            if is_graph_input or is_graph_output or is_core_div_mismatch:
                # graph input itself cannot be pinned, but we may be able to clone
                # graph output has to go back to HBM
                # if buf users have diff core-splits -> cause cross-core LX read/write
                meta_data["lx_compatible"] = False

        # attempt to place the optimized buffers into LX
        def try_layout_with_fallback(
            strategies: Sequence[LayoutSolver],
            buffers: Sequence[LifetimeBoundBuffer]
        ) -> AllocationResult | None:
            if not buffers:
                return []

            final_layout = None
            for strategy in strategies:
                current_layout = strategy.plan_layout(buffers)

                # Check if this strategy places all buffers
                if all([buffer.component == Component.LX for buffer in current_layout]):
                    return current_layout  # Exit early if optimal

                final_layout = current_layout  # Store the last attempt

            # Return the best found or the last attempt
            return final_layout

        filtered_buffers = [
            LifetimeBoundBuffer(
                name, item["size"], item["start_time"], item["end_time"]
            )
            for name, item in buffer_list.items()
            if item["lx_compatible"]
        ]
        allocation = try_layout_with_fallback(
            self.layout_planning,
            filtered_buffers)

        if allocation:
            self.push_allocation(allocation)
            return

        logger.warning("LX layout planning failed. All buffers will reside in HBM")


def scratchpad_planning(
    operations: list[Operation],
    strategy: Optional[AbstractAllocator] = None,
) -> None:
    # Operations are in topological order (guaranteed by GraphLowering).
    # Core division has already been done.
    # Stickification has already been done (therefore all ComputedBuffers have FixedTiledLayouts).
    if not strategy:
        strategy = DefaultAllocator()
    strategy.plan_allocation(operations)
