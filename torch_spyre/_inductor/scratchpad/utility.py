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
from torch._inductor.graph import GraphLowering
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer


def op_output_good_for_lx_reuse(org_op_name: str, op_list: list[str]) -> bool:
    return any(op in org_op_name for op in op_list)


def is_permissible_op(graph: GraphLowering, op_list: list[str]) -> dict[str, bool]:
    buf_permissible_op = {}
    for op in graph.operations:
        rw = op.get_read_writes()
        for mem_dep in rw.writes:
            buffer_name = mem_dep.name
            allowed_output_op = op_output_good_for_lx_reuse(
                op.origin_node.target._opname, op_list
            )
            buf_permissible_op[buffer_name] = allowed_output_op

    return buf_permissible_op


def get_buffer_users(graph: GraphLowering) -> dict[str, list[str]]:
    buf_users_read_and_write: dict[str, list[str]] = {}
    for op in graph.operations:
        rw = op.get_read_writes()
        for dep in rw.reads | rw.writes:  # union of the OrderedSets
            buf = dep.name  # buffer name, i.e. a str
            buf_users_read_and_write[buf] = buf_users_read_and_write.get(buf, []) + [
                op.name
            ]
    return buf_users_read_and_write


def determine_core_division(graph: GraphLowering) -> dict[str, bool]:
    core_div_mismatch: dict[str, bool] = {}
    users = get_buffer_users(graph)
    for buf_name, users_rw in users.items():
        # this dict includes graph input and output
        core_div_mismatch[buf_name] = True
        user_ops = [op for op in graph.operations if op.name in users_rw]
        if all([hasattr(op, "op_it_space_splits") for op in user_ops]):
            # graph input and output can have only 1 read or 1 write user.
            u0_split = user_ops[0].op_it_space_splits  # a list like [16, 1]
            core_div_mismatch[buf_name] = all(
                u0_split == u.op_it_space_splits for u in user_ops[1:]
            )
    return core_div_mismatch


def calculate_buffer_statistics(graph: GraphLowering) -> dict[str, dict[str, int]]:
    # this can be compressed more with some comprehensions.
    buf_read_counts: dict[str, int] = {}
    buf_write_counts: dict[str, int] = {}

    for op in graph.operations:
        rw = op.get_read_writes()
        read_names = op.get_read_names()  # <- Comprehension here
        for dep in rw.reads | rw.writes:
            buf = dep.name
            if buf in read_names:
                buf_read_counts[buf] = buf_read_counts.get(buf, 0) + 1
            else:
                buf_write_counts[buf] = buf_write_counts.get(buf, 0) + 1

    all_bufs = buf_read_counts.keys() | buf_write_counts.keys()
    return {
        buf: {
            "reads": buf_read_counts.get(buf, 0),
            "writes": buf_write_counts.get(buf, 0),
        }
        for buf in all_bufs
    }


def mem_usage_by_buffer(graph: GraphLowering) -> dict[str, int]:
    mem_usage = {}
    rw_list = [op.get_read_writes for op in graph.operations]
    deps = [rw.reads | rw.writes for rw in rw_list]
    bufs = [graph.get_buffer(dep.name) for dep in deps]

    def get_device_size(buf):
        dev_layout = buf.layout.device_layout
        dev_size = math.prod(dev_layout.device_size[:-1]) * 128
        return dev_size

    mem_usage = {buf: get_device_size(buf) for buf in bufs}

    return mem_usage


def calculate_liveness(graph: GraphLowering) -> tuple[dict[str, int], dict[str, int]]:
    liveness_start = {}
    liveness_end = {}
    for i, op in enumerate(graph.operations):
        rw = op.get_read_writes()
        for mem_dep in rw.reads | rw.writes:
            buffer_name = mem_dep.name
            if buffer_name not in liveness_start:
                liveness_start[buffer_name] = i
            liveness_end[buffer_name] = i + 1
    return liveness_start, liveness_end


def push_allocation(graph: GraphLowering, buffers: list[LifetimeBoundBuffer]):
    # push the allocation into the code generation
    for b in buffers:
        if b.address is not None:
            buf = graph.get_buffer(b.name)
            layout = buf.get_layout()
            layout.allocation["lx"] = b.address
