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
from torch._inductor.ir import Operation, ComputedBuffer
from torch_spyre._inductor import config


def calculate_liveness(
    mem_usage: dict[str, dict[str, bool | int]], ops: Operation
) -> dict:
    for i, op in enumerate(ops):
        rw = op.get_read_writes()
        for mem_dep in rw.reads | rw.writes:
            buf_name = mem_dep.name
            if buf_name not in mem_usage:
                mem_usage[buf_name] = {}
            if "liveness_start" not in mem_usage[buf_name]:
                mem_usage[buf_name]["liveness_start"] = i
            mem_usage[buf_name]["liveness_end"] = i + 1
    return mem_usage


def mem_usage_by_op(
    graph: GraphLowering,
    ops: list[ComputedBuffer],
    core_div_mismatch: dict[str, bool] = {},
) -> tuple[dict, dict]:
    """
    Get a summary of memory usage of the given operation. Two types of info can be found
    1. Name lists, e.g. mem_usage["all_inputs"], or "all_outputs", "all_buf_used"
    2. Detailed info of individual buf, e.g. mem_usage[<buf_name>], which has
        "is_input", "size", "core_div_mismatch", "last_usage" fields
    NOTE:
    if a buf is not in core_div_mismatch => it has no users => graph output
    if a buf is on release_next => it's the last time it'll be used => allow inplace
    """
    mem_usage: dict = {}
    for op in ops:
        rw = op.get_read_writes()
        mem_usage[op.name] = {
            "all_inputs": [],
            "all_outputs": [],
        }
        num_cores = math.prod(
            [s for p in getattr(op, "op_it_space_splits", ()) for s in p.values()]
        )

        for is_input, deps in [(True, rw.reads), (False, rw.writes)]:
            for dep in deps:
                buf = graph.get_buffer(dep.name)
                dev_layout = buf.layout.device_layout
                dev_size = (
                    math.prod(dev_layout.device_size[:-1]) * 128
                )  # num_sticks * bytes_per_stick
                mem_usage[op.name][dep.name] = {
                    "is_input": is_input,
                    "size": dev_size,
                    "size_per_core": dev_size // num_cores,
                    "core_div_mismatch": core_div_mismatch.get(dep.name, False),
                }

                if is_input:
                    mem_usage[op.name]["all_inputs"].append(dep.name)
                else:
                    mem_usage[op.name]["all_outputs"].append(dep.name)
        mem_usage[op.name]["all_buf_used"] = (
            mem_usage[op.name]["all_inputs"] + mem_usage[op.name]["all_outputs"]
        )
    return mem_usage, calculate_liveness({}, ops)


def buf_analysis(operations: list[Operation]):
    """
    First, find out the last time each buffer was used. {buf1: idx_last_used, ...}
    Turn it into {idx_last_used+1:[buf1, ], ...}, ie. buffers to be deleted at given idx
    Then check work division -> If any of the operations on a given buffer has different
    work division => should not pin this buffer to LX
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


def get_buffer_users(graph: GraphLowering) -> dict[str, list[Operation]]:
    buf_users_read_and_write: dict[str, list[Operation]] = {}
    for op in graph.operations:
        rw = op.get_read_writes()
        for dep in rw.reads | rw.writes:  # union of the OrderedSets
            buf = dep.name  # buffer name, i.e. a str
            buf_users_read_and_write[buf] = buf_users_read_and_write.get(buf, []) + [op]
    return buf_users_read_and_write


def get_ncores_for_buffers(graph: GraphLowering) -> dict[str, int]:
    """
    Return a dictionary mapping buffer names to the number of cores
    used by all the operations that uses the buffer.
    If there is a core division mismatch return -1 instead of the
    number of cores.
    """
    result: dict[str, int] = {}
    using_multicore = config.sencores > 1
    buf_users_read_and_write = get_buffer_users(graph)
    for buf_name, users_rw in buf_users_read_and_write.items():
        # this dict includes graph input and output
        if using_multicore:
            # graph input and output can have only 1 read or 1 write user.
            u0_split = users_rw[0].op_it_space_splits  # a list like [16, 1]
            same_core_div = all(u0_split == u.op_it_space_splits for u in users_rw[1:])
            num_cores = (
                math.prod([s for p in u0_split for s in p.values()])
                if same_core_div
                else -1
            )
        else:
            num_cores = 1
        result[buf_name] = num_cores
    return result


def calculate_buffer_statistics(graph: GraphLowering) -> dict[str, dict[str, int]]:
    buf_read_counts: dict[str, int] = {}
    buf_write_counts: dict[str, int] = {}

    for op in graph.operations:
        rw = op.get_read_writes()
        read_names = op.get_read_names()
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
