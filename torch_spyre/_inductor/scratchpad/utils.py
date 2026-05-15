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
from torch._inductor.ir import Operation
from torch_spyre._inductor import config


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
