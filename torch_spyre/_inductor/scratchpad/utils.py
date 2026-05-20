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
from typing import Callable

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


def calculate_liveness(graph: GraphLowering) -> dict:
    liveness: dict[str, dict[str, bool | int]] = {}
    for i, op in enumerate(graph.operations):
        rw = op.get_read_writes()
        for mem_dep in rw.reads | rw.writes:
            buf_name = mem_dep.name
            if buf_name not in liveness:
                liveness[buf_name] = {}
            if "liveness_start" not in liveness[buf_name]:
                liveness[buf_name]["liveness_start"] = i
            liveness[buf_name]["liveness_end"] = i + 1
    return liveness


def mem_usage_by_op(graph: GraphLowering, ignore: list[str] = []) -> dict:
    """
    Get a summary of memory usage of the given operation. Two types of info can be found
    1. Name lists, e.g. mem_usage["all_inputs"], or "all_outputs", "all_buf_used"
    2. Detailed info of individual buf, e.g. mem_usage[<buf_name>], which has
        "is_input", "size", "core_div_mismatch", "last_usage" fields
    NOTE:
    if a buf is not in core_div_mismatch => it has no users => graph output
    if a buf is on release_next => it's the last time it'll be used => allow inplace
    """
    num_cores = get_ncores_for_buffers(graph)
    mem_usage: dict = {}
    for op in graph.operations:
        rw = op.get_read_writes()
        mem_usage[op.name] = {"all_inputs": []}
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
                    "size_per_core": dev_size // num_cores[op.name],
                    "core_div_mismatch": num_cores[op.name] < 0,
                }

                if is_input:
                    mem_usage[op.name]["all_inputs"].append(dep.name)
    return mem_usage


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


class GraphView(GraphLowering):
    def __init__(
        self, graph: GraphLowering, predicate: Callable[[GraphLowering], Operation]
    ):
        object.__init__(self)
        self.__dict__.update(graph.__dict__)
        self.graph = graph
        self.operations = predicate(graph)
