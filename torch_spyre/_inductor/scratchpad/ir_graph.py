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

"""The solve's graph built straight from the IR (Phase 3c).

:func:`build_graph_from_ir` makes one node per op output and per graph input
the allocator plans (which buffers take part is the allocator's decision),
each filled from the IR: its size from the device layout, its uses and
producing tick from liveness, its op and split space, and a read edge per
``MemoryDep`` read between two of them. The graph's division mode is a build
setting. :func:`compare_graphs` checks it against the graph built from the
allocator's buffers; ``LX_IR_GRAPH_CHECK=<file>`` records that agreement per
solve.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Collection
from typing import Any, Literal, Mapping, Optional

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering

from torch_spyre._inductor.scratchpad.buffer_graph import (
    ResourceNode,
    SolveSettings,
    SpyreOptimizationGraph,
    WorkSplit,
)
from torch_spyre._inductor.scratchpad.plan_solver import BufferType, ceil_div


def _device_bytes(lowering: GraphLowering, name: str) -> int:
    from torch_spyre._C import get_device_size_in_bytes

    return get_device_size_in_bytes(
        lowering.get_buffer(name).get_layout().device_layout
    )


def build_graph_from_ir(
    lowering: GraphLowering,
    names: Collection[str],
    alignment: int,
    capacity: int,
    settings: SolveSettings,
    division_mode: Literal["open", "committed"] = "open",
    spaces: Mapping[str, Any] = {},
) -> SpyreOptimizationGraph:
    """The graph of the buffers named ``names``, from the IR alone. In the
    ``"committed"`` mode every work split is committed upstream; in
    ``"open"`` an op with a split space in ``spaces`` is left open."""
    from torch_spyre._inductor.pass_utils import op_read_writes
    from torch_spyre._inductor.scratchpad.utils import calculate_liveness

    graph = SpyreOptimizationGraph(
        alignment=alignment,
        capacity=capacity,
        settings=settings,
        division_mode=division_mode,
    )
    lifetimes = calculate_liveness(lowering)
    ops = {op.get_name(): op for op in lowering.operations}
    outputs = set(lowering.get_output_names())
    inputs = set(lowering.graph_input_names)
    for name in names:
        uses = lifetimes.get(name) or []
        if not uses:
            continue
        nbytes = _device_bytes(lowering, name)
        is_input = name in inputs
        op = None if is_input else ops.get(name)
        space = spaces.get(name) if division_mode == "open" else None
        graph.nodes[name] = ResourceNode(
            name=name,
            size=ceil_div(nbytes, alignment),
            start_time=uses[0],
            end_time=uses[-1] + 1,
            read_count=len(uses) - (0 if is_input else 1),
            first_use_is_read=is_input,
            boundary=(
                BufferType.Input
                if is_input
                else BufferType.Output
                if name in outputs
                else BufferType.Intermediate
            ),
            sym_core_divs={},
            sym_tile_counts={},
            produced_at=None if is_input else uses[0],
            op=op,
            space=space,
            uses=tuple(uses),
            nbytes=nbytes,
            work_split=None if space is not None else WorkSplit.COMMITTED,
        )
    for name, node in graph.nodes.items():
        op = ops.get(name)
        if op is None or node.produced_at is None:
            continue
        reads = Counter(
            d.name
            for d in op_read_writes(op).reads
            if isinstance(d, MemoryDep) and d.name in graph.nodes and d.name != name
        )
        for parent, count in reads.items():
            graph.nodes[parent].add_child(node, reads=count)
    graph.link_execution_order()
    return graph


def compare_graphs(ir: SpyreOptimizationGraph, ref: SpyreOptimizationGraph) -> dict:
    """Agreement of the IR graph with the buffer graph: the node set, and
    per node its producing tick, size, nominal lifetime and read edges.
    Lifetime overrides and edges the allocator adds or drops by convention
    (drained collectives, loop-carry updates, in-place-only edges) show up
    as differences, by kind."""
    out: Counter = Counter()
    diffs: list = []
    out["missing_nodes"] = len(set(ref.nodes) - set(ir.nodes))
    out["extra_nodes"] = len(set(ir.nodes) - set(ref.nodes))
    for name, r in ref.nodes.items():
        n = ir.nodes.get(name)
        if n is None:
            diffs.append(("missing", name))
            continue
        for field in ("produced_at", "size", "uses"):
            if getattr(n, field) == getattr(r, field):
                out[f"{field}_ok"] += 1
            else:
                out[field] += 1
                diffs.append((field, name, getattr(n, field), getattr(r, field)))
        if (n.start_time, n.end_time) == (r.start_time, r.end_time):
            out["lifetime_ok"] += 1
        else:
            out["lifetime_override"] += 1
        got = {e.src.name: e.reads for e in n.inputs}
        want = {e.src.name: e.reads for e in r.inputs if e.reads}
        if got == want:
            out["reads_ok"] += 1
        else:
            out["reads"] += 1
            diffs.append(
                (
                    "reads",
                    name,
                    sorted(set(got) - set(want)),
                    sorted(set(want) - set(got)),
                )
            )
    return {"counts": dict(out), "diffs": diffs[:20]}


def record_agreement(
    lowering: GraphLowering, buffers, alignment: int, division_mode: str
) -> Optional[dict]:
    """With ``LX_IR_GRAPH_CHECK`` set, build both graphs for this solve and
    append their agreement to that file."""
    path = os.environ.get("LX_IR_GRAPH_CHECK")
    if not path:
        return None
    from torch_spyre._inductor.scratchpad.buffer_graph import build_graph
    from torch_spyre._inductor.scratchpad.plan_solver import RelayoutCopyBuffer

    real = [b for b in buffers if not isinstance(b, RelayoutCopyBuffer)]
    settings = SolveSettings(None, False, lambda *a, **k: 0)
    ref = build_graph(real, alignment, 1 << 40, settings)
    ir = build_graph_from_ir(
        lowering,
        [b.name for b in real if b.name in ref.nodes],
        alignment,
        1 << 40,
        settings,
        division_mode,  # type: ignore[arg-type]
    )
    record = compare_graphs(ir, ref)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record
