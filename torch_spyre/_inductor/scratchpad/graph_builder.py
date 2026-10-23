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

"""The solve's graph built from the IR in the committed division mode
(placement only): one node per op output and per graph input with a
lifetime, carrying its per-core footprint under the division work division
committed, its residency verdict and its in-place parents; a committed LX
relayout's destinations as transforms pinned on their source. The open
mode's graph comes from the table strategy's candidate generator
(``TableCandidates.prepare_graph``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from torch._inductor.graph import GraphLowering

from torch_spyre._inductor import config
from torch_spyre._inductor.scratchpad.drain_plans import (
    _handoff_child_start,
    _handoff_parent_end,
)
from torch_spyre._inductor.scratchpad import residency_rules
from torch_spyre._inductor.scratchpad.buffer_graph import (
    SolveSettings,
    SpyreOptimizationGraph,
)
from torch_spyre._inductor.scratchpad.lx_relayout import (
    LXRelayoutPlan,
    collect_lx_relayout_plans,
    materialized_lx_relayouts,
)
from torch_spyre._inductor.scratchpad.utils import (
    calculate_liveness,
    counted_loop_lifetime_overrides,
    get_ncores_for_buffers,
    mem_usage_by_buf,
    round_up_to_alignment,
)

if TYPE_CHECKING:
    from torch_spyre._inductor.scratchpad.allocator import ScratchpadAllocator

ALIGNMENT = 128  # bytes per unit: one stick, what every solver aligns to
_LX_ALLOCATION_GRANULARITY_BYTES = 128


def _unposted(*args, **kwargs) -> int:
    raise AssertionError("the solver sets the solve settings")


def empty_graph(capacity_bytes: int, division_mode: str) -> SpyreOptimizationGraph:
    return SpyreOptimizationGraph(
        alignment=ALIGNMENT,
        capacity=capacity_bytes // ALIGNMENT,
        settings=SolveSettings(None, False, _unposted),
        division_mode=division_mode,  # type: ignore[arg-type]
    )


@dataclass
class _Entry:
    """One node's incoming data while the builder still adjusts it."""

    nbytes: int
    uses: list[int]
    first_use_is_read: bool = False
    in_place: list[str] = field(default_factory=list)
    reason: Optional[str] = None
    start_override: Optional[int] = None
    end_override: Optional[int] = None
    lx_view: Any = None
    plans: list[LXRelayoutPlan] = field(default_factory=list)
    pinned: list[dict] = field(default_factory=list)


def placement_graph(
    alloc: ScratchpadAllocator,
    lowering: GraphLowering,
    lx_relayout_plans: Optional[Sequence[LXRelayoutPlan]] = None,
) -> SpyreOptimizationGraph:
    """The committed-mode graph of ``lowering``. A solver that places a
    committed relayout's source with its destinations
    (``supports_paired_buffers``) gets the relayout plans: collected here
    when ``lx_relayout_plans`` is ``None``, else the ones given."""
    plans: Sequence[LXRelayoutPlan] = ()
    if not alloc.capabilities.supports_paired_buffers:
        if config.lx_planner_relayout:
            alloc_logger().debug(
                "LX relayout is not supported by %s; continuing without relayout",
                alloc.solver.name,
            )
        paired = False
    else:
        paired = True
        if lx_relayout_plans is None:
            plans = collect_lx_relayout_plans(lowering)
        elif not config.lx_planner_relayout or config.ktir_emitter:
            plans = []
        else:
            if materialized_lx_relayouts(lowering):
                raise RuntimeError(
                    "LX relayout planning requires an unmaterialized graph"
                )
            plans = lx_relayout_plans
    entries = _entries(alloc, lowering, plans)
    if paired:
        _relayout_destinations(alloc, lowering, entries)
    graph = empty_graph(alloc.size, "committed")
    for name, e in entries.items():
        node = graph.add_ir_node(
            name,
            nbytes=e.nbytes,
            uses=e.uses,
            first_use_is_read=e.first_use_is_read,
            residency_reason=e.reason,
            start_override=e.start_override,
            end_override=e.end_override,
            lx_view=e.lx_view,
            lx_relayout_plans=e.plans,
        )
        for dest in e.pinned:
            graph.pin_transform(node, **dest)
    for name, e in entries.items():
        graph.add_reads(graph.nodes[name], (), e.in_place)
    graph.link_execution_order()
    return graph


def alloc_logger():
    from torch_spyre._inductor.scratchpad.allocator import logger

    return logger


def _entries(
    alloc: ScratchpadAllocator,
    graph: GraphLowering,
    lx_relayout_plans: Sequence[LXRelayoutPlan],
) -> dict[str, _Entry]:
    """One entry per buffer, barred or not; only buffers with no lifetime are
    skipped (an unused graph input has no ``uses[0]``). An ineligible buffer
    carries its ``residency_reason`` and the solver declines to place it.

    Graph inputs are pinned by cloning them into LX rather than placed
    directly, so their verdict comes from ``_input_residency_reason`` and
    their footprint is computed here rather than read off ``mem_usage``
    (which covers ops only)."""
    lifetimes = calculate_liveness(graph)
    start_overrides, end_overrides = counted_loop_lifetime_overrides(graph)
    ncores, ncores_reasons, lx_views = get_ncores_for_buffers(graph)
    mem_usage = mem_usage_by_buf(graph)
    for plan in lx_relayout_plans:
        name = plan.source_name
        if name not in mem_usage:
            continue
        ncores[name] = plan.source_view.num_cores or plan.num_cores
        ncores_reasons.pop(name, None)
        lx_views[name] = plan.source_view
        # Only sources exist in the graph here. Each private destination
        # receives its own view and bound in _relayout_destinations. Shared
        # sources keep the largest packed footprint; ordinary buffers keep
        # mem_usage_by_buf's equal-share size unchanged.
        mem_usage[name]["size_per_core"] = max(
            mem_usage[name]["size_per_core"], plan.source_footprint_bytes
        )
        mem_usage[name]["core_div_mismatch"] = False

    # Divisions are already committed on this path, so a core-division
    # mismatch between a buffer's users is fatal and is checked here.
    reasons = residency_rules._residency_reasons(
        graph,
        list(mem_usage),
        division_is_fixed=True,
        lifetimes=lifetimes,
        ncores=ncores,
        ncores_reasons=ncores_reasons,
        planned_lx_buffers=alloc._planned_lx_buffer_names(lx_relayout_plans),
        lx_relayout_plans=lx_relayout_plans,
    )
    in_place = _in_place(alloc, graph, mem_usage, lifetimes, reasons)
    entries: dict[str, _Entry] = {}
    for name, info in mem_usage.items():
        uses = lifetimes.get(name, [])
        if not uses:
            continue
        entries[name] = _Entry(
            # An unsized (-1) or core-div-mismatched (negative) entry is always
            # barred, so its footprint is never used; clamp it so a nonsense
            # size can never look placeable.
            nbytes=max(0, info["size_per_core"]),
            uses=list(uses),
            in_place=list(in_place.get(name, [])),
            reason=reasons.get(name),
            start_override=start_overrides.get(name),
            end_override=end_overrides.get(name),
            lx_view=lx_views.get(name),
        )
    # A graph-input clone can only ever be an in-place *parent*: it is pinned
    # to LX and dies at its last read, and its source stays in HBM. The value
    # is letting the consumer that performs that last read reuse the clone's
    # slot for its own output.
    outputs = dict(entries)
    for input_name in graph.graph_input_names:
        uses = lifetimes.get(input_name, [])
        if not uses:
            continue
        reason = residency_rules._input_residency_reason(
            graph,
            input_name,
            uses,
            ncores=ncores,
            ncores_reasons=ncores_reasons,
            division_is_fixed=True,
        )
        clone_size = _input_footprint(graph, input_name, ncores)
        entries[input_name] = _Entry(
            nbytes=clone_size,
            uses=list(uses),
            first_use_is_read=True,
            reason=reason,
            start_override=start_overrides.get(input_name),
            end_override=end_overrides.get(input_name),
            lx_view=lx_views.get(input_name),
        )
        # Reverse-parent edge (issue #3212): let the input clone's last
        # consumer reuse the clone's LX slot in place. The op at the clone's
        # last-use tick both reads the clone and writes its own output, so
        # the single-handoff-tick invariant holds for that consumer alone.
        # Only when the clone can reside and the consumer has a matching
        # per-core size, device layout, a pointwise producer, and no
        # core-division mismatch is there anything safe to merge.
        if reason is not None:
            continue
        consumer_op = graph.operations[uses[-1]]
        consumer = outputs.get(consumer_op.name)
        if consumer is None or input_name in consumer.in_place:
            continue
        # A multi-output op carries a MultiOutputLayout with no single
        # ``device_layout``, so it cannot alias one input clone in place.
        consumer_layout = graph.get_buffer(consumer_op.name).get_layout()
        input_layout = graph.get_buffer(input_name).layout
        if not hasattr(consumer_layout, "device_layout") or not hasattr(
            input_layout, "device_layout"
        ):
            continue
        if residency_rules._inplace_edge_ok(
            child_pointwise_inputs=residency_rules._op_inputs_good_for_lx_inplace(
                consumer_op
            ),
            parent_name=input_name,
            child_size_per_core=consumer.nbytes,
            parent_size_per_core=clone_size,
            child_device_layout=consumer_layout.device_layout,
            parent_device_layout=input_layout.device_layout,
            child_start=_handoff_child_start(
                consumer_op.name, lifetimes, start_overrides
            ),
            parent_end=_handoff_parent_end(input_name, lifetimes, end_overrides),
            child_core_div_mismatch=mem_usage[consumer_op.name]["core_div_mismatch"],
        ):
            consumer.in_place.append(input_name)
    for plan in lx_relayout_plans:
        entries[plan.source_name].plans.append(plan)
    return entries


def _input_footprint(graph: GraphLowering, name: str, ncores: dict[str, int]) -> int:
    """Per-core LX footprint of a cloned graph input, or 0 when it has no
    computable one (in which case ``_input_residency_reason`` has already
    barred it, so the value is never used)."""
    from torch_spyre._C import get_device_size_in_bytes

    layout = getattr(graph.get_buffer(name), "layout", None)
    dev_layout = getattr(layout, "device_layout", None)
    num_cores = ncores.get(name, -1)
    if dev_layout is None or num_cores < 1:
        return 0
    return get_device_size_in_bytes(dev_layout) // num_cores


def _in_place(
    alloc: ScratchpadAllocator,
    graph: GraphLowering,
    mem_usage: dict,
    lifetimes: dict[str, list[int]],
    reasons: dict[str, Optional[str]],
) -> dict[str, list[str]]:
    """In-place reuse candidates: ``buf -> [inputs whose slot it may take]``.

    Only buffers that may actually reside are considered on either side of
    the pair. A barred buffer has no LX slot to hand over or inherit, so
    pairing with one is meaningless -- and it would let two unsized (-1)
    sentinels match each other on size."""
    start_overrides, end_overrides = counted_loop_lifetime_overrides(graph)
    allow: dict[str, list[str]] = {}
    allowed = {
        op.name: residency_rules._op_inputs_good_for_lx_inplace(op)
        for op in graph.operations
    }
    for buf_name, info in mem_usage.items():
        allow[buf_name] = []
        if not allowed.get(buf_name):
            continue
        if reasons.get(buf_name) is not None or not lifetimes.get(buf_name):
            continue
        out_start = _handoff_child_start(buf_name, lifetimes, start_overrides)
        out_layout = graph.get_buffer(buf_name).get_layout().device_layout
        for input_buf in info["op_inputs"]:
            if input_buf not in mem_usage or not lifetimes[input_buf]:
                continue
            if reasons.get(input_buf) is not None:
                continue
            in_layout = graph.get_buffer(input_buf).get_layout().device_layout
            if residency_rules._inplace_edge_ok(
                child_pointwise_inputs=allowed[buf_name],
                parent_name=input_buf,
                child_size_per_core=info["size_per_core"],
                parent_size_per_core=mem_usage[input_buf]["size_per_core"],
                child_device_layout=out_layout,
                parent_device_layout=in_layout,
                child_start=out_start,
                parent_end=_handoff_parent_end(input_buf, lifetimes, end_overrides),
                child_core_div_mismatch=info["core_div_mismatch"],
            ):
                allow[buf_name].append(input_buf)
    return allow


def _relayout_destinations(
    alloc: ScratchpadAllocator, graph: GraphLowering, entries: dict[str, _Entry]
) -> None:
    """Each committed relayout's destination, pinned on its source: live from
    the transfer (a half tick before its first consumer) to its last. Once
    relayout owns every later read, the source's loop-extended lifetime moves
    to its destinations."""
    op_index = {op.get_name(): i for i, op in enumerate(graph.operations)}
    found = []
    invalid = set()
    for name, source in entries.items():
        for plan in source.plans:
            ticks = [op_index[c] for c in plan.consumer_names]
            assert all(tick in source.uses for tick in ticks)
            if source.reason is not None:
                invalid.add(plan.source_name)
            else:
                found.append((name, plan, ticks))
    if invalid:
        found = [f for f in found if f[0] not in invalid]
        for name in invalid:
            entries[name].plans = []
    planned = {name for name, _, _ in found}
    for e in entries.values():
        e.in_place = [p for p in e.in_place if p not in planned]
    if not found:
        return
    for e in entries.values():
        e.uses = [2 * use + 1 for use in e.uses]
        if e.start_override is not None:
            e.start_override *= 2
        if e.end_override is not None:
            e.end_override *= 2
    # Adjacent half-ticks rely on DSCs within a bundle executing serially;
    # otherwise the allocator's lifetime reuse is unsound beyond relayout too.
    by_source: dict[str, list[tuple[LXRelayoutPlan, list[int]]]] = {}
    for name, plan, ticks in found:
        by_source.setdefault(name, []).append((plan, ticks))
    for name, items in by_source.items():
        source = entries[name]
        original_start, original_end = source.start_override, source.end_override
        transfers = []
        for plan, original in items:
            consumer_ticks = [2 * tick + 1 for tick in original]
            transfer = consumer_ticks[0] - 1
            transfers.append(transfer)
            source.uses = sorted(
                {use for use in source.uses if use not in consumer_ticks} | {transfer}
            )
            nominal_end = consumer_ticks[-1] + 1
            source.pinned.append(
                {
                    "name": plan.destination_name,
                    "nbytes": round_up_to_alignment(
                        plan.destination_footprint_bytes or source.nbytes,
                        _LX_ALLOCATION_GRANULARITY_BYTES,
                    ),
                    "uses": [transfer, *consumer_ticks],
                    "start_override": original_start,
                    "end_override": (
                        original_end
                        if original_end is not None and original_end > nominal_end
                        else None
                    ),
                    "lx_view": plan.destination_view,
                }
            )
        remaining = source.uses[0 if source.first_use_is_read else 1 :]
        if not any(use > max(transfers) for use in remaining):
            source.start_override = None
            source.end_override = None
