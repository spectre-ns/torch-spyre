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

"""Candidate generation for the table encoding of the LX layout solve.

:class:`TableCandidates` derives what the ``TableStrategy`` enumerates: each
op's candidate core divisions and coarse tilings (the menus), the view tables
between a producer's and a consumer's candidates (``cd_parent_matches``,
including the loop-carry intersection and input-clone overrides), the priced
relayout candidates, capped per edge, and the relayout copies built from them.
It hands them to the solvers as ``CoreDivisionBuffer`` objects. The allocator keeps
the incoming data these read (residency reasons, lifetimes, in-place gates,
drain plans) and the commit path that applies the solved divisions.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any, Optional
import sympy
from torch._inductor.ir import (
    ComputedBuffer,
    Operation,
    Pointwise,
    Reduction,
)
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch_spyre._inductor.pass_utils import (
    PerCoreView,
    op_read_writes,
)
from torch_spyre._C import get_device_size_in_bytes
from torch_spyre._inductor.work_division import (
    OpSplitSpace,
    ResidencyEdge,
    build_op_split_space,
    build_residency_edge,
    enumerate_work_division_candidates,
    has_resolved_work_div_hint,
    _core_division,
    _view_for_div,
)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    TileSpec,
    BufferType,
    RelayoutCopyBuffer,
    build_relayout_copy,
)
from torch_spyre._inductor.scratchpad.utils import (
    clone_at_graph_boundaries,
    mem_usage_by_buf,
    calculate_liveness,
    get_buffer_users,
    ops_in_offset_mutation_component,
    counted_loop_lifetime_overrides,
)
from torch_spyre._inductor.constants import (
    BATCH_MATMUL_FP8_OP,
    KEEP_BY_INDEX_OP,
)
from torch_spyre._inductor import config
from torch_spyre._inductor.loop_info import (
    LoopCarryRecord,
    ReadCopyElisionRecord,
)
from torch_spyre._inductor.padding import is_restickify_op
from torch_spyre._inductor.scratchpad.lx_relayout import (
    _unsupported_relayout_transition_reason,
    core_domain_rejection,
    grouped_gather_rejection,
    LXRelayoutPlan,
    partition_footprint,
    RelayoutCandidate,
    solver_relayout_edge_context,
    solver_relayout_pair_cost,
    work_division_from_view,
)
from torch_spyre._inductor.op_spec import TensorWorkDivision

from torch_spyre._inductor.scratchpad.allocator import (
    _COST_PARAMS,
    _DivisionMap,
    _coarse_tile_loop_trip_count,
    _drain_lifetime_end_overrides,
    _drained_collectives,
    _drop_read_distance_violations,
    _enum_split_options,
    _find_distinct_matmul_splits,
    _fixed_core_division,
    _fused_layout_group_ops,
    _handoff_child_start,
    _handoff_parent_end,
    _intern_view_group,
    _is_cpu_host_buffer,
    _is_indirect_access_op,
    _is_windowed_pool,
    _legal_fixed_division,
    _legal_split_options,
    _reads_offset_slice,
    logger,
    validated_drain_plans,
)

if TYPE_CHECKING:
    from torch_spyre._inductor.scratchpad.allocator import CoOptimizingAllocator
    from torch_spyre._inductor.scratchpad.coarse_tiling import TileReads


class TableCandidates:
    """The candidate derivations of one :class:`CoOptimizingAllocator`."""

    def __init__(self, allocator: CoOptimizingAllocator) -> None:
        self.allocator = allocator
        self._relayout_pair_costs: dict[tuple, Optional[float]] = {}
        # Per enumerated op, its divisions as a space (lx_division_encoding).
        self.lazy_spaces: dict[str, Any] = {}

    def _prepare_buffers(
        self,
        graph: GraphLowering,
        *,
        lx_relayout_plans: list[LXRelayoutPlan] | None = None,
    ) -> Sequence[Any]:
        # Joint selection derives its own divisions; fixed-division plans do not apply.
        # Validate the post-loop drain plans exactly once, here, before the
        # solver runs: the residency gate, the lifetime extension and the push
        # must all consume this same object.  Re-deriving it after the solve
        # could disagree with what residency actually admitted.
        self.allocator._validated_drain_plans = validated_drain_plans(
            graph, division_is_fixed=False
        )
        in_place = self._determine_in_place_division_invariant(graph)
        division_map = self._division_map(graph, allow_deferred_read_candidates=True)
        divisions = division_map.divisions
        pending = {
            op.name: op
            for op in graph.operations
            if hasattr(op, "_read_copy_elision_record")
            and is_restickify_op(op, graph)
            and divisions[op.name] != [_fixed_core_division(op)]
        }
        while True:
            buffers = self._build_cd_bound_buffers(graph, in_place, division_map)
            if not pending:
                return buffers
            pricing = {
                op.get_name(): op for op in self._pricing_operations(graph, buffers)
            }
            rejected = [
                name
                for name, op in pending.items()
                if pricing.get(name) is op
                or not self._direct_read_candidates_priced(
                    pricing.get(name), divisions[name], buffers
                )
            ]
            if not rejected:
                return buffers
            for name in rejected:
                op = pending.pop(name)
                fixed = _fixed_core_division(op)
                assert fixed.cores_used <= config.sencores, (
                    f"{name}: fixed direct-read division over the "
                    f"{config.sencores}-core budget"
                )
                divisions[name] = _legal_fixed_division(
                    op, [fixed], "unproved or unpriced direct read"
                )
                division_map.enumerated.discard(name)
            # Menus affect input clones and relayouts. Rebuild their actual
            # allocation context and recheck the remaining expanded reads.
            # Rejection is monotonic, so this terminates after at most one pin
            # per deferred read, without changing the graph or the late proof.

    @staticmethod
    def _direct_read_candidates_priced(op, divisions, buffers) -> bool:
        from torch_spyre._inductor.cost_model import transport_dma_cost_available
        from torch_spyre._inductor.dump_cost_model import extract_op_features
        from torch_spyre._inductor.scratchpad.sa_cooptimizer import _work_slices

        if op is None or not any(buf.name == op.get_name() for buf in buffers):
            return False
        is_lx = {buf.name: buf.sym_is_lx for buf in buffers}
        return all(
            transport_dma_cost_available(
                extract_op_features(op, _work_slices(op, division), is_lx=is_lx),
                _COST_PARAMS,
            )
            for division in divisions
        )

    @staticmethod
    def _pricing_operations(graph, buffers):
        from torch_spyre._inductor.read_copy_elision import (
            project_transport_read_copies,
        )

        return project_transport_read_copies(
            graph,
            {buf.name: [cd.splits for cd in buf.core_divisions] for buf in buffers},
            relayout_sources={
                buf.relayout_parent
                for buf in buffers
                if isinstance(buf, RelayoutCopyBuffer)
            },
        )

    def _division_map(
        self, graph: GraphLowering, *, allow_deferred_read_candidates: bool = False
    ) -> "_DivisionMap":
        """Per-op core-division candidates for the joint-division solve.

        Every op gets at least one ``CoreDivision`` so the slicing-match gate can
        constrain it. Pointwise / Reduction ops get the enumerated candidates;
        every other op falls back to its committed symbol-keyed division. No
        op-kind pre-filter -- residency is
        gated per buffer (``_residency_by_buf``) and by the solver, so ineligible
        ops still participate as producers/consumers in the match.

        Constant-offset writes, their storage aliases and their forward
        dependency closure keep their committed division. This protects offset
        address arithmetic even in HBM, where residency edges do not constrain
        division choices. Upstream producers remain free to optimize. See
        ``utils.ops_in_offset_mutation_component``. As with every fixed pin,
        an illegal committed division raises ``Unsupported`` rather than
        bypassing hard work-division constraints.

        Whatever the path, every candidate returned is within the ``sencores`` budget
        -- asserted here because nothing downstream re-checks it (issue #4387).
        """
        max_cores = config.sencores
        profiles, matmul_roles = _find_distinct_matmul_splits(graph.operations)
        # Tilings are offered per op (``_tiling_candidates``), but whether an op
        # sits inside a for_each_tile region, and who reads its output, are
        # graph-level facts.
        self._prescribed_ops: frozenset[str] = frozenset()
        self._readers_by_name: dict[str, list[Operation]] = {}
        if config.auto_coarse_tiling and self.allocator._tilings_from_menu:
            from torch_spyre._inductor.scratchpad.coarse_tiling import (
                prescribed_regions,
            )

            self._prescribed_ops = frozenset(
                name
                for region in prescribed_regions(graph.operations)
                for name in region.names
            )
            for reader in graph.operations:
                for name in {dep.name for dep in reader.get_read_writes().reads}:
                    self._readers_by_name.setdefault(name, []).append(reader)

        # Ops pinned to their committed (work-division) division: each guard
        # detects a distinct wrong-code or scheduling hazard the joint solver
        # would hit by re-slicing the op, and all share the one remedy -- keep the
        # fixed division. A resolved user work_div hint takes the same remedy,
        # last so a hazard guard's reason is the one logged: work division
        # already committed the hint, and the pin is whole-op -- unhinted dims
        # keep their committed split of 1. The graph-level group sets are
        # loop-invariant, so build them once here rather than rescanning
        # graph.operations for every op.
        offset_mutation_ops = ops_in_offset_mutation_component(graph)
        layout_group_reason = _fused_layout_group_ops(
            graph,
            {
                KEEP_BY_INDEX_OP: "keep_by_index layout group",
                BATCH_MATMUL_FP8_OP: "fp8 matmul layout group",
            },
        )
        result = {}
        enumerated = set()
        for op in graph.operations:
            reason: Optional[str] = None
            if _is_cpu_host_buffer(op):
                reason = "cpu/host buffer"
            elif op.name in offset_mutation_ops:
                reason = "offset mutation forward closure"
            elif _is_windowed_pool(op):
                reason = "windowed pool"
            elif op.name in layout_group_reason:
                reason = layout_group_reason[op.name]
            elif _is_indirect_access_op(op):
                reason = "indirect access entry split"
            elif _reads_offset_slice(op):
                reason = "offset slice read"
            elif (
                is_restickify_op(op, graph)
                and hasattr(op, "_read_copy_elision_record")
                and not (
                    allow_deferred_read_candidates
                    and config.read_copy_elision
                    and isinstance(op._read_copy_elision_record, ReadCopyElisionRecord)
                )
            ):
                # The preparation path may provisionally enumerate ordinary
                # legal candidates, but retains them only after proving and
                # pricing the direct read in the actual allocation context.
                # Other callers and unrecognized records keep the fixed pin.
                reason = "deferred direct graph-input read"
            elif (
                not config.ignore_work_division_hints
                and isinstance(op, ComputedBuffer)
                and has_resolved_work_div_hint(op)
            ):
                reason = "user work_div hint"

            if reason is not None:
                divs = _legal_fixed_division(op, [_fixed_core_division(op)], reason)
            elif self.allocator.prune and isinstance(op, ComputedBuffer):
                divs = [
                    _core_division(op, splits)
                    for splits in _legal_split_options(
                        op, _enum_split_options(op, profiles, matmul_roles)
                    )
                ]
                if not divs:
                    divs = _legal_fixed_division(
                        op, [_fixed_core_division(op)], "empty pruned candidate set"
                    )
            else:
                divs, is_enumeration = self._enumerate_core_divisions(op, max_cores)
                if is_enumeration:
                    enumerated.add(op.name)
            if not divs:
                raise Unsupported(f"{op.name}: no legal core-division candidates.")
            # The core budget is an invariant of the MENU, not of its consumers: both
            # engines pin an op's split symbols to one enumerated candidate, so nothing
            # downstream re-checks the product -- and `_matmul_split_cost`'s own budget
            # guard no-ops on symbolic splits, scoring an over-budget division NEGATIVE.
            # A minimizing solve then finds it maximally attractive rather than
            # rejecting it (issue #4387). Only two of the three paths above take
            # `max_cores` -- `_legal_split_options` asks nothing about a core budget --
            # so check the menu itself, unconditionally: it is one product per candidate.
            over = [d for d in divs if d.cores_used > max_cores]
            assert not over, (
                f"{op.name}: enumerated core divisions over the {max_cores}-core "
                f"budget: "
                + ", ".join(f"{d.label} ({d.cores_used} cores)" for d in over)
            )
            result[op.name] = divs

        return _DivisionMap(result, enumerated)

    def _enumerate_core_divisions(
        self, op: Operation, max_cores: int
    ) -> tuple[list[CoreDivision], bool]:
        """Enumerate and deduplicate symbol-keyed candidates for one operation,
        with whether they are the *whole* legal cross product.

        Operations without an enumerable concrete iteration space retain their
        committed division, and say so (see :class:`_DivisionMap`).
        Deduplication uses local symbol names only within this operation;
        cross-operation compatibility is derived from ``PerCoreView`` instead.
        """
        fixed = [_fixed_core_division(op)]
        if not isinstance(op, ComputedBuffer) or not isinstance(
            op.data, (Pointwise, Reduction)
        ):
            return fixed, False
        try:
            candidates = enumerate_work_division_candidates(op, max_cores)
        except Unsupported as exc:
            return _legal_fixed_division(op, fixed, str(exc)), False
        cds: list[CoreDivision] = []
        seen: set[tuple] = set()
        options: list = []
        # Each tiling option gets its own division enumeration: the legal split
        # set is tiling-relative -- a tiled dim has fewer/other divisors and a
        # smaller per-core span. The untiled option reuses the already-enumerated
        # `candidates`; a non-empty option re-enumerates on the tiled frame.
        # Divisions are symbol-keyed and a tiling rescales indices without
        # renaming loop symbols, so both frames' keys are directly comparable.
        for tiling in self._tiling_candidates(op, max_cores):
            tile_splits: tuple[tuple[sympy.Symbol, int], ...] = ()
            if tiling.is_untiled:
                tiled_candidates = candidates
            else:
                from torch_spyre._inductor.scratchpad.coarse_tiling import (
                    try_resolve_tile_axis_loop_vars,
                )

                loop_vars, reason = try_resolve_tile_axis_loop_vars(op, tiling)
                if loop_vars is None:
                    logger.debug(
                        "skip tiling %s for %s: %s", tiling.label, op.name, reason
                    )
                    continue
                tile_splits = tuple(
                    (sym, axis.count) for sym, axis in zip(loop_vars, tiling.axes)
                )
                try:
                    tiled_candidates = enumerate_work_division_candidates(
                        op, max_cores, tiling=tiling
                    )
                except Unsupported as exc:
                    logger.debug(
                        "skip tiled division for %s under %s: %s",
                        op.name,
                        tiling.label,
                        exc,
                    )
                    continue
            if config.lx_division_encoding == "lazy":
                from torch_spyre._inductor.scratchpad.lazy_division import LazyOption
                from torch_spyre._inductor.work_division import (
                    work_division_context_for_op,
                )

                context = work_division_context_for_op(op, max_cores, tiling)
                if context is not None and tiled_candidates:
                    options.append(LazyOption(tiling, tile_splits, context))
            for candidate in tiled_candidates:
                division = _core_division(op, candidate, tiling, tile_splits)
                key = (
                    tuple(
                        sorted(
                            division.splits.items(),
                            key=lambda item: str(item[0]),
                        )
                    ),
                    division.reduction_syms,
                    tiling,
                )
                if key not in seen:
                    seen.add(key)
                    cds.append(division)
        if cds:
            if options:
                from torch_spyre._inductor.scratchpad.lazy_division import LazySpace

                self.lazy_spaces[op.get_name()] = LazySpace(op, options, max_cores)
            return cds, True
        return _legal_fixed_division(op, fixed, "no enumerable candidate"), False

    def _tiling_candidates(self, op: Operation, max_cores: int) -> list[TileSpec]:
        """Coarse-tiling options to pair with ``op``'s divisions.

        Unless ``auto_coarse_tiling`` is on and the solver takes its tilings off
        this menu (``tilings_from_menu()``) the only option is the untiled
        ``TileSpec``, so enumeration and every downstream plan stay
        bit-identical to today. When it does, the op is offered the
        output-axis tilings it could take, minus any whose per-core read span
        would still exceed the read-distance limit (``MAX_SPAN_BYTES``); the
        untiled option is dropped too when the op's own full-size read
        overflows, and an op with no fitting tiling raises ``Unsupported`` (see
        :func:`_drop_read_distance_violations`).

        Filtered by op kind:

        - **Restickify** offers nothing: no tiling form is correct
          post-stickification.
        - **Everything else** -- pointwise, reduction, *and matmul* -- offers only
          output-axis tilings (``is_clean``). That single filter, plus the
          enumerator's refusal to emit the stick (innermost) dim, already drops
          the numerically fragile forms for every op kind, so matmul needs no
          guard of its own: a matmul's K/reduction axis is not an output axis and
          is excluded (reduction tiling routes through the accumulator/combine
          path, ~2 orders off CPU -- see ``_mlp_case``), and its N/stick dim is
          never emitted. What remains for a matmul is row/M-axis output tiling,
          which is correct and backend-accepted (see
          ``test_hint_matmul_row_tiling``), so a discovered tiling on that axis
          is honored the same as for any other op.

        (The ``_resize_device_layout`` gap #3218 that rejects a tile-sized read
        copy is confined to the span-overflow path, where a matmul feeds a
        differently-shaped consumer -- a separate mechanism, still xfailed, not
        reached by the ordinary M-axis output tiling offered here.)

        An op already tiled (``loop_info`` set -- by the ``spyre_hint`` pass
        pre-stickification, a ``for_each_tile`` loop, or a prior apply) is left
        untouched, so the solve never re-tiles or un-tiles it, and so is every
        other op inside a ``for_each_tile`` region (``_prescribed_ops``): the
        user's loop covers it even where no level stamped it. The solve reads no
        hints of its own.
        """
        untiled = [TileSpec()]
        if getattr(self.allocator, "_suppress_tiling", False):
            return untiled
        if not (config.auto_coarse_tiling and self.allocator._tilings_from_menu):
            return untiled
        if getattr(op, "loop_info", None) is not None:
            return untiled
        if op.get_operation_name() in getattr(self, "_prescribed_ops", ()):
            return untiled
        if self.allocator._get_op_name(op) == "restickify":
            return untiled
        from torch_spyre._inductor.wsr.enumerate_tilings import enumerate_tile_options

        try:
            # enumerate_tile_options returns untiled first; the is_clean filter
            # keeps it and the output-only specs, preserving that order.
            readers = getattr(self, "_readers_by_name", {}).get(op.get_name(), ())
            options = [
                t for t in enumerate_tile_options(op, readers=readers) if t.is_clean
            ]
        except Unsupported:
            options = list(untiled)

        # Discovery offers the op tilings the solve is free to pick, but
        # the CP-SAT cost model carries no span term, so a discovered TileSpec is
        # never otherwise checked against the hardware read-distance limit
        # (MAX_SPAN_BYTES) the span-overflow planner enforces. Drop any candidate
        # whose per-core read span would still overflow that limit -- including
        # the untiled option when the op's own full-size read overflows -- so the
        # solve can never choose a tiling that reintroduces the very span
        # violation coarse tiling exists to prevent, and abort when none fit.
        return _drop_read_distance_violations(op, options, max_cores)

    def _determine_in_place_division_invariant(
        self, graph: GraphLowering
    ) -> dict[str, list[str]]:
        """Co-opt in-place candidates: keep only the *division-invariant*
        preconditions here and defer the division-dependent ones to the solver.

        The per-core size match and core-division compatibility depend on the
        division the ILP has not yet chosen, so they are enforced in the solver
        (``eff_size`` equality + the ``cd_parent_matches`` gate). What stays as a
        pre-filter is division-invariant: lifetime adjacency
        (``in_end == out_start``, the single-tick-handoff invariant the solver's
        no-overlap relaxation relies on but cannot re-derive) and identical device
        layouts (required for the storage to alias).
        """
        allow_inplace: dict[str, list[str]] = {}
        mem_usage = mem_usage_by_buf(graph)
        in_place_allowed = {
            op.name: self.allocator._op_inputs_good_for_lx_inplace(op)
            for op in graph.operations
        }
        lifetimes = calculate_liveness(graph)
        lifetime_start_overrides, lifetime_end_overrides = (
            counted_loop_lifetime_overrides(graph)
        )
        for buf_name, info in mem_usage.items():
            allow_inplace[buf_name] = []
            if not in_place_allowed[buf_name]:
                continue
            # Unplaceable producers (e.g. a ``MultiOutputLayout`` tuple op like
            # max-with-indices) carry no ``device_layout``: their storage cannot
            # alias an input, so skip rather than raise ``AttributeError``.
            out_layout = graph.get_buffer(buf_name).layout
            if not hasattr(out_layout, "device_layout"):
                continue
            out_start = _handoff_child_start(
                buf_name, lifetimes, lifetime_start_overrides
            )
            out_ten_layout = out_layout.device_layout
            for input_buf in info["op_inputs"]:
                # Graph inputs / constants now appear in ``op_inputs`` but are not
                # solver buffers, so they can't be in-place aliasing parents (the
                # solver's ``_check_in_place_relationships`` would fail to resolve
                # them). Skip them, matching the base allocator's guard.
                if input_buf not in mem_usage or not lifetimes[input_buf]:
                    continue
                in_layout = graph.get_buffer(input_buf).layout
                if not hasattr(in_layout, "device_layout"):
                    continue
                in_ten_layout = in_layout.device_layout
                # The division-invariant edge gate (layout match + single handoff
                # tick; per-core size and core-division deferred to the solver). The
                # ``division_invariant`` mode also applies the per-input
                # pointwise-eligibility test (``input_buf in in_place_allowed``): for
                # pointwise-tagged ops that is every read (a no-op), but for a
                # non-tagged Pointwise op it drops an input read at a different index
                # than the output write -- which must not be aliased over the output.
                if self.allocator._inplace_edge_ok(
                    child_pointwise_inputs=in_place_allowed[buf_name],
                    parent_name=input_buf,
                    child_device_layout=out_ten_layout,
                    parent_device_layout=in_ten_layout,
                    child_start=out_start,
                    parent_end=_handoff_parent_end(
                        input_buf, lifetimes, lifetime_end_overrides
                    ),
                    division_invariant=True,
                ):
                    allow_inplace[buf_name].append(input_buf)
        return allow_inplace

    def _residency_by_buf(
        self,
        graph: GraphLowering,
        mem_usage: dict,
        lifetimes: dict[str, list[int]],
        drain_plans: Collection[str] = (),
    ) -> dict[str, Optional[str]]:
        """Per-buffer residency verdict: ``None`` if the buffer may be pinned in
        LX, else the reason it may not.

        Every buffer is handed to the solver so it participates in the slicing
        match, but participation is not residency. The predicate is the shared
        one in :meth:`_buffer_residency_reason` -- the same list the placement
        path uses -- with ``division_is_fixed=False``: this allocator *chooses*
        each op's core division, so pre-rejecting a buffer whose users disagree
        under the upstream-committed division would be premature. The solver's
        ``cd_parent_matches`` slicing gate decides that instead. ``ncores`` is
        therefore not needed here.
        """
        return self.allocator._residency_reasons(
            graph,
            list(mem_usage),
            division_is_fixed=False,
            lifetimes=lifetimes,
            drain_plans=drain_plans,
        )

    def _build_cd_bound_buffers(
        self,
        graph: GraphLowering,
        in_place: Optional[dict[str, list[str]]],
        division_map: "_DivisionMap",
    ) -> list[CoreDivisionBuffer]:
        """Build the ``CoreDivisionBuffer``s handed to the solver.

        Every buffer carries its candidate divisions and is sized by its
        *total* device footprint plus its producer edges (``parent_proj``); the
        solver picks a division and divides by its ``output_partition``.
        Counted-loop lifetimes still filter unsafe in-place handoffs before the
        solver compares those total footprints.

        Each buffer also carries the same two relations *per candidate*, for a
        solver that generates divisions rather than indexing the list: the
        residency edge to each divided producer, and -- where
        :class:`_DivisionMap` allows one and the solver is one that generates
        (see :attr:`_solver_generates_divisions`) -- its op's split space.
        """
        divisions = division_map.divisions
        # Per-plan interning of relayout destination views (see
        # _cd_parent_relayouts): group ids are only meaningful within one plan.
        self._relayout_view_groups: dict[str, dict[PerCoreView, int]] = {}
        lifetimes = calculate_liveness(graph)
        lifetime_start_overrides, lifetime_end_overrides = (
            counted_loop_lifetime_overrides(graph)
        )
        # A planned drain is a new post-solve op that reads the carry storage
        # after the loop, so a graph-output storage must stay live to the graph
        # exit.  This is inside the solve, so the extension only removes reuse
        # and stays priced -- it adds no cost term and no capacity.
        drain_plans = self.allocator._validated_drain_plans
        if drain_plans:
            _drain_lifetime_end_overrides(
                lifetime_end_overrides, drain_plans, len(graph.operations)
            )
        # A collective has no division pair with any resident buffer, so its
        # edge to a carry would force the carry into HBM.  A planned drain is
        # what that collective reads instead, so the edge does not exist.
        drained_collectives = _drained_collectives(drain_plans)
        mem_usage = mem_usage_by_buf(graph)
        in_place = {} if in_place is None else in_place
        op_by_name = {op.name: op for op in graph.operations}
        position_by_name = {op.name: index for index, op in enumerate(graph.operations)}
        graph_output_names = set(graph.get_output_names())
        readers_by_name: dict[str, list[ComputedBuffer]] = {}
        if self.allocator._solver_chooses_tilings:
            for reader in graph.operations:
                if isinstance(reader, ComputedBuffer):
                    for name in {read.name for read in reader.get_read_writes().reads}:
                        readers_by_name.setdefault(name, []).append(reader)

        prep_cache: dict = {}
        buffers: list[CoreDivisionBuffer] = []
        residency_by_buf = self._residency_by_buf(
            graph, mem_usage, lifetimes, drain_plans
        )

        # Resolve every compiler-tagged carry before constructing any buffer.
        # If its aliased update cannot be represented as a physical-ownership
        # edge, fail closed by leaving the storage in HBM.  The storage usually
        # precedes its update in graph order, so doing this up front avoids
        # discovering the malformed contract after its solver record is built.
        carry_update_edges: dict[str, ResidencyEdge] = {}
        for update_op in graph.operations:
            record = getattr(update_op, "_loop_carry_record", None)
            if not isinstance(record, LoopCarryRecord):
                continue
            if record.update_name != update_op.get_name():
                continue
            edge = self._loop_carry_update_edge(update_op, op_by_name, prep_cache)
            if edge is None:
                if residency_by_buf.get(record.storage_name) is None:
                    residency_by_buf[record.storage_name] = (
                        "loop carry update ownership unavailable"
                    )
                continue
            if residency_by_buf.get(record.storage_name) is None:
                carry_update_edges[record.update_name] = edge

        input_clone_matches: dict[str, dict[str, list[tuple[int, int]]]] = {}
        # Consumer op name -> input clones for which it is the last reader, and so
        # may reuse the clone's LX slot in place (reverse-parent, #3212). Stays
        # empty unless cloning is on, making the reverse-parent block in the output
        # loop a no-op otherwise.
        last_consumer_clones: dict[str, list[str]] = {}
        if clone_at_graph_boundaries():
            buffer_users = get_buffer_users(graph)
            for input_name in self._eligible_clone_inputs(graph, lifetimes):
                consumers = [op for op in buffer_users.get(input_name, [])]
                divs, matches = self._clone_divisions_and_matches(
                    input_name, consumers, divisions, prep_cache
                )
                # No division matched any consumer -> the clone has no valid core
                # division and could never reside, so don't hand an unplaceable
                # buffer to the solver (it would trip the >=1-division invariant).
                # The input simply stays in HBM, as it would uncloned.
                if not divs:
                    continue
                input_clone_matches[input_name] = matches
                residency_by_buf[input_name] = None
                # Only the op at the clone's last-use tick both reads the clone and
                # writes its own output, so only it satisfies the single handoff
                # tick for reusing the clone's slot in place.
                last_use = lifetimes[input_name][-1]
                last_consumer_clones.setdefault(
                    graph.operations[last_use].name, []
                ).append(input_name)
                dev_layout = graph.get_buffer(input_name).layout.device_layout
                size = get_device_size_in_bytes(dev_layout)
                buffers.append(
                    CoreDivisionBuffer(
                        input_name,
                        size,
                        lifetimes[input_name],
                        first_use_is_read=True,
                        in_place_parents=[],
                        core_divisions=divs,
                        parents=[],
                        cd_parent_matches={},
                        residency_reason=None,
                        lifetime_start_override=lifetime_start_overrides.get(
                            input_name
                        ),
                        lifetime_end_override=lifetime_end_overrides.get(input_name),
                        boundary=BufferType.Input,
                    )
                )

        for output_name, info in mem_usage.items():
            uses = lifetimes[output_name]

            op = op_by_name.get(output_name)
            residency_reason = residency_by_buf[output_name]

            buf_divisions = divisions[output_name]
            # Drain plans can extend a parent's lifetime after the in-place
            # candidates were computed. Recheck adjacency with the same final
            # bounds handed to the solver so stale handoffs cannot reach it.
            parents = [
                parent
                for parent in in_place.get(output_name, [])
                if _handoff_parent_end(parent, lifetimes, lifetime_end_overrides)
                == _handoff_child_start(
                    output_name, lifetimes, lifetime_start_overrides
                )
            ]
            size = info["size"]  # total footprint; solver divides per chosen cd
            parent_proj = [
                parent
                for parent in info["op_inputs"]
                if drained_collectives.get(parent) != output_name
            ]
            residency_edges = self._parent_residency_edges(
                op, parent_proj, op_by_name, prep_cache, residency_by_buf
            )
            if config.lx_view_relation == "derived":
                # The strategy derives these edges' relation from their
                # ResidencyEdge; only edges whose table is overridden below
                # (loop carries, input clones) are materialized here.
                carry_edge = carry_update_edges.get(output_name)
                special = set(input_clone_matches) | set(
                    self._loop_carry_read_edges(op, carry_update_edges, prep_cache)
                )
                if carry_edge is not None:
                    special.add(carry_edge.buf_name)
                tabled = {p: e for p, e in residency_edges.items() if p in special}
            else:
                tabled = residency_edges
            cd_parent_matches = self._cd_parent_matches(
                tabled, buf_divisions, divisions
            )

            # A for_each_tile update writes through a MutationLayout whose
            # dependency is named after the update op, even though the bytes
            # belong to the persistent carry storage.  Model that write as a
            # producer -> consumer edge so a resident carry is legal only when
            # the solver chooses identical physical ownership for the initial
            # storage and every update.  A relayout cannot satisfy an in-place
            # write, so this edge deliberately has match pairs only.
            #
            # Folded into cd_parent_matches BEFORE _cd_parent_relayouts runs
            # (not after, as a prior version of this did): a sibling parent's
            # own relayout candidates get capped against cd_parent_matches
            # inside that call, and a cap that ran before the carry's matches
            # existed could prune away the one candidate that would have let
            # that sibling land on a division the carry already occupies for
            # free (cyang49's PR #5057 review). original_parent_proj (the
            # pre-carry snapshot) is what every "already a direct read"
            # membership check below is against, so a storage this same op
            # also reads directly keeps getting the intersected (update AND
            # read) match pairs either block alone would have produced,
            # independent of which block runs first -- and is what
            # _cd_parent_relayouts is called with: no carry-touched storage
            # name is ever added to parent_proj before that call, so it is
            # never offered as a relayout candidate at all (a relayout copy
            # taken before the update would be stale, whether or not the
            # storage happens to also be an ordinary read). Carry-touched
            # names are appended to the real parent_proj only afterward.
            original_parent_proj = list(parent_proj)
            carry_storage_names: list[str] = []
            carry_edge = carry_update_edges.get(output_name)
            if carry_edge is not None:
                storage_name = carry_edge.buf_name
                update_matches = carry_edge.match_pairs(
                    divisions[storage_name],
                    buf_divisions,
                )
                if storage_name in original_parent_proj:
                    # If the update also reads the carry directly, both that
                    # read and the aliased write must agree with its storage.
                    read_matches = cd_parent_matches.get(storage_name)
                    if read_matches is None:
                        update_matches = []
                    else:
                        update_set = set(update_matches)
                        update_matches = [p for p in read_matches if p in update_set]
                carry_storage_names.append(storage_name)
                cd_parent_matches[storage_name] = update_matches

            # A read of a carry update reads the storage's bytes, so it must
            # agree with the storage's ownership too; otherwise a resident
            # carry reaches a reader that slices it differently, and kernel
            # preparation demotes it (#4990). A relayout copy taken before the
            # update would be stale, so this edge also has match pairs only.
            for storage_name, read_edge in self._loop_carry_read_edges(
                op, carry_update_edges, prep_cache
            ).items():
                read_matches = read_edge.match_pairs(
                    divisions[storage_name],
                    buf_divisions,
                )
                # Also checks carry_storage_names, not just original_parent_proj:
                # a storage that is THIS op's own carry-update target (recorded
                # above) but not a direct read never enters original_parent_proj,
                # so without this a self-overlap -- two sequential mutations of
                # one carry, where this op both updates storage S and reads
                # another update that also writes S -- would overwrite the
                # update-edge's match_pairs with the unfiltered read-edge ones
                # instead of intersecting them, same loss of constraint the
                # direct-read branch above exists to prevent.
                if storage_name in original_parent_proj or storage_name in (
                    carry_storage_names
                ):
                    known = set(cd_parent_matches.get(storage_name, []))
                    read_matches = [p for p in read_matches if p in known]
                if storage_name not in carry_storage_names:
                    carry_storage_names.append(storage_name)
                cd_parent_matches[storage_name] = read_matches

            relayout_parent_names = [
                parent for parent in parent_proj if parent not in carry_storage_names
            ]
            # Merge clone-supplied free divisions BEFORE the call: the
            # cap-and-rank step inside _cd_parent_relayouts pins the candidate
            # group that would let a parent join a division another parent
            # already occupies for free, and an input clone's reusable slot
            # (#3212) is one such free division -- merged only after the call,
            # a positive lx_solver_relayout_groups_per_edge cap could prune the
            # one group matching it and silently drop a residency option
            # (cyang49's PR #5057 review). The carry branch above has already
            # run, so the intersection it applies to a carry storage's
            # match_pairs is unaffected by this merge; the re-merge after
            # the call below stays for carry storages that only join
            # parent_proj between the two points.
            for input_name in parent_proj:
                if input_name in input_clone_matches:
                    cd_parent_matches[input_name] = input_clone_matches[input_name][
                        output_name
                    ]
            cd_parent_relayouts = self._cd_parent_relayouts(
                graph,
                op,
                buf_divisions,
                relayout_parent_names,
                divisions,
                op_by_name,
                prep_cache,
                residency_by_buf,
                cd_parent_matches,
            )
            for storage_name in carry_storage_names:
                if storage_name not in parent_proj:
                    parent_proj.append(storage_name)

            for input_name in parent_proj:
                if input_name in input_clone_matches:
                    cd_parent_matches[input_name] = input_clone_matches[input_name][
                        output_name
                    ]

            # Reverse-parent edge (#3212): when this op is an input clone's last
            # reader, let it reuse the clone's LX slot in place. Division-invariant
            # gate (pointwise child reading the clone + matching device layout;
            # single tick guaranteed by "last reader"); per-core size and core
            # division are deferred to the solver, which also gates the merge on the
            # cd_parent_matches entry set just above. Multi-output ops carry a
            # MultiOutputLayout with no single device_layout and cannot alias one
            # clone, so they are skipped.
            out_layout = graph.get_buffer(output_name).layout
            for clone_name in last_consumer_clones.get(output_name, []):
                if clone_name in parents:
                    continue
                clone_layout = graph.get_buffer(clone_name).layout
                if (
                    op is None
                    or not hasattr(out_layout, "device_layout")
                    or not hasattr(clone_layout, "device_layout")
                ):
                    continue
                if self.allocator._inplace_edge_ok(
                    child_pointwise_inputs=self.allocator._op_inputs_good_for_lx_inplace(
                        op
                    ),
                    parent_name=clone_name,
                    child_device_layout=out_layout.device_layout,
                    parent_device_layout=clone_layout.device_layout,
                    child_start=_handoff_child_start(
                        output_name, lifetimes, lifetime_start_overrides
                    ),
                    parent_end=_handoff_parent_end(
                        clone_name, lifetimes, lifetime_end_overrides
                    ),
                    division_invariant=True,
                ):
                    parents.append(clone_name)

            buffer = CoreDivisionBuffer(
                output_name,
                size,
                uses,
                # An op output is a computed buffer: ``uses[0]`` is the
                # producing write, as on the placement path above. (Only the
                # input-clone loop above sets this True.)
                first_use_is_read=False,
                in_place_parents=parents,
                core_divisions=buf_divisions,
                parents=parent_proj,
                cd_parent_matches=cd_parent_matches,
                cd_parent_relayouts=cd_parent_relayouts,
                residency_edges=residency_edges,
                residency_reason=residency_reason,
                lifetime_start_override=lifetime_start_overrides.get(output_name),
                lifetime_end_override=lifetime_end_overrides.get(output_name),
                op_position=position_by_name.get(output_name),
                boundary=BufferType.Output
                if output_name in graph_output_names
                else BufferType.Intermediate,
            )
            if (
                op is not None
                and output_name in division_map.enumerated
                and self.allocator._solver_generates_divisions
            ):
                buffer.division_space = self._division_space(
                    op, readers_by_name.get(output_name, ())
                )
            if output_name in division_map.enumerated:
                buffer.lazy_space = self.lazy_spaces.get(output_name)
            if self.allocator._solver_chooses_tilings and isinstance(
                op, ComputedBuffer
            ):
                buffer.tile_reads = self._tile_reads(graph, op, op_by_name)
            buffers.append(buffer)
        buffers.extend(self._relayout_copy_buffers(buffers, self.allocator.size))
        return buffers

    @staticmethod
    def _tile_reads(
        graph: GraphLowering, op: ComputedBuffer, op_by_name: dict[str, Operation]
    ) -> dict[str, "TileReads"]:
        """``CoreDivisionBuffer.tile_reads`` for ``op``: one per computed
        buffer it reads."""
        # Local import: ``coarse_tiling`` imports this module.
        from torch_spyre._inductor.scratchpad.coarse_tiling import TileReads

        return {
            name: TileReads(graph, producer, op)
            for name in sorted({read.name for read in op.get_read_writes().reads})
            if isinstance(producer := op_by_name.get(name), ComputedBuffer)
        }

    @staticmethod
    def _loop_carry_update_edge(
        update_op: Optional[Operation],
        op_by_name: dict[str, Operation],
        prep_cache: dict,
    ) -> Optional[ResidencyEdge]:
        """Return the physical-ownership edge for an aliased carry update.

        Inductor names the mutation write after ``update_op`` rather than the
        storage it aliases.  Renaming that dependency lets the ordinary
        :class:`ResidencyEdge` machinery interpret its index against the carry
        storage's device layout.
        """
        if update_op is None:
            return None
        record = getattr(update_op, "_loop_carry_record", None)
        if not isinstance(record, LoopCarryRecord):
            return None
        if record.update_name != update_op.get_name():
            return None
        storage_op = op_by_name.get(record.storage_name)
        if storage_op is None:
            return None
        storage_write = next(
            (
                dep
                for dep in op_read_writes(storage_op).writes
                if dep.name == record.storage_name and isinstance(dep, MemoryDep)
            ),
            None,
        )
        update_write = next(
            (
                dep
                for dep in op_read_writes(update_op).writes
                if dep.name == record.update_name and isinstance(dep, MemoryDep)
            ),
            None,
        )
        if storage_write is None or update_write is None:
            return None
        return ResidencyEdge(
            buf_name=record.storage_name,
            parent_op=storage_op,
            consumer_op=update_op,
            write_dep=storage_write,
            read_deps=(update_write.rename({record.update_name: record.storage_name}),),
            prep_cache=prep_cache,
        )

    @staticmethod
    def _loop_carry_read_edges(
        consumer_op: Optional[Operation],
        carry_update_edges: dict[str, ResidencyEdge],
        prep_cache: dict,
    ) -> dict[str, ResidencyEdge]:
        """Storage-ownership edges for ``consumer_op``'s reads of carry updates.

        A carry update writes through the carry's storage, so a later read of the
        update's name reads the storage's bytes. The update itself is never an LX
        buffer, so its ordinary producer -> consumer edge is excluded, and without
        this edge nothing ties a resident storage's ownership to that reader. Each
        returned edge, keyed by storage name, compares the storage's write with the
        reader's accesses renamed onto the storage, as
        :meth:`_loop_carry_update_edge` does for the update's own write. A
        reader that reaches one storage through several reads gets one edge
        holding them all.
        """
        if consumer_op is None:
            return {}
        edges: dict[str, ResidencyEdge] = {}
        for dep in op_read_writes(consumer_op).reads:
            carry_edge = carry_update_edges.get(dep.name)
            if carry_edge is None or not isinstance(dep, MemoryDep):
                continue
            storage_name = carry_edge.buf_name
            read = dep.rename({dep.name: storage_name})
            edge = edges.get(storage_name)
            if edge is not None:
                edge.read_deps += (read,)
                continue
            edges[storage_name] = ResidencyEdge(
                buf_name=storage_name,
                parent_op=carry_edge.parent_op,
                consumer_op=consumer_op,
                write_dep=carry_edge.write_dep,
                read_deps=(read,),
                prep_cache=prep_cache,
            )
        return edges

    @staticmethod
    def _relayout_copy_buffers(
        buffers: Sequence[CoreDivisionBuffer],
        capacity: int | None = None,
    ) -> list[RelayoutCopyBuffer]:
        """One :class:`RelayoutCopyBuffer` per relayout group enumerated across
        ``buffers``: the destination the solver places, live from the group's
        first consumer to its last, carrying every priced candidate that lands
        on it. A group whose source is not among the buffers has nothing to
        shuffle from and gets no copy; the solver then ignores its candidates.

        ``capacity`` is the planner's per-core LX budget. A group whose
        destination span alone exceeds it can never be resident, so building a
        copy for it only adds a buffer the solver must place and prove out; such
        groups get no copy either (the residency gate ignores candidates whose
        group has none). On the 304-op decode attention graph these copies are a
        measurable share of a model whose presolve alone outlived the time limit.
        """
        by_name = {b.name: b for b in buffers}
        groups: dict[tuple[str, int], list[RelayoutCandidate]] = {}
        for consumer in buffers:
            for candidates in consumer.cd_parent_relayouts.values():
                for candidate in candidates:
                    groups.setdefault(candidate.group_key, []).append(candidate)
        ticks = {b.name: b.start_time for b in buffers}
        copies: list[RelayoutCopyBuffer] = []
        oversized = 0
        for (parent, group), candidates in sorted(groups.items()):
            source = by_name.get(parent)
            if source is None:
                logger.debug(
                    "[lx solver relayout] %s/g%d: source is not in the solve; "
                    "no copy built",
                    parent,
                    group,
                )
                continue
            copy = build_relayout_copy(source, group, candidates, ticks)
            if capacity is not None and copy.per_core_footprint > capacity:
                oversized += 1
                continue
            copies.append(copy)
        if oversized:
            logger.debug(
                "[lx solver relayout] %d relayout group(s) skipped: destination "
                "span exceeds the %d-byte LX budget",
                oversized,
                capacity,
            )
        return copies

    def _division_space(
        self, op: Operation, readers: Sequence[ComputedBuffer] = ()
    ) -> Optional[OpSplitSpace]:
        """The op's split space; ``readers`` are the ops reading its output.

        The tiling half is attached only for the solver that can use it (see
        :attr:`_solver_chooses_tilings`): deriving it costs a stick-alignment
        analysis per output dim, so an engine that would ignore the answer does
        not pay for it.
        """
        # Local: enumerate_tilings imports scratchpad.coarse_tiling, which
        # imports this module.
        from torch_spyre._inductor.wsr.enumerate_tilings import build_tiling_space

        tiling = (
            build_tiling_space(op, readers=readers)
            if self.allocator._solver_chooses_tilings
            else None
        )
        return build_op_split_space(op, config.sencores, tiling=tiling)

    def _eligible_clone_inputs(
        self, graph: GraphLowering, lifetimes: dict[str, list[int]]
    ) -> list[str]:
        """Graph inputs eligible to be cloned into LX.

        The same shared predicate the placement path's input loop uses, with
        ``division_is_fixed=False``: the core division is deferred to the solver,
        since a clone can take any division for which some valid choice
        satisfies all its children. That constraint is enforced by requiring
        each child to match the parent, not by pre-rejecting here.
        """
        return [
            name
            for name in graph.graph_input_names
            if self.allocator._input_residency_reason(
                graph, name, lifetimes.get(name, []), division_is_fixed=False
            )
            is None
        ]

    def _clone_divisions_and_matches(
        self,
        input_name: str,
        consumers: list[Operation],
        divisions: dict[str, list[CoreDivision]],
        prep_cache: dict,
    ) -> tuple[list[CoreDivision], dict[str, list[tuple[int, int]]]]:
        """Determine the core divisions which are applicable to the clone
        node based on the read per core views of the clone's consumers.

        A consumer that *broadcast-reads* the input -- its view covers fewer
        cores than its division runs, because it splits an axis the input does
        not have -- is skipped. There is no single-base LX broadcast, so the
        cores without a local copy would read stale scratchpad; the same
        predicate rejects the buffer in ``get_ncores_for_buffers``, and by then
        the division is committed and ``_post_solve`` can only raise. This is
        the whole broadcast defense on this edge: unlike a producer-consumer
        edge, a clone has no write-view of its own for
        ``ResidencyEdge.match_pairs`` to compare against, since the clone's
        view *is* the consumer's.

        The applicable core divisions are found and returned as a list of
        ``CoreDivision`` objects. The mapping such that the clone output
        per core view matches a given op's read per core view is returned
        where the mapping exists for each consumer. When solved the parent
        output per core view must match that of all consumers to be placed.
        This forces correctness at solve time rather than pre-pruning by
        finding the intersection of core divisions.
        """
        clone_divs: list[CoreDivision] = []
        clone_views: list[PerCoreView] = []
        matches: dict[str, list[tuple[int, int]]] = {}
        for consumer in consumers:
            cname = consumer.get_name()
            consumer_divs = divisions[cname]
            rw = op_read_writes(consumer)
            read_dep = next(
                (
                    r
                    for r in rw.reads
                    if r.name == input_name and isinstance(r, MemoryDep)
                ),
                None,
            )
            write = next((w for w in rw.writes if isinstance(w, MemoryDep)), None)
            if read_dep is None or write is None:
                matches[cname] = []
                continue
            views = self._views_for_divs(
                consumer, read_dep, input_name, consumer_divs, prep_cache
            )
            pairs: list[tuple[int, int]] = []
            for j, (view, _, repr_ok) in enumerate(views):
                if not repr_ok:
                    continue
                # ``num_cores`` is the division's core count; the split product
                # is what the view covers. They differ exactly on a broadcast
                # read (see the docstring).
                if math.prod(f for _, f in view.work_slice_dims) != view.num_cores:
                    continue
                k = next(
                    (
                        idx
                        for idx, candidate in enumerate(clone_views)
                        if candidate.same_partition(view)
                    ),
                    None,
                )
                if k is None:
                    cd = consumer_divs[j]
                    per_sym = cd.splits
                    # Project onto the input: a consumer split on an axis the
                    # input lacks (e.g. a matmul's free dim) does not slice the
                    # input, so it must not count toward the clone's cores.
                    # Otherwise the cores_used check below cannot reject a
                    # broadcast read.
                    read_syms = read_dep.index.free_symbols
                    k = len(clone_divs)
                    clone_divs.append(
                        CoreDivision(
                            splits={
                                sym: split
                                for sym, split in per_sym.items()
                                if split > 1 and sym in read_syms
                            }
                        )
                    )  # a clone op cannot have a reduction split
                    clone_views.append(view)
                # No core-count check here: ``same_partition`` already demands
                # equal core counts, and the guard above ties each view's count
                # to its division's.
                pairs.append((k, j))
            matches[cname] = pairs
        # An empty ``clone_divs`` means no consumer matched the clone under any
        # division, so it has no valid core division. Return it empty rather than
        # fabricating a whole-buffer fallback that no consumer matches: the caller
        # drops such a clone (it can never reside), keeping it out of the solver's
        # >=1-division invariant.
        return clone_divs, matches

    @staticmethod
    def _parent_residency_edges(
        consumer_op: Optional[Operation],
        parent_names: list[str],
        op_by_name: dict[str, Operation],
        prep_cache: dict,
        residency_by_buf: dict[str, Optional[str]],
    ) -> dict[str, ResidencyEdge]:
        """One :class:`ResidencyEdge` per divided producer this op reads, which
        decides both which candidate pairs are compatible and which producers
        are excluded from matching altogether. A producer with no entry can host
        no residency for this consumer at all."""
        if consumer_op is None:
            return {}
        consumer_reads = op_read_writes(consumer_op).reads
        edges: dict[str, ResidencyEdge] = {}
        for parent in parent_names:
            if parent not in op_by_name:
                continue
            edge = build_residency_edge(
                parent,
                op_by_name[parent],
                consumer_op,
                consumer_reads,
                residency_by_buf.get(parent, "not in graph"),
                prep_cache,
            )
            if edge is not None:
                edges[parent] = edge
        return edges

    def _cd_parent_matches(
        self,
        edges: dict[str, ResidencyEdge],
        consumer_divs: list[CoreDivision],
        divisions: dict[str, list[CoreDivision]],
    ) -> dict[str, list[tuple[int, int]]]:
        """Physical slicing-match pairs for each divided producer this op reads:
        the pair-table materialization of :meth:`_parent_residency_edges`.

        The pairing is the per-core-view comparison ``get_ncores_for_buffers``
        uses -- correct across reductions/reshapes, where a coeff-keyed
        signature would conflate axes.
        """
        return {
            parent: edge.match_pairs(
                divisions[parent],
                consumer_divs,
            )
            for parent, edge in edges.items()
        }

    @staticmethod
    def _cap_relayout_groups(
        parent: str,
        consumer: str,
        candidates: list[RelayoutCandidate],
        consumer_divs: list[CoreDivision],
        consumer_costs: dict[int, float] | None = None,
        pinned_consumer_divisions: frozenset[int] = frozenset(),
    ) -> list[RelayoutCandidate]:
        """Keep the candidates of the ``config.lx_solver_relayout_groups_per_edge``
        cheapest destination views of one (source, consumer) edge.

        Each distinct consumer read partition is its own
        relayout group, and every group becomes a copy buffer the solver must
        place, though the consumer will read through at most one of them. A
        group is ranked by copy plus consumer execution cost, not copy cost
        alone: a cheap copy can feed an expensive matmul division. This is a
        shortlist estimate, not the whole-graph objective. Ties favor more
        consumer cores, the solver's existing preference. Dropping a group only
        removes an option:
        a consumer division without a copy is treated exactly like an unpriced
        pair (match for free or spill), and every fired relayout is still
        certified at materialization.

        ``pinned_consumer_divisions`` (divisions another parent already
        matches for free, per ``_cd_parent_matches``) are kept regardless of
        cost rank: cost alone can't see that reaching one of them lets every
        parent land on the same division, so the cap must not drop it.
        """
        cap = config.lx_solver_relayout_groups_per_edge
        if cap <= 0 or not candidates:
            return candidates
        by_group: dict[int, list[RelayoutCandidate]] = {}
        for candidate in candidates:
            by_group.setdefault(candidate.group, []).append(candidate)
        if len(by_group) <= cap:
            return candidates

        pinned_groups = {
            group
            for group, members in by_group.items()
            if any(c.consumer_division in pinned_consumer_divisions for c in members)
        }

        def rank(item: tuple[int, list[RelayoutCandidate]]) -> tuple:
            group, members = item
            best = min(
                c.cost_ns
                + (
                    consumer_costs[c.consumer_division]
                    if consumer_costs is not None
                    else 0.0
                )
                for c in members
            )
            cores = max(consumer_divs[c.consumer_division].cores_used for c in members)
            return (best, -cores, group)

        remaining = cap - len(pinned_groups)
        ranked = [item for item in by_group.items() if item[0] not in pinned_groups]
        cost_kept = {
            group for group, _ in sorted(ranked, key=rank)[: max(remaining, 0)]
        }
        kept = pinned_groups | cost_kept
        logger.debug(
            "[lx solver relayout] %s -> %s: keeping %d of %d destination views"
            " (%d pinned by a sibling parent match)",
            parent,
            consumer,
            len(kept),
            len(by_group),
            len(pinned_groups),
        )
        return [c for c in candidates if c.group in kept]

    @staticmethod
    def _relayout_consumer_costs(consumer_op, consumer_divs, parent, candidates):
        """Reuse the execution model to shortlist copies feeding this consumer.

        Price this input in LX and the remaining arguments in HBM. The solver
        still decides their actual placement and prices complete bundles.
        Extract once per consumer division, not once per source/destination pair.
        """
        from torch_spyre._inductor.cost_model import predict_ops
        from torch_spyre._inductor.dump_cost_model import extract_op_features
        from torch_spyre._inductor.scratchpad.sa_cooptimizer import _work_slices

        is_lx = {dep.name: False for dep in op_read_writes(consumer_op).reads}
        is_lx[consumer_op.get_name()] = False
        is_lx[parent] = True
        candidate_work_slices = [
            _work_slices(consumer_op, division) for division in consumer_divs
        ]
        return {
            j: float(
                predict_ops(
                    [
                        extract_op_features(
                            consumer_op,
                            _work_slices(consumer_op, consumer_divs[j]),
                            is_lx=is_lx,
                            candidate_work_slices=candidate_work_slices,
                        )
                    ],
                    params=_COST_PARAMS,
                )
            )
            for j in sorted({c.consumer_division for c in candidates})
        }

    def _cd_parent_relayouts(
        self,
        graph: GraphLowering,
        consumer_op: Optional[Operation],
        consumer_divs: list[CoreDivision],
        parent_names: list[str],
        divisions: dict[str, list[CoreDivision]],
        op_by_name: dict[str, Operation],
        prep_cache: dict,
        residency_by_buf: dict[str, Optional[str]],
        cd_parent_matches: Optional[dict[str, list[tuple[int, int]]]] = None,
    ) -> dict[str, list[RelayoutCandidate]]:
        """Priced relayout candidates for each divided producer this op reads.

        Sibling of :meth:`_cd_parent_matches`: where that method records the
        division pairs whose per-core views are EQUAL (residency is free), this
        one records the pairs whose views DIFFER but are relayout-compatible -
        the producer could stay LX-resident by paying a shuffle - as
        :class:`RelayoutCandidate` records priced by the fitted relayout law. The division-independent edge gates (single non-indirect
        write, activation source, pointwise-or-matmul consumer, ...) mirror
        ``collect_lx_relayout_plans``; the per-pair gates (permutation
        compatibility, projectable ownership on both frames, the law's fitted
        split range) live in ``solver_relayout_pair_cost``. Gated on the
        solver's ``decides_lx_relayouts``.

        ``cd_parent_matches`` (the caller's own ``_cd_parent_matches`` result)
        gives each parent's cap-and-rank step the other parents' already-free
        divisions, so it won't drop the one group that would let this parent
        join them.
        """
        if (
            not self.allocator._decides_lx_relayouts
            or config.ktir_emitter
            or config.lx_view_relation == "derived"
            or not config.lx_solver_relayouts
        ):
            return {}
        if consumer_op is None:
            return {}
        relayouts: dict[str, list[RelayoutCandidate]] = {}
        for parent in parent_names:
            if parent not in op_by_name:
                continue
            # The source must be residency-eligible: a relayout keeps it in LX.
            if residency_by_buf.get(parent, "not in graph") is not None:
                continue
            parent_op = op_by_name[parent]
            context = solver_relayout_edge_context(
                graph, parent_op, consumer_op, parent, op_by_name
            )
            if context is None:
                continue
            write_dep, read_dep, prod_coords, cons_coords, prod_space, cons_space = (
                context
            )
            parent_divs = divisions[parent]
            # Same per-candidate view screens as _cd_parent_matches: the source
            # gets LX-pinned exactly like a matched producer, so the same
            # coherence bars apply (partial-reduction write, unrepresentable
            # slicing). Movement is checked on the complete per-core views.
            prod_views: list[Optional[PerCoreView]] = [
                view if repr_ok and not partial else None
                for view, partial, repr_ok in self._views_for_divs(
                    parent_op, write_dep, parent, parent_divs, prep_cache
                )
            ]
            cons_views: list[Optional[PerCoreView]] = [
                view if repr_ok else None
                for view, _partial, repr_ok in self._views_for_divs(
                    consumer_op, read_dep, parent, consumer_divs, prep_cache
                )
            ]
            device_dims = list(parent_op.layout.device_layout.device_size)
            out_elems = math.prod(device_dims)
            dtype_bytes = parent_op.get_dtype().itemsize

            # Ownership must project into loop symbols on both frames (the
            # committed path's work_division_from_view gates), cached per view:
            # several candidates often induce the same view. The projected
            # division is kept (not just a bool) because the transition gate
            # below compares source and destination divisions for equality.
            projected: dict[tuple, Optional[TensorWorkDivision]] = {}

            def _projected(view, coords, space, frame) -> Optional[TensorWorkDivision]:
                key = (view, frame)
                if key not in projected:
                    try:
                        projected[key] = work_division_from_view(
                            view, device_dims, coords, space
                        )
                    except ValueError:
                        projected[key] = None
                return projected[key]

            # A coarse-tiled CANDIDATE can never host a relayout, for the same
            # reasons a coarse-tiled op cannot in general (a tiled candidate's
            # buffer is per-tile scratch). The loop_info edge gate covers
            # hint-materialized tiling decided before the solve; once the
            # unified-tiling work (#3923) makes tiling a per-candidate solver
            # choice, cd.tiling is the only marker. Written via getattr so it
            # is inert until that lands.
            def _candidate_tiled(cd) -> bool:
                tiling = getattr(cd, "tiling", None)
                return tiling is not None and not getattr(tiling, "is_untiled", True)

            # solver_relayout_edge_context's same-tile-group exception admits a
            # coarse-tiled parent/consumer pair: that candidate's copy runs
            # once per loop trip, not once total, so its price must scale by
            # the loop's own trip count or the solver sees an under-priced
            # option that could out-rank a correctly-priced alternative whose
            # real total cost is actually lower (cyang49's PR #5057 review).
            loop_trip_count = _coarse_tile_loop_trip_count(parent_op)

            # The per-core LX span of a view (#3440's reservation for a relayout
            # member). Unavailable (non-standard arrangement, an unsplittable
            # stick axis) is a decline, as on the committed path: without a
            # size the copy cannot be placed, so the pair is never offered.
            spans: dict[PerCoreView, Optional[int]] = {}

            def _span(view: PerCoreView) -> Optional[int]:
                if view not in spans:
                    try:
                        spans[view] = partition_footprint(parent_op.layout, view)
                    except (TypeError, ValueError) as exc:
                        logger.debug(
                            "[lx solver relayout] %s: span unavailable for view %s: %s",
                            parent,
                            dict(view.work_slice_dims),
                            exc,
                        )
                        spans[view] = None
                return spans[view]

            # Graph-wide cache: the same (source view, destination view, cores,
            # tensor geometry) recurs across structurally identical ops (the
            # unrolled KV blocks of attention), and pricing it re-runs the movement
            # gate's per-core owner comparison each time.
            pair_cost = self._relayout_pair_costs
            candidates: list[RelayoutCandidate] = []
            # Destination views are interned per parent across every consumer
            # of this solve: two consumers whose candidates land on the same
            # physical view of the same parent share one group, hence one
            # shuffle (interned by ownership, so spelling cannot split a group).
            view_groups = self._relayout_view_groups.setdefault(parent, {})
            for i, pv in enumerate(prod_views):
                if pv is None or _candidate_tiled(parent_divs[i]):
                    continue
                for j, cv in enumerate(cons_views):
                    # Physical ownership, not structural equality: two slot
                    # spellings of one partition are a match, never a shuffle.
                    if cv is None or pv.same_partition(cv):
                        continue
                    if _candidate_tiled(consumer_divs[j]):
                        continue
                    ncores = parent_divs[i].cores_used
                    dst_cores = consumer_divs[j].cores_used
                    # The committed collector's consumer rules (#3440): the core
                    # domains, then the gather rule on the destination view; the
                    # geometric half is movement_supported's, inside the pair cost.
                    if core_domain_rejection(ncores, dst_cores):
                        continue
                    if grouped_gather_rejection(consumer_op, ncores, cv):
                        continue
                    if _projected(pv, prod_coords, prod_space, "prod") is None:
                        continue
                    # The relayout copy iterates the producer's shape, so the
                    # destination view must also project on the producer frame
                    # (materialize_lx_relayouts commits it there and raises).
                    if _projected(cv, prod_coords, prod_space, "prod") is None:
                        continue
                    src_division = _projected(pv, cons_coords, cons_space, "cons")
                    dst_division = _projected(cv, cons_coords, cons_space, "cons")
                    if src_division is None or dst_division is None:
                        continue
                    # Distinct per-core views that collapse to one logical work
                    # division would codegen as a plain identity copy with the
                    # cross-core movement silently omitted (#3926); the solver
                    # must never be offered such a pair.
                    if (
                        _unsupported_relayout_transition_reason(
                            src_division, dst_division
                        )
                        is not None
                    ):
                        continue
                    source_span, destination_span = _span(pv), _span(cv)
                    if (
                        source_span is None
                        or destination_span is None
                        or destination_span > self.allocator.size
                    ):
                        continue
                    key = (
                        pv,
                        cv,
                        ncores,
                        dst_cores,
                        tuple(device_dims),
                        out_elems,
                        dtype_bytes,
                    )
                    if key not in pair_cost:
                        pair_cost[key] = solver_relayout_pair_cost(
                            pv,
                            cv,
                            ncores,
                            device_dims,
                            out_elems,
                            dtype_bytes,
                            destination_num_cores=dst_cores,
                        )
                    cost = pair_cost[key]
                    if cost is None:
                        continue
                    # pair_cost is a graph-wide cache keyed on geometry alone
                    # (shared across parents), so the trip multiplier is
                    # applied to this candidate's own cost, never written
                    # back into the cached value.
                    cost = cost * loop_trip_count
                    candidates.append(
                        RelayoutCandidate(
                            parent=parent,
                            consumer=consumer_op.get_name(),
                            source_division=i,
                            consumer_division=j,
                            group=_intern_view_group(view_groups, cv),
                            source_view=pv,
                            destination_view=cv,
                            cost_ns=cost,
                            source_footprint_bytes=source_span,
                            destination_footprint_bytes=destination_span,
                        )
                    )
            consumer_costs = None
            cap = config.lx_solver_relayout_groups_per_edge
            if cap > 0 and len({c.group for c in candidates}) > cap:
                try:
                    consumer_costs = self._relayout_consumer_costs(
                        consumer_op, consumer_divs, parent, candidates
                    )
                except (ValueError, RuntimeError, TypeError) as exc:
                    logger.warning(
                        "relayout shortlist consumer cost unavailable for %s: %s; "
                        "using copy cost only",
                        consumer_op.get_name(),
                        exc,
                    )
            sibling_pinned = frozenset(
                j
                for other_parent, pairs in (cd_parent_matches or {}).items()
                if other_parent != parent
                for _, j in pairs
            )
            candidates = self._cap_relayout_groups(
                parent,
                consumer_op.get_name(),
                candidates,
                consumer_divs,
                consumer_costs,
                sibling_pinned,
            )
            if candidates:
                relayouts[parent] = candidates
                logger.debug(
                    "[lx solver relayout] %s -> %s: %d priced candidate pair(s), "
                    "cheapest %.1f ns",
                    parent,
                    consumer_op.get_name(),
                    len(candidates),
                    min(c.cost_ns for c in candidates),
                )
        return relayouts

    @staticmethod
    def _views_for_divs(op, dep, buf_name, divs: list[CoreDivision], prep_cache: dict):
        """Per-core views of ``buf_name`` for each candidate division of ``op``.

        The candidate-invariant prep is computed once and shared through
        ``prep_cache``, so cost scales with the op rather than its candidate
        count. A tiled division is reported unrepresentable: these views pair
        clones and relayout copies on core ownership alone, and a tiled
        candidate would also have to agree tile by tile.
        """
        return [
            _view_for_div(op, dep, buf_name, cd, prep_cache)
            if not cd.tile_splits
            else (PerCoreView((), (), num_cores=cd.cores_used), False, False)
            for cd in divs
        ]
