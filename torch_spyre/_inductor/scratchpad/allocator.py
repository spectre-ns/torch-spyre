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

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, cast

import sympy
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    Operation,
    TensorBox,
)

from torch_spyre._C import get_device_size_in_bytes
from torch_spyre._inductor import config
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.pass_utils import (
    PerCoreView,
    commit_iteration_space_ownership,
    iteration_space_from_op,
)
from torch_spyre._inductor.scratchpad.buffer_graph import (
    SOLVER_CHOSE_SPILL,
    ResourceNode,
    SpyreOptimizationGraph,
    Transform,
)
from torch_spyre._inductor.scratchpad.encoding_strategy import (
    CandidateTables,
    EnumerationFactory,
)
from torch_spyre._inductor.scratchpad.graph_editor import (
    GraphEditor,
)
from torch_spyre._inductor.scratchpad import pricing
from torch_spyre._inductor.scratchpad.drain_plans import (
    DrainPlan,
    _drained_collectives,
)
from torch_spyre._inductor.scratchpad.division_menus import (
    _split_option_is_legal,
)
from torch_spyre._inductor.scratchpad.residency_rules import (
    _extern_kernel_in_live_range,
    _get_op_name,
    _multi_output_extern_kernel_in_live_range,
)
from torch_spyre._inductor.scratchpad.graph_solvers import (
    BufferSolver,
    GraphSolver,
)
from torch_spyre._inductor.scratchpad.lx_relayout import (
    ChosenRelayout,
    FiredRelayoutGroup,
    LXRelayoutPlan,
    materialize_lx_relayouts,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    LifetimeBoundBuffer,
    SolveError,
    TileSpec,
    ceil_div,
)
from torch_spyre._inductor.scratchpad.sa_cooptimizer import SaCoOptimizingSolver
from torch_spyre._inductor.scratchpad.utils import (
    counted_loop_entry,
    counted_loop_group_path,
    get_buffer_users,
    get_ncores_for_buffers,
    round_up_to_alignment,
)
from torch_spyre._inductor.work_division_constraints import (
    JOINT_TILING_AND_DIVISION_ATTR,
)
from torch_spyre._inductor.wsr.coarse_tile import (
    IterationFrame,
    capture_iteration_frame,
    iteration_symbol_remap,
)


logger = get_inductor_logger("scratchpad.allocator")


# Keep these values synchronized with Deeptools' LX memory tracker:
#
# * ``SenSystemDef`` removes 64 KiB of the physical 2 MiB LX for program and
#   debug data (``dsc/sysdef.cpp``).
# * ``MemTrackBundle::initializeMemoryTrackers`` uses one 128-byte stick as the
#   LX allocation granularity (``sharedtools/mem_track_bundle.cpp``).
#
# Torch and the backend compiler independently consume ``DXP_LX_FRAC_AVAIL``:
# dbo reads it in ``dbo/src/Transforms/ProgramLayout.cpp`` with the same 0.2
# default.  The ``DXP_`` prefix is historical -- the variable is a cross-compiler
# contract, so it cannot be renamed from this side alone without silently
# reintroducing the ownership mismatch of issue #3222 (Torch would read the new
# name while the backend kept defaulting the old one).  These constants define
# the fixed part of that contract.
_LX_PHYSICAL_CAPACITY_BYTES = 2 << 20
_LX_PROGRAM_DEBUG_RESERVATION_BYTES = 64 << 10
_LX_TRACKER_CAPACITY_BYTES = (
    _LX_PHYSICAL_CAPACITY_BYTES - _LX_PROGRAM_DEBUG_RESERVATION_BYTES
)
_LX_ALLOCATION_GRANULARITY_BYTES = 128


def _clear_loop_membership_metadata(op: Operation) -> None:
    """Drop loop/carry metadata from a clone placed outside its counted loop.

    ``copy_op_metadata`` copies the storage's (drain) or the first consumer's
    (input clone) attributes onto the clone, but a post-loop drain and a
    hoisted pre-loop input clone are neither loop members nor carries: a
    surviving ``loop_info`` would re-group the clone into the counted loop at
    scheduling time (``_loop_group_id``) and give it another op's per-read tile
    advance in codegen (``_general_tile_advance``), and the records would
    misclassify it in ``TableCandidates._build_graph``.  Only the two drain branches
    and the hoisted-input branch of ``_push_allocation`` call this; every other
    clone keeps today's metadata-copy behavior.
    """

    for attr in ("loop_info", "_loop_carry_record"):
        if hasattr(op, attr):
            delattr(op, attr)


def _hoisted_input_clone_entry(
    graph: GraphLowering, name: str, users: Sequence[Operation]
) -> Optional[Operation]:
    """Where an LX clone of graph input ``name`` may run once, before its loop.

    An input clone copies the whole input (``clone_lowering`` over the input's
    full ranges), never a per-trip window: each consumer keeps its own index,
    tile advance included, and only the buffer it names changes.  So when the
    first consumer runs inside a counted loop, re-running the clone on every
    trip rewrites the same LX bytes with the same values; it can run once
    before the loop.  ``counted_loop_lifetime_overrides`` already reserves the
    input's LX address from that loop's entry to its end, so the move changes
    no address, no lifetime and no capacity -- only how often the HBM read
    happens.

    Returns the loop's entry operation, or ``None`` (clone stays where it is
    today) when the first consumer is not a counted-loop member, when any
    operation mutates the input (a per-trip clone would then observe the
    writes), or when an opaque extern kernel runs between the loop entry and
    the first consumer (the clone's LX bytes would be live across it, which
    the residency gate only checked from the first use on), or when the
    input's last reader is its outermost loop's last member: no lifetime end
    override widens the clone then, so the reverse-parent in-place edge
    (``_handoff_parent_end``) may hand its slot to that reader, and the next
    trip would read the overwritten bytes that a per-trip clone re-copies.
    A multi-output fallback anywhere in the outer loop's span also prevents
    hoisting: it can run before the loop and overwrite the clone's LX bytes,
    and context switching does not bracket it.
    """
    if not users:
        return None
    entry = counted_loop_entry(graph.operations, users[0])
    if entry is None:
        return None
    for op in graph.operations:
        try:
            if name in op.get_mutation_names():
                return None
        except NotImplementedError:
            return None
    start = graph.operations.index(entry)
    first_use = graph.operations.index(users[0])
    if _extern_kernel_in_live_range(graph, list(range(start, first_use + 1))):
        return None
    outer = counted_loop_group_path(entry)[:1]
    end = max(
        i
        for i, op in enumerate(graph.operations)
        if counted_loop_group_path(op)[:1] == outer
    )
    if _multi_output_extern_kernel_in_live_range(graph, [start, end]):
        return None
    last_outer = counted_loop_group_path(users[-1])[:1]
    if last_outer and not any(
        counted_loop_group_path(op)[:1] == last_outer
        for op in graph.operations[graph.operations.index(users[-1]) + 1 :]
    ):
        return None
    return entry


def _assert_drain_plan_committed(
    graph: GraphLowering,
    storage: ResourceNode,
    solved: "SolvedPlan",
    op_by_name: dict[str, Operation],
    plan: DrainPlan,
) -> None:
    """Fail-fast checks for a planned drain the solver committed to LX.

    Reaching this function means the pre-solve plan was accepted and the
    solver chose the storage resident, so every fact below must hold; a
    failure is an internal error, not a decline (all refusal happens before
    the solve, in :func:`validated_drain_plans`, and the HBM fallback simply
    never gets here).  The ownership check recomputes the carry edge's
    admitted ``(storage, update)`` division pairs from the chosen divisions --
    the same geometry ``encode_read`` gated on -- so a resident
    storage whose update does not actually match its slicing can never emit a
    drain.
    """

    if plan.anchor_op not in graph.operations:
        raise AssertionError(
            f"drain plan for {plan.storage_name}: lowered anchor "
            f"{plan.anchor_op.get_name()} is no longer in graph.operations; "
            "the plan was validated pre-solve, so this is an internal error"
        )
    if getattr(plan.loop_origin, "graph", None) is not graph.graph:
        raise AssertionError(
            f"drain plan for {plan.storage_name}: retained FX loop origin "
            f"{plan.loop_origin} is not in the current lowering graph"
        )
    storage_op = graph.get_buffer(plan.storage_name)
    update_op = op_by_name.get(plan.update_name)
    if update_op is None:
        raise AssertionError(
            f"drain plan for {plan.storage_name}: tagged update "
            f"{plan.update_name} is missing from graph.operations"
        )
    if (
        getattr(storage_op, "iteration_space_ownership", None) is None
        or getattr(update_op, "iteration_space_ownership", None) is None
    ):
        raise AssertionError(
            f"drain plan for {plan.storage_name}: committed physical "
            "ownership is missing on the storage or its update"
        )
    update = solved.graph.nodes.get(plan.update_name)
    storage_menu = solved.tables.menus.get(storage.name)
    update_menu = solved.tables.menus.get(plan.update_name)
    if update is None or not storage_menu or not update_menu:
        raise AssertionError(
            f"drain plan for {plan.storage_name}: storage or update is not a "
            "core-division solver node"
        )
    chosen = (solved.division_index(storage), solved.division_index(update))
    if None in chosen:
        raise AssertionError(
            f"drain plan for {plan.storage_name}: solver left the storage or "
            "update division unchosen"
        )
    from torch_spyre._inductor.scratchpad.table_candidates import TableCandidates

    edge = TableCandidates._loop_carry_update_edge(update_op, op_by_name, {})
    pairs = (
        edge.match_pairs(list(storage_menu), list(update_menu))
        if edge is not None
        else []
    )
    if chosen not in pairs:
        raise AssertionError(
            f"drain plan for {plan.storage_name}: committed divisions "
            f"{chosen} are not an admitted carry pair"
        )


@dataclass
class SolvedPlan:
    """A solved graph as the commit path reads it: the nodes and their
    decisions, the strategy and its candidates (a node's chosen division is
    one of them), and the solver that produced it."""

    graph: SpyreOptimizationGraph
    strategy: Any
    solver: Optional[GraphSolver]
    tables: CandidateTables

    @classmethod
    def of_buffers(
        cls, buffers: Sequence[LifetimeBoundBuffer], solver: Any = None
    ) -> "SolvedPlan":
        """A solved plan given in buffer form (tests): the graph of
        ``buffers`` in byte units, each decision read off its buffer
        (``address``, ``chosen_division``, ``lx_view``, ``spill_reason``)."""
        from torch_spyre._inductor.scratchpad.buffer_graph import (
            build_graph,
            empty_graph,
        )

        graph = build_graph(buffers, 1, 1 << 62, empty_graph(0).settings)
        tables = CandidateTables.of_buffers(buffers)
        strategy = EnumerationFactory().strategy_from_tables(tables)
        by_name = {b.name: b for b in buffers}
        for node in graph.get_nodes():
            b = by_name[node.name]
            node.resident = b.address is not None
            node.offset = b.address
            node.spill_reason = b.spill_reason
            chosen = getattr(b, "chosen_division", None)
            if chosen is not None and node.name in tables.menus:
                strategy.solved[node] = chosen
        return cls(graph, strategy, solver, tables)

    @property
    def nodes(self) -> list[ResourceNode]:
        return list(self.graph.get_nodes())

    def address(self, item: ResourceNode | Transform) -> Optional[int]:
        """The item's LX address in bytes, or ``None`` when it is not
        resident."""
        if not item.resident or item.offset is None:
            return None
        return item.offset * self.graph.alignment

    def division_index(self, node: ResourceNode) -> Optional[int]:
        """The index of the node's chosen division in its menu."""
        return self.strategy.solved.get(node)

    def division(self, node: ResourceNode) -> Optional[CoreDivision]:
        """The node's chosen division, or ``None`` when it was committed
        upstream."""
        i = self.division_index(node)
        return None if i is None else self.tables.menus[node.name][i]

    def per_core_size(self, node: ResourceNode) -> int:
        cd = self.division(node)
        if cd is None:
            return node.nbytes
        return ceil_div(node.nbytes, cd.output_partition * cd.tiling.output_tile_count)

    def chosen_relayouts(self) -> list[ChosenRelayout]:
        """Every fired relayout: the candidate a read went through, and its
        copy's address in bytes."""
        out = []
        for node in self.nodes:
            for edge in node.inputs:
                t = edge.transform
                if t is not None and t.offset is not None:
                    out.append(
                        ChosenRelayout(self.strategy.fired[edge], t.offset).scaled(
                            self.graph.alignment
                        )
                    )
        return out

    def report(self) -> list[CoreDivisionBuffer]:
        """The solved plan in buffer form, for the cost-expression dump."""
        from torch_spyre._inductor.scratchpad.graph_solvers import buffers_of

        out: list[CoreDivisionBuffer] = []
        for b in buffers_of(self.graph, self.tables):
            node = self.graph.nodes[b.name]
            b.address = self.address(node)
            b.chosen_division = self.division_index(node)
            b.spill_reason = node.spill_reason
            out.append(b)
        return out


class ScratchpadOptimizationPass(ABC):
    """
    Abstract class for optimization passes which are implemented to improve
    a graph's overall scratchpad memory utilization and/or memory latency.
    """

    @abstractmethod
    def apply_pass(self, graph: GraphLowering):
        """
        Accepts a candidate graph to be optimized and evaluated for scratchpad memory allocation.
        `graph` will be mutated according in an implementation defined way. The order and
        number of nodes in the graph may change as a result of an optimization pass.

        Args:
            graph (GraphLowering): The graph to be optimized for scratchpad memory allocation
        """
        pass


class ScratchpadAllocator:
    """
    Class for allocating on scratchpad
    """

    def __init__(
        self,
        solver: GraphSolver,
        size: int,
        pre_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
        post_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
        division_mode: Literal["open", "committed"] = "committed",
        prune: bool = False,
    ):
        """Configure the allocator with a graph solver and graph passes.

        Args:
            solver: The graph solver that fills the solve's open decisions;
                its capabilities say what the solve offers it.
            size: LX size
            pre_optimization_passes: Graph passes applied before layout planning.
                Defaults to no passes.
            post_optimization_passes: Graph passes applied after layout planning.
                Defaults to no passes.
            division_mode: ``"committed"`` keeps the divisions work division
                committed upstream; ``"open"`` chooses each buffer's core
                division (and coarse tiling) jointly with its placement, which
                needs a solver that chooses divisions.
            prune: Enable heuristic pruning of the core-division search space.
        """
        self.solver = solver
        self.capabilities = self.solver.capabilities
        if division_mode == "open" and not self.capabilities.chooses_core_divisions:
            raise ValueError(
                f"{self.solver.name} cannot choose core divisions; the open "
                "division mode needs a solver that does"
            )
        if pre_optimization_passes is None:
            pre_optimization_passes = []
        if post_optimization_passes is None:
            post_optimization_passes = []

        # Populated during plan_allocation: maps buffer/op name → reason string.
        # Stamped by _record_spill_reasons from the solver's own spill_reasons
        # (the declared residency verdict, or its capacity check)
        # (for the solver decision). Reset at the start of each plan_allocation.
        self.pre_optimization_passes = pre_optimization_passes
        self.post_optimization_passes = post_optimization_passes
        self.size = size
        # The post-loop drain plans the last solve's candidates were validated
        # against: what the lifetimes, the push and its assertions consume.
        self._validated_drain_plans: dict[str, DrainPlan] = {}
        self.division_mode = division_mode
        self.prune = prune
        # Per op ``_apply_chosen_tilings`` tiled, its pre-apply iteration
        # symbols to its live ones; read through :meth:`_live_splits`.
        self._tiling_symbol_remaps: dict[str, dict[sympy.Symbol, sympy.Symbol]] = {}
        self.factory = EnumerationFactory(self.capabilities)
        # Candidate generation belongs to the solve's table encoding; this
        # allocator dispatches the solve and applies the result.
        self.candidates = self.factory.candidates(size, prune, committed=not self.joint)

    @property
    def joint(self) -> bool:
        """Whether divisions are open: chosen jointly with placement."""
        return self.division_mode == "open"

    def plan_allocation(self, graph: GraphLowering):
        """Run pre-passes, plan LX for ``graph``, then run post-passes.

        The allocator coordinates; it builds no buffers. The solve's graph is
        built from the IR (:meth:`_build_graph`), the configured solver fills
        its decisions (:meth:`_solve`), a solve that chose coarse tilings
        re-plans the tiled graph (:meth:`_materialize_selection`), and the
        solved graph is committed back to the IR: relayout plans, divisions,
        views, addresses.

        Args:
            graph: Lowered graph whose buffers will be assigned LX scratchpad
                addresses where viable.
        """
        self._run_passes(self.pre_optimization_passes, graph)
        plan = self._solve(graph)
        plan = self._materialize_selection(graph, plan)
        accepted_lx_relayouts = self._finalize_lx_relayout_allocation(plan, graph)
        self._post_solve(graph, plan, accepted_lx_relayouts)
        reasons = self._get_spill_reasons(plan)
        self._push_allocation(graph, plan, accepted_lx_relayouts)
        self._log_lx_pinning(graph, reasons)
        self._run_passes(self.post_optimization_passes, graph)

    @staticmethod
    def _run_passes(
        passes: Sequence[ScratchpadOptimizationPass], graph: GraphLowering
    ) -> None:
        for p in passes:
            p.apply_pass(graph)

    def _build_graph(
        self, graph: GraphLowering
    ) -> tuple[SpyreOptimizationGraph, CandidateTables]:
        """The solve's graph, built from the IR, and the table strategy's
        candidates for it."""
        built = self.candidates.prepare_graph(graph)
        self._validated_drain_plans = self.candidates.drain_plans
        return built

    def _solve(self, graph: GraphLowering) -> SolvedPlan:
        """Build the graph and solve it under the priced cost expression."""
        solve_graph, tables = self._build_graph(graph)
        strategy = self.factory.strategy_from_tables(tables)
        plan = SolvedPlan(solve_graph, strategy, self.solver, tables)
        cost_expr, bundle_terms = pricing.cost_expr(graph, plan, self.factory)
        self.solver.solve(solve_graph, strategy, cost_expr)
        if config.dump_cost_expr_file and cost_expr is not None:
            pricing.dump_cost_expr(graph, plan, cost_expr, bundle_terms, self.size)
        return plan

    def _materialize_selection(
        self, graph: GraphLowering, plan: SolvedPlan
    ) -> SolvedPlan:
        """Act on what the solve *chose* before the choice is committed.

        A solve that chose coarse tilings and asks to re-plan
        (``replans_after_tiling``) has ``CoarseTilingPass`` apply exactly those
        choices (mutating the IR the same way a pre-stickification hint
        would), and the graph is built and solved again over the materialized
        IR, so new boundary buffers get placed and the applied ops get their
        final divisions. The second solve enumerates no further tilings
        (``TableCandidates.suppress_tiling``), so it terminates.

        Ordering is solve-before-apply: a ``SolveError`` from the first solve
        propagates over the *unmutated* graph, so ``scratchpad_planning``'s
        committed replan never runs on a half-tiled graph (a second-solve
        ``SolveError`` falls back over the fully-tiled graph, which is a valid
        outcome). A committed solve selects nothing to materialize.
        """
        if not self.joint or not self.capabilities.replans_after_tiling:
            return plan
        choices = self._chosen_tilings(graph, plan)
        if logger.isEnabledFor(logging.DEBUG):
            for name, spec in choices.items():
                logger.debug("chosen_tiling: %s -> %s", name, spec.label)
        if not choices:
            return plan

        from torch_spyre._inductor.scratchpad.coarse_tiling import CoarseTilingPass

        op_count = len(graph.operations)
        CoarseTilingPass(choices).apply_pass(graph)
        assert len(graph.operations) >= op_count, (
            "coarse tiling apply must not drop operations"
        )
        # Re-plan over the materialized tiling; no further tiling, so the second
        # solve only places.
        self.candidates.suppress_tiling = True
        try:
            return self._solve(graph)
        finally:
            self.candidates.suppress_tiling = False

    def _finalize_lx_relayout_allocation(
        self, plan: SolvedPlan, graph: GraphLowering
    ) -> list[LXRelayoutPlan]:
        """Turn the solver's fired relayouts into materializable plans.

        The solver already guaranteed what a materialization needs: the
        copy's residency IS the decision, the served consumers'
        division pairs are pinned by the literals that require the copy
        resident, and the 2D no-overlap kept source and copy disjoint. The
        fired candidates carry the views the enumeration priced, so nothing is
        re-derived here: the fired edges regroup by (source, destination view),
        the committed divisions and the copy's address are checked against
        them, and ``materialize_lx_relayouts`` gets one plan per fired group.
        """
        nodes = plan.graph.nodes
        plans: list[LXRelayoutPlan] = []
        for group in FiredRelayoutGroup.from_chosen(plan.chosen_relayouts()):
            source = nodes[group.parent]
            address = plan.address(source)
            assert plan.division_index(source) == group.source_division, (
                f"relayout group {group.parent}/g{group.group} chose source "
                f"division {group.source_division} but "
                f"{plan.division_index(source)} was committed"
            )
            assert address is not None, (
                f"relayout source {group.parent} has no LX address"
            )
            for member in group.members:
                consumer = nodes[member.candidate.consumer]
                (edge,) = [e for e in consumer.inputs if e.src is source]
                assert edge.transform is not None
                copy_address = plan.address(edge.transform)
                assert copy_address == group.destination_address, (
                    f"relayout group {group.parent}/g{group.group}: consumers read "
                    f"the copy at {group.destination_address} but it was placed "
                    f"at {copy_address}"
                )
                assert (
                    plan.division_index(consumer) == member.candidate.consumer_division
                ), (
                    f"relayout pair ({group.source_division}, "
                    f"{member.candidate.consumer_division}) disagrees with committed "
                    f"division {plan.division_index(consumer)} on {group.parent} -> "
                    f"{consumer.name}"
                )
            plans.append(group.plan(address))
        return plans

    def _post_solve(
        self,
        graph: GraphLowering,
        plan: SolvedPlan,
        accepted_lx_relayouts: Sequence[LXRelayoutPlan],
    ) -> None:
        """Commit the chosen tilings, divisions and views before the push.

        Apply before commit: ``commit_iteration_space_ownership`` builds the
        ownership off ``iteration_space_from_op``, whose symbols come from the
        write dep's ranges -- which ``_divide_ranges`` invalidates."""
        tiled = self._apply_chosen_tilings(graph, plan)
        self._check_priced_footprints(graph, plan, tiled)
        # The divisions must be committed such that any buffer clones can
        # correctly pull the selected core division from the dependent buffers
        # when the graph is updated with clones in ``_push_allocation``.
        commit_divisions(graph, plan, self._live_splits)
        # After the commit, and before the views below are derived from it: a
        # copy-out the apply mints is not a node, so nothing above gave it a
        # division and it would be judged against its tiled op's as a mismatch.
        self._commit_copy_out_divisions(graph, plan, tiled)
        commit_lx_views(
            graph,
            plan,
            accepted_lx_relayouts,
            drained_readers=_drained_collectives(self._validated_drain_plans),
        )
        log_solver_decisions(graph, plan)

    def _get_spill_reasons(self, plan: SolvedPlan) -> dict:
        """Why each node the solve left out of LX did not land there: the
        solver's ``spill_reason`` -- the declared verdict
        (``residency_reason``), its capacity check, or its own choice."""
        return {
            n.name: n.spill_reason or SOLVER_CHOSE_SPILL
            for n in plan.nodes
            if not n.resident
        }

    def _log_lx_pinning(self, graph: GraphLowering, reasons: dict) -> None:
        """Log the final LX pinning decision for every op in the graph."""
        # Skip the per-op getattr walk unless DEBUG is on.
        if not logger.isEnabledFor(logging.DEBUG):
            return
        for op in graph.operations:
            reason = reasons.get(op.name, "lx")
            logger.debug(
                "lx_pinning: %s (%s) → %s",
                op.name,
                _get_op_name(op),
                reason,
            )

    def _push_allocation(
        self,
        graph: GraphLowering,
        plan: SolvedPlan,
        accepted_lx_relayouts: list[LXRelayoutPlan],
    ):
        """Push the solved placement into the IR. This includes cloning graph
        inputs and graph outputs:

        - A graph input B that is allocated into LX means that it is cloned; call the clone C. The
        downstream users of B are now made to use C. The LX allocation is effectuated by assigning
        it to C.

        - A graph output B that is allocated into LX means that it is cloned; call the clone C.
        Nothing changes for the downstream users. The LX allocation is effectuated by assigning it
        to B itself. The graph is made to have C as its output.

        - A buffer that is neither a graph input nor a graph output gets the LX allocation assigned
        to itself.  If it is a loop carry whose drain plan names a collective, a post-loop clone C
        is also inserted and the collective is made to read C."""
        outputs = set(graph.get_output_names())
        inputs = set(graph.graph_input_names)

        buffer_users = get_buffer_users(graph)
        graph_editor = GraphEditor(graph)
        drain_plans = self._validated_drain_plans
        op_by_name = {op.name: op for op in graph.operations} if drain_plans else {}

        for b in plan.nodes:
            address = plan.address(b)
            if address is None or b.name.startswith("__spyre_lx_relayout__:"):
                continue

            buf = graph.get_buffer(b.name)
            if b.name in inputs:
                # A loop-invariant input clone runs once, before the counted
                # loop its consumers run in, instead of on every trip.
                hoist_before = _hoisted_input_clone_entry(
                    graph, b.name, buffer_users[b.name]
                )
                new_buffer = graph_editor.push_allocation_with_clone(
                    buf,
                    buffer_users[b.name],
                    input=True,
                    lx_view=b.lx_view,
                    lower_before=hoist_before,
                )
                if hoist_before is not None:
                    _clear_loop_membership_metadata(new_buffer)
                self._set_one_allocation(new_buffer, address, b.lx_view)

            elif b.name in outputs:
                drain_plan = drain_plans.get(b.name)
                if drain_plan is not None:
                    # Post-loop materialization of a resident carry that is
                    # also the graph output.  The plan was validated before the
                    # solve and the solver committed this buffer to LX, so
                    # every fact it relies on must still hold here: a missing
                    # anchor or ownership is an internal error, never a late
                    # fallback that would return the pre-loop value.
                    _assert_drain_plan_committed(graph, b, plan, op_by_name, drain_plan)
                    new_buffer = graph_editor.push_allocation_with_clone(
                        buf,
                        [],
                        input=False,
                        private=True,
                        after_fx=drain_plan.loop_origin,
                        lower_anchor=drain_plan.anchor_op,
                    )
                    # Drain-only metadata hygiene: the clone copied the
                    # storage's attributes, but it is neither a loop member nor
                    # a carry.
                    _clear_loop_membership_metadata(new_buffer)
                else:
                    new_buffer = graph_editor.push_allocation_with_clone(
                        buf, buffer_users[b.name], input=False
                    )
                self._set_one_allocation(buf, address, b.lx_view)
                graph_editor.change_graph_output(buf, new_buffer)

            else:
                drain_plan = drain_plans.get(b.name)
                if drain_plan is not None and drain_plan.collective is not None:
                    # The collective needs an HBM operand: give it a post-loop
                    # copy of the resident carry, exactly like the graph-output
                    # drain above, and leave the carry itself in LX.
                    _assert_drain_plan_committed(graph, b, plan, op_by_name, drain_plan)
                    drained = graph_editor.push_allocation_with_clone(
                        buf,
                        [],
                        input=False,
                        private=True,
                        after_fx=drain_plan.loop_origin,
                        lower_anchor=drain_plan.anchor_op,
                    )
                    _clear_loop_membership_metadata(drained)
                    graph_editor.replace_collective_operand(
                        drain_plan.collective, b.name, drained
                    )
                self._set_one_allocation(buf, address, b.lx_view)

        # Keep graph mutation last and in pre-scheduling: solver retries require
        # the original graph, and post-grad no-op elimination has already run.
        materialize_lx_relayouts(graph, accepted_lx_relayouts)

    def _set_one_allocation(
        self,
        buf: TensorBox | ComputedBuffer,
        address: int,
        lx_view: PerCoreView | None,
    ) -> None:
        if lx_view is None:
            raise RuntimeError(
                f"LX placement for {buf.get_name()} has no accepted physical ownership"
            )
        layout = buf.get_layout()
        layout.allocation["lx"] = address
        layout.lx_view = lx_view

    # ---- the joint (open division mode) solve ----

    def _chosen_tilings(
        self, graph: GraphLowering, plan: SolvedPlan
    ) -> dict[str, TileSpec]:
        """The non-empty tiling the solve chose for each op, keyed by operation
        name (the key ``CoarseTilingPass``/``derive_tiling_groups`` consume)."""
        op_by_name = {op.name: op for op in graph.operations}
        choices: dict[str, TileSpec] = {}
        for node in plan.nodes:
            op = op_by_name.get(node.name)
            cd = plan.division(node)
            if op is None or cd is None:
                continue
            if not cd.tiling.is_untiled:
                choices[op.get_operation_name()] = cd.tiling
        return choices

    def _apply_chosen_tilings(
        self,
        graph: GraphLowering,
        plan: SolvedPlan,
    ) -> dict[str, TileSpec]:
        """Apply the solver's chosen coarse tilings, returning them by buffer name.

        The only consumer a chosen ``TileSpec`` has. Without it the search prices
        the *per-tile* footprint and the packer lays LX out by it while the graph
        writes the full extent, so the reserved interval is a fraction of the
        real one.

        **The anneal's placement stands.** Applying the tiling is what makes its
        addresses true rather than stale, so there is no second placement round:
        re-running a placement engine here would decouple the layout from the
        divisions and tilings it was jointly chosen with, which is the coupling
        this allocator exists for. The companion buffers the apply mints (a
        full-extent ``full_buf`` per op whose output escapes its tiling group)
        were not in the joint state, so they get no LX address and stay in HBM
        -- which is what the solver's ``_companion_bytes`` prices them as.

        **A refusal raises.** The search is supposed to propose only tilings
        ``coarse_tile`` accepts -- ``OpSplitSpace.admits`` is meant to be
        complete with respect to it -- so a refusal is a defect in that model
        rather than a graph to route around, and dropping the tiling here would
        also invalidate the addresses already spaced for it.
        """
        if not self.factory.applies_tilings:
            return {}
        op_by_name = {op.name: op for op in graph.operations}
        # By buffer name; ``choices`` by operation name, as ``CoarseTilingPass``
        # takes them.
        tiled: dict[str, TileSpec] = {}
        choices: dict[str, TileSpec] = {}
        for node in plan.nodes:
            cd = plan.division(node)
            if cd is None or cd.tiling.is_untiled:
                continue
            tiled[node.name] = cd.tiling
            choices[op_by_name[node.name].get_operation_name()] = cd.tiling
        if not choices:
            return {}
        # Local import: ``coarse_tiling`` imports ``ScratchpadOptimizationPass``
        # from this module, so a top-level import would be circular.
        from torch_spyre._inductor.scratchpad.coarse_tiling import CoarseTilingPass

        # A tiling comes only from a generated division, which only an
        # enumerated ``ComputedBuffer`` has (``TableCandidates._build_graph``).
        frames = {
            name: capture_iteration_frame(cast(ComputedBuffer, op_by_name[name]))
            for name in tiled
        }
        chosen = ", ".join(
            f"{name}={spec.label}" for name, spec in sorted(choices.items())
        )
        try:
            CoarseTilingPass(choices).apply_pass(graph)
        except Unsupported as exc:
            exc.add_note(f"auto_coarse_tiling chose: {chosen}")
            raise
        self._remap_tiled_symbols(graph, frames)
        # See ``JOINT_TILING_AND_DIVISION_ATTR``. Looked up afresh: the apply
        # replaces an op it redirects to a companion's full buffer
        # (``replace_computed_buffer_body``), so a pre-apply object is stale.
        op_by_name = {op.name: op for op in graph.operations}
        for name in tiled:
            setattr(op_by_name[name], JOINT_TILING_AND_DIVISION_ATTR, True)
        logger.info("applied a coarse tiling to %d op(s): %s", len(choices), chosen)
        return tiled

    def _remap_tiled_symbols(
        self, graph: GraphLowering, frames: dict[str, IterationFrame]
    ) -> None:
        """Record how the apply renumbered each tiled op's iteration symbols, and
        move the op's span floors through it.

        The solve keyed its splits by the pre-apply symbols. A tile that shrinks
        a dim to extent 1 drops that dim's symbol and renumbers every later one,
        so a pre-apply key read against the live op names the next axis. The
        floors ``span_reduction_pass`` committed are keyed the same way and are
        read against the live op by ``meets_span_floors``.
        """
        live = {op.name: op for op in graph.operations}
        self._tiling_symbol_remaps = {}
        for name, frame in frames.items():
            # ``replace_computed_buffer_body`` keeps it a ``ComputedBuffer``.
            op = cast(ComputedBuffer, live[name])
            self._tiling_symbol_remaps[name] = iteration_symbol_remap(op, frame)
            floors = getattr(op, "_work_division_span_min_splits", None)
            if floors:
                op._work_division_span_min_splits = self._live_splits(name, floors)

    def _live_splits(
        self, name: str, splits: dict[sympy.Symbol, int]
    ) -> dict[sympy.Symbol, int]:
        """``splits``, keyed by op ``name``'s pre-apply symbols, re-keyed to its
        live ones. Identity for an op the apply did not tile."""
        remap = self._tiling_symbol_remaps.get(name)
        if remap is None:
            return dict(splits)
        lost = sorted(
            str(sym)
            for sym, factor in splits.items()
            if sym not in remap and factor > 1
        )
        if lost:
            raise Unsupported(
                f"{name}: the coarse tiling left no axis for the split on "
                f"{', '.join(lost)}; it would commit as 1"
            )
        return {remap[sym]: factor for sym, factor in splits.items() if sym in remap}

    def _commit_copy_out_divisions(
        self,
        graph: GraphLowering,
        plan: SolvedPlan,
        tiled: Iterable[str],
    ) -> None:
        """Give each copy-out the apply minted its tiled op's core division.

        ``_insert_copy_op`` (wsr/coarse_tile.py) builds the copy-out over the
        tiled op's own ``data.ranges`` and reads the per-tile buffer, so it
        keeps no division of its own -- one core -- while the tiled op commits
        whatever the solve chose. ``get_ncores_for_buffers`` then reports the
        mismatch and withholds the tiled buffer's view, and a resident one
        raises in :meth:`_post_solve`.

        The copy's iteration symbols are the tiled op's output symbols in order
        (``iteration_space_from_op`` lists reductions after), so the splits map
        positionally and only a reduction split is dropped.
        """
        ops = {getattr(op, "name", None): op for op in graph.operations}
        for source in tiled:
            # Named as ``_insert_copy_op`` names it.
            copy = ops.get(graph.qualify_name(f"coarse_tile_copy_{source}"))
            cd = plan.division(plan.graph.nodes[source])
            if copy is None:
                # Expected when the op's output stays inside its tiling group.
                logger.debug("%s: no copy-out to give its core division", source)
                continue
            if cd is None:
                continue
            splits = self._live_splits(source, cd.splits)
            symbols = list(iteration_space_from_op(copy))
            source_symbols = list(iteration_space_from_op(graph.get_buffer(source)))
            mapped = {
                symbols[position]: factor
                for position, symbol in enumerate(source_symbols)
                if position < len(symbols) and (factor := splits.get(symbol, 1)) > 1
            }
            commit_iteration_space_ownership(copy, mapped)

    def _check_priced_footprints(
        self,
        graph: GraphLowering,
        plan: SolvedPlan,
        tiled: Mapping[str, TileSpec],
    ) -> None:
        """Every tiled buffer's applied per-core footprint fits the one it was
        placed at, or this raises.

        The search divides the buffer's *total* size by
        ``output_partition * output_tile_count``; the apply divides the op's
        ranges per dim and rebuilds the device layout through
        ``_resize_device_layout``. Those agree only if resizing divides the
        device byte size exactly -- plausible, since ``build_tiling_space`` never
        tiles the stick dim, but per-dim device padding could re-round. An
        applied footprint *larger* than the priced one would overlap whatever
        was packed above this buffer's LX interval; a smaller one only leaves
        part of the interval unused, so it is logged rather than refused.
        """
        for node in plan.nodes:
            tiling = tiled.get(node.name)
            if tiling is None:
                continue
            layout = graph.get_buffer(node.name).layout
            device_layout = getattr(layout, "device_layout", None)
            if device_layout is None:
                raise Unsupported(
                    f"{node.name}: tiled as {tiling.label} but carries no device "
                    "layout to check the applied footprint against"
                )
            # ``_apply_chosen_tilings`` skips a node with no chosen division,
            # so anything in ``tiled`` has one.
            cd = plan.division(node)
            assert cd is not None
            partition = cd.output_partition
            applied = ceil_div(get_device_size_in_bytes(device_layout), partition)
            priced = plan.per_core_size(node)
            if applied < priced:
                logger.debug(
                    "%s: applied per-core footprint %d bytes under tiling %s is "
                    "below the %d it was placed at",
                    node.name,
                    applied,
                    tiling.label,
                    priced,
                )
            elif applied > priced:
                raise Unsupported(
                    f"{node.name}: placed at a per-core footprint of {priced} "
                    f"bytes under tiling {tiling.label} ({node.nbytes} bytes over "
                    f"{partition} cores x {tiling.output_tile_count} tiles), but "
                    f"the applied graph gives {applied}; its LX address is spaced "
                    "for a footprint the graph does not have"
                )


def _lx_planning_size() -> int:
    """Return the frontend LX reservation, matching Deeptools exactly.

    The shared Torch/DXP contract partitions Deeptools' allocatable LX capacity,
    not the physical 2 MiB.  The frontend reserves
    ``1 - DXP_LX_FRAC_AVAIL`` from address zero, truncates the fractional byte
    count to an integer, and rounds that reservation up to the memory tracker's
    128-byte allocation granularity.  DXP marks that interval unavailable and
    allocates at or above the returned exclusive upper bound.  This is the
    ownership boundary whose mismatch was reported in torch-spyre issue #3222,
    not a safety margin.
    """
    backend_fraction = config.dxp_lx_frac_avail
    if not 0.0 <= backend_fraction <= 1.0:
        raise ValueError("DXP_LX_FRAC_AVAIL must be >=0 and <=1")

    frontend_reservation = int(_LX_TRACKER_CAPACITY_BYTES * (1.0 - backend_fraction))
    return round_up_to_alignment(frontend_reservation, _LX_ALLOCATION_GRANULARITY_BYTES)


def commit_divisions(
    graph: GraphLowering,
    plan: SolvedPlan,
    live_splits: Optional[
        Callable[[str, dict[sympy.Symbol, int]], dict[sympy.Symbol, int]]
    ] = None,
) -> None:
    """Commit the solver's chosen symbol-keyed division for every buffer.

    The solver optimizes a core division for all buffers, not just resident
    ones: a resident producer and its consumers are pinned by
    ``TableStrategy.encode_read`` to one shared
    slicing (so those commits are mutually consistent), while a spilled
    buffer is free of that gate -- its accesses round-trip through HBM,
    which re-slices on load -- so it takes its most parallel candidate.
    Committing the spilled buffers' divisions too lets the joint solve
    optimize work division across the whole graph, not only the LX-resident
    region.

    ``live_splits`` re-keys a buffer's chosen splits to its op's live iteration
    symbols, for an op an apply has renumbered since the solve.
    """
    op_by_name = {op.name: op for op in graph.operations}
    for node in plan.nodes:
        op = op_by_name.get(node.name)
        cd = plan.division(node)
        if op is None or cd is None:
            continue
        if not hasattr(op, "iteration_space_ownership"):
            # The guard (#4062) means "only refine a division the
            # work-division pass established", and its real subjects are the
            # fallback ops (SpyreConstantFallback / SpyreEmptyFallback):
            # they carry no iteration space to own, and the solver leaves
            # their splits empty, so both tests below skip them.
            #
            # An op ``CoarseTilingPass`` synthesises is a different case. It
            # is created *after* the work-division pass, so it was never
            # offered ownership -- not deliberately denied it -- yet the
            # joint solve still enumerates candidates for it, gates it
            # through ``cd_parent_matches`` against its producer, and picks
            # a division consistent with that producer's. Skipping it here
            # drops a decision the solve made: the copy stays undivided
            # while its producer commits divided, and ``_post_solve``'s
            # ownership check then rejects a pair the solver never made
            # inconsistent ("op 'bufN' ref PerCoreView(... num_cores=32) !=
            # 'coarse_tile_copy_bufN' PerCoreView((), (), num_cores=1)").
            # Mint ownership for it so the choice lands.
            if not isinstance(op, ComputedBuffer) or not cd.splits:
                continue
        splits = live_splits(node.name, cd.splits) if live_splits else cd.splits
        if not _split_option_is_legal(op, splits):
            raise Unsupported(
                f"{op.name}: chosen split violates hard domain "
                f"(division {cd.label}, tiling {cd.tiling.label}, ranges "
                f"{[str(r) for r in getattr(op.data, 'ranges', [])]})"
            )
        commit_iteration_space_ownership(op, splits)


def commit_lx_views(
    graph: GraphLowering,
    plan: SolvedPlan,
    accepted_lx_relayouts: Sequence[LXRelayoutPlan],
    drained_readers: Optional[Mapping[str, str]] = None,
) -> None:
    """Set ``lx_view``, the per-core view LX holds it under, on every resident
    node of ``plan``."""
    # A solver-fired relayout source stays resident under ITS committed view
    # while the consumer it feeds will read the shuffled copy under another.
    # The judge runs on the pre-materialization graph, where that consumer
    # still reads the source directly, so it reports the pair as a
    # mismatch and withholds a view. The plan carries the source view the
    # enumeration priced and the solver committed, so it is authoritative
    # here - the same precedence the fixed-division allocator gives
    # ``plan.source_view`` when it builds its buffers.
    source_views = {
        plan.source_name: plan.source_view for plan in accepted_lx_relayouts
    }
    # Likewise a drained carry's collective still reads the carry here; the
    # push repoints it at the drain, so it does not constrain the carry.
    _, reasons, views = get_ncores_for_buffers(graph, drained_readers=drained_readers)
    # A relayout copy is a transform, not a node: materialize_lx_relayouts
    # creates its destination, carrying the plan's view.
    for node in plan.nodes:
        if not node.resident:
            continue
        view = source_views.get(node.name) or views.get(node.name)
        if view is None:
            reason = reasons.get(node.name, "physical ownership was not accepted")
            raise Unsupported(f"{node.name}: {reason}")
        node.lx_view = view


def log_solver_decisions(graph: GraphLowering, plan: SolvedPlan) -> None:
    """Dump what the joint solve actually decided, per buffer.

    The solve's own output is otherwise invisible: the spill log reports
    residency but not the chosen division or tiling, and nothing reports
    whether that choice survived ``commit_divisions`` -- which silently
    skips any op lacking ``iteration_space_ownership``, i.e. every op
    synthesised after the work-division pass ran. Pairing this against the
    emitted ``OpSpec`` work slices is how a decided-but-discarded division
    shows up.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    op_by_name = {op.name: op for op in graph.operations}
    for buf in plan.nodes:
        cd = plan.division(buf)
        op = op_by_name.get(buf.name)
        info = getattr(op, "loop_info", None)
        group = getattr(info, "loop_group_id", None)
        propagation = getattr(info, "propagation", None)
        logger.debug(
            "solver_out: %s group=%s kind=%s loop=%s div=%s tiling=%s lx=%s "
            "size=%s committed=%s",
            buf.name,
            group if group is not None else "-",
            getattr(propagation, "kind", "-"),
            getattr(info, "loop_count", "-"),
            cd.label if cd is not None else "-",
            cd.tiling.label if cd is not None else "-",
            plan.address(buf),
            buf.nbytes,
            "yes"
            if getattr(op, "iteration_space_ownership", None) is not None
            else "NO(skipped)",
        )


def _graph_solver(size: int) -> GraphSolver:
    """The graph solver ``layout_solver`` names: CP-SAT posting the graph, or
    the annealing co-optimizer."""
    if config.layout_solver == "cpsat":
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import ModelSolver

        return ModelSolver(size)
    if config.layout_solver == "simulated_annealing":
        return BufferSolver(SaCoOptimizingSolver, size)
    raise ValueError(f"Invalid layout_solver config option '{config.layout_solver}'.")


def select_allocator() -> ScratchpadAllocator:
    """The scratchpad allocator for the configured solver: the one place that
    maps config to an (allocator, solver) pair, so the allocator never
    inspects config. With ``co_optimizing_lx_planning`` it chooses core
    divisions (and coarse tilings) jointly with placement; without, every op
    keeps the division work division committed.

    ``LxContextSwitchingPass`` replaces PR3683's blanket "never pin a buffer
    to LX across an extern kernel" guard with per-buffer dump/restore around
    the risky call; both are gated by the same flag, so the pass list is
    empty exactly when that guard is active."""
    # lx_context_switching imports ScratchpadOptimizationPass from this module.
    from torch_spyre._inductor.scratchpad.lx_context_switching import (
        LxContextSwitchingPass,
    )

    size = _lx_planning_size()
    return ScratchpadAllocator(
        _graph_solver(size),
        size=size,
        division_mode="open" if config.co_optimizing_lx_planning else "committed",
        post_optimization_passes=(
            [LxContextSwitchingPass()] if config.enable_lx_context_switching else []
        ),
    )


def scratchpad_planning(
    graph: GraphLowering, allocator: Optional[ScratchpadAllocator] = None
) -> None:
    """Assign LX scratchpad addresses to eligible buffers in a lowered graph.

    Called after stickification and core-division are complete. Graph operations
    are expected to be in topological order as guaranteed by GraphLowering.

    Args:
        graph: Lowered graph to plan scratchpad memory for.
        allocator: Allocator strategy to use. Defaults to the config-selected
            allocator (see :func:`select_allocator`).
    """
    if allocator is None:
        allocator = select_allocator()
    try:
        allocator.plan_allocation(graph)
        return
    except SolveError as error:
        # Strong exception guarantee: SolveError comes from the solve, before the
        # allocator commits divisions, relayouts or addresses, and select_allocator
        # configures no pre-passes, so the graph is unchanged.
        failed = error
    if allocator.joint:
        # Keep the failed allocator's post-allocation passes: they protect LX
        # buffers across FallbackKernel calls (see select_allocator).
        logger.info(
            "LX layout solve failed with layout_solver=%s (%s); replanning with "
            "the committed work divisions",
            config.layout_solver,
            failed,
        )
        size = _lx_planning_size()
        try:
            ScratchpadAllocator(
                _graph_solver(size),
                size=size,
                post_optimization_passes=allocator.post_optimization_passes,
            ).plan_allocation(graph)
            return
        except SolveError as error:
            failed = error
    logger.info("LX layout solve failed (%s); nothing is placed in LX", failed)
