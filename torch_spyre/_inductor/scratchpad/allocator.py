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
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Optional

import torch
from torch._inductor.ir import (
    TensorBox,
    ComputedBuffer,
    Operation,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    Reduction,
    ExternKernel,
)
from torch._inductor.graph import GraphLowering

from torch_spyre._inductor.pass_utils import (
    apply_splits_from_index_coeff,
    iteration_space_from_op,
    splits_by_index_coeff,
    op_read_writes,
    _prepare_per_core_view,
    _per_core_view_from_prep,
)
from torch_spyre._inductor.work_division import enumerate_work_division_candidates
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    CoOptimizingSolver,
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
    MemoryPlanSolver,
)
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    BestFitLayoutSolver,
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.passes import (
    ScratchpadOptimizationPass,
)
from torch_spyre._inductor.scratchpad.utils import (
    OP_OUTPUT_GOOD_FOR_LX_REUSE,
    clone_at_graph_boundaries,
    mem_usage_by_buf,
    calculate_liveness,
    get_ncores_for_buffers,
    get_buffer_users,
    buffer_not_read_in_full,
    GraphView,
    get_op_pointwise_inputs,
    _would_produce_lx_back_gap,
)
from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor

from torch_spyre._inductor import config
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.pass_utils import _is_matmul_op

logger = get_inductor_logger("scratchpad.allocator")


class ScratchpadAllocator(ABC):
    """
    Abstract class for all implementations of ScratchpadAllocator
    """

    def __init__(self) -> None:
        # Populated during plan_allocation: maps buffer/op name → reason string.
        # Stamped by _filter_ops, _build_bound_buffers, and plan_allocation
        # (for the solver decision). Reset at the start of each plan_allocation.
        self.reject_reasons: dict[str, str] = {}

    @abstractmethod
    def plan_allocation(self, graph: GraphLowering):
        """
        Accepts a graph to be considered for scratchpad memory according
        to its composition and the specific implementation used.

        Args:
            graph (GraphLowering): Graph to be considered for scratchpad planning
        """
        pass

    def _get_op_name(self, op: Any) -> str:
        target = getattr(getattr(op, "origin_node", None), "target", None)
        org_op_name = (
            getattr(target, "_opname", None)
            or getattr(target, "__name__", None)
            or getattr(target, "name", None)
            or str(target)
        )
        return org_op_name

    def _op_output_good_for_lx_reuse(self, op: Any) -> bool:
        return (
            isinstance(op, ComputedBuffer)
            and not isinstance(op.layout, MutationLayoutSHOULDREMOVE)
            and (
                config.allow_all_ops_in_lx_planning
                or (self._get_op_name(op) in OP_OUTPUT_GOOD_FOR_LX_REUSE)
                # Clones are only pinned when the boundary-clone path is on; they
                # are never in the whitelist, so without this they'd be ineligible
                # and the inserted clones would not land in LX.
                or (config.lx_boundary_clones and self._get_op_name(op) == "clone")
            )
        )

    def _op_inputs_good_for_lx_inplace(self, op: Any) -> list[str]:
        target = getattr(getattr(op, "origin_node", None), "target", None)
        if target is None:
            return []
        reads = [dep.name for dep in op.get_read_writes().reads]
        if torch.Tag.pointwise in target.tags:
            # If the op is tagged as pointwise by pytorch upstream
            # allow all inputs. Does not work for all ops
            return reads
        # Fallback/constant ops (e.g. SpyreConstantFallback) carry no Pointwise
        # ``data`` to inspect; they can't in-place-alias an input, so allow none.
        # The greedy path never reaches here (``_determine_in_place`` iterates
        # ``_filter_ops``-filtered ops), but the co-opt path inspects every op.
        data = getattr(op, "data", None)
        if data is None:
            return []
        return get_op_pointwise_inputs(data)

    def _filter_ops(
        self,
        graph: GraphLowering,
        cache: Optional[dict] = None,
    ) -> list[Operation]:
        core_div_mismatch = get_ncores_for_buffers(graph, cache)
        drop_list = set()

        # filter out by permitted operations
        for op in graph.operations:
            if not self._op_output_good_for_lx_reuse(op):
                drop_list.add(op.name)
                self.reject_reasons[op.name] = "op not allowed"

        # filter out core division mismatches
        for key, mismatch in core_div_mismatch.items():
            if mismatch == -1:
                drop_list.add(key)
                self.reject_reasons[key] = "core div mismatch"

        # filter out intermediates read partially (sliced / multi-offset): the
        # single-base LX path mis-addresses such reads (see
        # buffer_not_read_in_full / compute_ops._start_addr_data), e.g. an
        # inner-dim slice x[:, :, 32:96] feeding a chained op. _build_bound_buffers
        # applies the same guard to graph input/output clones; this covers the
        # intermediate buffers. Overrides allow_all_ops_in_lx_planning by design.
        # Only check ops still eligible above: ops already dropped include
        # non-ComputedBuffer outputs (e.g. multi-output) whose layouts have no
        # size for buffer_not_read_in_full to inspect.
        drop_list.update(
            op.name
            for op in graph.operations
            if op.name not in drop_list and buffer_not_read_in_full(graph, op.name)
        )

        if not clone_at_graph_boundaries():
            # Without clone support, graph outputs cannot be LX-pinned: the caller
            # holds an HBM reference and there is no clone to redirect it to.
            # graph_input_names is a no-op here (inputs are not in graph.operations),
            # but kept for symmetry with _build_bound_buffers, which handles inputs
            # separately when clone is available.
            drop_list.update(graph.get_output_names())
            drop_list.update(graph.graph_input_names)

        return [op for op in graph.operations if op.name not in drop_list]

    def _build_bound_buffers(
        self,
        graph: GraphLowering,
        in_place: Optional[dict[str, list[str]]],
        mem_usage: dict,
        lifetimes: dict[str, list[int]],
        cache: Optional[dict] = None,
    ) -> list[LifetimeBoundBuffer]:
        in_place = {} if in_place is None else in_place
        buffers: list[LifetimeBoundBuffer] = []
        graph_output_names = set(graph.get_output_names())
        cloning_allowed = clone_at_graph_boundaries()
        for output_name, info in mem_usage.items():
            uses = lifetimes[output_name]
            if len(uses) <= 1:
                self.reject_reasons[output_name] = "single use"
                continue  # output is not read (only the write, or never touched)
            if any(isinstance(graph.operations[u], ExternKernel) for u in uses):
                self.reject_reasons[output_name] = "extern kernel user"
                continue
            if output_name in graph_output_names and not cloning_allowed:
                self.reject_reasons[output_name] = "graph output (no clone)"
                continue  # we can only allocate graph outputs if we're allowed to clone
            if output_name in graph_output_names and buffer_not_read_in_full(
                graph, output_name
            ):
                # A pinned graph output is cloned for the HBM return; if a
                # consumer reads it partially (sliced / multi-offset), SDSC
                # mis-addresses the single-base LX buffer. Don't pin it.
                continue
            if _would_produce_lx_back_gap(graph, output_name, uses):
                self.reject_reasons[output_name] = "lx back gap"
                continue

            uses = lifetimes[output_name]
            parents = in_place.get(output_name, [])
            size = info["size_per_core"]
            buffers.append(
                LifetimeBoundBuffer(
                    output_name,
                    size,
                    uses,
                    first_use_is_read=False,
                    in_place_parents=parents,
                )
            )

        if cloning_allowed:
            ncores = get_ncores_for_buffers(graph, cache)
            for input_name in graph.graph_input_names:
                uses = lifetimes[input_name]
                if len(uses) <= 1:
                    # Input read only once, or not at all. A non-input that's read only once still
                    # saves a roundtrip to HBM if it is allocated in LX, but the input is already
                    # present in HBM and would need to be cloned to LX explicitly, which costs one
                    # transfer anyway.
                    continue
                if not GraphEditor.all_uses_are_rewritable(graph, uses):
                    continue
                if buffer_not_read_in_full(graph, input_name):
                    # A consumer reads this input partially -- a sliced/
                    # multi-offset read (e.g. x[:, 0:512] + x[:, 512:1024], or
                    # x[:, :, 0:64]). The clone would be pinned to LX, which
                    # SDSC addresses by a single base, so partial reads
                    # mis-address and produce wrong results.
                    continue
                num_cores = ncores.get(input_name, -1)
                if num_cores < 0:
                    continue  # core division mismatch across consumers
                if _would_produce_lx_back_gap(graph, input_name, uses):
                    self.reject_reasons[input_name] = "lx back gap"
                    continue
                buf = graph.get_buffer(input_name)
                dev_layout = buf.layout.device_layout
                dev_size = math.prod(dev_layout.device_size[:-1]) * 128
                buffers.append(
                    LifetimeBoundBuffer(
                        input_name,
                        dev_size // num_cores,
                        uses,
                        first_use_is_read=True,
                        in_place_parents=[],
                    )
                )

        return buffers

    def _determine_in_place(
        self,
        graph: GraphLowering,
        graph_view: "GraphView",
        mem_usage: dict,
        lifetimes: dict[str, list[int]],
    ) -> dict[str, list[str]]:
        allow_inplace: dict[str, list[str]] = {}
        in_place_allowed = {
            op.name: self._op_inputs_good_for_lx_inplace(op)
            for op in graph_view.operations
        }
        for buf_name, info in mem_usage.items():
            allow_inplace[buf_name] = []
            if not in_place_allowed[buf_name]:
                continue
            out_start = lifetimes[buf_name][0]
            out_ten_layout = graph.get_buffer(buf_name).layout.device_layout
            out_size = info["size_per_core"]
            for input_buf in info["op_inputs"]:
                in_end = lifetimes[input_buf][-1]  # inclusive last use
                in_ten_layout = graph.get_buffer(input_buf).layout.device_layout
                in_size = mem_usage[input_buf]["size_per_core"]
                inp_i_size_match = out_size == in_size
                inp_i_lay_match = out_ten_layout == in_ten_layout
                inp_i_eol = in_end == out_start  # same op reads input and writes output
                no_core_div_mismatch = not info["core_div_mismatch"]
                if (
                    input_buf in in_place_allowed[buf_name]
                    and inp_i_size_match
                    and inp_i_lay_match
                    and inp_i_eol
                    and no_core_div_mismatch
                ):
                    allow_inplace[buf_name].append(input_buf)
        return allow_inplace

    def _generate_buffers(
        self,
        graph: GraphLowering,
        cache: Optional[dict] = None,
        timings: Optional[dict[str, float]] = None,
        lifetimes: Optional[dict[str, list[int]]] = None,
    ) -> list[Operation]:
        # Build graph_view + mem_usage once and share; both helpers below treat
        # them read-only. `lifetimes` is split-invariant, so the co-opt search
        # passes it in (computed here only for the single-shot path).
        # get_read_writes() is memoized per op by `op_read_writes`, so the
        # per-leaf core-div check doesn't re-trace it across leaves.
        #
        # TODO: graph_view + mem_usage still rebuilt per leaf; only their
        #   split-dependent part is the (cached) core-div check, so the rest
        #   could be hoisted out of the per-leaf path too.
        t0 = time.perf_counter()
        graph_view = GraphView(graph, lambda g: self._filter_ops(g, cache))
        t1 = time.perf_counter()
        mem_usage = mem_usage_by_buf(graph_view, cache)
        t2 = time.perf_counter()
        if timings is not None:
            timings["graph_view"] += t1 - t0
            timings["mem_usage"] += t2 - t1

        if lifetimes is None:
            lifetimes = calculate_liveness(graph)

        in_place = self._determine_in_place(graph, graph_view, mem_usage, lifetimes)
        buffers = self._build_bound_buffers(
            graph, in_place, mem_usage, lifetimes, cache
        )
        return buffers

    def _log_plan_summary(
        self,
        label: str,
        allocation: Sequence[LifetimeBoundBuffer],
        footprint_of,
        hbm: Optional[int] = None,
        baseline: Optional[int] = None,
    ) -> None:
        """One concise INFO summary of the planning result, in the style of
        upstream Inductor's ``torch/_inductor/memory.py`` (resident count + peak vs
        limit; HBM traffic vs baseline when the solver reports it). Per-op /
        per-timestep detail stays at DEBUG (``_log_lx_pinning`` and the solvers)."""
        if not logger.isEnabledFor(logging.INFO):
            return
        resident = [b for b in allocation if b.address is not None]
        lx_peak = max((b.address + footprint_of(b) for b in resident), default=0)
        limit = getattr(self.layout_planning, "limit", _lx_planning_size())
        if hbm is not None:
            logger.info(
                "LX planning [%s]: %d/%d buffers resident, HBM traffic %d bytes "
                "(baseline %d), LX peak %d/%d bytes",
                label,
                len(resident),
                len(allocation),
                hbm,
                baseline,
                lx_peak,
                limit,
            )
        else:
            logger.info(
                "LX planning [%s]: %d/%d buffers resident, LX peak %d/%d bytes",
                label,
                len(resident),
                len(allocation),
                lx_peak,
                limit,
            )

    def _log_lx_pinning(self, graph: GraphLowering) -> None:
        """Log the final LX pinning decision for every op in the graph."""
        # Skip the per-op getattr walk unless DEBUG is on.
        if not logger.isEnabledFor(logging.DEBUG):
            return
        for op in graph.operations:
            reason = self.reject_reasons.get(op.name, "lx")
            logger.debug(
                "lx_pinning: %s (%s) → %s",
                op.name,
                self._get_op_name(op),
                reason,
            )

    def _push_allocation(
        self, graph: GraphLowering, buffers: Sequence[LifetimeBoundBuffer]
    ):
        """Push the allocation into the code generation. This includes cloning graph inputs and
        graph outputs:

        - A graph input B that is allocated into LX means that it is cloned; call the clone C. The
        downstream users of B are now made to use C. The LX allocation is effectuated by assigning
        it to C.

        - A graph output B that is allocated into LX means that it is cloned; call the clone C.
        Nothing changes for the downstream users. The LX allocation is effectuated by assigning it
        to B itself. The graph is made to have C as its output.

        - A buffer that is neither a graph input nor a graph output gets the LX allocation assigned
        to itself."""
        outputs = set(graph.get_output_names())
        inputs = set(graph.graph_input_names)

        buffer_users = get_buffer_users(graph)
        graph_editor = GraphEditor(graph)

        for b in buffers:
            if b.address is None:
                continue

            buf = graph.get_buffer(b.name)
            if b.name in inputs:
                new_buffer = graph_editor.push_allocation_with_clone(
                    buf, b.address, buffer_users[b.name], input=True
                )
                self._set_one_allocation(new_buffer, b.address)

            elif b.name in outputs:
                new_buffer = graph_editor.push_allocation_with_clone(
                    buf, b.address, buffer_users[b.name], input=False
                )
                self._set_one_allocation(buf, b.address)
                graph_editor.change_graph_output(buf, new_buffer)

            else:
                self._set_one_allocation(buf, b.address)

    def _set_one_allocation(self, buf: TensorBox | ComputedBuffer, address: int):
        layout = buf.get_layout()
        layout.allocation["lx"] = address


def _lx_planning_size() -> int:
    """LX scratchpad bytes available to the layout solver."""
    return int((2 << 20) * (1.0 - config.dxp_lx_frac_avail))


def _fixed_core_division(op: Operation) -> CoreDivision:
    """The op's upstream-committed division (``op.op_it_space_splits``) as a single
    pinned :class:`CoreDivision`; a never-divided op yields a one-core empty split.
    """
    seed: tuple[dict, dict] = getattr(op, "op_it_space_splits", None) or ({}, {})
    return CoreDivision(output_splits=dict(seed[0]), reduction_splits=dict(seed[1]))


class DefaultAllocator(ScratchpadAllocator):
    def __init__(
        self,
        layout_planning: MemoryPlanSolver | None = None,
        pre_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
        post_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
    ):
        """Configure the allocator with an optional solver and graph passes.

        Args:
            layout_planning: Solver that assigns LX addresses to lifetime-bound
                buffers. Defaults to GreedyLayoutSolver sized to available LX memory.
            pre_optimization_passes: Graph passes applied before layout planning.
                Defaults to no passes.
            post_optimization_passes: Graph passes applied after layout planning.
                Defaults to no passes.
        """
        # No config inspection here: the config -> (allocator, solver) mapping
        # lives in ``select_allocator``. A bare ``DefaultAllocator()`` defaults to
        # the greedy solver; any other solver is injected explicitly.
        if layout_planning is None:
            layout_planning = GreedyLayoutSolver(_lx_planning_size())
        if pre_optimization_passes is None:
            pre_optimization_passes = []
        if post_optimization_passes is None:
            post_optimization_passes = []

        super().__init__()
        self.pre_optimization_passes = pre_optimization_passes
        self.post_optimization_passes = post_optimization_passes
        self.layout_planning = layout_planning

    def plan_allocation(self, graph: GraphLowering):
        """Run pre-passes, assign LX addresses to eligible buffers, then run post-passes.

        Args:
            graph: Lowered graph whose buffers will be assigned LX scratchpad
                addresses where viable.
        """
        self.reject_reasons = {}
        for p in self.pre_optimization_passes:
            p.apply_pass(graph)
        # Placement-only: the gap solver consumes plain LifetimeBoundBuffers (whose
        # ``size`` is already per-core) and reads none of the co-optimization
        # metadata. Buffers ineligible for LX are already flagged non-placeable by
        # ``_generate_buffers``.
        buffers = self._generate_buffers(graph)
        allocation = self.layout_planning.plan_layout(buffers, log_lx_usage=True)
        for b in allocation:
            if b.address is None:
                self.reject_reasons[b.name] = "no room on scratchpad"
        self._push_allocation(graph, allocation)
        self._log_plan_summary(
            type(self.layout_planning).__name__, allocation, lambda b: b.size
        )
        self._log_lx_pinning(graph)
        for p in self.post_optimization_passes:
            p.apply_pass(graph)


class CoOptimizingAllocator(ScratchpadAllocator):
    def __init__(
        self,
        layout_planning: "CoOptimizingSolver",
        search_space: Optional[set[str]] = None,
        pre_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
        post_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
    ):
        """Joint core-division + LX-placement allocator.

        Args:
            layout_planning: a :class:`CoOptimizingSolver` (``DfsLayoutSolver`` or
                ``CpSatLayoutSolver``) that jointly chooses divisions + placement.
                Injected by :func:`select_allocator`, which also handles the
                cpsat-unavailable fallback (to the DFS solver), so this allocator
                never inspects config or builds a solver.
            search_space: enabled candidate-division modes (subset of
                ``{"complete", "axis_swaps", "matmul_only"}``); defaults to
                ``{"complete"}``.
            pre_optimization_passes / post_optimization_passes: graph passes run
                before / after layout planning (default none).
        """
        super().__init__()
        self.layout_planning = layout_planning
        self.search_space: set[str] = search_space or {"complete"}
        self.pre_optimization_passes = pre_optimization_passes or []
        self.post_optimization_passes = post_optimization_passes or []

    def plan_allocation(self, graph: GraphLowering):
        """Run pre-passes, jointly solve core-division + LX placement, commit the
        chosen divisions, then run post-passes."""
        self.reject_reasons = {}
        for p in self.pre_optimization_passes:
            p.apply_pass(graph)
        buffers = self._generate_cd_buffers(graph, self._division_map(graph))
        allocation = self.layout_planning.plan_layout_and_core_divs(buffers)
        # Commit divisions before pushing: clone materialization in
        # ``_push_allocation`` (``push_allocation_with_clone``) re-keys the clone
        # from the consumer's ``op_it_space_splits``, which ``_commit_divisions``
        # writes from the solver's chosen divisions.
        self._commit_divisions(graph, allocation)
        self._push_allocation(graph, allocation)
        for p in self.post_optimization_passes:
            p.apply_pass(graph)

        # Surface the solver's per-buffer spill causes so the LX-pinning debug
        # log reports why each buffer landed in HBM, on par with the other
        # allocators.
        self.reject_reasons = dict(getattr(self.layout_planning, "spill_reasons", {}))
        self._log_plan_summary(
            type(self.layout_planning).__name__,
            allocation,
            self._resident_footprint,
            *self._hbm_traffic(allocation),
        )
        self._log_lx_pinning(graph)

    @staticmethod
    def _resident_footprint(b: CoreDivisionBuffer) -> int:
        """Per-core LX footprint of ``b`` under its chosen division (``size`` is the
        total device footprint on a ``CoreDivisionBuffer``)."""
        if b.chosen_division is None:
            return b.size
        part = b.core_divisions[b.chosen_division].output_partition
        return math.ceil(b.size / part)

    def _hbm_traffic(self, allocation: Sequence[CoreDivisionBuffer]) -> tuple[int, int]:
        """``(achieved, baseline)`` HBM traffic for the plan summary: baseline is
        the all-spilled cost, achieved charges resident buffers their
        ``boundary_cost`` and spilled buffers their ``_spill_cost`` (the same
        objective the solver minimized)."""
        n_children: dict[str, int] = {}
        for b in allocation:
            for p in b.parents:
                n_children[p] = n_children.get(p, 0) + 1
        spill = self.layout_planning._spill_cost
        baseline = sum(spill(b, n_children.get(b.name, 0)) for b in allocation)
        achieved = sum(
            b.boundary_cost
            if b.address is not None
            else spill(b, n_children.get(b.name, 0))
            for b in allocation
        )
        return achieved, baseline

    def _division_map(self, graph: GraphLowering) -> dict[str, list[CoreDivision]]:
        """Per-op core-division candidates for the joint-division solve.

        Every op gets at least one ``CoreDivision`` so the slicing-match gate can
        constrain it. Pointwise / Reduction ops get the enumerated candidates
        selected by ``self.search_space``; every other op falls back to a single
        fixed division read off its committed ``op_it_space_splits``. No op-kind
        pre-filter -- residency is gated per buffer (``residency_allowed``) and by
        the solver, so ineligible ops still participate in the match.
        """
        max_cores = config.sencores
        return {
            op.name: self._enumerate_core_divisions(op, max_cores)
            for op in graph.operations
        }

    def _fixed_division(self, op: Operation) -> CoreDivision:
        """The op's upstream-committed division (``op.op_it_space_splits``) as a
        single pinned CoreDivision; a never-divided op yields a one-core empty
        split. Used as the seed (always kept) and the fallback for ops with no
        enumerable candidates, so every buffer carries at least one division.
        """
        return _fixed_core_division(op)

    def _enumerate_core_divisions(
        self, op: Operation, max_cores: int
    ) -> list[CoreDivision]:
        """Core-division candidates for one op: the complete work-division set
        (``enumerate_work_division_candidates``) with a ``self.search_space``
        filter applied. The committed split (seed) is always kept at index 0.

        The search-space modes only *select* from the one generic enumeration; no
        candidate is hand-constructed:

        * ``complete`` -- keep every enumerated candidate.
        * ``axis_swaps`` -- keep single-output-dim splits (one output axis split,
          no reduction split).
        * ``matmul_only`` -- keep every candidate for matmul ops; other ops keep
          only the seed.
        """
        fixed = self._fixed_division(op)
        if not isinstance(op, ComputedBuffer) or not isinstance(
            op.data, (Pointwise, Reduction)
        ):
            return [fixed]
        rw = op_read_writes(op)
        write = next(iter(rw.writes), None)

        # this is essentially a dead branch but serves as a type narrowing below
        if write is None:
            return [fixed]
        write_index = write.index
        first_read = next(iter(rw.reads), None)
        read_index = first_read.index if first_read is not None else write_index

        try:
            candidates = enumerate_work_division_candidates(op, max_cores)
        except Unsupported as exc:
            # Symbolic stick dims etc. can't be enumerated; leave the op on its
            # upstream-chosen split (fixed division).
            logger.debug("skip joint division for %s: %s", op.name, exc)
            return [fixed]

        # Encode + dedup by slicing signature; the seed is index 0 and always kept.
        all_cds: list[CoreDivision] = [fixed]
        seen: set[tuple] = {self._division_key(fixed)}
        for cand in candidates:
            out_s, red_s = splits_by_index_coeff(cand, write_index, read_index)
            cd = CoreDivision(output_splits=out_s, reduction_splits=red_s)
            key = self._division_key(cd)
            if key in seen:
                continue
            seen.add(key)
            all_cds.append(cd)

        modes = self.search_space
        is_mm = _is_matmul_op(op)
        kept: list[CoreDivision] = [all_cds[0]]  # seed always kept
        for cd in all_cds[1:]:
            keep = (
                ("complete" in modes)
                or ("axis_swaps" in modes and self._is_single_axis_split(cd))
                or ("matmul_only" in modes and is_mm)
            )
            if keep:
                kept.append(cd)
        return kept

    @staticmethod
    def _division_key(cd: CoreDivision) -> tuple:
        return (
            tuple(sorted(cd.output_splits.items())),
            tuple(sorted(cd.reduction_splits.items())),
        )

    @staticmethod
    def _is_single_axis_split(cd: CoreDivision) -> bool:
        """True iff ``cd`` splits exactly one output axis and no reduction axis --
        the ``axis_swaps`` subset selected from the complete enumeration."""
        return (
            not cd.reduction_splits
            and sum(1 for f in cd.output_splits.values() if f > 1) == 1
        )

    def _commit_divisions(
        self,
        graph: GraphLowering,
        allocation: Sequence[CoreDivisionBuffer],
    ) -> None:
        """Write the solver's chosen division back to ``op.op_it_space_splits``
        for *every* buffer the solver assigned one.

        The solver optimizes a core division for all buffers, not just resident
        ones: a resident producer and its consumers are pinned by
        ``_implicate_core_division`` to one shared slicing (so those commits are
        mutually consistent), while a spilled buffer is free of that gate -- its
        accesses round-trip through HBM, which re-slices on load -- so it takes
        its most parallel candidate. Committing the spilled buffers' divisions
        too lets the joint solve optimize work division across the whole graph,
        not only the LX-resident region.
        """
        op_by_name = {op.name: op for op in graph.operations}
        for buf in allocation:
            op = op_by_name.get(buf.name)
            if op is None or buf.chosen_division is None:
                continue
            cd = buf.core_divisions[buf.chosen_division]
            op.op_it_space_splits = (
                dict(cd.output_splits),
                dict(cd.reduction_splits),
            )

    def _generate_cd_buffers(
        self,
        graph: GraphLowering,
        divisions: dict[str, list[CoreDivision]],
    ) -> list[CoreDivisionBuffer]:
        in_place = self._determine_in_place_division_invariant(graph)
        buffers = self._build_cd_bound_buffers(graph, in_place, divisions)
        return buffers

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
            op.name: self._op_inputs_good_for_lx_inplace(op) for op in graph.operations
        }
        lifetimes = calculate_liveness(graph)
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
            out_start = lifetimes[buf_name][0]
            out_ten_layout = out_layout.device_layout
            for input_buf in info["op_inputs"]:
                in_layout = graph.get_buffer(input_buf).layout
                if not hasattr(in_layout, "device_layout"):
                    continue
                in_end = lifetimes[input_buf][-1]  # inclusive last use
                in_ten_layout = in_layout.device_layout
                inp_i_lay_match = out_ten_layout == in_ten_layout
                inp_i_eol = in_end == out_start  # same op reads input, writes output
                if inp_i_lay_match and inp_i_eol:
                    allow_inplace[buf_name].append(input_buf)
        return allow_inplace

    def _residency_by_buf(
        self,
        graph: GraphLowering,
        mem_usage: dict,
        op_by_name: dict[str, Operation],
        lifetimes: dict[str, list[int]],
    ) -> dict[str, Optional[str]]:
        """Per-buffer residency verdict: ``None`` if the buffer may be pinned
        (resident) in LX, else the reason it may not.

        Every buffer is handed to the solver so it participates in the slicing
        match, but participation is not residency. A buffer may be *pinned* only
        if its producing op clears ``_op_output_good_for_lx_reuse``, has no
        ExternKernel consumer (extern ops read from HBM), is not the target of an
        in-place mutation, is off a graph boundary, is read in full (offset reads
        mis-address a single LX base), would not produce a backGapCore_ (the
        backend supports backGap for HBM but not LX), and is actually read.
        Otherwise it stays non-resident (carrying the reason) so it doesn't
        orphan its neighbours. The reason strings mirror the ``DefaultAllocator``
        ``reject_reasons`` vocabulary where the checks overlap.

        Note: core-division consistency is *not* pre-filtered here (unlike the
        gap allocators' "core div mismatch" drop); the joint solver enforces it
        via the ``cd_parent_matches`` slicing gate instead.
        """
        # Targets of a ``MutationLayoutSHOULDREMOVE`` op (e.g. a ``cat`` dest
        # filled by per-input ``copy_`` slices): the producing op reads nothing
        # -- its data arrives via offset writes -- so pinning it to one LX base
        # mis-addresses. The mutating ops are rejected by
        # ``_op_output_good_for_lx_reuse``, but their target is a normal layout
        # that would otherwise pass, so exclude it explicitly. Computed once so
        # the predicate stays linear in the graph.
        mutated_buffers = {
            op.layout.target.get_name()
            for op in graph.operations
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE)
        }
        graph_output_names = set(graph.get_output_names())
        return {
            name: self._residency_reason(
                graph,
                op_by_name.get(name),
                name,
                lifetimes[name],
                mutated_buffers,
                graph_output_names,
            )
            for name in mem_usage
        }

    def _residency_reason(
        self,
        graph: GraphLowering,
        op: Optional[Operation],
        name: str,
        uses: list[int],
        mutated_buffers: set[str],
        graph_output_names: set[str],
    ) -> Optional[str]:
        """The first check ``name`` fails (the reason it may not reside), or
        ``None`` if it clears them all. Order matters: the back-gap probe (last)
        touches ``device_layout``, so the earlier guards ensure it only runs on a
        non-mutation ``ComputedBuffer`` that is read in full."""
        if op is None or not self._op_output_good_for_lx_reuse(op):
            return "op not allowed"
        if any(isinstance(graph.operations[u], ExternKernel) for u in uses):
            return "extern kernel user"
        if name in mutated_buffers:
            return "mutation target"
        if name in graph_output_names and not clone_at_graph_boundaries():
            # The caller holds the HBM reference; without a clone to redirect the
            # return through, a pinned graph output would never be written to HBM.
            return "graph output (no clone)"
        if buffer_not_read_in_full(graph, name):
            # A pinned graph output is cloned for the HBM return (see
            # ``_push_allocation``); a partial/offset read of a single LX base
            # (input or output) mis-addresses under SDSC, so never pin it.
            return "partial/offset read"
        if len(uses) <= 1:
            return "single use"
        if _would_produce_lx_back_gap(graph, name, uses):
            return "lx back gap"
        return None

    def _build_cd_bound_buffers(
        self,
        graph: GraphLowering,
        in_place: Optional[dict[str, list[str]]],
        divisions: dict[str, list[CoreDivision]],
    ) -> list[CoreDivisionBuffer]:
        """Build the ``CoreDivisionBuffer``s handed to the solver.

        Every buffer carries its candidate ``divisions`` and is sized by its
        *total* device footprint plus its producer edges (``parent_proj``); the
        solver picks a division and divides by its ``output_partition``. Because
        all buffers are on the same total scale, ``in_place_parents`` need no
        filtering."""
        lifetimes = calculate_liveness(graph)
        mem_usage = mem_usage_by_buf(graph)
        in_place = {} if in_place is None else in_place
        op_by_name = {op.name: op for op in graph.operations}
        graph_output_names = set(graph.get_output_names())

        # Caches the candidate-invariant view prep (``_prepare_per_core_view``)
        # keyed by (op, dep, buf), so a parent read by several consumers prepares
        # its write-view once and each op's sympy work is reused across divisions.
        prep_cache: dict = {}
        buffers: list[CoreDivisionBuffer] = []
        # Residency for every buffer up front: ``_cd_parent_matches`` consults the
        # same map so it never matches against a never-resident parent. Computed
        # before the loop because a parent can appear later than its consumer.
        residency_by_buf = self._residency_by_buf(
            graph, mem_usage, op_by_name, lifetimes
        )

        # loop through the used buffers and determine the valid core division
        # of all the child nodes. Matches is the make between eqivalent per core
        # views from the valid division for the clone op and relative to the
        # child operations.
        input_clone_divs: dict[str, list[CoreDivision]] = {}
        input_clone_matches: dict[str, dict[str, list[tuple[int, int]]]] = {}
        if clone_at_graph_boundaries():
            buffer_users = get_buffer_users(graph)
            for input_name in self._eligible_clone_inputs(graph, lifetimes):
                consumers = [
                    op
                    for op in buffer_users.get(input_name, [])
                    if input_name in {d.name for d in op_read_writes(op).reads}
                ]
                divs, matches = self._clone_divisions_and_matches(
                    input_name, consumers, divisions, prep_cache
                )
                input_clone_divs[input_name] = divs
                input_clone_matches[input_name] = matches
                # The raw input is not a resident *producer* (it has no op to take
                # a write-view from); its clone edges are added to consumers below.
                # A non-None reason keeps ``_cd_parent_matches`` from matching the
                # raw input as a producer, mirroring the default for graph inputs.
                residency_by_buf[input_name] = "input clone"

        for output_name, info in mem_usage.items():
            uses = lifetimes[output_name]

            op = op_by_name.get(output_name)
            residency_reason = residency_by_buf[output_name]

            buf_divisions = divisions[output_name]
            parents = in_place.get(output_name, [])
            size = info["size"]  # total footprint; solver divides per chosen cd
            parent_proj = list(info["op_inputs"])
            cd_parent_matches = self._cd_parent_matches(
                op,
                buf_divisions,
                parent_proj,
                divisions,
                op_by_name,
                prep_cache,
                residency_by_buf,
            )

            # Add any graph-input clone this op reads as an extra producer edge,
            # with the compatible-by-construction pairs computed up front. Done
            # here (not in ``_cd_parent_matches``) because the input has no op to
            # take a write-view from.
            for input_name, per_consumer in input_clone_matches.items():
                if output_name in per_consumer:
                    parent_proj.append(input_name)
                    cd_parent_matches[input_name] = per_consumer[output_name]

            # Every main-loop buffer is produced by an in-graph op, so spilling it
            # costs that producer's HBM write (saved when resident -> written to
            # LX). A graph output additionally still writes HBM once when resident
            # (the clone returns it), so it carries that as boundary_cost; the two
            # net to "pinning saves only the internal re-reads", which is correct.
            boundary_cost = size if output_name in graph_output_names else 0
            buffers.append(
                CoreDivisionBuffer(
                    output_name,
                    size,
                    uses,
                    first_use_is_read=True,
                    in_place_parents=parents,
                    core_divisions=buf_divisions,
                    parents=parent_proj,
                    cd_parent_matches=cd_parent_matches,
                    residency_reason=residency_reason,
                    boundary_cost=boundary_cost,
                    spill_write_cost=size,
                )
            )

        buffers.extend(self._cd_input_buffers(graph, lifetimes, input_clone_divs))
        return buffers

    def _eligible_clone_inputs(
        self, graph: GraphLowering, lifetimes: dict[str, list[int]]
    ) -> list[str]:
        """Graph inputs eligible to be cloned into LX, applying the same
        correctness guards as the placement path's input loop
        (``_build_bound_buffers``): the input must be read more than once, every
        consumer must be a rewritable (Pointwise/Reduction) op, and it must be
        read in full (a single LX base can't address a partial/multi-offset read).

        Unlike the placement path we do NOT gate on ``get_ncores_for_buffers``
        agreement: the joint solver re-divides consumers and the slicing-match
        gate makes them converge on one shared slicing, or leaves the input in
        HBM. A committed-division mismatch is therefore not a blocker here.
        """
        eligible: list[str] = []
        for input_name in graph.graph_input_names:
            uses = lifetimes[input_name]
            if len(uses) <= 1:
                continue
            if not GraphEditor.all_uses_are_rewritable(graph, uses):
                continue
            if buffer_not_read_in_full(graph, input_name):
                continue
            eligible.append(input_name)
        return eligible

    def _clone_divisions_and_matches(
        self,
        input_name: str,
        consumers: list[Operation],
        divisions: dict[str, list[CoreDivision]],
        prep_cache: dict,
    ) -> tuple[list[CoreDivision], dict[str, list[tuple[int, int]]]]:
        """Candidate divisions for an input clone, plus per-consumer
        ``(clone_div_idx, consumer_div_idx)`` match pairs.

        The clone is an identity copy of ``input_name``, so for any slicing it
        writes its per-core view of ``input_name`` equals a consumer's read-view
        of that same slicing. The divisions worth offering are exactly the
        distinct representable read-views the consumers exhibit; each is realized
        as a ``CoreDivision`` by re-keying the consumer's split onto the buffer's
        own read index -- the identical re-keying ``push_allocation_with_clone``
        applies when it later materializes the clone op. A ``(clone, consumer)``
        pair is compatible iff their views match AND their total core counts agree
        (the same broadcast-axis guard as ``_cd_parent_matches``: equal view but
        unequal core count means the consumer splits an axis the clone, sliced
        only on ``input_name``'s own axes, cannot replicate).
        """
        clone_divs: list[CoreDivision] = []
        clone_views: list[tuple] = []  # parallel: the view each clone div reproduces
        matches: dict[str, list[tuple[int, int]]] = {}
        for consumer in consumers:
            cname = consumer.get_name()
            consumer_divs = divisions[cname]
            rw = op_read_writes(consumer)
            read_dep = next(
                (r for r in rw.reads if r.name == input_name and hasattr(r, "index")),
                None,
            )
            write = next((w for w in rw.writes if hasattr(w, "index")), None)
            if read_dep is None or write is None:
                continue
            iter_space = iteration_space_from_op(consumer)
            views = self._views_for_divs(
                consumer, read_dep, input_name, consumer_divs, prep_cache
            )
            pairs: list[tuple[int, int]] = []
            for j, (view, _, repr_ok) in enumerate(views):
                if not repr_ok:
                    continue
                k = next((idx for idx, v in enumerate(clone_views) if v == view), None)
                if k is None:
                    cd = consumer_divs[j]
                    per_sym = apply_splits_from_index_coeff(
                        (cd.output_splits, cd.reduction_splits),
                        write.index,
                        read_dep.index,
                        iter_space,
                    )
                    clone_out, _ = splits_by_index_coeff(
                        per_sym, read_dep.index, read_dep.index
                    )
                    k = len(clone_divs)
                    clone_divs.append(
                        CoreDivision(
                            output_splits=clone_out, reduction_splits={}
                        )  # a clone op cannot have a division split
                    )
                    clone_views.append(view)
                if clone_divs[k].cores_used == consumer_divs[j].cores_used:
                    pairs.append((k, j))
            if pairs:
                matches[cname] = pairs
        # Every buffer must carry >=1 division (``_assert_core_divisions_enumerated``);
        # a whole-buffer fallback also lets a whole-read consumer match.
        if not clone_divs:
            clone_divs.append(CoreDivision())
        return clone_divs, matches

    def _cd_input_buffers(
        self,
        graph: GraphLowering,
        lifetimes: dict[str, list[int]],
        input_clone_divs: dict[str, list[CoreDivision]],
    ) -> list[CoreDivisionBuffer]:
        """Build the ``CoreDivisionBuffer`` for each clone-eligible graph input.

        Sized by the input's *total* device footprint (the solver divides by the
        chosen division's ``output_partition``, matching the rest of the CD path).
        ``residency_reason=None`` (residency allowed) and ``boundary_cost=size``
        because a resident input clone still reads HBM once. It has no LX parent
        (``parents=[]``); its
        consumers carry the match pairs that gate its residency.
        """
        out: list[CoreDivisionBuffer] = []
        for input_name, divs in input_clone_divs.items():
            dev_layout = graph.get_buffer(input_name).layout.device_layout
            size = math.prod(dev_layout.device_size[:-1]) * 128
            out.append(
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
                    boundary_cost=size,
                )
            )
        return out

    def _cd_parent_matches(
        self,
        consumer_op: Optional[Operation],
        consumer_divs: list[CoreDivision],
        parent_names: list[str],
        divisions: dict[str, list[CoreDivision]],
        op_by_name: dict[str, Operation],
        prep_cache: dict,
        residency_by_buf: dict[str, Optional[str]],
    ) -> dict[str, list[tuple[int, int]]]:
        """Physical slicing-match pairs for each divided producer this op reads.

        For producer ``P`` feeding this consumer, a ``(P_div_idx,
        consumer_div_idx)`` pair is compatible iff the two divisions induce the
        *same per-core slicing of ``P``* (``P``'s write-view equals the
        consumer's read-view, both via ``_per_core_view_on_buf`` in ``P``'s
        device-dim frame) AND use the *same total core count*. This is the
        per-core-view comparison ``get_ncores_for_buffers`` uses -- correct across
        reductions/reshapes, where a coeff-keyed signature would conflate axes.

        Excluded from matching (producer then falls back to HBM, always correct):
        a producer that can never be resident (``residency_by_buf`` reason is not
        ``None``); a producer candidate whose write carries a partial reduction
        (output not final); and either side's candidate whose slicing of ``P`` is
        unrepresentable -- we never pin on a slicing we cannot verify.
        """
        if consumer_op is None:
            return {}
        matches: dict[str, list[tuple[int, int]]] = {}
        consumer_reads = op_read_writes(consumer_op).reads
        for parent in parent_names:
            # A never-resident producer always reads from HBM, so its division
            # can't constrain the consumer -- skip the match (and the write-index
            # lookup below, undefined for StarDep writers). A missing entry
            # defaults to non-resident (sentinel reason), never None.
            if residency_by_buf.get(parent, "not in graph") is not None:
                continue
            parent_divs = divisions[parent]
            parent_op = op_by_name[parent]
            write_dep = next(
                (
                    w
                    for w in op_read_writes(parent_op).writes
                    if w.name == parent and hasattr(w, "index")
                ),
                None,
            )
            read_dep = next(
                (r for r in consumer_reads if r.name == parent and hasattr(r, "index")),
                None,
            )
            if write_dep is None or read_dep is None:
                continue

            # Producer view per candidate on its own output ``parent``. ``None``
            # marks a candidate that cannot host a readable residency: a
            # partial-reduction write, or an unrepresentable slicing of ``parent``.
            prod_views: list[Optional[tuple]] = [
                view if (repr_ok and not partial) else None
                for view, partial, repr_ok in self._views_for_divs(
                    parent_op, write_dep, parent, parent_divs, prep_cache
                )
            ]
            # Consumer read-views: same unrepresentable guard. A clean empty view
            # (the split doesn't slice ``parent`` -> reads it whole) is
            # representable and legitimately matches a whole-buffer producer.
            cons_views: list[Optional[tuple]] = [
                view if repr_ok else None
                for view, _partial, repr_ok in self._views_for_divs(
                    consumer_op, read_dep, parent, consumer_divs, prep_cache
                )
            ]

            # A matched pair needs equal per-core slicing of ``parent`` AND equal
            # *total* core count. Equal views alone aren't enough: a producer on N
            # and consumer on M>N cores can share an identical (possibly empty)
            # slicing while the consumer's extra cores -- split on a broadcast axis
            # -- hold no copy and would read a stale/partial LX buffer. The joint
            # solver re-divides per buffer and can hit this, hence the gate; a
            # rejected pair just falls back to HBM.
            pairs = [
                (i, j)
                for i, pv in enumerate(prod_views)
                if pv is not None
                for j, cv in enumerate(cons_views)
                if cv is not None
                and pv == cv
                and parent_divs[i].cores_used == consumer_divs[j].cores_used
            ]
            matches[parent] = pairs
        return matches

    @staticmethod
    def _views_for_divs(op, dep, buf_name, divs, prep_cache: dict):
        """Per-core views of ``buf_name`` for each candidate division of ``op``.

        Prepares the candidate-invariant context once (``_prepare_per_core_view``
        -- the sympy-heavy op-level work) and evaluates every candidate from it
        via ``_per_core_view_from_prep``, so cost scales with the op rather than
        its candidate count.

        ``prep_cache`` is keyed by ``(op name, dep, buf_name)``: a producer's
        write-dep and a consumer's read-dep on the same buffer can be equal
        ``MemoryDep``s, so the op name keeps their preps distinct while a parent
        read by several consumers reuses its write-view prep.
        """
        key = (op.get_name(), dep, buf_name)
        out = []
        for cd in divs:
            coeff = (cd.output_splits, cd.reduction_splits)
            # Build the prep only when a candidate actually has a split:
            # ``_per_core_view_from_prep`` returns the whole-buffer view for a
            # no-split candidate before touching the prep, so a never-divided op
            # (e.g. a StarDep write with no ``.index``) is never prepared.
            if any(n > 1 for d in coeff for n in d.values()) and key not in prep_cache:
                prep_cache[key] = _prepare_per_core_view(op, dep, buf_name)
            out.append(_per_core_view_from_prep(prep_cache.get(key), coeff))
        return out


_PLACEMENT_SOLVERS: dict[str, type[MemoryPlanSolver]] = {
    "greedy": GreedyLayoutSolver,
    "bestfit": BestFitLayoutSolver,
    "firstfit": FirstFitLayoutSolver,
}


def _make_cpsat_solver(size: int) -> Optional["CoOptimizingSolver"]:
    """Build the CP-SAT layout solver, or ``None`` when ortools is unavailable.

    Imported lazily so this module (and every non-cpsat path) loads without
    ortools installed; ``CpSatLayoutSolver.__init__`` raises ``ImportError`` when
    ortools (``cp_model``) is missing, which we translate to ``None`` so the
    caller can fall back to the pure-Python DFS solver.
    """
    try:
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            CpSatLayoutSolver,
        )

        return CpSatLayoutSolver(size)
    except ImportError as exc:
        logger.warning(
            "cpsat layout solver unavailable (%s); falling back to the DFS "
            "co-optimizing solver.",
            exc,
        )
        return None


def _make_dfs_solver(size: int, inner: MemoryPlanSolver) -> "CoOptimizingSolver":
    """Build the pure-Python DFS co-optimizing solver with the given inner
    placement solver. Imported lazily to keep this module import-light."""
    from torch_spyre._inductor.scratchpad.dfs_solver import DfsLayoutSolver

    return DfsLayoutSolver(size, inner_solver=inner)


def _resolve_search_space(solver_name: str) -> set[str]:
    """Candidate-division modes for a co-optimizing solver. An explicit
    ``config.co_opt_search_space`` (comma list) wins; otherwise auto by solver:
    cpsat -> ``{complete}``, dfs -> ``{axis_swaps, matmul_only}``."""
    raw = config.co_opt_search_space.strip()
    if raw:
        return {m.strip() for m in raw.split(",") if m.strip()}
    return {"complete"} if solver_name == "cpsat" else {"axis_swaps", "matmul_only"}


def select_allocator() -> ScratchpadAllocator:
    """Build the scratchpad allocator and inject its layout solver from config.

    This is the single place that maps config to an (allocator, solver) pair, so
    the allocators themselves take an explicit solver and never inspect config.
    The solver name is also the co-optimization switch:

    * ``layout_solver == "cpsat"`` -> joint core-division + LX placement via
      :class:`CoOptimizingAllocator` on :class:`CpSatLayoutSolver`; when ortools
      is absent it falls back to the pure-Python DFS solver (still co-optimizing).
    * ``layout_solver == "dfs"`` -> :class:`CoOptimizingAllocator` on
      :class:`DfsLayoutSolver`, whose inner placement solver is ``config.dfs_solver``.
    * ``layout_solver in {greedy, bestfit, firstfit}`` -> placement-only
      :class:`DefaultAllocator` with that gap-based solver.
    """
    size = _lx_planning_size()
    solver_name = config.layout_solver

    if solver_name == "cpsat":
        solver = _make_cpsat_solver(size)
        if solver is None:
            solver = _make_dfs_solver(size, _PLACEMENT_SOLVERS[config.dfs_solver](size))
        return CoOptimizingAllocator(
            layout_planning=solver, search_space=_resolve_search_space("cpsat")
        )

    if solver_name == "dfs":
        inner = _PLACEMENT_SOLVERS[config.dfs_solver](size)
        return CoOptimizingAllocator(
            layout_planning=_make_dfs_solver(size, inner),
            search_space=_resolve_search_space("dfs"),
        )

    try:
        solver_cls = _PLACEMENT_SOLVERS[solver_name]
    except KeyError:
        raise ValueError(f"Invalid layout_solver config option '{solver_name}'.")
    return DefaultAllocator(layout_planning=solver_cls(size))


def scratchpad_planning(
    graph: GraphLowering,
    allocator: Optional[ScratchpadAllocator] = None,
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
    allocator.plan_allocation(graph)
