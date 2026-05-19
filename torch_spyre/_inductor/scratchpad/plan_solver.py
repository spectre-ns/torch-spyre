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

import numpy as np
import math

from dataclasses import dataclass, field, replace
from typing import Optional
from abc import ABC, abstractmethod
from collections import defaultdict

try:
    from ortools.sat.python import cp_model as cp_model

    _ORTOOLS_AVAILABLE = True
except ImportError:
    _ORTOOLS_AVAILABLE = False


@dataclass
class LifetimeBoundBuffer:
    """
    Defines the data fields required for a plan solver.
    """

    name: str
    size: int
    start_time: int
    end_time: int
    address: Optional[int] = None
    in_place_parents: list[str] = field(default_factory=list)


class MemoryPlanSolver(ABC):
    """
    An abstract class for defining algorithms which solve
    memory layout patterns based on provided sizes, lifetimes.
    """

    def __init__(self, size: int, alignment: int = 128):
        """Initialize the solver with a fixed scratchpad capacity and alignment.

        Args:
            size (int): Total scratchpad size in bytes. Buffers whose aligned
                placement would exceed this limit are evicted (address=None).
            alignment (int): Byte alignment boundary. Every buffer is placed at
                the next address that is a multiple of this value. Defaults to 128
                (one Spyre stick).
        """
        self.limit = size
        self.alignment = alignment
        self.usage: list[LifetimeBoundBuffer] = []

    @abstractmethod
    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        """
        Utilizes an implementation defined algorithm to determine
        if and where buffers should be placed in scratchpad memory based
        on their attributes.

        Args:
            buffers (list[LifetimeBoundBuffer]): The set of candidate buffers for memory planning

        Returns:
            list[LifetimeBoundBuffer]: The set of buffers with their placements defined.
        """
        pass


class GreedyLayoutSolver(MemoryPlanSolver):
    def _get_lowest_addr_in_use(self):
        return min(
            (rec.address for rec in self.usage if rec.address is not None),
            default=0,
        )

    def _get_highest_addr_in_use(self):
        return max(
            (rec.address + rec.size for rec in self.usage if rec.address is not None),
            default=0,
        )

    def _find_free_block(self, size_needed: int) -> Optional[int]:
        assert all(x.address is not None for x in self.usage)
        curr_lo = self._get_lowest_addr_in_use()
        curr_hi = self._get_highest_addr_in_use()
        if self.limit < size_needed:
            return None

        if not self.usage or curr_lo >= size_needed:
            return 0

        address = math.ceil(curr_hi / self.alignment) * self.alignment
        if address + size_needed <= self.limit:
            return address

        # Search for a gap between existing allocations
        self.usage.sort(key=lambda x: (x.address is None, x.address))
        for i in range(len(self.usage) - 1):
            assert (current_address := self.usage[i].address) is not None
            assert (next_address := self.usage[i + 1].address) is not None
            frag_st = (
                math.ceil((current_address + self.usage[i].size) / self.alignment)
                * self.alignment
            )
            if next_address - frag_st >= size_needed:
                return frag_st

        return None

    def _try_allocate(self, buffer: LifetimeBoundBuffer):
        # Check if the current buffer can be in-placed
        for in_place_opt in buffer.in_place_parents:
            matched_obj = next((u for u in self.usage if u.name == in_place_opt), None)
            if matched_obj is not None and buffer.size <= matched_obj.size:
                buffer.address = matched_obj.address
                self.usage.append(buffer)
                self.usage.remove(matched_obj)
                return None

        # Decide where to allocate the block from
        addr = self._find_free_block(buffer.size)

        # Push the allocation result to the buffer and the usage table
        if addr is not None:
            buffer.address = addr
            self.usage.append(buffer)
        else:
            buffer.address = None

    def _try_deallocate(self, bufs: list[LifetimeBoundBuffer] | LifetimeBoundBuffer):
        if isinstance(bufs, LifetimeBoundBuffer):
            bufs = [bufs]

        for buf in bufs:
            if buf in self.usage:
                self.usage.remove(buf)

    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        """Allocates addresses to the provided buffer list

        Accepts a set of buffers with pre-defined sizes and lifetimes. These buffers are
        allocated addresses with 0 -> `limit` where the maximum starting address of
        buffers are at most `self.limit` - `LifetimeBoundBuffer.size` - 1. The algorithm
        increments through logical time where time increments 1 unit for each
        step in a computation graph. At each step the lifetimes of all buffers are
        evaluated for allocation and deallocation based on its lifetime relative
        to the time being evaluated. As an optimization, times where no buffers
        enter or exit scope are not evaluated.

        When a buffer enters scope, the current usage is evaluated in the following
        manner:
            1. Check if there is a permissible in-place buffer already allocated
            2. Is there enough space from address 0 -> first usage.
            3. Is there enough space for the current buffer from the max address
                to the maximum memory address. Allocate as current_max + 1 + alignment.
            4. Is there space between allocations. Check for gaps between current
                allocations and find where gaps exceed current size. Allocate if
                current gap is larger than current size + alignment.

        Args:
            buffers (list[LifetimeBoundBuffer]): The set of buffers to be planned.

        Returns:
            list[LifetimeBoundBuffer]: The supplied buffers with addresses assigned.
        """
        if not buffers:
            return []
        assert all(buf.address is None for buf in buffers), (
            "Buffers cannot be previously or partially planned"
        )

        self.usage = []

        # Walk through all transition points in chronological order.
        # Include end_time + 1 so deallocation fires even when no other
        # buffer starts or ends at that tick.
        times = set()
        for b in buffers:
            times.add(b.start_time)
            times.add(b.end_time)
        sorted_times = sorted(times)

        for idx in sorted_times:
            # Deallocate all expired buffers before allocating new ones so that
            # freed slots are immediately available at the same time step.
            for buffer in buffers:
                if idx == buffer.end_time:
                    self._try_deallocate(buffer)

            for buffer in buffers:
                if idx == buffer.start_time:
                    self._try_allocate(buffer)

        return buffers


@dataclass
class _InPlaceCandidate:
    src: str
    dst: str


def _enumerate_dag_paths(
    candidates: list[_InPlaceCandidate], max_path_length: int = 8
) -> list[list[str]]:
    """Enumerate all simple paths of length >= 2 in the candidate DAG.

    max_path_length bounds the number of tensors per path to prevent
    combinatorial explosion on large graphs.
    """
    succ: dict[str, list[str]] = {}
    nodes: set[str] = set()
    for c in candidates:
        succ.setdefault(c.src, []).append(c.dst)
        nodes.add(c.src)
        nodes.add(c.dst)

    paths: list[list[str]] = []

    def dfs(path: list[str]) -> None:
        if len(path) >= 2:
            paths.append(list(path))
        if len(path) >= max_path_length:
            return
        for nxt in succ.get(path[-1], []):
            if nxt not in path:
                path.append(nxt)
                dfs(path)
                path.pop()

    for n in nodes:
        dfs([n])

    return paths


class OrToolsMemoryPlanSolver(MemoryPlanSolver):
    """
    Joint memory planning with in-place op selection using CP-SAT.

    Supports DAG-structured in-place candidates (derived from each buffer's
    in_place) and spillage when tensors exceed the buffer capacity.
    The objective minimizes total spill cost (primary) and peak buffer
    usage (secondary). The heuristic field on each buffer is the per-tensor
    spill cost weight (default 1.0 when None).
    """

    def __init__(
        self,
        buffer_size: int,
        alignment: int = 1,
        time_limit_seconds: float = 10.0,
        max_path_length: int = 8,
    ) -> None:
        if not _ORTOOLS_AVAILABLE:
            raise ImportError(
                "ortools is required for OrToolsMemoryPlanSolver. "
                "Install it with: pip install ortools"
            )
        self._buffer_size = buffer_size // alignment
        self._alignment = alignment
        self._time_limit_seconds = time_limit_seconds
        self._max_path_length = max_path_length

    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        candidates = [
            _InPlaceCandidate(src=parent, dst=b.name)
            for b in buffers
            for parent in b.in_place
        ]
        for b in buffers:
            b.size = int(np.ceil(b.size / self._alignment))

        offsets, spilled = self._solve(buffers, candidates)
        for _, o in offsets.items():
            o *= self._alignment

        return [
            replace(b, address=None if b.name in spilled else offsets.get(b.name))
            for b in buffers
        ]

    def _solve(
        self,
        tensors: list[LifetimeBoundBuffer],
        candidates: list[_InPlaceCandidate],
    ) -> tuple[dict[str, int], set[str]]:
        M = self._buffer_size
        SCALE = 1000
        SPILL_WEIGHT = M + 1

        by_name = {t.name: t for t in tensors}
        for c in candidates:
            if c.src not in by_name or c.dst not in by_name:
                raise ValueError(f"Candidate {c} references unknown tensor")

        model = cp_model.CpModel()

        ######################################
        # Define all optimization variables
        ######################################
        peak_var = model.new_int_var(0, M, "peak")

        offset: dict[str, cp_model.IntVar] = {
            t.name: model.new_int_var(0, max(0, M - t.size), f"off_{t.name}")
            for t in tensors
        }

        in_buffer = {t.name: model.new_bool_var(f"in_buf_{t.name}") for t in tensors}

        paths = _enumerate_dag_paths(candidates, max_path_length=self._max_path_length)

        merge_groups = [
            {"tensors": p, "var": model.new_bool_var(f"merge_{'_'.join(p)}")}
            for p in paths
        ]

        groups_containing: dict[str, list[dict]] = defaultdict(list)
        for mg, n in ((mg, n) for mg in merge_groups for n in mg["tensors"]):
            groups_containing[n].append(mg)

        solo: dict[str, cp_model.IntVar] = {
            t.name: in_buffer[t.name]
            if t.name not in groups_containing
            else model.new_bool_var(f"solo_{t.name}")
            for t in tensors
        }

        time_intervals = [
            model.new_optional_fixed_size_interval_var(
                t.start_time,
                (t.end_time + 1) - t.start_time,
                solo[t.name],
                f"x_solo_{t.name}",
            )
            for t in tensors
        ]
        memory_intervals = [
            model.new_optional_interval_var(
                offset[t.name],
                t.size,
                offset[t.name] + t.size,
                solo[t.name],
                f"y_solo_{t.name}",
            )
            for t in tensors
        ]

        m_off: dict[tuple, cp_model.IntVar] = {}
        for mg in merge_groups:
            names = tuple(mg["tensors"])
            var = mg["var"]
            ts = [by_name[n] for n in names]
            size = max(t.size for t in ts)
            start = min(t.start_time for t in ts)
            end = max(t.end_time for t in ts) + 1
            max_off = max(0, M - size)

            m_off[names] = model.new_int_var(0, max_off, f"off_merge_{'_'.join(names)}")
            time_intervals.append(
                model.new_optional_fixed_size_interval_var(
                    start, end - start, var, f"x_merge_{'_'.join(names)}"
                )
            )
            memory_intervals.append(
                model.new_optional_interval_var(
                    m_off[names],
                    size,
                    m_off[names] + size,
                    var,
                    f"y_merge_{'_'.join(names)}",
                )
            )

        ######################################
        # Define constraints
        ######################################

        for t in tensors:
            if t.size > M:
                model.add(in_buffer[t.name] == 0)

        for name, gs in groups_containing.items():
            if len(gs) > 1:
                model.add(sum(g["var"] for g in gs) <= 1)

        for mg in merge_groups:
            for n in mg["tensors"]:
                model.add_implication(mg["var"], in_buffer[n])

        for n, gs in groups_containing.items():
            s = solo[n]
            for mg in groups_containing[n]:
                model.add_implication(mg["var"], ~s)
            model.add_implication(~in_buffer[n], ~s)
            model.add(s + sum(mg["var"] for mg in gs) == in_buffer[n])

        for t in tensors:
            model.add(offset[t.name] + t.size <= peak_var).only_enforce_if(solo[t.name])

        for mg in merge_groups:
            names = tuple(mg["tensors"])
            var = mg["var"]
            ts = [by_name[n] for n in names]
            size = max(t.size for t in ts)
            if size > M:
                model.add(var == 0)
                continue
            for n in names:
                model.add(offset[n] == m_off[names]).only_enforce_if(var)
            model.add(m_off[names] + size <= peak_var).only_enforce_if(var)

        model.add_no_overlap_2d(time_intervals, memory_intervals)

        ######################################
        # Define objective function
        ######################################

        spill_cost_expr = SPILL_WEIGHT * sum(
            int(
                round(
                    t.size * (t.heuristic if t.heuristic is not None else 1.0) * SCALE
                )
            )
            * (1 - in_buffer[t.name])
            for t in tensors
        )

        model.minimize(spill_cost_expr + peak_var)

        ######################################
        # Solve
        ######################################
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self._time_limit_seconds
        status = solver.solve(model)

        # This should never happen as the optimizer has a release valve
        # to allow spilling. Thus, a non-optimal solution would be to
        # spill all buffers but obviously that is not desirable.
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"OrTools memory planner failed to converge with status: {status}"
            )

        spilled = {t.name for t in tensors if not solver.value(in_buffer[t.name])}
        offsets_out = {
            t.name: solver.value(offset[t.name])
            for t in tensors
            if solver.value(in_buffer[t.name])
        }
        return offsets_out, spilled
