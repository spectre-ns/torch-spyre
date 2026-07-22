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

"""DEBUG-level reporting of a solved scratchpad layout.

Extracted from the solvers' old ``plan_layout(..., log_lx_usage=True)`` path so
every allocator can emit the same per-timestep LX occupancy trace after it
commits a plan, independent of which solver produced it. Reporting is not part
of the solver contract: only the greedy solver ever implemented the flag, so
the same debug output was silently unavailable under every other solver.
"""

import logging
from collections.abc import Sequence

from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.logging_utils import get_inductor_logger


# The floor for every drop cause: what a solver reports when it has nothing
# more specific to say than that the buffer did not fit. Solvers may set a
# sharper reason on the buffer instead; this is what fills in when they don't.
NOT_PLACED = "solver could not place buffer"


def spill_reasons(buffers: Sequence[LifetimeBoundBuffer]) -> dict[str, str]:
    """Return ``name -> drop cause`` for every buffer the solve left in HBM.

    The cause rides on the buffer's ``residency_reason``, whether the allocator
    barred it up front or the solver left it unplaced. A solver that says
    nothing means only that it could not place the buffer, so fill in
    :data:`NOT_PLACED` here rather than at each allocator -- the gap heuristics
    attribute nothing, and this keeps their reports in the same vocabulary as
    CP-SAT's.
    """
    return {
        b.name: b.residency_reason or NOT_PLACED for b in buffers if b.address is None
    }


def record_scratchpad_allocation(
    size: int, buffers: Sequence[LifetimeBoundBuffer]
) -> None:
    """Log the peak and per-timestep LX usage of a planned buffer set at DEBUG.

    ``buffers`` are the solved buffers (``address`` set for resident ones,
    ``None`` for spilled). No-op unless DEBUG logging is enabled, so the
    per-timestep walk is never paid on a normal compile.
    """
    logger = get_inductor_logger("scratchpad.plan_solver")
    if not logger.isEnabledFor(logging.DEBUG) or not buffers:
        return

    for name, reason in sorted(spill_reasons(buffers).items()):
        logger.debug("%s -> HBM: %s", name, reason)

    # ``address`` can legitimately be 0, so test ``is not None`` -- an ``if
    # buf.address`` truthiness check would drop a buffer placed at the base.
    resident = [b for b in buffers if b.address is not None]
    peak_usage = max(
        (b.address + b.size for b in resident if b.address is not None), default=0
    )
    start_time = min(b.start_time for b in buffers)
    end_time = max(b.end_time for b in buffers)

    logger.debug("scratchpad limit: %d KB", size // 1024)
    logger.debug("scratchpad peak usage: %d KB", peak_usage // 1024)
    for idx in range(start_time, end_time):
        live = []
        # Sum by distinct address: an in-place reuse places two buffers
        # (a dying parent and its just-born child) at the same address
        # for one overlapping tick, and the child's region is contained
        # in the parent's. Counting both would double-count the shared
        # slot, so track the max size per address and sum those.
        size_by_addr: dict[int, int] = {}
        for b in buffers:
            if b.address is not None and b.start_time <= idx < b.end_time:
                live.append(f"{b.name}_{b.size // 1024}KB@{hex(b.address)}")
                size_by_addr[b.address] = max(size_by_addr.get(b.address, 0), b.size)
        used = sum(size_by_addr.values())
        logger.debug("t=%d: %d KB  [%s]", idx, used // 1024, ", ".join(live))
