# Copyright 2025 The Torch-Spyre Authors.
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

"""Opt-in dumping of the LoopLevel IR (Inductor ``Operation`` list).

A no-op unless ``SPYRE_DUMP_IR`` is set; see :mod:`dump_common` for the sink
and environment-variable contract.

The dump is wired into ``CustomPreSchedulingPasses`` at two points: before any
Spyre pre-scheduling pass runs, and after all of them, so the two records can
be diffed to see what the middle-end did. The rendering reuses
``passes._format_operations`` so there is a single source of truth.

Note that HBM ``pool`` allocations are assigned later (at the post-fusion
``memory_planning`` pass), so they are not yet present in the "after" record;
LX ``allocation`` and ``op_it_space_splits`` are present.
"""

from torch._inductor.ir import Operation

from .dump_common import banner, dump_enabled, emit


def dump_loop_ir(
    operations: list[Operation],
    label: str = "LoopLevel IR (after pre-scheduling passes)",
) -> None:
    """Print the LoopLevel IR; no-op unless SPYRE_DUMP_IR is set.

    A debug dump must never break compilation, so any formatting error is
    reported and swallowed.
    """
    if not dump_enabled():
        return
    # Lazy import to avoid an import cycle (passes imports this module) and to
    # reuse the existing formatter rather than duplicating it.
    from .passes import _format_operations

    try:
        body = _format_operations(operations)
        emit(f"{banner(label)}\n{body}[{len(operations)} ops]\n")
    except Exception as exc:  # noqa: BLE001 - instrumentation must not raise
        emit(f"[SPYRE_DUMP_IR] failed to dump LoopLevel IR: {exc!r}")
