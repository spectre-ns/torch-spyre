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

"""Opt-in host-side timing of Spyre kernel launches.

Enabled by ``SPYRE_PROFILE=1``. ``kernel_timer`` wraps the device launch in
``SpyreSDSCKernelRunner.run`` and accumulates per-kernel wall-clock; a summary
is printed at process exit (to stderr, or to ``SPYRE_PROFILE_FILE``). A no-op
unless the env var is set.

Device execution is asynchronous (``executeProgramAsync``), so by default this
measures host-side *dispatch* latency (~microseconds), not device compute. Set
``SPYRE_PROFILE_SYNC=1`` to block on the device (``torch_spyre._C.synchronize``)
after each launch -- then the recorded time is that kernel's actual **device**
latency (serialized, so it kills pipelining; intended for profiling, not
production). This works for *any* op, including small ones the end-to-end
wall-clock harness can't resolve below its ~70us host-per-call floor.

Because the device is a static dataflow engine, per-kernel latency is
deterministic; the reported ``min`` strips host-side jitter and is the value to
trust.
"""

import atexit
import contextlib
import math
import os
import sys
import time

_TRUTHY = {"1", "true", "yes", "on"}

# name -> {count, total_ns, min_ns, max_ns}
_records: dict[str, dict] = {}
_atexit_registered = False
_report_at_exit = True


def set_report_at_exit(enabled: bool) -> None:
    """Enable/disable the automatic per-kernel report at process exit.

    Callers that print the report themselves (e.g. the bench examples) disable
    it to avoid a duplicate dump.
    """
    global _report_at_exit
    _report_at_exit = enabled


def profile_enabled() -> bool:
    """Return True when SPYRE_PROFILE requests kernel timing."""
    return os.environ.get("SPYRE_PROFILE", "").strip().lower() in _TRUTHY


def profile_sync_enabled() -> bool:
    """Return True when SPYRE_PROFILE_SYNC asks for a device sync per kernel.

    With it set, ``kernel_timer`` blocks on the device after each launch, so the
    recorded time is the kernel's actual **device** latency (serialized) rather
    than just async dispatch. Off by default — syncing serializes execution.
    """
    return os.environ.get("SPYRE_PROFILE_SYNC", "").strip().lower() in _TRUTHY


def _device_synchronize() -> None:
    """Block until the device finishes queued work (no-op if unavailable)."""
    try:
        import torch_spyre._C as _C

        _C.synchronize()
    except Exception:  # noqa: BLE001 - no runtime/sync -> fall back to dispatch timing
        pass


def reset() -> None:
    """Clear accumulated records (useful for tests / between benchmark phases)."""
    _records.clear()


def records() -> dict[str, dict]:
    """Return the raw per-kernel timing records."""
    return _records


@contextlib.contextmanager
def kernel_timer(name: str):
    """Time the wrapped device launch and accumulate it under ``name``.

    No-op (zero overhead beyond the env check) unless SPYRE_PROFILE is set.
    """
    if not profile_enabled():
        yield
        return
    _ensure_atexit()
    sync = profile_sync_enabled()
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        if sync:
            _device_synchronize()
        elapsed = time.perf_counter_ns() - start
        rec = _records.get(name)
        if rec is None:
            rec = {"count": 0, "total_ns": 0, "min_ns": math.inf, "max_ns": 0}
            _records[name] = rec
        rec["count"] += 1
        rec["total_ns"] += elapsed
        rec["min_ns"] = min(rec["min_ns"], elapsed)
        rec["max_ns"] = max(rec["max_ns"], elapsed)


def format_report() -> str:
    """Render the accumulated per-kernel timings as a text table."""
    if not _records:
        return "[SPYRE_PROFILE] no kernel launches recorded"
    lines = [
        "==== SPYRE_PROFILE: per-kernel host launch timing "
        "(min strips host jitter) ====",
        f"{'kernel':<40}{'count':>7}{'min_us':>11}{'mean_us':>11}{'max_us':>11}",
    ]
    for name in sorted(_records):
        rec = _records[name]
        count = rec["count"]
        min_us = rec["min_ns"] / 1000.0
        mean_us = (rec["total_ns"] / count) / 1000.0 if count else 0.0
        max_us = rec["max_ns"] / 1000.0
        lines.append(
            f"{name:<40}{count:>7}{min_us:>11.3f}{mean_us:>11.3f}{max_us:>11.3f}"
        )
    if profile_sync_enabled():
        lines.append(
            "[note] SPYRE_PROFILE_SYNC on: times are per-kernel DEVICE latency "
            "(serialized); trust the min."
        )
    else:
        lines.append(
            "[note] host-side DISPATCH only (async launch). Set SPYRE_PROFILE_SYNC=1 "
            "for per-kernel device latency."
        )
    return "\n".join(lines)


def report() -> None:
    """Emit the timing report to SPYRE_PROFILE_FILE or stderr."""
    text = format_report()
    dest = os.environ.get("SPYRE_PROFILE_FILE")
    if dest:
        with open(dest, "a", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    else:
        sys.stderr.write(text)
        sys.stderr.write("\n")
        sys.stderr.flush()


def _atexit_report() -> None:
    if _report_at_exit:
        report()


def _ensure_atexit() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_atexit_report)
        _atexit_registered = True
