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

"""Shared sink for the opt-in pipeline dumps (``dump_fx_graph``, ``dump_loop_ir``).

All dumping is gated by the ``SPYRE_DUMP_IR`` environment variable; when it is
unset the dump entry points are no-ops, so they are safe to leave wired into
the pass pipeline permanently. Output goes to stderr by default, or is appended
to the file named by ``SPYRE_DUMP_IR_FILE`` when that variable is set.
"""

import os
import sys

_TRUTHY = {"1", "true", "yes", "on"}


def dump_enabled() -> bool:
    """Return True when SPYRE_DUMP_IR requests dumping."""
    return os.environ.get("SPYRE_DUMP_IR", "").strip().lower() in _TRUTHY


def emit(text: str) -> None:
    """Write one dump record to the configured sink (file or stderr)."""
    dest = os.environ.get("SPYRE_DUMP_IR_FILE")
    if dest:
        with open(dest, "a", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    else:
        sys.stderr.write(text)
        sys.stderr.write("\n")
        sys.stderr.flush()


def banner(title: str) -> str:
    """Return a boxed section header for a dump record."""
    bar = "=" * 78
    return f"{bar}\n==== {title}\n{bar}"
