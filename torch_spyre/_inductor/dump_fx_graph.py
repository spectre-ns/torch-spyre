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

"""Opt-in dumping of the ATen FX graph for inspecting the Spyre pipeline.

A no-op unless ``SPYRE_DUMP_IR`` is set; see :mod:`dump_common` for the sink
and environment-variable contract.
"""

import torch
import torch.fx

from .dump_common import banner, dump_enabled, emit


def _format_fx_graph(graph: torch.fx.Graph) -> str:
    """Render each FX node, annotated with its fake-tensor metadata."""
    lines = []
    for node in graph.nodes:
        line = node.format_node()
        if line is None:
            line = f"{node.op}: {node.name}"
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            line += f"    # {val.dtype} {tuple(val.shape)} {val.device}"
        lines.append(line)
    return "\n".join(lines)


def dump_fx_graph(
    graph: torch.fx.Graph,
    label: str = "ATen FX graph (post-grad, pre-lowering)",
) -> None:
    """Print the ATen FX graph; no-op unless SPYRE_DUMP_IR is set.

    Wired into ``CustomPostPasses`` so it runs on the post-grad FX graph just
    before Inductor lowers it to LoopLevel IR. A debug dump must never break
    compilation, so any formatting error is reported and swallowed.
    """
    if not dump_enabled():
        return
    try:
        body = _format_fx_graph(graph)
        emit(f"{banner(label)}\n{body}\n[{len(graph.nodes)} nodes]\n")
    except Exception as exc:  # noqa: BLE001 - instrumentation must not raise
        emit(f"[SPYRE_DUMP_IR] failed to dump FX graph: {exc!r}")
