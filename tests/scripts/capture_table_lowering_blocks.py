#!/usr/bin/env python3
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

"""Capture the common blocks' cost objectives for test_cpsat_table_lowering_blocks.py.

Run it by hand on a Spyre host, from the repository root, with torch_spyre built:

  python tests/scripts/capture_table_lowering_blocks.py
  python tests/scripts/capture_table_lowering_blocks.py --replay-dir /tmp/blocks

It runs each block's own test once, with the inductor caches off so every graph is
compiled and solved, and wraps ilp_solver_ortools.cost_expr_to_cpsat to record the
cost expression and the two maps the CP-SAT solver hands it. Each expression is kept
with its inputs, in the form TableLowering takes them: a buffer's symbol reads its
division, with its menu's values, and any other symbol reads its own variable, with
its domain. A symbol the solver maps to a constant is replaced by it. A block whose
test solves more than once keeps the objective that reads the most symbols.

It captures under the default config. With auto coarse tiling on, the allocator hands
CP-SAT no cost expression.

With --replay-dir, each kept objective is also written in the replay's capture format:
the expression, the CP-SAT model and the two maps as the printer received them.

Rerun it when a block's objective changes shape. The tests check that a block lowers,
not what it costs, so a change to the cost model's prices needs no new capture.
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FIXTURE = Path("tests/cpsat_table_lowering_blocks.json.gz")

# Each common block and the test that builds it.
BLOCKS = {
    "flash_attention": (
        "tests/inductor/test_building_blocks.py::TestBuildingBlocks::test_flash_attention"
    ),
    "causal_sdpa": (
        "tests/inductor/test_building_blocks.py::TestBuildingBlocks"
        "::test_causal_sdpa_unpadded_kv_no_inf"
    ),
    "simple_attention": (
        "tests/inductor/test_building_blocks.py::TestBuildingBlocks::test__simple_attn"
    ),
    "gqa_decode": (
        "tests/inductor/test_sliding_window_attention.py::TestSlidingWindowAttention"
        "::test_decode_gqa"
    ),
    "rms_norm": (
        "tests/inductor/test_building_blocks.py::TestBuildingBlocks"
        "::test_chained_rms_norm_fp32_upcast"
    ),
}


def _variable_name(model, var, names: dict) -> str:
    """A name for the variable, unique in the fixture: its own, or v<index>."""
    if var.index not in names:
        name = model.proto.variables[var.index].name or f"v{var.index}"
        if name in names.values():
            name = f"{name}_v{var.index}"
        names[var.index] = name
    return names[var.index]


def _inputs(model, sym_map: dict, buffer_map: dict, expr):
    """The expression with constants replaced, its variables' domains, and each
    free symbol's input: its variable's name and its value at each of its values."""
    import sympy
    from ortools.sat.python import cp_model

    names: dict = {}
    variables: dict = {}
    inputs: dict = {}
    constants: dict = {}

    def variable(var) -> str:
        if not isinstance(var, cp_model.IntVar):
            raise ValueError(f"{var!r} is not an IntVar")
        name = _variable_name(model, var, names)
        domain = list(model.proto.variables[var.index].domain)
        if len(domain) != 2:
            raise ValueError(f"{name}: its domain {domain} has gaps")
        variables[name] = domain
        return name

    for symbol in sorted(expr.free_symbols, key=lambda s: s.name):
        key = symbol.name
        wrapper, raw = buffer_map.get(key, (None, None))
        if wrapper is not None:
            inputs[key] = [variable(wrapper.division), [float(x) for x in raw]]
            continue
        if key not in sym_map:
            raise ValueError(f"{key} has no entry in the solver's maps")
        value = sym_map[key]
        if isinstance(value, (int, float)):
            constants[symbol] = sympy.Float(value)
            continue
        if key.startswith("_division_of_"):
            value = value.division
        name = variable(value)
        lo, hi = variables[name]
        inputs[key] = [name, [float(x) for x in range(lo, hi + 1)]]
    return expr.xreplace(constants), variables, inputs


def _replay_context(sym_map: dict, buffer_map: dict) -> dict:
    """The two maps in the replay's capture format, keyed by variable index."""
    wrappers: dict = {}

    def wrap(w):
        key = w.division.index
        if key not in wrappers:
            b = w.buffer
            wrappers[key] = {
                "division": key,
                "name": getattr(b, "name", None),
                "n": len(b.core_divisions),
                "splits": [
                    {repr(k): int(v) for k, v in cd.splits.items()}
                    for cd in b.core_divisions
                ],
                "sym_core_divs": {
                    repr(k): s.name for k, s in getattr(b, "sym_core_divs", {}).items()
                },
                "sym_division": getattr(getattr(b, "sym_division", None), "name", None),
            }
        return key

    syms: dict = {}
    for name, v in sym_map.items():
        if name.startswith("_division_of_"):
            syms[name] = {"wrapper": wrap(v)}
        elif isinstance(v, (int, float)):
            syms[name] = {"const": v}
        elif hasattr(v, "index"):
            syms[name] = {"var": v.index, "bool": bool(getattr(v, "is_boolean", False))}
    bufs = {
        name: {
            "wrapper": wrap(w) if w is not None else None,
            "raw": [int(x) for x in raw],
        }
        for name, (w, raw) in buffer_map.items()
    }
    return {"sym_map": syms, "buffer_map": bufs, "wrappers": list(wrappers.values())}


class _Recorder:
    """A pytest plugin: every cost expression the CP-SAT solver lowers, by test."""

    def __init__(self, replay: bool) -> None:
        self.replay = replay
        self.test: str | None = None
        self.solves: list[dict] = []

    def pytest_runtest_setup(self, item) -> None:
        self.test = item.nodeid

    def wrap(self, original):
        def cost_expr_to_cpsat(model, sym_map, buffer_map, cost_expr):
            import sympy

            # Before the printer runs: it adds helper variables to model and sym_map.
            expr, variables, inputs = _inputs(model, sym_map, buffer_map, cost_expr)
            solve = {
                "test": self.test,
                "expr": sympy.srepr(expr),
                "variables": variables,
                "inputs": inputs,
            }
            if self.replay:
                solve["model"] = str(model.proto)
                solve["context"] = _replay_context(sym_map, buffer_map)
                solve["replay_expr"] = sympy.srepr(cost_expr)
            self.solves.append(solve)
            return original(model, sym_map, buffer_map, cost_expr)

        return cost_expr_to_cpsat


def _commit() -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "torch_spyre"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {"commit": head, "torch_spyre_modified": bool(dirty)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=FIXTURE, help="the fixture to write")
    ap.add_argument(
        "--replay-dir", type=Path, help="also write each block in the replay's format"
    )
    args = ap.parse_args(argv)

    # Before torch is imported: it reads these when it starts.
    os.environ.pop("AUTO_COARSE_TILING", None)
    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="table_blocks_")

    import pytest
    import torch  # noqa: F401  # before torch_spyre: torch autoloads it as a backend

    from torch_spyre._inductor import config
    from torch_spyre._inductor.scratchpad import ilp_solver_ortools

    if config.auto_coarse_tiling:
        print("auto_coarse_tiling is on: CP-SAT would get no cost expression")
        return 1
    recorder = _Recorder(replay=args.replay_dir is not None)
    ilp_solver_ortools.cost_expr_to_cpsat = recorder.wrap(
        ilp_solver_ortools.cost_expr_to_cpsat
    )
    status = pytest.main(
        ["-q", "-p", "no:cacheprovider", *BLOCKS.values()], plugins=[recorder]
    )

    blocks = {}
    for name, test in BLOCKS.items():
        solves = [s for s in recorder.solves if s["test"] == test]
        if not solves:
            print(f"{name}: {test} handed CP-SAT no cost expression")
            return 1
        index, kept = max(enumerate(solves), key=lambda e: len(e[1]["inputs"]))
        blocks[name] = {
            "test": test,
            "solve": index,
            "solves": len(solves),
            "expr": kept["expr"],
            "variables": kept["variables"],
            "inputs": kept["inputs"],
        }
        if args.replay_dir is not None:
            out = args.replay_dir / name
            out.mkdir(parents=True, exist_ok=True)
            (out / "expr.srepr").write_text(kept["replay_expr"])
            (out / "model.pbtxt").write_text(kept["model"])
            context = {"test": test, **kept["context"]}
            (out / "context.json").write_text(json.dumps(context))
        print(f"{name}: solve {index} of {len(solves)}, {len(kept['inputs'])} inputs")

    fixture = {
        "version": 1,
        "source": {**_commit(), "config": {"auto_coarse_tiling": False}},
        "pytest_status": int(status),
        "blocks": blocks,
    }
    data = json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode()
    with (
        open(args.out, "wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
        ) as f,
    ):
        f.write(data)
    print(f"wrote {args.out}, {args.out.stat().st_size} bytes")
    return 0 if status == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
