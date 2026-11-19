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

"""The table lowering on the cost objectives of common blocks: flash attention, causal
SDPA, simple attention, GQA decode and RMSNorm, each captured from its own test by
tests/scripts/capture_table_lowering_blocks.py. No device and no compile.

The lowering cannot lower a real objective yet, so each block is marked to fail with
TableLoweringError until a stage lowers it. The marks are strict: a block that starts to
lower fails the run until its mark is removed, so the marks show which blocks lower.

A block that lowers is checked at three assignments of its variables, each at its
lowest value, at its highest, and at a seeded random value: minimizing and maximizing
the objective agree, and equal COST_SCALE times the expression evaluated by sympy,
within the rounding bound.
"""

import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pytest
import sympy

# ortools is not installed on s390x or ppc64le (pyproject.toml), and the table module
# imports it at its top: skip the file there rather than fail to collect it.
pytest.importorskip("ortools.sat.python.cp_model")

from ortools.sat.python import cp_model
from sympy.functions.elementary.piecewise import ExprCondPair

from torch_spyre._inductor.scratchpad.cpsat_table_lowering import (
    COST_SCALE,
    TableLowering,
    TableLoweringError,
)
from torch_spyre._inductor.scratchpad.plan_solver import RelayoutCharge

FIXTURE = Path(__file__).with_name("cpsat_table_lowering_blocks.json.gz")
CAP = 8192
SEED = 0

BLOCKS = (
    "causal_sdpa",
    "flash_attention",
    "gqa_decode",
    "rms_norm",
    "simple_attention",
)

# Why each block cannot lower yet: the terms of its objective whose one table is past
# the cap. A stage that lowers a block removes its entry.
REFUSED = {
    "causal_sdpa": "a top-level Min of 10^8.2 entries",
    "flash_attention": "top-level Mins of 10^24.3 and 10^13.2, a Max of 10^6.5 entries",
    "gqa_decode": "top-level Mins of 10^15.1 and 10^8.1, a Max of 10^4.3 entries",
    "rms_norm": "top-level Mins of 10^8.4 and 10^5.7 entries",
    "simple_attention": "a top-level Min of 10^7.6 entries",
}


@dataclass(frozen=True)
class Block:
    """One captured objective, with each variable's domain and each symbol's input."""

    name: str
    test: str
    expr: sympy.Expr
    variables: dict[str, tuple[int, int]]
    inputs: dict[str, tuple[str, list[float]]]


def _parse(srepr: str) -> sympy.Expr:
    namespace = {k: getattr(sympy, k) for k in dir(sympy) if not k.startswith("_")}
    namespace.update(RelayoutCharge=RelayoutCharge, ExprCondPair=ExprCondPair)
    return eval(srepr, {"__builtins__": {}}, namespace)


@pytest.fixture(scope="module")
def blocks() -> dict[str, Block]:
    """The fixture's blocks, parsed once: flash attention's expression takes seconds."""
    with gzip.open(FIXTURE, "rt") as f:
        data = json.load(f)
    return {
        name: Block(
            name=name,
            test=b["test"],
            expr=_parse(b["expr"]),
            variables={v: (lo, hi) for v, (lo, hi) in b["variables"].items()},
            inputs={s: (v, values) for s, (v, values) in b["inputs"].items()},
        )
        for name, b in data["blocks"].items()
    }


def _assignments(block: Block) -> list[dict[str, int]]:
    rng = random.Random(SEED)
    variables = sorted(block.variables.items())
    return [
        {v: lo for v, (lo, _) in variables},
        {v: hi for v, (_, hi) in variables},
        {v: rng.randint(lo, hi) for v, (lo, hi) in variables},
    ]


def _expected(block: Block, assignment: dict[str, int]) -> float:
    point = {}
    for symbol in block.expr.free_symbols:
        variable, values = block.inputs[symbol.name]
        lo, _ = block.variables[variable]
        point[symbol] = values[assignment[variable] - lo]
    return COST_SCALE * float(block.expr.xreplace(point))


def _solve_at(model, objective, variables, assignment) -> tuple[float, float]:
    """The objective's minimum and maximum with every variable fixed."""
    on = model.new_bool_var("assignment")
    for name, value in assignment.items():
        model.add(variables[name] == value).only_enforce_if(on)
    model.clear_assumptions()
    model.add_assumptions([on])
    found = []
    for sense in (model.minimize, model.maximize):
        sense(objective)
        solver = cp_model.CpSolver()
        status = solver.solve(model)
        assert status == cp_model.OPTIMAL, solver.status_name(status)
        found.append(solver.objective_value)
    return found[0], found[1]


def test_fixture_holds_the_common_blocks(blocks):
    assert sorted(blocks) == list(BLOCKS)


@pytest.mark.parametrize("name", BLOCKS)
def test_every_symbol_has_one_value_per_value_of_its_variable(blocks, name):
    block = blocks[name]
    assert {s.name for s in block.expr.free_symbols} <= set(block.inputs)
    for symbol, (variable, values) in block.inputs.items():
        lo, hi = block.variables[variable]
        assert len(values) == hi - lo + 1, symbol


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                raises=TableLoweringError, strict=True, reason=REFUSED[name]
            ),
        )
        if name in REFUSED
        else name
        for name in BLOCKS
    ],
)
def test_block_lowers_exactly(blocks, name):
    block = blocks[name]
    model = cp_model.CpModel()
    variables = {
        v: model.new_int_var(lo, hi, v) for v, (lo, hi) in block.variables.items()
    }
    symbols = {s.name: s for s in block.expr.free_symbols}
    inputs = {
        symbols[s]: (variables[v], values)
        for s, (v, values) in block.inputs.items()
        if s in symbols
    }
    before = len(model.proto.constraints)
    objective = TableLowering(inputs, cap=CAP).lower(block.expr).cp_expr(model)
    # Half a unit per constraint the lowering adds, and half a unit more: at least the
    # (T + 1) / 2 units of a sum of T rounded lists.
    bound = (len(model.proto.constraints) - before + 1) / 2
    for assignment in _assignments(block):
        low, high = _solve_at(model, objective, variables, assignment)
        assert low == high
        assert abs(low - _expected(block, assignment)) <= bound
