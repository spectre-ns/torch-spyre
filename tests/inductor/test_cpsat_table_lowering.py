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
"""The table lowering of a CP-SAT cost objective: the base mechanics, one table per
node, every value by lambdify. No device and no compile: each test builds its inputs
by hand and solves the lowered model with CP-SAT.

*Exact* means: every assignment of the inputs has exactly one solution, so each
position and the cost are determined by the inputs, and that solution's cost is
COST_SCALE times the expression evaluated by sympy, within the rounding bound.
"""

import itertools

import numpy as np
import pytest
import sympy
from ortools.sat.python import cp_model
from sympy.functions.elementary.piecewise import ExprCondPair

from torch_spyre._inductor.scratchpad.cpsat_table_lowering import (
    COST_SCALE,
    UNIT,
    Combine,
    CpConstant,
    CpForm,
    CpIndex,
    CpNode,
    CpSymbol,
    CpTable,
    Input,
    TableLowering,
    TableLoweringError,
    values_on_grid,
)
from torch_spyre._inductor.scratchpad.plan_solver import RelayoutCharge

CAP = 8192


class Case:
    """A CP-SAT model with inputs made by hand: each input is a variable and the
    column of every symbol that reads it."""

    def __init__(self) -> None:
        self.model = cp_model.CpModel()
        self.inputs: dict[sympy.Symbol, CpTable] = {}
        self.variables: list[cp_model.IntVar] = []
        self.values: dict[cp_model.IntVar, list[int]] = {}

    def add_input(self, name: str, **columns: list[float]) -> CpIndex:
        """A buffer's division: a variable in 0 .. n-1 picking a row of its menu, and
        one column per symbol of the buffer."""
        sizes = {len(column) for column in columns.values()}
        assert len(sizes) == 1
        (size,) = sizes
        var = self.model.new_int_var(0, size - 1, f"division_{name}")
        position = CpIndex(Input(var, size=size))
        for symbol_name, column in columns.items():
            self.inputs[sympy.Symbol(symbol_name, positive=True)] = CpTable(
                values=column, index=position
            )
        self.variables.append(var)
        self.values[var] = list(range(size))
        return position

    def integer(self, name: str, lo: int, size: int) -> CpIndex:
        """An integer variable with its own values, lo .. lo + size - 1: its position
        is the value offset by lo."""
        domain = list(range(lo, lo + size))
        var = self.model.new_int_var(lo, lo + size - 1, name)
        position = CpIndex(Input(var, size=size, lo=lo))
        self.inputs[sympy.Symbol(name, integer=True)] = CpTable(
            values=domain, index=position
        )
        self.variables.append(var)
        self.values[var] = list(domain)
        return position

    def symbol(self, name: str) -> sympy.Symbol:
        (symbol,) = (s for s in self.inputs if s.name == name)
        return symbol

    def lower(
        self, expr: sympy.Basic, cap: int = CAP
    ) -> tuple[CpNode, cp_model.LinearExprT]:
        root = TableLowering(self.inputs, cap=cap).lower(expr)
        return root, root.cp_expr(self.model)

    def solutions(self, exprs: list) -> list[tuple[int, ...]]:
        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True
        found: list[tuple[int, ...]] = []

        class Collect(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self) -> None:
                found.append(tuple(self.value(e) for e in exprs))

        status = solver.solve(self.model, Collect())
        assert status == cp_model.OPTIMAL, solver.status_name(status)
        return found

    def sympy_value(self, expr: sympy.Basic, assignment: tuple[int, ...]) -> float:
        """expr evaluated by sympy where the variables take assignment."""
        values = {v: np.array([x]) for v, x in zip(self.variables, assignment)}
        point = {}
        for symbol, table in self.inputs.items():
            position = int(np.ravel(table.index.positions(values))[0])
            value = float(table.values[position])
            point[symbol] = (
                sympy.Integer(int(value)) if value.is_integer() else sympy.Float(value)
            )
        return float(expr.xreplace(point))

    def assert_exact(self, expr: sympy.Basic, cap: int = CAP) -> CpNode:
        plan, objective = self.lower(expr, cap)
        cost = self.model.new_int_var(-(2**60), 2**60, "cost")
        self.model.add(cost == objective)
        rows = self.solutions(self.variables + [cost])
        assignments = list(itertools.product(*(self.values[v] for v in self.variables)))
        assert len(rows) == len(assignments)
        assert sorted(row[:-1] for row in rows) == sorted(assignments)
        bound = 0.5  # one table, rounded once
        for row in rows:
            exact = COST_SCALE * self.sympy_value(expr, row[:-1])
            assert abs(row[-1] - exact) <= bound, (row, exact)
        return plan


# ----------------------------------------------------------------------- the walk


def test_the_factory_makes_children_first_and_shares_a_subexpression():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    shared = a**0.5 + b
    expr = shared * a + sympy.log(shared)
    before = sympy.srepr(expr)
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(expr)
    order = list(nodes)
    for expr_node, node in nodes.items():
        for child in node.children:
            assert order.index(child.expr) < order.index(expr_node)
    holders = [n for n in nodes.values() if any(c is nodes[shared] for c in n.children)]
    assert len(holders) == 2
    assert len(order) == len(set(order))
    assert sympy.srepr(expr) == before


def test_a_piecewise_arm_gets_no_node():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    expr = sympy.Piecewise((a, a > 2), (a**2, True))
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(expr)
    assert not any(isinstance(e, ExprCondPair) for e in nodes)
    assert [c.expr for c in nodes[expr].children] == [a, a > 2, a**2, sympy.true]


def test_each_node_is_a_symbol_a_constant_or_an_operation():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    root2 = sympy.sqrt(2)
    expr = 3 * a + root2 * a**0.5
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(expr)
    assert type(nodes[a]) is CpSymbol
    assert nodes[a].column is case.inputs[a]
    assert type(nodes[sympy.Integer(3)]) is CpConstant
    assert type(nodes[root2]) is CpConstant  # an operation on numbers
    assert type(nodes[a**0.5]) is CpNode
    assert type(nodes[expr]) is CpNode


def test_a_node_gathers_its_columns_from_its_children():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    expr = a**0.5 + 2 * b
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(expr)
    assert dict(nodes[expr].columns) == {a: case.inputs[a], b: case.inputs[b]}
    assert all(nodes[expr].columns[s] is case.inputs[s] for s in (a, b))
    assert dict(nodes[2 * b].columns) == {b: case.inputs[b]}
    assert dict(nodes[sympy.Integer(2)].columns) == {}


def test_a_node_realizes_itself_when_it_is_read():
    """No lowering is passed in, and nothing visits: a node reads its children and
    its cap, the first time its form is read."""
    case = Case()
    position = case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    leaf = CpSymbol(a, (), CAP, column=case.inputs[a])
    half = CpConstant(sympy.Float(0.5), (), CAP)
    root = CpNode(a**0.5, (leaf, half), CAP)
    assert "form" not in vars(root)
    assert root.form is not None
    assert root.form.index == position
    assert list(root.form.values) == pytest.approx([1.0, 2**0.5, 2.0])
    small = CpNode(a**0.5, (leaf, half), 2)
    with pytest.raises(
        TableLoweringError,
        match="one table over its inputs would hold 3 entries, past the cap of 2",
    ):
        small.form
    with pytest.raises(TableLoweringError, match="no input"):
        CpSymbol(a, (), CAP).form


def test_a_table_past_the_cap_is_never_evaluated(monkeypatch):
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    calls: list[sympy.Basic] = []
    lambdify = sympy.lambdify

    def counting(args, expr, **kwargs):
        calls.append(expr)
        return lambdify(args, expr, **kwargs)

    monkeypatch.setattr(sympy, "lambdify", counting)
    root = TableLowering(case.inputs, cap=5).lower(a * b)
    with pytest.raises(TableLoweringError, match="6 entries, past the cap of 5"):
        root.form
    assert calls == []


def test_a_table_builds_its_own_lookup_and_a_constant_one_adds_nothing():
    case = Case()
    position = case.integer("y", 1, 4)
    before = len(case.model.proto.variables), len(case.model.proto.constraints)
    constant = CpTable(values=[2.0, 2.0, 2.0, 2.0], index=position)
    assert constant.cp_expr(case.model) == 2 * COST_SCALE
    after = len(case.model.proto.variables), len(case.model.proto.constraints)
    assert after == before
    table = CpTable(values=[1.0, 2.0, 4.0, 8.0], index=position)
    target = table.cp_expr(case.model)
    rows = dict(case.solutions([case.variables[0], target]))
    assert rows == {1: 1024, 2: 2048, 3: 4096, 4: 8192}


def test_lower_realizes_nothing_and_a_refusal_leaves_the_model_as_it_was():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    root = TableLowering(case.inputs, cap=CAP).lower(a**0.5 * b)
    assert type(root) is CpNode and root.expr == a**0.5 * b
    assert "form" not in vars(root)
    root.cp_expr(case.model)
    size = len(case.model.proto.constraints), len(case.model.proto.variables)
    wide = TableLowering(case.inputs, cap=4).lower(a**0.5 * b)
    with pytest.raises(TableLoweringError, match="past the cap of 4"):
        wide.cp_expr(case.model)
    assert (len(case.model.proto.constraints), len(case.model.proto.variables)) == size


# ------------------------------------------------------------------------- values


def test_a_form_is_abstract_and_a_table_is_a_form():
    case = Case()
    position = case.add_input("a", a=[1.0, 2.0, 4.0])
    with pytest.raises(TypeError):
        CpForm()  # type: ignore[abstract]

    class Partial(CpForm):
        """A form with no cp_expr."""

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]
    table = CpTable(values=[1.0, 2.0, 4.0], index=position)
    assert isinstance(table, CpForm)
    assert table.finite and table.entries == 3
    assert not CpTable(values=[1.0, np.inf, 4.0], index=position).finite
    a = case.symbol("a")
    assert isinstance(TableLowering(case.inputs, cap=CAP).lower(a**0.5).form, CpForm)


def test_a_list_must_have_its_index_size():
    case = Case()
    position = case.add_input("a", a=[1.0, 2.0, 4.0])
    with pytest.raises(ValueError):
        CpTable(values=[1.0, 2.0], index=position)


def test_an_index_refuses_parts_its_step_does_not_take():
    case = Case()
    a = case.add_input("a", a=[1.0, 2.0, 4.0])
    b = case.add_input("b", b=[1.0, 3.0])
    with pytest.raises(ValueError):
        CpIndex(Combine(), (a,))
    with pytest.raises(ValueError):
        CpIndex(Combine(), (a, b, a))
    with pytest.raises(ValueError):
        CpIndex(Input(case.variables[0], size=3), (a,))
    assert CpIndex(Combine(), (a, b)).size == 6


def test_a_table_takes_values_and_index_only_and_its_list_is_read_only():
    case = Case()
    position = case.add_input("a", a=[1.0, 2.0, 4.0])
    with pytest.raises(TypeError):
        CpTable(values=[1.0, 2.0, 3.0], index=position, constant=1.0)  # type: ignore[call-arg]
    source = np.array([1.0, 2.0, 3.0])
    table = CpTable(values=source, index=position)
    source[0] = 9.0
    assert table.values[0] == 1.0
    with pytest.raises(ValueError):
        table.values[0] = 5.0
    with pytest.raises(AttributeError):
        table.values = np.zeros(3)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        table.index = UNIT  # type: ignore[misc]


def test_positions_agree_with_cp_sat_at_every_assignment():
    case = Case()
    x = case.integer("x", 2, 4)
    y = case.integer("y", -1, 3)
    joint = CpIndex(Combine(), (x, y))
    rows = case.solutions(case.variables + [joint.cp_expr(case.model)])
    values = {v: np.array([r[k] for r in rows]) for k, v in enumerate(case.variables)}
    assert list(joint.positions(values)) == [r[-1] for r in rows]
    assert sorted(r[-1] for r in rows) == list(range(12))


# --------------------------------------------------------------- values_on_grid


def test_values_on_grid_is_row_major_with_the_last_position_fastest():
    case = Case()
    a = case.add_input("a", a=[0.0, 1.0])
    b = case.add_input("b", b=[0.0, 1.0, 2.0])
    c = case.add_input("c", c=[0.0, 1.0, 2.0, 3.0])
    sa, sb, sc = case.symbol("a"), case.symbol("b"), case.symbol("c")
    values = values_on_grid(100 * sa + 10 * sb + sc, case.inputs, [a, b, c])
    expected = [
        100 * i + 10 * j + k for i in range(2) for j in range(3) for k in range(4)
    ]
    assert list(values) == expected


def test_values_on_grid_moves_two_symbols_of_one_position_together():
    case = Case()
    p = case.add_input("a", d0=[1.0, 2.0, 4.0], d1=[8.0, 4.0, 2.0])
    d0, d1 = case.symbol("d0"), case.symbol("d1")
    assert list(values_on_grid(d0 * d1, case.inputs, [p])) == [8.0, 8.0, 8.0]


def test_values_on_grid_broadcasts_a_constant():
    case = Case()
    p = case.add_input("a", a=[1.0, 2.0, 4.0])
    values = values_on_grid(sympy.Integer(5), case.inputs, [p])
    assert list(values) == [5.0, 5.0, 5.0]


def test_a_value_that_is_not_finite_is_refused():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    with pytest.raises(TableLoweringError):
        case.lower(1 / (a - 2))


def test_a_node_is_lambdified_once_and_only_when_its_form_is_read(monkeypatch):
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0])
    a, b = case.symbol("a"), case.symbol("b")
    calls: list[sympy.Basic] = []
    lambdify = sympy.lambdify

    def counting(args, expr, **kwargs):
        calls.append(expr)
        return lambdify(args, expr, **kwargs)

    monkeypatch.setattr(sympy, "lambdify", counting)
    shared = a**0.5 + b
    expr = shared * a + sympy.log(shared)
    nodes = TableLowering(case.inputs, cap=CAP).rewrite(expr)
    assert calls == []  # making the nodes realizes nothing
    computed = [e for e in nodes if e.free_symbols and not isinstance(e, sympy.Symbol)]
    for _ in range(2):
        for e in computed:
            assert nodes[e].form is not None
    assert len(calls) == len(computed) == len(set(calls))
    calls.clear()
    TableLowering(case.inputs, cap=CAP).lower(expr).cp_expr(case.model)
    assert calls == [expr]  # one table per node: only the root is read


# ---------------------------------------------------------------- exact lowering


def test_one_table_over_two_inputs_of_different_sizes_is_exact():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    case.add_input("b", b=[1.0, 3.0, 5.0, 7.0])
    a, b = case.symbol("a"), case.symbol("b")
    plan = case.assert_exact(a**2 * b + b**0.5 / a)
    index = plan.form.index
    assert index.size == 12


def test_variables_above_and_below_zero_read_their_entries():
    case = Case()
    case.integer("x", 2, 4)
    case.integer("y", -1, 3)
    x, y = case.symbol("x"), case.symbol("y")
    case.assert_exact(x**1.5 * y + 3 * x)


def test_a_piecewise_with_an_infinite_arm_not_taken_is_exact():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    case.assert_exact(sympy.Piecewise((1 / (a - 2), a > 2), (0.5, True)))


def test_a_piecewise_with_no_arm_true_is_refused():
    case = Case()
    case.add_input("a", a=[1.0, 4.0])
    a = case.symbol("a")
    with pytest.raises(TableLoweringError):
        case.lower(sympy.Piecewise((a, a > 3)))


def test_functions_numpy_does_not_share_give_sympys_values():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    case.assert_exact(sympy.sec(a / 4) + sympy.sinc(a))


def test_a_relayout_charge_is_one_table_over_its_literal_and_division():
    case = Case()
    model = case.model
    lit = model.new_bool_var("is_lx")
    division = model.new_int_var(0, 2, "division_src")
    lit_index = CpIndex(Input(lit, size=2))
    div_index = CpIndex(Input(division, size=3))
    is_lx = sympy.Symbol("is_lx", integer=True)
    div = sympy.Symbol("division_src", integer=True, nonnegative=True)
    case.inputs[is_lx] = CpTable(values=[0, 1], index=lit_index)
    case.inputs[div] = CpTable(values=[0, 1, 2], index=div_index)
    case.variables += [lit, division]
    case.values[lit], case.values[division] = [0, 1], [0, 1, 2]
    expr = RelayoutCharge(is_lx, div, 0, 65, 147)
    plan = case.assert_exact(expr)
    index = plan.form.index
    assert index.size == 6


def test_an_undefined_function_is_refused():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    with pytest.raises(TableLoweringError):
        case.lower(sympy.Function("f")(a))


def test_a_coefficient_is_applied_before_the_rounding():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    _, objective = case.lower(7 * a**0.5)
    cost = case.model.new_int_var(0, 2**40, "cost")
    case.model.add(cost == objective)
    rows = dict((r[0], r[1]) for r in case.solutions([case.variables[0], cost]))
    assert rows[1] == 10137  # rint(1024 * 7 * 2**0.5), not 7 * rint(1024 * 2**0.5)


def test_a_constant_list_adds_no_variable_and_a_constant_objective_is_an_int():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    variables = len(case.model.proto.variables)
    _, objective = case.lower(a / a + 2)
    assert objective == 3 * COST_SCALE
    assert isinstance(objective, int)
    assert len(case.model.proto.variables) == variables


def test_an_entry_too_large_for_int64_sums_is_refused():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a = case.symbol("a")
    with pytest.raises(TableLoweringError, match="2\\^50"):
        case.lower(1e14 * a)


def test_a_symbol_with_no_input_is_refused_by_name():
    case = Case()
    case.add_input("a", a=[1.0, 2.0, 4.0])
    a, z = case.symbol("a"), sympy.Symbol("z", positive=True)
    with pytest.raises(TableLoweringError) as caught:
        case.lower(a + z)
    assert caught.value.node == z
    assert "no input" in caught.value.reason


def test_the_naive_lowering_explodes_past_the_cap():
    """One table per node puts the sum over all three buffers in one table, 21 * 11 *
    20 = 4620 entries, though no term reads more than one buffer. Under a cap of 4096
    the sum is refused, by name; keeping a sum linear is the next optimization."""
    case = Case()
    case.add_input("a", a=[float(v) for v in range(1, 22)])
    case.add_input("b", b=[float(v) for v in range(1, 12)])
    case.add_input("c", c=[float(v) for v in range(1, 21)])
    a, b, c = case.symbol("a"), case.symbol("b"), case.symbol("c")
    expr = a**0.5 + b**0.5 + c**0.25
    with pytest.raises(
        TableLoweringError, match="4620 entries, past the cap of 4096"
    ) as caught:
        case.lower(expr, cap=4096)
    assert caught.value.node == expr
    plan, _ = case.lower(expr, cap=8192)
    index = plan.form.index
    assert index.size == 4620
