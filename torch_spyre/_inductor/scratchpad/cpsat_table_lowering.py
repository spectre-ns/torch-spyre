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

"""A sympy cost expression lowered to a CP-SAT objective from exact tables.

Every cost value is computed before the solve, in floating point, and CP-SAT only
works out which entry of a table applies. A node's value is a *table*: a float list
``values`` and a ``CpIndex``, the position in it that the decision variables fix,
``value == values[index]``. The lowering has three layers:

- values: ``CpIndex`` and the forms, ``CpForm`` and its one kind here, ``CpTable``,
  hold no model;
- the nodes: ``TableLowering`` is a factory that makes one ``CpNode`` per node of the
  expression, of a class chosen by the node's sympy class. Each node realizes its
  own form, here a table, from its children the first time it is read, so a node
  nobody reads is never realized. The plan, the root's ``CpNode``, holds no model
  either;
- the build: the root's ``cp_expr`` reads its form, which builds itself into the
  model (``CpTable.cp_expr``), rounded onto ``COST_SCALE``, its position built by
  ``CpIndex.cp_expr``. Only ``cp_expr`` methods touch the model.

This is the base: one table per node, every value computed by ``sympy.lambdify`` over
the node's inputs, and the root read as one lookup. It lowers an expression whose
inputs fit the cap together and refuses anything wider. Nothing imports it yet.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, TypeAlias

import numpy as np
import sympy
from ortools.sat.python import cp_model
from sympy.functions.elementary.piecewise import ExprCondPair

from torch_spyre._inductor.scratchpad.plan_solver import RelayoutCharge

# One cost unit is COST_SCALE objective units: a table entry is
# rint(COST_SCALE * value).
COST_SCALE = 1024
# The largest table entry: a sum of thousands of entries stays inside int64.
MAX_ENTRY = 2**50

# What CP-SAT takes as a linear expression: a LinearExpr, an IntVar or an int.
CpExpr: TypeAlias = cp_model.LinearExprT
# Each variable's values at a set of assignments, one array entry per assignment.
Values: TypeAlias = Mapping[cp_model.IntVar, np.ndarray]


class TableLoweringError(Exception):
    """An objective the tables cannot lower. ``node`` is the sympy node left with no
    form, or None for a refusal outside the walk; ``reason`` says why.

    It subclasses none of RuntimeError, TypeError and ValueError, so an ``except``
    around the objective that catches those cannot turn it into a different solve.
    """

    def __init__(self, node: sympy.Basic | None, reason: str) -> None:
        super().__init__(reason if node is None else f"{reason}: {node}")
        self.node = node
        self.reason = reason


# --------------------------------------------------------------------------- values


class IndexMap(ABC):
    """One mapping step of a CpIndex: from its parts' positions to a position
    0 .. size - 1, in numpy and in CP-SAT. Each kind of step is a subclass; CpIndex
    and the nodes do not change when one is added."""

    @abstractmethod
    def size_from(self, parts: Sequence[CpIndex]) -> int:
        """The number of positions the step maps onto. Raises ValueError for parts it
        does not take: the wrong count, or sizes its own data does not fit."""

    @property
    @abstractmethod
    def entries(self) -> int:
        """The table entries the step puts in the model, 0 for an arithmetic step."""

    @abstractmethod
    def positions(
        self, values: Values, index: CpIndex, parts: Sequence[np.ndarray]
    ) -> np.ndarray:
        """The step in numpy: index's position at each assignment in values, from its
        parts' positions there. A leaf step reads its variable's values."""

    @abstractmethod
    def cp_expr(
        self, model: cp_model.CpModel, index: CpIndex, parts: Sequence[CpExpr]
    ) -> CpExpr:
        """The step in model: the same position as a CP-SAT expression, from its
        parts' expressions. A leaf step reads its variable."""


def _no_parts(step: str, parts: Sequence[CpIndex]) -> None:
    if parts:
        raise ValueError(f"{step} is a leaf and takes no parts, got {len(parts)}")


@dataclass(frozen=True)
class Input(IndexMap):
    """A leaf: a variable's own position, v -> v - lo, for a domain lo .. lo + size - 1
    with no gaps. A division or a literal has lo 0. The variable is already in the
    model."""

    var: cp_model.IntVar
    size: int
    lo: int = 0

    def size_from(self, parts: Sequence[CpIndex]) -> int:
        _no_parts("Input", parts)
        if self.size < 1:
            raise ValueError(f"Input of size {self.size}")
        return self.size

    @property
    def entries(self) -> int:
        return 0

    def positions(
        self, values: Values, index: CpIndex, parts: Sequence[np.ndarray]
    ) -> np.ndarray:
        return np.asarray(values[self.var]) - self.lo

    def cp_expr(
        self, model: cp_model.CpModel, index: CpIndex, parts: Sequence[CpExpr]
    ) -> CpExpr:
        return self.var - self.lo if self.lo else self.var


@dataclass(frozen=True)
class Locate(IndexMap):
    """A leaf: the position of a variable whose domain has gaps, v -> its place in
    domain, the domain's values ascending."""

    var: cp_model.IntVar
    domain: tuple[int, ...]

    def size_from(self, parts: Sequence[CpIndex]) -> int:
        _no_parts("Locate", parts)
        if not self.domain or list(self.domain) != sorted(set(self.domain)):
            raise ValueError(
                f"Locate needs distinct ascending values, got {self.domain}"
            )
        return len(self.domain)

    @property
    def entries(self) -> int:
        return len(self.domain)

    def positions(
        self, values: Values, index: CpIndex, parts: Sequence[np.ndarray]
    ) -> np.ndarray:
        return np.searchsorted(np.asarray(self.domain), values[self.var])

    def cp_expr(
        self, model: cp_model.CpModel, index: CpIndex, parts: Sequence[CpExpr]
    ) -> CpExpr:
        pos = model.new_int_var(0, index.size - 1, f"pos_{self.var.name}")
        model.add_element(pos, list(self.domain), self.var)
        return pos


@dataclass(frozen=True)
class Unit(IndexMap):
    """A leaf: the one position of a constant, on no input."""

    def size_from(self, parts: Sequence[CpIndex]) -> int:
        _no_parts("Unit", parts)
        return 1

    @property
    def entries(self) -> int:
        return 0

    def positions(
        self, values: Values, index: CpIndex, parts: Sequence[np.ndarray]
    ) -> np.ndarray:
        return np.zeros((), dtype=np.int64)

    def cp_expr(
        self, model: cp_model.CpModel, index: CpIndex, parts: Sequence[CpExpr]
    ) -> CpExpr:
        return 0


@dataclass(frozen=True)
class Combine(IndexMap):
    """(left, right) -> left * right.size + right. The right part's size is a
    constant, read from index.parts, so this is linear: no two variables are
    multiplied."""

    def size_from(self, parts: Sequence[CpIndex]) -> int:
        if len(parts) != 2:
            raise ValueError(f"Combine takes two parts, got {len(parts)}")
        return parts[0].size * parts[1].size

    @property
    def entries(self) -> int:
        return 0

    def positions(
        self, values: Values, index: CpIndex, parts: Sequence[np.ndarray]
    ) -> np.ndarray:
        return parts[0] * index.parts[1].size + parts[1]

    def cp_expr(
        self, model: cp_model.CpModel, index: CpIndex, parts: Sequence[CpExpr]
    ) -> CpExpr:
        return parts[0] * index.parts[1].size + parts[1]


@dataclass(frozen=True)
class CpIndex:
    """A map from the inputs' values to a position in one table, 0 .. size - 1: its
    mapping applied to its parts.

    A value, with no model in it. Two nodes that make the same position make equal
    CpIndexes, which hash alike. Made from
    the leaf steps by IndexMap steps alone, so the final index of a term is a function
    of every input it was made from.
    """

    mapping: IndexMap
    parts: tuple[CpIndex, ...] = ()

    def __post_init__(self) -> None:
        """Refuses parts its mapping does not take, by asking for size."""
        self.size

    @cached_property
    def size(self) -> int:
        """mapping.size_from(parts): computed, never passed in."""
        return self.mapping.size_from(self.parts)

    def positions(self, values: Values) -> np.ndarray:
        """The numpy side: its position at each assignment in values, its parts'
        positions first. cp_expr is the CP-SAT side of the same map."""
        parts = [part.positions(values) for part in self.parts]
        return self.mapping.positions(values, self, parts)

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The CP-SAT side: this position as an expression in model, its mapping's
        cp_expr on its parts' expressions, parts first, a part it reaches twice built
        once. Each call builds anew."""
        built: dict[CpIndex, CpExpr] = {}
        for step in self.under():
            parts = [built[part] for part in step.parts]
            built[step] = step.mapping.cp_expr(model, step, parts)
        return built[self]

    def under(self) -> list[CpIndex]:
        """This index and every index it is made from, each once, parts first."""
        seen: dict[CpIndex, None] = {}
        stack: list[tuple[CpIndex, bool]] = [(self, False)]
        while stack:
            index, expanded = stack.pop()
            if expanded:
                seen[index] = None
            elif index not in seen:
                stack.append((index, True))
                stack.extend((part, False) for part in reversed(index.parts))
        return list(seen)


UNIT = CpIndex(Unit())


def _read_only(values: Any, index: CpIndex) -> np.ndarray:
    array = np.array(values, dtype=float)
    if array.shape != (index.size,):
        raise ValueError(
            f"a list of shape {array.shape} on an index of size {index.size}"
        )
    array.setflags(write=False)
    return array


class CpForm(ABC):
    """What a node is realized as: the one thing a reader and the objective take of
    it. A value, with no model in it, that builds its own CP-SAT side. Each kind of
    form is a subclass: here a CpTable, and from Stage 3 a CpLinear too, the form a
    sum is realized as."""

    @abstractmethod
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The form in model, as a CP-SAT expression in COST_SCALE units."""


@dataclass(frozen=True, eq=False, kw_only=True)
class CpTable(CpForm):
    """A node's table: values[index], made together and never apart. values is the
    value at each position, float until cp_expr rounds it, stored as a read-only
    copy; a list whose length is not the index's size is refused. Frozen, with no
    model in it, and compared by identity, its list being an array."""

    values: np.ndarray
    index: CpIndex

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _read_only(self.values, self.index))

    @property
    def entries(self) -> int:
        """The entries the table puts in the model, what the cap bounds: its list,
        and the remaps under its index. Counted before any rounding, so a list that
        turns out constant still counts in full."""
        return len(self.values) + sum(i.mapping.entries for i in self.index.under())

    @property
    def finite(self) -> bool:
        """Whether every value in its list is a finite number."""
        return bool(np.all(np.isfinite(self.values)))

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The table in model, values[index] in COST_SCALE units: the list rounded
        once, within half a unit of its float value, then a number when it is
        constant, and otherwise one add_element on its index's position into a new
        variable whose domain is the rounded values. The position is built only when
        the table is not constant, so a constant table adds nothing to the model.
        Raises TableLoweringError for an entry that is not finite or past
        MAX_ENTRY."""
        if not self.finite:
            raise TableLoweringError(None, "a table entry that is not finite")
        scaled = np.rint(COST_SCALE * self.values)
        largest = float(np.max(np.abs(scaled)))
        if largest > MAX_ENTRY:
            raise TableLoweringError(
                None, f"a table entry of {largest:.3g}, past 2^50 = {MAX_ENTRY}"
            )
        entries = [int(v) for v in scaled]
        if len(set(entries)) == 1:
            return entries[0]
        target = model.new_int_var_from_domain(
            cp_model.Domain.from_values(sorted(set(entries))), "table"
        )
        model.add_element(self.index.cp_expr(model), entries, target)
        return target


# Each symbol's column, the symbol's value at each row of its input, on that input.
Inputs: TypeAlias = Mapping[sympy.Symbol, CpTable]


# ----------------------------------------------------------- values by lambdify


def _compile(expr: sympy.Basic) -> tuple[list[sympy.Symbol], Callable[..., Any]]:
    symbols = sorted(expr.free_symbols, key=sympy.default_sort_key)
    return symbols, sympy.lambdify(symbols, expr, modules="numpy")


def values_on_grid(
    expr: sympy.Basic,
    columns: Mapping[sympy.Symbol, CpTable],
    positions: Sequence[CpIndex],
) -> np.ndarray:
    """expr at every joint assignment of positions, row-major, the last position
    fastest: entry i1 * s1 + ... + im * sm, s_k the product of the sizes after k.

    Each free symbol of expr is read from its column in columns, at its own position,
    which must be one of positions; two symbols on one position move together. expr
    is lambdified with the numpy module on each call: a node makes one call, when it
    builds its table. A function that takes scalars only, as RelayoutCharge's
    ``_imp_`` does, is evaluated point by point. Raises TableLoweringError for a
    function with no numeric value.

    A value need not be finite: a Piecewise arm is infinite where it is not taken,
    and a Piecewise with no arm true is NaN. A list is refused for that only when it
    is read into the objective (CpNode.read).
    """
    shape = tuple(p.size for p in positions)
    grid = np.indices(shape).reshape(len(shape), -1) if shape else np.zeros((0, 1), int)
    count = grid.shape[1]
    axis = {p: k for k, p in enumerate(positions)}
    symbols, func = _compile(expr)
    arguments = []
    for symbol in symbols:
        table = columns[symbol]
        arguments.append(table.values[grid[axis[table.index]]])
    with np.errstate(all="ignore"):
        try:
            try:
                result = np.asarray(func(*arguments), dtype=float)
            except TypeError:
                result = np.array(
                    [func(*(a[i] for a in arguments)) for i in range(count)],
                    dtype=float,
                )
        except NameError as exc:
            raise TableLoweringError(expr, f"no numeric value ({exc})") from exc
    return np.broadcast_to(result, (count,)).astype(float)


# ------------------------------------------------------------------------ the walk


def children_of(expr: sympy.Basic) -> tuple[sympy.Basic, ...]:
    """A node's children: its arguments, except a Piecewise, whose children are its
    arms' values and conditions, and a RelayoutCharge, whose children are its literal
    and its division (its prices are data)."""
    if isinstance(expr, sympy.Piecewise):
        return tuple(part for pair in expr.args for part in pair.args)
    if isinstance(expr, RelayoutCharge):
        return expr.args[:2]
    return expr.args


@dataclass(eq=False)
class CpNode:
    """The CP-SAT counterpart of one sympy node, which realizes itself.

    expr is the sympy node, injected and only read: it says which operation the node
    is, and gives its arguments and its free symbols. The CpNodes form the same graph
    as the sympy nodes: children holds the CpNodes of children_of(expr), made before
    it, and a subexpression shared by two parents is one CpNode that both hold. A
    node reads only its children: the columns of its inputs are gathered from them.
    cap bounds its table. No model is involved until cp_expr.

    A node's form, a CpForm, is what a reader and the objective take of it: in this
    stage always its table, one CpTable. Its columns and its form are computed the first
    time they are read, and kept. This class is an operation over free symbols;
    CpSymbol and CpConstant are the leaves. A subclass changes how a node is
    realized by overriding realize.
    """

    expr: sympy.Basic
    children: tuple[CpNode, ...]
    cap: int

    @cached_property
    def columns(self) -> Mapping[sympy.Symbol, CpTable]:
        """Each free symbol's column on its input, gathered from the children."""
        columns: dict[sympy.Symbol, CpTable] = {}
        for child in self.children:
            columns.update(child.columns)
        return columns

    @cached_property
    def form(self) -> CpForm:
        """The node's form, realized the first time it is read; realize raises
        TableLoweringError naming the node when it has none."""
        return self.realize()

    def realize(self) -> CpForm:
        """Realizes the node: one table over its inputs, their positions combined in
        the order of their symbols sorted by sympy.default_sort_key, the last
        fastest. Raises TableLoweringError naming the node when that table would pass
        the cap, one past the cap never evaluated, and when a value of it is not
        finite."""
        symbols = sorted(self.columns, key=sympy.default_sort_key)
        positions = list(dict.fromkeys(self.columns[s].index for s in symbols))
        under = {i: None for p in positions for i in p.under()}
        entries = math.prod(p.size for p in positions) + sum(
            i.mapping.entries for i in under
        )
        if entries > self.cap:
            raise TableLoweringError(
                self.expr,
                f"one table over its inputs would hold {entries} entries, past the "
                f"cap of {self.cap}",
            )
        index = positions[0]
        for position in positions[1:]:
            index = CpIndex(Combine(), (index, position))
        values = values_on_grid(self.expr, self.columns, positions)
        return self._finite(CpTable(values=values, index=index))

    def _finite(self, table: CpTable) -> CpTable:
        """table, refused naming the node when a value of it is not finite."""
        if not table.finite:
            raise TableLoweringError(self.expr, "a value that is not finite")
        return table

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The node in model, in COST_SCALE units: its form built
        (CpTable.cp_expr), one lookup or a number. Reading the form realizes the
        node, so a refusal comes before anything is added to the model. Each call
        builds anew."""
        return self.form.cp_expr(model)


@dataclass(eq=False)
class CpSymbol(CpNode):
    """A symbol: its column on its input, injected by the factory from the inputs,
    or None for a symbol with no input, which raises when it is read."""

    column: CpTable | None = None

    @cached_property
    def columns(self) -> Mapping[sympy.Symbol, CpTable]:
        """Its own column. Raises TableLoweringError naming the symbol when it has no
        input, so a parent that reads it is refused by the symbol's name."""
        if self.column is None:
            raise TableLoweringError(self.expr, "a symbol with no input")
        return {self.expr: self.column}

    def realize(self) -> CpTable:
        """Its column."""
        (column,) = self.columns.values()
        return self._finite(column)


@dataclass(eq=False)
class CpConstant(CpNode):
    """A node with no free symbol: a number, true or false, or an operation on
    numbers. Its form is its value on the Unit position."""

    def realize(self) -> CpTable:
        """Its value, one entry on the Unit position."""
        value = np.array([_constant(self.expr)])
        return self._finite(CpTable(values=value, index=UNIT))


class TableLowering:
    """Plans cost expressions as tables: a factory for CpNodes. It makes one CpNode
    per node, of the class the node's sympy class calls for, and hands it what it
    reads; the nodes do the rest. It holds no model, so planning cannot write to one,
    and it keeps nothing between calls of lower.

    inputs gives each symbol's column on its input, which the factory hands to the
    symbol's CpSymbol; cap bounds the entries of one table, its list and the remaps
    under its index, and is handed to every node.
    """

    def __init__(self, inputs: Inputs, *, cap: int) -> None:
        self.inputs = inputs
        self.cap = cap

    def lower(self, root: sympy.Basic) -> CpNode:
        """root as a plan: root's CpNode, over the graph of nodes under it. Nothing is
        realized here; the root's cp_expr reads it, which realizes it, and through
        it every node it reads."""
        return self.rewrite(root)[root]

    def rewrite(self, root: sympy.Basic) -> dict[sympy.Basic, CpNode]:
        """Each node under root, once, as a CpNode holding its children's CpNodes,
        made after them, in the order of sympy.postorder_traversal with its repeats
        dropped. An arm's ExprCondPair gets no CpNode. Nothing is realized here;
        root is not changed."""
        nodes: dict[sympy.Basic, CpNode] = {}
        for expr in dict.fromkeys(sympy.postorder_traversal(root)):
            if isinstance(expr, ExprCondPair):
                continue
            nodes[expr] = self._node(expr, tuple(nodes[c] for c in children_of(expr)))
        return nodes

    def _node(self, expr: sympy.Basic, children: tuple[CpNode, ...]) -> CpNode:
        """expr's CpNode: a CpSymbol holding its input's column, a CpConstant for a
        node with no free symbol, and a CpNode for every other node."""
        if isinstance(expr, sympy.Symbol):
            return CpSymbol(expr, children, self.cap, column=self.inputs.get(expr))
        if not expr.free_symbols:
            return CpConstant(expr, children, self.cap)
        return CpNode(expr, children, self.cap)


def _constant(expr: sympy.Basic) -> float:
    if expr is sympy.true or expr is sympy.false:
        return float(bool(expr))
    try:
        return float(expr)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TableLoweringError(expr, "a constant with no numeric value") from exc
