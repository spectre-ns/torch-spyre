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

- values: ``CpIndex``, ``CpLinear`` and ``CpTable`` are frozen and hold no model;
- the walk: ``TableLowering`` makes one ``CpNode`` per node of the expression, and
  each node tabulates itself from its children. The plan it returns, a ``CpLinear``,
  holds no model either, so a refusal cannot touch one;
- the build: ``CpEmitter`` rounds the plan onto ``COST_SCALE`` and is the only code
  that calls CP-SAT.

This is the base: one table per node, every value computed by ``sympy.lambdify`` over
the node's inputs, and the root read as one lookup. It lowers an expression whose
inputs fit the cap together and refuses anything wider. Nothing imports it yet.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
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
    0 .. size - 1, in numpy and in CP-SAT. Each kind of step is a subclass; CpIndex,
    the walk and the emitter do not change when one is added."""

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
    CpIndexes, which hash alike, so an emitter builds the position once. Made from
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
        positions first. CpEmitter.index is the CP-SAT side of the same map."""
        parts = [part.positions(values) for part in self.parts]
        return self.mapping.positions(values, self, parts)

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


@dataclass(frozen=True, eq=False, kw_only=True)
class CpLinear:
    """A node read linearly: the float form an emitter collapses into a CP-SAT
    expression, a constant plus one list per index. Nothing in it is rounded, and
    nothing in it changes once it is made: its lists are read-only copies. Compared by
    identity, its lists being arrays."""

    constant: float = 0.0
    lists: Mapping[CpIndex, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuses a list whose length is not its index's size, and stores read-only
        copies."""
        lists = {
            index: _read_only(values, index) for index, values in self.lists.items()
        }
        object.__setattr__(self, "lists", MappingProxyType(lists))

    @property
    def entries(self) -> int:
        """The entries the form puts in the model: its lists, and the remaps under
        their indexes, each once. Counted before any rounding, so a list that turns
        out constant still counts in full."""
        under = {index: None for key in self.lists for index in key.under()}
        return sum(len(v) for v in self.lists.values()) + sum(
            index.mapping.entries for index in under
        )


class CpTable(CpLinear):
    """A node read as a table: values[index], made together and never apart. It is a
    CpLinear with one list on one index and nothing else, so a table is read linearly
    as it is. Its entries, the list plus the remaps under its index, are what the cap
    bounds."""

    def __init__(self, *, values: Any, index: CpIndex) -> None:
        super().__init__(lists={index: values})

    @property
    def values(self) -> np.ndarray:
        """Its one list: the value at each position, float until collapsed."""
        (values,) = self.lists.values()
        return values

    @property
    def index(self) -> CpIndex:
        """The index of its one list."""
        (index,) = self.lists
        return index


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
    is read into the objective (CpNode.as_linear).
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


@dataclass(frozen=True)
class Candidate:
    """One way to tabulate a node: what it is, the entries it would hold, known
    before its list is computed, and how to build it. A candidate past the cap is
    never built."""

    name: str
    entries: int
    build: Callable[[], CpTable]


@dataclass(eq=False)
class CpNode:
    """The CP-SAT counterpart of one sympy node, which tabulates itself.

    expr is the sympy node, injected and only read: it says which operation the node
    is, and gives its arguments and its free symbols. The CpNodes form the same graph
    as the sympy nodes: children holds the CpNodes of children_of(expr), made and
    visited before it, and a subexpression shared by two parents is one CpNode that
    both hold. A node reads only its children: the columns of its inputs are
    gathered from them. visit_op sets what a parent reads. No model is involved.

    This class is an operation over free symbols; CpSymbol and CpConstant are the
    leaves. A subclass changes how a node is tabulated by overriding candidates.
    """

    expr: sympy.Basic
    children: tuple[CpNode, ...]
    table: CpTable | None = field(default=None, init=False)

    @cached_property
    def columns(self) -> Mapping[sympy.Symbol, CpTable]:
        """Each free symbol's column on its input, gathered from the children."""
        columns: dict[sympy.Symbol, CpTable] = {}
        for child in self.children:
            columns.update(child.columns)
        return columns

    @cached_property
    def positions(self) -> list[CpIndex]:
        """The distinct positions of the node's columns, in the order of their symbols
        sorted by sympy.default_sort_key."""
        symbols = sorted(self.columns, key=sympy.default_sort_key)
        return list(dict.fromkeys(self.columns[s].index for s in symbols))

    def visit_op(self, cap: int) -> None:
        """The walk on this node, once, after its children: sets table. Raises
        TableLoweringError naming the node when it ends with no table."""
        self.table = self.tabulate(cap)
        if self.table is None:
            raise TableLoweringError(self.expr, self.refusal(cap))

    def tabulate(self, cap: int) -> CpTable | None:
        """The node's table: of the candidates under cap, the one with the fewest
        entries (rule 3), built; None when none fits. A candidate past cap is never
        built."""
        fits = [c for c in self.candidates() if c.entries <= cap]
        if not fits:
            return None
        return min(fits, key=lambda c: c.entries).build()

    def candidates(self) -> list[Candidate]:
        """The ways to tabulate the node. Here one: one table over its inputs, their
        positions combined in the order of positions, the last fastest."""
        under = {i: None for p in self.positions for i in p.under()}
        entries = math.prod(p.size for p in self.positions) + sum(
            i.mapping.entries for i in under
        )
        return [Candidate("one table over its inputs", entries, self._one_table)]

    def _one_table(self) -> CpTable:
        index = self.positions[0]
        for position in self.positions[1:]:
            index = CpIndex(Combine(), (index, position))
        values = values_on_grid(self.expr, self.columns, self.positions)
        return CpTable(values=values, index=index)

    def refusal(self, cap: int) -> str:
        """Why the node has no table under cap."""
        smallest = min(self.candidates(), key=lambda c: c.entries, default=None)
        if smallest is None:
            return "no way to tabulate it"
        return (
            f"{smallest.name} would hold {smallest.entries} entries, past the cap "
            f"of {cap}"
        )

    def as_linear(self) -> CpLinear:
        """The node read linearly: its table, a CpTable being a CpLinear. Raises
        TableLoweringError naming the node when a value of it is not finite: an
        infinite or NaN entry may sit in a list that a parent discards, but not in one
        read into the objective."""
        if self.table is None:
            raise TableLoweringError(self.expr, "read before it was visited")
        if not np.all(np.isfinite(self.table.values)):
            raise TableLoweringError(self.expr, "a value that is not finite")
        return self.table


@dataclass(eq=False)
class CpSymbol(CpNode):
    """A symbol: its column on its input, injected by the rewrite from the inputs,
    or None for a symbol with no input."""

    column: CpTable | None = None

    @cached_property
    def columns(self) -> Mapping[sympy.Symbol, CpTable]:
        """Its own column, or none."""
        return {} if self.column is None else {self.expr: self.column}

    def candidates(self) -> list[Candidate]:
        """Its column, the one way to tabulate it; none without an input."""
        column = self.column
        if column is None:
            return []
        return [Candidate("its column", column.entries, lambda: column)]

    def refusal(self, cap: int) -> str:
        if self.column is None:
            return "a symbol with no input"
        return super().refusal(cap)


@dataclass(eq=False)
class CpConstant(CpNode):
    """A node with no free symbol: a number, true or false, or an operation on
    numbers. Its table is its value on the Unit position."""

    def candidates(self) -> list[Candidate]:
        """Its value, one entry on the Unit position."""
        return [Candidate("its value", 1, self._value)]

    def _value(self) -> CpTable:
        return CpTable(values=[_constant(self.expr)], index=UNIT)


class TableLowering:
    """Plans cost expressions as tables: the walk that makes and visits one CpNode
    per node, each of which tabulates itself. It holds no model, so the walk cannot
    write to one, and it keeps nothing between calls of lower.

    inputs gives each symbol's column on its input, which the rewrite hands to the
    symbol's CpSymbol; cap bounds the entries of one table, its list and the remaps
    under its index, and is handed to each node's visit_op.
    """

    def __init__(self, inputs: Inputs, *, cap: int) -> None:
        self.inputs = inputs
        self.cap = cap

    def lower(self, root: sympy.Basic) -> CpLinear:
        """root as a plan: the form the objective reads, the root's table."""
        return self.rewrite(root)[root].as_linear()

    def rewrite(self, root: sympy.Basic) -> dict[sympy.Basic, CpNode]:
        """Each node under root, once, as a CpNode holding its children's CpNodes,
        visited after them, in the order of sympy.postorder_traversal with its repeats
        dropped. An arm's ExprCondPair gets no CpNode. Recursion-free; root is not
        changed."""
        nodes: dict[sympy.Basic, CpNode] = {}
        for expr in dict.fromkeys(sympy.postorder_traversal(root)):
            if isinstance(expr, ExprCondPair):
                continue
            node = self._node(expr, tuple(nodes[c] for c in children_of(expr)))
            node.visit_op(self.cap)
            nodes[expr] = node
        return nodes

    def _node(self, expr: sympy.Basic, children: tuple[CpNode, ...]) -> CpNode:
        """expr's CpNode: a CpSymbol holding its input's column, a CpConstant for a
        node with no free symbol, and a CpNode for every other node."""
        if isinstance(expr, sympy.Symbol):
            return CpSymbol(expr, children, column=self.inputs.get(expr))
        if not expr.free_symbols:
            return CpConstant(expr, children)
        return CpNode(expr, children)


def _constant(expr: sympy.Basic) -> float:
    if expr is sympy.true or expr is sympy.false:
        return float(bool(expr))
    try:
        return float(expr)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TableLoweringError(expr, "a constant with no numeric value") from exc


# ----------------------------------------------------------------------- the build


class CpEmitter:
    """Builds a plan into one CP-SAT model: the only code that calls CP-SAT. It owns
    the model and the memo of what it built, so a position, and a form, is built
    once per emitter."""

    def __init__(self, model: cp_model.CpModel) -> None:
        self.model = model
        self._indexes: dict[CpIndex, CpExpr] = {}
        self._forms: dict[CpLinear, CpExpr] = {}

    def index(self, index: CpIndex) -> CpExpr:
        """index's position as a CP-SAT expression: its mapping's cp_expr on its
        parts' expressions, parts first. Built once per equal CpIndex."""
        for step in index.under():
            if step not in self._indexes:
                parts = [self._indexes[part] for part in step.parts]
                self._indexes[step] = step.mapping.cp_expr(self.model, step, parts)
        return self._indexes[index]

    def table(self, values: np.ndarray, index: CpIndex) -> CpExpr:
        """values[index] as a CP-SAT expression: the list rounded once onto
        COST_SCALE, then a number when it is constant, and otherwise one add_element
        on self.index(index) into a new variable whose domain is the rounded values.
        Raises TableLoweringError for an entry that is not finite or past
        MAX_ENTRY."""
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise TableLoweringError(None, "a table entry that is not finite")
        scaled = np.rint(COST_SCALE * values)
        largest = float(np.max(np.abs(scaled)))
        if largest > MAX_ENTRY:
            raise TableLoweringError(
                None, f"a table entry of {largest:.3g}, past 2^50 = {MAX_ENTRY}"
            )
        entries = [int(v) for v in scaled]
        if len(set(entries)) == 1:
            return entries[0]
        target = self.model.new_int_var_from_domain(
            cp_model.Domain.from_values(sorted(set(entries))), "table"
        )
        self.model.add_element(self.index(index), entries, target)
        return target

    def linear(self, form: CpLinear) -> CpExpr:
        """form as a CP-SAT expression in COST_SCALE units, built once per form: its
        constant, plus self.table of each list. A sum of T lists and a constant is
        within (T + 1) / 2 units of its float value."""
        if form not in self._forms:
            constant = self.table(np.array([form.constant]), UNIT)
            terms = [self.table(v, index) for index, v in form.lists.items()]
            self._forms[form] = sum(terms, start=constant)
        return self._forms[form]
