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
works out which entry of a table applies: ``value == values[index]``. Three layers:

- values: ``Index``, a position, and ``AbstractRealization``, what a node is realized
  as: a list on a position or a linear form of its operands;
- the printer: ``TableLowering``, a sympy ``Printer`` whose print of a node is the
  node's realization. It prints a node's operands first, reads their tables, and
  computes the node's list on a position it builds from the node's inputs;
- the build: a realization's ``cp_expr`` rounds it onto ``COST_SCALE`` and adds it to
  the model. Each ``cp_expr`` but a leaf position's is a ``memoize_method``, so a
  position or realization held by several readers is built once per model. Only
  ``cp_expr`` methods touch the model.

A sum stays linear, so no table crosses two of its terms. A product of several
operands is gated by one that reads a boolean literal only and is 0 at one of its
values: its rest is the product with the literal at the other, so no table reads the
literal. A relayout charge is gated by its literal too. Any other product that
divides is one table, divided once. Any other product is a switch on one of its
operands where that has fewer entries than its one table, so no table crosses that
factor and the rest. Any other node is a table, and a parent that reads a sum, a
switch or a gate as a list reads its table. A table is over the node's inputs, over
its operands' positions, each ranked to its distinct values, or, for a Min or Max, by
rung: its ladder at the min or max of its operands' positions on it. It takes the way
with the fewest entries (rule 3), the next where that one's list is not finite, and
is refused when none fits the cap. Its list is the numpy core's function of its sympy
class (``cost_expr_numpy.py``, which the CP-SAT printer evaluates its candidates
with) applied to its operands' tables; a class the core has no function for is
refused. Nothing imports the module yet.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import reduce
from typing import Any, Self, TypeAlias

import numpy as np
import sympy
from ortools.sat.python import cp_model
from sympy.logic.boolalg import Boolean
from sympy.printing.printer import Printer

from torch_spyre._inductor.scratchpad.cost_expr_numpy import (
    divides,
    factors,
    numpy_eval,
    operands,
)
from torch_spyre._inductor.scratchpad.cpsat_expr_printer import memoize_method

# One cost unit is COST_SCALE objective units: a table entry is
# rint(COST_SCALE * value).
COST_SCALE = 1024
# The largest table entry: a sum of thousands of entries stays inside int64.
MAX_ENTRY = 2**50
# A switch writes a fraction p / q, q <= MAX_DENOMINATOR, each p * w < MAX_PRODUCT.
MAX_DENOMINATOR = 2**24
MAX_PRODUCT = 2**60
# A Min or Max whose parent reads it linearly stays linear where that needs under
# 1/LINEAR_MARGIN of its table's entries (rule 3): a table searches faster on a near
# tie.
LINEAR_MARGIN = 2

# What CP-SAT takes as a linear expression: a LinearExpr, an IntVar or an int.
CpExpr: TypeAlias = cp_model.LinearExprT


class TableLoweringError(Exception):
    """An objective the tables cannot lower. ``node`` is the sympy node refused, or
    None when the refusal names none; ``reason`` says why.

    It subclasses none of RuntimeError, TypeError and ValueError, so an ``except``
    around the objective that catches those cannot turn it into a different solve.
    """

    def __init__(self, node: sympy.Basic | None, reason: str) -> None:
        super().__init__(reason if node is None else f"{reason}: {node}")
        self.node = node
        self.reason = reason


# --------------------------------------------------------------------------- values


class Index(ABC):
    """A position: a map from the inputs' values to 0 .. size - 1. Each kind of step is
    a frozen dataclass holding the positions it is made from, so equal steps are one
    position."""

    size: int  # the number of positions

    @abstractmethod
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The position in model, its parts read through their cp_expr. A step's is a
        memoize_method, built once per model; a leaf's adds nothing to the model, so
        it is not."""

    @abstractmethod
    def entries(self) -> int:
        """The entries of the lookups it is read through."""

    @abstractmethod
    def inputs(self) -> tuple[InputIndex, ...]:
        """The input positions it is made from."""

    @abstractmethod
    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        """The position at every point of grid, which holds each input position's
        coordinate there: what cp_expr builds, computed."""


@dataclass(frozen=True)
class InputIndex(Index):
    """A leaf: a variable's own position, v - lo, for a variable whose values run from
    lo to lo + size - 1."""

    var: cp_model.IntVar
    size: int
    lo: int = 0

    def entries(self) -> int:
        return 0

    def inputs(self) -> tuple[InputIndex, ...]:
        return (self,)

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        return self.var - self.lo if self.lo else self.var

    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return grid[self]


@dataclass(frozen=True)
class UnitIndex(Index):
    """A leaf: the one position of a constant, on no input."""

    size: int = field(default=1, init=False)

    def entries(self) -> int:
        return 0

    def inputs(self) -> tuple[InputIndex, ...]:
        return ()

    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        return 0

    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return 0


@dataclass(frozen=True)
class CombineIndex(Index):
    """left * right.size + right: linear, as right.size is a constant."""

    left: Index
    right: Index
    size: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", self.left.size * self.right.size)

    def entries(self) -> int:
        return self.left.entries() + self.right.entries()

    def inputs(self) -> tuple[InputIndex, ...]:
        return self.left.inputs() + self.right.inputs()

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        left = self.left.cp_expr(model)
        return left * self.right.size + self.right.cp_expr(model)

    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return self.left.on_grid(grid) * self.right.size + self.right.on_grid(grid)


@dataclass(frozen=True)
class CompactIndex(Index):
    """remap[parent]: one lookup onto fewer positions, as an entry's value's rank."""

    parent: Index
    remap: tuple[int, ...]
    size: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", max(self.remap) + 1)

    def entries(self) -> int:
        return len(self.remap) + self.parent.entries()

    def inputs(self) -> tuple[InputIndex, ...]:
        return self.parent.inputs()

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        rank = model.new_int_var(0, self.size - 1, "rank")
        model.add_element(self.parent.cp_expr(model), list(self.remap), rank)
        return rank

    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        return np.asarray(self.remap)[self.parent.on_grid(grid)]


@dataclass(frozen=True)
class ExtremeIndex(Index):
    """The max of rungs, positions on one ascending ladder of size values, or their
    min where is_min: the ladder keeps order, so the Max of the rungs is the rung of
    the Max."""

    rungs: tuple[Index, ...]
    is_min: bool
    size: int

    def entries(self) -> int:
        return sum(rung.entries() for rung in self.rungs)

    def inputs(self) -> tuple[InputIndex, ...]:
        return tuple(i for rung in self.rungs for i in rung.inputs())

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        rung = model.new_int_var(0, self.size - 1, "rung")
        rungs = [r.cp_expr(model) for r in self.rungs]
        if self.is_min:
            model.add_min_equality(rung, rungs)
        else:
            model.add_max_equality(rung, rungs)
        return rung

    def on_grid(self, grid: Mapping[Index, np.ndarray]) -> Any:
        pick = np.minimum if self.is_min else np.maximum
        return reduce(pick, (rung.on_grid(grid) for rung in self.rungs))


def _read_only(values: Any, index: Index) -> np.ndarray:
    """values as a read-only list of index's size: bool when they are bool, as a
    condition's are, and float otherwise."""
    array = np.asarray(values)
    array = np.array(array, dtype=bool if array.dtype == bool else float)
    if array.shape != (index.size,):
        raise ValueError(
            f"a list of shape {array.shape} on an index of size {index.size}"
        )
    array.setflags(write=False)
    return array


def _rounded(values: Any) -> list[int]:
    """values rounded onto COST_SCALE."""
    scaled = np.rint(COST_SCALE * np.asarray(values, dtype=float))
    largest = float(np.max(np.abs(scaled)))
    if largest > MAX_ENTRY:
        raise TableLoweringError(
            None, f"a table entry of {largest:.3g}, past 2^50 = {MAX_ENTRY}"
        )
    return [int(v) for v in scaled]


class AbstractRealization(ABC):
    """What the printer returns for a node. It holds only finite values."""

    @abstractmethod
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """The realization in model, in COST_SCALE units, its operands read through
        their cp_expr. Each kind's is a memoize_method: a realization that several
        readers hold is one expression per model."""

    @abstractmethod
    def table(self, cap: int) -> TableRealization:
        """The realization as one table, for a parent that computes its own list from
        it. cap bounds the entries of a table that has to be made."""

    @abstractmethod
    def scaled(self, factor: float) -> AbstractRealization:
        """factor times the realization, applied in float before any rounding."""

    @abstractmethod
    def bounds(self) -> tuple[int, int]:
        """Its lowest and highest value in a model."""

    @abstractmethod
    def entries(self) -> int:
        """The table entries it adds to a model, which a node's choice weighs."""

    def sum_key(self) -> Hashable:
        """In a sum, terms with one key are one term (joined). By default the
        realization itself: a term of its own."""
        return self

    def joined(self, others: Sequence[Self]) -> AbstractRealization:
        """It and others, which have its sum_key, as one term. By default others are
        itself again, so the term is it times their count."""
        return self.scaled(1 + len(others))


@dataclass(frozen=True, eq=False, kw_only=True)
class TableRealization(AbstractRealization):
    """values[index]: a read-only list, one value per entry of index: bool for a
    condition, float for any other node."""

    values: np.ndarray
    index: Index

    def __post_init__(self) -> None:
        values = _read_only(self.values, self.index)
        if not np.all(np.isfinite(values)):
            raise TableLoweringError(None, "a value that is not finite")
        object.__setattr__(self, "values", values)

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """A constant or affine list adds no table; any other is one add_element."""
        entries = _rounded(self.values)
        step = entries[1] - entries[0] if len(entries) > 1 else 0
        if all(entry == entries[0] + i * step for i, entry in enumerate(entries)):
            if step == 0:
                return entries[0]
            return entries[0] + step * self.index.cp_expr(model)
        target = model.new_int_var_from_domain(
            cp_model.Domain.from_values(sorted(set(entries))), "table"
        )
        model.add_element(self.index.cp_expr(model), entries, target)
        return target

    def table(self, cap: int) -> TableRealization:
        """Itself: none is made, so cap is not read."""
        return self

    def scaled(self, factor: float) -> TableRealization:
        with np.errstate(over="ignore"):  # inf is refused by TableRealization
            return TableRealization(values=factor * self.values, index=self.index)

    def bounds(self) -> tuple[int, int]:
        entries = _rounded(self.values)
        return min(entries), max(entries)

    def entries(self) -> int:
        """Its list, counted before the rounding, and its position's lookups: none
        when it is one value."""
        if np.all(self.values == self.values[0]):
            return 0
        return len(self.values) + self.index.entries()

    def sum_key(self) -> Index:
        """Its position: lists on one position are added in float, one lookup and one
        rounding."""
        return self.index

    def joined(self, others: Sequence[Self]) -> TableRealization:
        with np.errstate(over="ignore"):  # inf is refused by TableRealization
            values = sum((other.values for other in others), self.values)
        return TableRealization(values=values, index=self.index)


class SymbolicRealization(AbstractRealization):
    """No list of its own: CP-SAT builds it from its operands' expressions."""

    def table(self, cap: int) -> TableRealization:
        """Its cheapest table from its parts' tables (_tabulated), or one _made."""
        parts = self._parts(cap)
        with np.errstate(over="ignore"):  # inf is refused by TableRealization
            return _tabulated(
                parts,
                cap,
                lambda _, lists: self._values_from_parts(lists),
                self._made(parts, cap),
            )

    @abstractmethod
    def _parts(self, cap: int) -> list[TableRealization]:
        """The tables it is a function of."""

    def _made(
        self, parts: list[TableRealization], cap: int
    ) -> tuple[TableRealization, ...]:
        """Its tables from parts that are no way of _tabulated's: none by default."""
        return ()

    @abstractmethod
    def _values_from_parts(self, lists: list[np.ndarray]) -> np.ndarray:
        """Its value from its parts' lists, each on one position."""


@dataclass(frozen=True, eq=False)
class SumRealization(SymbolicRealization):
    """sum(terms). A constant is a term on a UnitIndex."""

    terms: tuple[AbstractRealization, ...]

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        total = 0
        exprs = []
        for term in self.terms:
            expr = term.cp_expr(model)
            if isinstance(expr, int):
                total += expr
            else:
                exprs.append(expr)
        return cp_model.LinearExpr.sum(exprs) + total if exprs else total

    def _parts(self, cap: int) -> list[TableRealization]:
        return [term.table(cap) for term in self.terms]

    def _values_from_parts(self, lists: list[np.ndarray]) -> np.ndarray:
        return sum(lists[1:], lists[0])

    def scaled(self, factor: float) -> SumRealization:
        terms = tuple(term.scaled(factor) for term in self.terms)
        return SumRealization(terms)

    def bounds(self) -> tuple[int, int]:
        ends = [term.bounds() for term in self.terms]
        return sum(end[0] for end in ends), sum(end[1] for end in ends)

    def entries(self) -> int:
        return sum(term.entries() for term in self.terms)


@dataclass(frozen=True, eq=False)
class SwitchRealization(SymbolicRealization):
    """A product as factor * rest, the factor's value values[index] (rule 4). The rest
    is collapsed once into w, and one true/false per value, on where the factor takes
    it, sets g to the value times w. No table crosses factor and rest."""

    values: tuple[float, ...]
    index: Index
    rest: AbstractRealization

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """A whole value is g == v * w; a fraction, p / q with q = denominator(), is
        the floor: q * g <= p * w <= q * g + q - 1. A rest of several terms is one
        variable w, so p multiplies one coefficient, not each term's: CP-SAT 9.15's
        presolve calls a model with p times a list's step past 2^31 infeasible."""
        w = _one_variable(model, self.rest, "rest")
        q = self.denominator()
        cases = [model.new_bool_var("case") for _ in self.values]
        model.add_map_domain(self.index.cp_expr(model), cases)
        g = model.new_int_var(*self.bounds(), "switch")
        for value, case in zip(self.values, cases):
            if value.is_integer():
                model.add(g == int(value) * w).only_enforce_if(case)
            else:
                p = round(value * q)
                model.add(q * g <= p * w).only_enforce_if(case)
                model.add(p * w <= q * g + q - 1).only_enforce_if(case)
        return g

    def _parts(self, cap: int) -> list[TableRealization]:
        factor = TableRealization(values=self.values, index=self.index)
        return [factor, self.rest.table(cap)]

    def _values_from_parts(self, lists: list[np.ndarray]) -> np.ndarray:
        return lists[0] * lists[1]

    def scaled(self, factor: float) -> SwitchRealization:
        rest = self.rest.scaled(factor)
        return SwitchRealization(self.values, self.index, rest)

    @memoize_method
    def bounds(self) -> tuple[int, int]:
        """Memoized, and its rest's read once: a switch inside a switch reads it
        again, so a chain of switches is otherwise exponential in its depth."""
        q = self.denominator()
        rest = self.rest.bounds()
        ends = [
            int(value) * end if value.is_integer() else round(value * q) * end // q
            for value in self.values
            for end in rest
        ]
        return min(ends), max(ends)

    def entries(self) -> int:
        return self.rest.entries() + len(self.values) + self.index.entries()

    def sum_key(self) -> tuple[Index, tuple[float, ...]]:
        """Its factor: switches on one factor are one switch over their rests' sum."""
        return self.index, self.values

    def joined(self, others: Sequence[Self]) -> SwitchRealization:
        rest = _sum([switch.rest for switch in (self, *others)])
        return SwitchRealization(self.values, self.index, rest)

    def denominator(self) -> int:
        """q: 1 when every value is whole, otherwise the largest power of two up to
        MAX_DENOMINATOR that keeps every p * w under MAX_PRODUCT."""
        widest = max(abs(end) for end in self.rest.bounds())
        q = 1 if all(v.is_integer() for v in self.values) else MAX_DENOMINATOR
        while max(abs(round(v * q)) for v in self.values) * widest >= MAX_PRODUCT:
            if q == 1:
                raise TableLoweringError(None, "a switched product past 2^60")
            q //= 2
        return q


@dataclass(frozen=True, eq=False)
class ExtremeRealization(SymbolicRealization):
    """A Min (is_min) or Max kept linear: one lin_min or lin_max of its operands, each
    behind a variable of its own. Its parent reads it linearly, as a sum, a constant,
    a switch's or a gate's rest, or another Min or Max kept linear does; any other
    parent reads its table."""

    is_min: bool
    operands: tuple[AbstractRealization, ...]

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """Each operand one variable: a long sum inside lin_max made CP-SAT's presolve
        run out of memory."""
        operands = [_one_variable(model, o, "term") for o in self.operands]
        target = model.new_int_var(*self.bounds(), "min" if self.is_min else "max")
        if self.is_min:
            model.add_min_equality(target, operands)
        else:
            model.add_max_equality(target, operands)
        return target

    def _parts(self, cap: int) -> list[TableRealization]:
        return [operand.table(cap) for operand in self.operands]

    def _values_from_parts(self, lists: list[np.ndarray]) -> np.ndarray:
        return reduce(np.minimum if self.is_min else np.maximum, lists)

    def _made(
        self, parts: list[TableRealization], cap: int
    ) -> tuple[TableRealization, ...]:
        return (_by_rung(parts, self.is_min),)

    def scaled(self, factor: float) -> ExtremeRealization:
        """A negative factor turns a Min into a Max."""
        operands = tuple(operand.scaled(factor) for operand in self.operands)
        return ExtremeRealization(self.is_min == (factor >= 0), operands)

    def bounds(self) -> tuple[int, int]:
        ends = [operand.bounds() for operand in self.operands]
        pick = min if self.is_min else max
        return pick(lo for lo, _ in ends), pick(hi for _, hi in ends)

    def entries(self) -> int:
        return sum(operand.entries() for operand in self.operands)


def _one_variable(
    model: cp_model.CpModel, realization: AbstractRealization, name: str
) -> CpExpr:
    """realization's expression in model as one variable, or the number it is."""
    expr = realization.cp_expr(model)
    if isinstance(expr, (int, cp_model.IntVar)):
        return expr
    var = model.new_int_var(*realization.bounds(), name)
    model.add(var == expr)
    return var


@dataclass(frozen=True, eq=False)
class GateRealization(SymbolicRealization):
    """rest where a 0/1 literal is at kept, 0 or 1, and 0 where it is at its other
    value (rule 5). The literal is the true/false of the two cases, and rest does not
    read it."""

    literal: InputIndex
    kept: int
    rest: AbstractRealization

    @memoize_method
    def cp_expr(self, model: cp_model.CpModel) -> CpExpr:
        """g == w where the term is kept, g == 0 where it is not."""
        w = self.rest.cp_expr(model)
        g = model.new_int_var(*self.bounds(), "gate")
        var = self.literal.var
        on, off = (var, var.negated()) if self.kept else (var.negated(), var)
        model.add(g == w).only_enforce_if(on)
        model.add(g == 0).only_enforce_if(off)
        return g

    def _parts(self, cap: int) -> list[TableRealization]:
        literal = TableRealization(values=[0.0, 1.0], index=self.literal)
        return [literal, self.rest.table(cap)]

    def _values_from_parts(self, lists: list[np.ndarray]) -> np.ndarray:
        return np.where(lists[0] == self.kept, lists[1], 0.0)

    def scaled(self, factor: float) -> GateRealization:
        return replace(self, rest=self.rest.scaled(factor))

    def bounds(self) -> tuple[int, int]:
        lo, hi = self.rest.bounds()
        return min(lo, 0), max(hi, 0)

    def entries(self) -> int:
        return self.rest.entries()

    def sum_key(self) -> tuple[InputIndex, int]:
        """Its literal and kept value: gates on them are one gate over their rests'
        sum."""
        return self.literal, self.kept

    def joined(self, others: Sequence[Self]) -> GateRealization:
        return replace(self, rest=_sum([gate.rest for gate in (self, *others)]))


# --------------------------------------------------------------------- a node's list


def _values(expr: sympy.Basic, count: int, compute: Callable[[], Any]) -> np.ndarray:
    """expr's list of count values, from compute(): bool for a condition, float
    otherwise. A RecursionError is left to TableLowering.print."""
    dtype = bool if _is_condition(expr) else float
    with np.errstate(all="ignore"):
        try:
            result = np.asarray(compute(), dtype=dtype)
        except RecursionError:
            raise
        except Exception as exc:
            raise TableLoweringError(
                expr, f"no value ({type(exc).__name__}: {exc})"
            ) from exc
    return np.broadcast_to(result, (count,)).astype(dtype)


def _combined(positions: Sequence[Index]) -> Index:
    """positions as one position, the last fastest; a UnitIndex for none."""
    index = positions[0] if positions else UnitIndex()
    for position in positions[1:]:
        index = CombineIndex(index, position)
    return index


def _grid(positions: Sequence[Index]) -> dict[Index, np.ndarray]:
    """Each position's coordinate at every joint assignment of positions, row-major,
    the last fastest."""
    shape = tuple(p.size for p in positions)
    return dict(zip(positions, np.indices(shape).reshape(len(shape), -1)))


def _values_at(
    table: TableRealization, index: Index, grid: Mapping[Index, np.ndarray]
) -> np.ndarray:
    """table's list at each entry of index; grid holds the coordinates of the
    positions index combines."""
    # On a UnitIndex too a constant is broadcast: numpy computes x**0.5 as sqrt(x)
    # only for a broadcast exponent, and both sides of a tie must take one path.
    if table.index == index and index != UnitIndex():
        return table.values
    at = grid[table.index] if table.index in grid else table.index.on_grid(grid)
    return np.broadcast_to(table.values[at], (index.size,))


def _ways(
    parts: Sequence[TableRealization],
) -> list[tuple[list[Index], list[TableRealization]]]:
    """The positions a table of a function of parts can be on, each with the parts as
    read there: the parts' inputs, in the order of their variables; and the parts' own
    positions, ranked to their distinct values (_compact_to_unique_rows) when the
    parts are on several, so a function of one list adds no position. Parts on one
    position share one rank, of their rows; two positions that share an input pair
    their rows at two values of it too, which no assignment reaches."""
    on: dict[Index, list[int]] = {}
    for i, part in enumerate(parts):
        if part.index != UnitIndex():
            on.setdefault(part.index, []).append(i)
    inputs = {i for index in on for i in index.inputs()}
    positions: list[Index] = sorted(inputs, key=lambda i: i.var.index) or [UnitIndex()]
    ways = [(positions, list(parts))]
    own = list(parts)
    if len(on) > 1:
        for index, members in on.items():
            rank, lists = _compact_to_unique_rows(
                index, [parts[i].values for i in members]
            )
            for i, values in zip(members, lists):
                own[i] = TableRealization(values=values, index=rank)
    ranks = [own[members[0]].index for members in on.values()] or [UnitIndex()]
    if set(ranks) != set(positions):
        ways.append((ranks, own))
    return ways


def _size(positions: Sequence[Index]) -> int:
    return math.prod(position.size for position in positions)


def _tabulated(
    parts: Sequence[TableRealization],
    cap: int,
    compute: Callable[[int, list[np.ndarray]], Any],
    made: Sequence[TableRealization] = (),
) -> TableRealization:
    """A function of parts as its cheapest table (_cheapest), refused where none fits
    cap."""
    table = _cheapest(parts, cap, compute, made)
    if table is None:
        inputs, *own = [_size(way) for way, _ in _ways(parts)]
        raise TableLoweringError(
            None,
            f"one table over its inputs would hold {inputs} entries, past the cap of "
            f"{cap}" + "".join(f", and one from its parts {size}" for size in own),
        )
    return table


def _cheapest(
    parts: Sequence[TableRealization],
    cap: int,
    compute: Callable[[int, list[np.ndarray]], Any],
    made: Sequence[TableRealization] = (),
) -> TableRealization | None:
    """A function of parts as the table within cap with the fewest entries, its
    list's and its lookups' (rule 3): one of its ways (_ways), over its inputs on a
    tie, or one of made, tables already made; None where none fits. A way's list is
    compute(count, the parts' lists on it), and only the way taken is computed, but
    where its list is not finite, at a pair no assignment reaches (_ways), the next
    is."""
    ways = _ways(parts)
    sizes = [_size(way) for way, _ in ways]
    made = [table for table in made if table.index.size <= cap]
    fitting = sorted(
        (
            (_size(w) + sum(p.entries() for p in w), w, read)
            for (w, read), size in zip(ways, sizes)
            if size <= cap
        ),
        key=lambda f: f[0],
    )
    cheapest = min(made, key=lambda table: table.entries(), default=None)
    for entries, way, read in fitting:
        if cheapest is not None and cheapest.entries() < entries:
            break
        index, grid = _combined(way), _grid(way)
        lists = [_values_at(part, index, grid) for part in read]
        values = compute(index.size, lists)
        if np.all(np.isfinite(values)):
            return TableRealization(values=values, index=index)
    if cheapest is not None:
        return cheapest
    if not fitting:
        return None
    return TableRealization(values=values, index=index)


def _by_rung(parts: Sequence[TableRealization], is_min: bool) -> TableRealization:
    """A Min (is_min) or Max of parts by rung (rule 1): its ladder, the values it can
    take ascending, at the min or max of each part's rung on it. A Max is never below
    its largest part's minimum, so a value under that bound is clamped to it, and a Min
    mirrors it. So a part on no input only sets the bound, and a part at the bound
    everywhere decides nothing. Parts on one position are one rung."""
    pick = np.minimum if is_min else np.maximum
    on: dict[Index, np.ndarray] = {}
    for part in parts:
        values = part.values
        on[part.index] = pick(on[part.index], values) if part.index in on else values
    ends = [values.max() if is_min else values.min() for values in on.values()]
    bound = min(ends) if is_min else max(ends)
    ladder = np.unique(np.concatenate([pick(values, bound) for values in on.values()]))
    still = len(ladder) - 1 if is_min else 0  # the rung that decides nothing
    rungs: list[Index] = []
    for index, values in on.items():
        remap = tuple(np.searchsorted(ladder, pick(values, bound)).tolist())
        if set(remap) != {still}:
            identity = remap == tuple(range(index.size))
            rungs.append(index if identity else CompactIndex(index, remap))
    if len(rungs) > 1:
        return TableRealization(
            values=ladder, index=ExtremeIndex(tuple(rungs), is_min, len(ladder))
        )
    return TableRealization(values=ladder, index=rungs[0] if rungs else UnitIndex())


# ---------------------------------------------------------------------- the printer


def _is_condition(expr: sympy.Basic) -> bool:
    """Whether expr is a condition: a relational, a logical operation, true or false,
    and not a symbol."""
    return isinstance(expr, Boolean) and not isinstance(expr, sympy.Symbol)


class TableLowering(Printer):
    """A sympy Printer whose print of a node is its realization. A print method
    returns None to decline a node, which is then one table of its operands
    (_operation).

    inputs gives each symbol's variable and the symbol's value at each of the
    variable's values, lowest first; cap bounds one table's entries.
    A printer is for one set of inputs: it keeps each node's realization, so a shared
    subexpression is printed once.
    """

    def __init__(
        self,
        inputs: Mapping[sympy.Symbol, tuple[cp_model.IntVar, Sequence[float]]],
        *,
        cap: int,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.cap = cap
        self._realizations: dict[sympy.Basic, AbstractRealization] = {}
        # one instance per rank, so equal ranks are one lookup
        self._ranks: dict[Index, Index] = {}

    def print(self, expr: sympy.Basic) -> AbstractRealization:
        """expr's realization; raises TableLoweringError for what cannot be lowered."""
        try:
            return self._print(expr)
        except RecursionError as exc:
            # No node: printing one this deep into the message would recurse too.
            raise TableLoweringError(
                None,
                "an expression nested too deep to print within the recursion limit "
                f"of {sys.getrecursionlimit()}",
            ) from exc

    def _print(self, expr: sympy.Basic, **kwargs: Any) -> AbstractRealization:
        """sympy's dispatch, kept by node. A declined node is one table."""
        if expr not in self._realizations:
            try:
                realization = super()._print(expr, **kwargs)
                if realization is None:
                    realization = self._operation(expr, operands(expr))
            except TableLoweringError as exc:
                if exc.node is not None:
                    raise
                raise TableLoweringError(expr, exc.reason) from exc
            self._realizations[expr] = realization
        return self._realizations[expr]

    def doprint(self, expr: sympy.Basic) -> str:
        """Refused: this printer's print of a node is a realization, not a string."""
        raise NotImplementedError("TableLowering prints realizations: call print")

    def _print_Symbol(self, expr: sympy.Symbol) -> AbstractRealization:
        """A symbol's column on its variable. A domain with gaps waits for Stage 13."""
        if expr not in self.inputs:
            raise TableLoweringError(expr, "a symbol with no input")
        var, values = self.inputs[expr]
        if not isinstance(var, cp_model.IntVar):
            raise TableLoweringError(
                expr, f"a variable that is not an IntVar ({type(var).__name__})"
            )
        domain = list(var.proto.domain)  # its intervals, flattened: lowest first
        if len(domain) != 2:
            raise TableLoweringError(expr, "a variable whose domain has gaps")
        lo, hi = domain
        if len(values) != hi - lo + 1:
            raise TableLoweringError(
                expr,
                f"{len(values)} values for a variable with {hi - lo + 1}, {lo} to {hi}",
            )
        index = InputIndex(var, size=len(values), lo=lo)
        try:
            column = TableRealization(values=values, index=index)
        except (TypeError, ValueError) as exc:
            raise TableLoweringError(
                expr, f"values that are not one number each ({exc})"
            ) from exc
        return column

    def _print_Add(self, expr: sympy.Add) -> AbstractRealization | None:
        """A sum kept linear."""
        if not expr.free_symbols:
            return None
        operands = [self._print(operand) for operand in expr.args]
        return _sum(operands)

    def _print_Mul(self, expr: sympy.Mul) -> AbstractRealization | None:
        """A product of several operands is gated (_gated) where one of them is a
        gate, its constants in the gate's rest. Otherwise a product that divides
        (divides), 98*k/a, is declined: one table, divided once in the numpy core as
        lambdify prints it, and no table of 1/a, so 98*k/a is 2k at a = 49, where k
        times 98 * (1/49) is under 2k; where an operand has no table, it is a switch
        on that quotient of the others (_divided). Any other product of several
        operands is a switch (_switched), its constants applied to the rest. A constant
        times one operand is the operand scaled in float, as lambdify prints a/49:
        (1/49)*a, and any other product is declined."""
        constants = [factor for factor in factors(expr) if not factor.free_symbols]
        symbolic = [factor for factor in expr.args if factor.free_symbols]
        if len(symbolic) > 1:
            gated = self._gated(expr, expr.args)
            if gated is not None:
                return gated
        if any(divides(factor) for factor in expr.args):
            return self._divided(expr)
        if len(symbolic) == 1:
            realization: AbstractRealization | None = self._print(symbolic[0])
        else:
            realization = self._switched(expr, symbolic)
        if realization is None or not constants:
            return realization
        coefficient = math.prod(float(self._table(c).values[0]) for c in constants)
        return realization.scaled(coefficient)

    def _gated(
        self, expr: sympy.Basic, operands: Sequence[sympy.Basic]
    ) -> AbstractRealization | None:
        """expr, the product of operands, gated (_gate) by the first operand that is a
        gate. Its rest is the product with the gate's literal at its kept value, so
        nothing under the gate reads the literal. None when no operand is a gate."""
        for operand in operands:
            gate = self._literal(operand)
            if gate is not None:
                literal, kept = gate
                product = sympy.Mul(*operands)
                try:  # sympy evaluates the product at the kept value
                    kept_product = product.xreplace(self._at(product, literal, kept))
                except (TypeError, ValueError) as exc:
                    raise TableLoweringError(
                        expr, f"no value ({type(exc).__name__}: {exc})"
                    ) from exc
                rest = self._print(kept_product)
                return _gate(literal, kept, rest)
        return None

    def _literal(self, operand: sympy.Basic) -> tuple[InputIndex, int] | None:
        """The boolean literal that operand gates by, and its kept value: where operand
        reads that literal only and is 0 at exactly one of its values, as x, 1 - x and
        2 - 2*x are and 1 + x is not. None for any other operand."""
        positions = self._inputs(operand)
        one = len(positions) == 1 and not divides(operand)  # 1/a is never 0
        literal = positions[0] if one else None
        if isinstance(literal, InputIndex) and (literal.lo, literal.size) == (0, 2):
            zero = self._table(operand).values == 0
            if np.count_nonzero(zero) == 1:
                return literal, int(zero[0])
        return None

    def _at(
        self, expr: sympy.Basic, literal: InputIndex, value: int
    ) -> dict[sympy.Basic, sympy.Number]:
        """Each of expr's symbols on literal, at its value where literal is value."""
        return {
            symbol: _number(self._table(symbol).values[value])
            for symbol in expr.free_symbols
            if self._table(symbol).index == literal
        }

    def _switched(
        self, expr: sympy.Basic, operands: Sequence[sympy.Basic]
    ) -> AbstractRealization | None:
        """expr, the product of operands, as a switch (_switch) on the operand with
        the fewest distinct values among those with a table (_has_table), the first in
        expr's order on a tie, over the others' product. None where no operand has a
        table or the product keeps its table."""
        tables = {
            operand: self._table(operand)
            for operand in operands
            if self._has_table(operand)
        }
        if not tables:
            return None
        factor = min(tables, key=lambda o: len(np.unique(tables[o].values)))
        return self._switch_on(expr, factor, [o for o in operands if o != factor])

    def _divided(self, expr: sympy.Mul) -> AbstractRealization | None:
        """A product that divides, declined where each of its operands (operands)
        has a table. Otherwise a switch on the quotient of the factors whose operands
        have one, its table divided once, over the product of the others, as a sum of
        residency literals too wide for any table is: S / (d0 * d1)."""
        wide = [
            factor
            for factor, operand in zip(factors(expr), operands(expr))
            if operand.free_symbols and not self._has_table(operand)
        ]
        if not wide:
            return None
        quotient = sympy.Mul(*(factor for factor in expr.args if factor not in wide))
        return self._switch_on(expr, quotient, wide)

    def _switch_on(
        self, expr: sympy.Basic, factor: sympy.Basic, others: Sequence[sympy.Basic]
    ) -> AbstractRealization | None:
        """expr as a switch (_switch) on factor's table, ranked, over the product of
        others."""
        table = self._table(factor)
        index, (values,) = _compact_to_unique_rows(table.index, [table.values])
        index = self._ranks.setdefault(index, index)
        rest = self._print(sympy.Mul(*others))
        positions = tuple(self._inputs(expr))
        return _switch(tuple(values.tolist()), index, rest, positions, self.cap)

    def _print_Min(self, expr: sympy.Min) -> AbstractRealization | None:
        return self._extreme(expr, is_min=True)

    def _print_Max(self, expr: sympy.Max) -> AbstractRealization | None:
        return self._extreme(expr, is_min=False)

    def _extreme(
        self, expr: sympy.Basic, *, is_min: bool
    ) -> AbstractRealization | None:
        """A Min or Max as its cheapest table, by rung (_by_rung) among its ways, or
        kept linear (ExtremeRealization, rule 3): where an operand has no table
        (_has_table), none of its tables fits the cap, or the linear one needs under
        1/LINEAR_MARGIN of its table's entries. Declined where it reads no symbol."""
        if not expr.free_symbols:
            return None
        operands = tuple(self._print(operand) for operand in expr.args)
        linear = ExtremeRealization(is_min, operands)
        if not all(self._has_table(operand) for operand in expr.args):
            return linear
        parts = [self._table(operand) for operand in expr.args]
        table = _cheapest(
            parts,
            self.cap,
            lambda count, lists: _values(expr, count, lambda: numpy_eval(expr, lists)),
            (_by_rung(parts, is_min),),
        )
        if table is None or LINEAR_MARGIN * linear.entries() < table.entries():
            return linear
        return table

    def _print_RelayoutCharge(self, expr: sympy.Function) -> AbstractRealization | None:
        """RelayoutCharge(is_lx, division, *prices), is_lx times the price at the
        division, gated (_gated) as that product over its prices, one list on the
        division. Declined where is_lx reads no variable or is no gate."""
        is_lx, *others = expr.args
        if not is_lx.free_symbols:
            return None
        return self._gated(expr, [is_lx, expr.func(sympy.S.One, *others)])

    def emptyPrinter(self, expr: sympy.Basic) -> None:
        """Any other node is declined. sympy's own returns str(expr)."""
        return None

    def _operation(
        self,
        expr: sympy.Basic,
        operands: Sequence[sympy.Basic],
        made: Sequence[TableRealization] = (),
    ) -> TableRealization:
        """expr as its table (_tabulated) of its operands, or one of made. An operation
        on numbers is computed the same way, on a UnitIndex, so sqrt(19) equals
        sqrt(a) at a = 19 (sympy's float is one double below)."""
        if not operands:
            return TableRealization(values=[_constant(expr)], index=UnitIndex())
        return _tabulated(
            [self._table(operand) for operand in operands],
            self.cap,
            lambda count, lists: _values(expr, count, lambda: numpy_eval(expr, lists)),
            made,
        )

    def _fits(self, expr: sympy.Basic) -> bool:
        """Whether one table over expr's inputs is within the cap."""
        return math.prod(position.size for position in self._inputs(expr)) <= self.cap

    def _has_table(self, expr: sympy.Basic) -> bool:
        """Whether expr has a table within the cap: it is one, as a Max by rung over
        more inputs than one table holds is, or one over its inputs fits."""
        return isinstance(self._print(expr), TableRealization) or self._fits(expr)

    def _inputs(self, expr: sympy.Basic) -> list[Index]:
        """The positions of expr's inputs, in the order of its symbols."""
        symbols = sorted(expr.free_symbols, key=sympy.default_sort_key)
        return list(dict.fromkeys(self._table(symbol).index for symbol in symbols))

    def _table(self, expr: sympy.Basic) -> TableRealization:
        """expr's table within the cap; a refusal of that table names expr."""
        realization = self._print(expr)
        try:
            return realization.table(self.cap)
        except TableLoweringError as exc:
            raise TableLoweringError(expr, exc.reason) from exc


def _sum(operands: Sequence[AbstractRealization]) -> SumRealization:
    """operands added, kept linear. Terms with one sum_key are joined into one: lists
    on one position, constants among them, added in float, one lookup and one
    rounding; switches on one factor, and gates on one literal and kept value, one
    over their rests' sum. Any other operand, as sqrt(2) * (a + b) is (sympy
    distributes only a number), stays one term."""
    groups: dict[Hashable, list[AbstractRealization]] = {}
    for operand in operands:
        groups.setdefault(operand.sum_key(), []).append(operand)
    terms = (g[0] if len(g) == 1 else g[0].joined(g[1:]) for g in groups.values())
    return SumRealization(tuple(terms))


def _switch(
    values: tuple[float, ...],
    index: Index,
    rest: AbstractRealization,
    positions: tuple[Index, ...],
    cap: int,
) -> AbstractRealization | None:
    """values[index] * rest. A factor of one value is a constant on its inputs: it
    scales the rest. Otherwise the switch where one table over positions is past cap
    or has more entries (rule 3), and None where that table is kept. The table from
    the product's parts is not weighed: it pairs the factor's ranks with the rest's,
    which never has fewer entries than the switch."""
    if len(values) == 1:
        return rest.scaled(values[0])
    switch = SwitchRealization(values, index, rest)
    entries = math.prod(position.size for position in positions)
    return switch if entries > cap or switch.entries() < entries else None


def _gate(
    literal: InputIndex,
    kept: int,
    rest: AbstractRealization,
) -> AbstractRealization:
    """rest where literal is at kept and 0 at its other value (rule 5). A rest of one
    value makes it one list on the literal, affine: no gate and no table."""
    if isinstance(rest, TableRealization) and len(np.unique(rest.values)) == 1:
        values = np.where(np.arange(2) == kept, float(rest.values[0]), 0.0)
        return TableRealization(values=values, index=literal)
    return GateRealization(literal, kept, rest)


def _number(value: float) -> sympy.Number:
    """value as a sympy number: an Integer when it is whole."""
    return (
        sympy.Integer(int(value)) if float(value).is_integer() else sympy.Float(value)
    )


def _compact_to_unique_rows(
    index: Index, lists: Sequence[np.ndarray]
) -> tuple[Index, list[np.ndarray]]:
    """index compacted to the unique rows of lists, each a list on index, in the order
    of their first entries: the rank of each entry's row, a CompactIndex, or index
    itself when no row repeats; and each list's value at each rank."""
    rows = list(zip(*(values.tolist() for values in lists)))
    distinct = list(dict.fromkeys(rows))
    if len(distinct) == len(rows):
        return index, list(lists)
    rank = {row: i for i, row in enumerate(distinct)}
    remap = tuple(rank[row] for row in rows)
    return CompactIndex(index, remap), [np.asarray(column) for column in zip(*distinct)]


def _constant(expr: sympy.Basic) -> float | bool:
    """The value of a node with no operand: a number, or true or false as a bool."""
    if expr is sympy.true or expr is sympy.false:
        return bool(expr)
    try:
        return float(expr)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TableLoweringError(expr, "a constant with no numeric value") from exc
