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

"""One op's legal core divisions posted as constraints, not listed (Phase 2f,
experimental).

A :class:`LazySpace` is the op plus its tiling options, each with the
:class:`WorkDivisionContext` of its per-tile frame. :func:`post_division`
posts one split var per axis and a tiling choice, and per option, under its
literal, what :meth:`WorkDivisionContext.is_legal` checks: the factor domains,
the core budget, at most one reduction split, the blocked axes and hard split
domains, the span floors, and the per-core span cap. The menu
``enumerate_work_division_candidates`` builds is the set of solutions.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import sympy

from torch_spyre._inductor import work_division as wd
from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision, TileSpec

Var = Any


@dataclass(frozen=True)
class LazyOption:
    """One tiling the op may take, with its per-tile frame."""

    tiling: TileSpec
    tile_splits: tuple[tuple[sympy.Symbol, int], ...]
    context: wd.WorkDivisionContext


@dataclass(eq=False)
class LazySpace:
    """An op's divisions as per-axis choices under each tiling option."""

    op: Any
    options: list[LazyOption]
    max_cores: int
    _points: Optional[list[tuple[int, dict]]] = field(default=None, repr=False)
    _divisions: dict = field(default_factory=dict, repr=False)

    @property
    def axes(self) -> list[sympy.Symbol]:
        return list(dict.fromkeys(a for o in self.options for a in o.context.axes))

    def domain(self, axis: sympy.Symbol) -> list[int]:
        """Every factor ``axis`` takes under some option (1 where it is not an
        axis of that option's frame)."""
        values = {1}
        for o in self.options:
            if axis in o.context.axes:
                values.update(o.context.factor_domain(axis))
        return sorted(values)

    def points(self) -> list[tuple[int, dict]]:
        """``(option, splits)`` over each option's factor domains within the
        core budget: a superset of the legal divisions, cheap to list, that
        the strategy tabulates per-division quantities over. Legality is
        posted separately; a quantity's value at an illegal point is never
        read."""
        if self._points is None:
            out = []
            for k, o in enumerate(self.options):
                axes = o.context.axes
                domains = [o.context.factor_domain(a) or [1] for a in axes]
                for combo in itertools.product(*domains):
                    if math.prod(combo) <= self.max_cores:
                        out.append((k, dict(zip(axes, combo))))
            self._points = out
        return self._points

    def division(self, option: int, splits: dict) -> CoreDivision:
        key = (option, tuple(sorted((str(a), v) for a, v in splits.items())))
        if key not in self._divisions:
            o = self.options[option]
            self._divisions[key] = wd._core_division(
                self.op, dict(splits), o.tiling, o.tile_splits
            )
        return self._divisions[key]


def _span_terms(ctx: wd.WorkDivisionContext, td, domain: dict) -> tuple[list, int]:
    """Per outer device dim with a loop symbol: ``(stride_bytes, {axis: {split:
    term}})``; that dim's per-core extent is ``1 + sum(terms)``. ``domain`` is
    each axis's posted domain, so every term table is total."""
    device_size = td.layout.device_layout.device_size
    itemsize = td.layout.dtype.itemsize
    dims = []
    for d, coord in enumerate(td.device_coords[:-1]):
        if not coord.free_symbols:
            continue
        per_axis = {}
        for v in coord.free_symbols:
            term = coord.subs(dict.fromkeys(coord.free_symbols - {v}, 0))
            base = int(term.subs(v, 0))
            size = wd._effective_size(v, ctx.it_space, ctx.symbol_meta)
            values = domain.get(v, [1]) if v in ctx.it_space_adjusted else [1]
            per_axis[v] = {s: int(term.subs(v, size // s - 1)) - base for s in values}
        stride = math.prod(device_size[d + 1 :]) * itemsize
        dims.append((stride, per_axis))
    return dims, itemsize


def post_division(
    model, space: LazySpace, name: str
) -> tuple[Var | int, dict[sympy.Symbol, Var], list[Var | int], Var | int]:
    """Post ``space``'s legal divisions: ``(option, splits, option literals,
    cores)``. ``option`` and ``cores`` are constants when nothing is left to
    choose."""
    axes = space.axes
    domain = {a: space.domain(a) for a in axes}
    splits: dict[sympy.Symbol, Var] = {}
    for a in axes:
        var = model.IntVar(min(domain[a]), max(domain[a]), f"split_{name}_{a}")
        if len(domain[a]) < max(domain[a]) - min(domain[a]) + 1:
            model.AddAllowedAssignments([var], [(d,) for d in domain[a]])
        splits[a] = var
    k = len(space.options)
    option: Var | int
    if k == 1:
        option, lits = 0, [1]
    else:
        option = model.IntVar(0, k - 1, f"tiling_{name}")
        lits = []
        for i in range(k):
            lit = model.BoolVar(f"tiling_{name}_is_{i}")
            model.Add(option == i).OnlyEnforceIf(lit)
            model.Add(option != i).OnlyEnforceIf(lit.Not())
            lits.append(lit)
    cores: Var | int = 1
    for a in axes:
        if isinstance(cores, int):
            cores = splits[a]
            continue
        product = model.IntVar(1, space.max_cores, f"cores_{name}_{a}")
        model.AddMultiplicationEquality(product, [cores, splits[a]])
        cores = product
    if not isinstance(cores, int):
        model.Add(cores <= space.max_cores)
    for i, o in enumerate(space.options):
        _post_option(model, o.context, splits, domain, lits[i], f"{name}_{i}")
    return option, splits, lits, cores


def _enforce(constraint, lit) -> None:
    if lit is not True:
        constraint.OnlyEnforceIf(lit)


def _post_option(model, ctx, splits, domain, lit, name: str) -> None:
    """Under ``lit``, :meth:`WorkDivisionContext.is_legal` for ``ctx``."""
    if isinstance(lit, int):
        lit = True

    def infeasible() -> None:
        if lit is True:
            raise ValueError(f"{name}: the only tiling option has no legal division")
        model.Add(lit == 0)

    for a, var in splits.items():
        allowed = set(ctx.factor_domain(a)) if a in ctx.axes else {1}
        if not allowed:
            infeasible()
            return
        for value in domain[a]:
            if value not in allowed:
                _enforce(model.Add(var != value), lit)
    split_reductions = [v for v in ctx.reduction_vars if v in splits]
    if len(split_reductions) > 1:
        reductions = []
        for v in split_reductions:
            b = model.BoolVar(f"red_{name}_{v}")
            model.Add(splits[v] > 1).OnlyEnforceIf(b)
            model.Add(splits[v] == 1).OnlyEnforceIf(b.Not())
            reductions.append(b)
        _enforce(model.Add(sum(reductions) <= 1), lit)
    for v in ctx.constraints.blocked:
        if v in splits:
            _enforce(model.Add(splits[v] == 1), lit)
    for v, allowed in ctx.constraints.allowed_splits.items():
        if v in splits:
            for value in domain[v]:
                if value not in allowed:
                    _enforce(model.Add(splits[v] != value), lit)
    for v, minimum in ctx.span_min_splits.items():
        if v in splits:
            _enforce(model.Add(splits[v] >= minimum), lit)
        elif minimum > 1:
            infeasible()
            return
    for ti, td in enumerate(ctx.tensor_deps):
        dims, itemsize = _span_terms(ctx, td, domain)
        if itemsize > wd.MAX_SPAN_BYTES:
            infeasible()
            return
        earlier: list = []
        for di, (stride, per_axis) in enumerate(dims):
            parts: list = []
            for v, table in per_axis.items():
                # A split this frame does not adjust leaves the term at its
                # value for split 1.
                if v in splits and v in ctx.axes and len(table) > 1:
                    lo, hi = min(table.values()), max(table.values())
                    if lo == hi:
                        parts.append(lo)
                        continue
                    t = model.IntVar(lo, hi, f"span_{name}_{ti}_{di}_{v}")
                    model.AddAllowedAssignments([splits[v], t], list(table.items()))
                    parts.append(t)
                else:
                    parts.append(table.get(1, 0))
            cap = wd.MAX_SPAN_BYTES // stride
            if all(isinstance(p, int) for p in parts):
                size = 1 + sum(parts)
                if size == 1:
                    continue
                # A fixed extent above one decides the span unless an earlier
                # dim already did.
                if size > cap:
                    if not earlier:
                        infeasible()
                        return
                    _enforce(model.AddBoolOr(earlier), lit)
                break
            # Unbounded below: the term tables are posted for every option, and
            # a split legal elsewhere can make this frame's extent negative.
            size_var = model.IntVar(-(1 << 40), 1 << 40, f"extent_{name}_{ti}_{di}")
            model.Add(size_var == 1 + sum(parts))
            gt1 = model.BoolVar(f"gt1_{name}_{ti}_{di}")
            model.Add(size_var > 1).OnlyEnforceIf(gt1)
            model.Add(size_var <= 1).OnlyEnforceIf(gt1.Not())
            model.Add(size_var <= cap).OnlyEnforceIf(
                [gt1, *[e.Not() for e in earlier], *_lits(lit)]
            )
            earlier.append(gt1)


def _lits(lit) -> list:
    return [] if lit is True else [lit]


def legal_points(space: LazySpace) -> set[tuple[int, tuple]]:
    """Every ``(option, splits over space.axes)`` the posted constraints admit,
    by enumerating the solutions of the division model alone: the agreement
    check against the menu."""
    from ortools.sat.python import cp_model

    from torch_spyre._inductor.scratchpad.backend import CpSat

    backend = CpSat()
    option, splits, _, _ = post_division(backend, space, "check")
    axes = space.axes
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_search_workers = 1
    found: set = set()

    class Collect(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self) -> None:
            k = option if isinstance(option, int) else self.Value(option)
            found.add((k, tuple(self.Value(splits[a]) for a in axes)))

    solver.Solve(backend.model, Collect())
    return found


def menu_points(
    space: LazySpace, menu: Sequence[CoreDivision]
) -> set[tuple[int, tuple]]:
    """The menu in :func:`legal_points`' terms."""
    by_tiling = {o.tiling: k for k, o in enumerate(space.options)}
    axes = space.axes
    return {
        (by_tiling.get(cd.tiling, -1), tuple(cd.splits.get(a, 1) for a in axes))
        for cd in menu
    }
