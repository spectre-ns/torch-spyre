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

"""The solver interface the encoding strategy posts to (Phase 3a, prototype).

:class:`Backend` is shaped like ``pywraplp.Solver`` -- ``IntVar`` /
``BoolVar`` / ``Add`` / ``Minimize`` / ``Solve`` / ``Value`` -- plus the
operations the model uses that ``pywraplp`` lacks: enforcement
(``OnlyEnforceIf``), Boolean clauses, element and table, products, division,
max, optional intervals and the 2-D no-overlap. The strategy and the objective
ladder post only through it; backends differ in how they solve.

Every backend here posts through one recorded CP model (``CpSat``'s), which the
others then solve their own way: ``Mip`` linearizes it onto ``pywraplp``
(SCIP / HiGHS / CBC: big-M enforcement, one-hot element and table, McCormick
products, pairwise no-overlap disjunctions), ``BruteForce`` searches it
exhaustively. Recording once keeps the cost-expression printer, which builds
``cp_model`` expressions, working for every backend.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional, Protocol, Sequence

from google.protobuf import text_format
from ortools.linear_solver import pywraplp
from ortools.sat import cp_model_pb2
from ortools.sat.python import cp_model

Var = Any
LinearExpr = Any


class Constraint(Protocol):
    def OnlyEnforceIf(self, *lits: Var) -> "Constraint": ...


class Backend(Protocol):
    """What the strategy and the ladder post to."""

    model: cp_model.CpModel  # the recorded model (the printer builds on it)

    def IntVar(self, lb: int, ub: int, name: str) -> Var: ...
    def BoolVar(self, name: str) -> Var: ...
    def Constant(self, value: int) -> Var: ...
    def Add(self, bounded: object) -> Constraint: ...
    def AddBoolOr(self, lits: Sequence[Var]) -> Constraint: ...
    def AddBoolAnd(self, lits: Sequence[Var]) -> Constraint: ...
    def AddAtMostOne(self, lits: Sequence[Var]) -> Constraint: ...
    def AddImplication(self, a: Var, b: Var) -> Constraint: ...
    def AddElement(
        self, index: Var, values: Sequence[int], target: Var
    ) -> Constraint: ...
    def AddAllowedAssignments(
        self, vars: Sequence[Var], tuples: Sequence[Sequence[int]]
    ) -> Constraint: ...
    def AddMaxEquality(
        self, target: Var, exprs: Sequence[LinearExpr]
    ) -> Constraint: ...
    def NewOptionalIntervalVar(
        self,
        start: LinearExpr,
        size: LinearExpr,
        end: LinearExpr,
        present: Var,
        name: str,
    ) -> Any: ...
    def AddNoOverlap2D(self, x: Sequence[Any], y: Sequence[Any]) -> Constraint: ...
    def Minimize(self, expr: LinearExpr) -> None: ...
    def Maximize(self, expr: LinearExpr) -> None: ...
    def Solve(self) -> int: ...
    def StatusName(self, status: int) -> str: ...
    def WallTime(self) -> float: ...
    def ObjectiveValue(self) -> float: ...
    def Value(self, expr: LinearExpr) -> int: ...
    def BooleanValue(self, lit: Var) -> bool: ...
    def NumVariables(self) -> int: ...
    def NumConstraints(self) -> int: ...


class CpSat:
    """Pass-through to ``cp_model``: the default and the reference."""

    def __init__(self, model: Optional[cp_model.CpModel] = None) -> None:
        self.model = model if model is not None else cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.record = None  # (solver, model) -> status; the solver's stats hook
        self._objective = False

    # ---- posting ----
    def IntVar(self, lb, ub, name):
        return self.model.new_int_var(lb, ub, name)

    def BoolVar(self, name):
        return self.model.new_bool_var(name)

    def Constant(self, value):
        return self.model.new_constant(value)

    def Add(self, bounded):
        return self.model.add(bounded)

    def AddBoolOr(self, lits):
        return self.model.add_bool_or(lits)

    def AddBoolAnd(self, lits):
        return self.model.add_bool_and(lits)

    def AddAtMostOne(self, lits):
        return self.model.add_at_most_one(lits)

    def AddImplication(self, a, b):
        return self.model.add_implication(a, b)

    def AddElement(self, index, values, target):
        return self.model.add_element(index, values, target)

    def AddAllowedAssignments(self, vars, tuples):
        return self.model.add_allowed_assignments(vars, tuples)

    def AddMaxEquality(self, target, exprs):
        return self.model.add_max_equality(target, exprs)

    def NewOptionalIntervalVar(self, start, size, end, present, name):
        return self.model.new_optional_interval_var(start, size, end, present, name)

    def AddNoOverlap2D(self, x, y):
        return self.model.add_no_overlap_2d(x, y)

    def Minimize(self, expr):
        self.model.minimize(expr)

    def Maximize(self, expr):
        self.model.maximize(expr)

    # ---- solving ----
    def Solve(self, objective: bool = False) -> int:
        if self.record is not None:
            return self.record(self.solver, self.model, objective=objective)
        return self.solver.Solve(self.model)

    def StatusName(self, status):
        return self.solver.StatusName(status)

    def WallTime(self):
        return self.solver.WallTime()

    def ObjectiveValue(self):
        return self.solver.ObjectiveValue()

    def Value(self, expr):
        return self.solver.Value(expr)

    def BooleanValue(self, lit):
        return self.solver.BooleanValue(lit)

    def NumVariables(self):
        return len(self.model.proto.variables)

    def NumConstraints(self):
        return len(self.model.proto.constraints)


class _SolvedRecording(CpSat):
    """A backend that solves the recorded model with its own engine and
    answers ``Value`` from that solution."""

    def __init__(self, model=None):
        super().__init__(model)
        self._values: Optional[list] = None
        self._objective_value = 0.0
        self._wall = 0.0
        self._status = cp_model.UNKNOWN

    def _proto(self):
        return text_format.Parse(str(self.model.proto), cp_model_pb2.CpModelProto())

    def StatusName(self, status):
        return cp_model.CpSolver().StatusName(status)

    def WallTime(self):
        return self._wall

    def ObjectiveValue(self):
        return self._objective_value

    def Value(self, expr):
        if isinstance(expr, (int, float)):
            return int(expr)
        flat = cp_model.FlatIntExpr(expr)
        return int(flat.offset) + sum(
            int(c) * self._lit(v) for v, c in zip(flat.vars, flat.coeffs)
        )

    def _lit(self, v):
        if isinstance(v, cp_model_helper_not()):
            return 1 - self._values[v.negated().index]
        return self._values[v.index]

    def BooleanValue(self, lit):
        return bool(self.Value(lit))


def cp_model_helper_not():
    from ortools.sat.python import cp_model_helper

    return cp_model_helper.NotBooleanVariable


class Mip(_SolvedRecording):
    """The recorded model, linearized onto a ``pywraplp`` MIP solver with a
    zero relative gap, so OPTIMAL is exact."""

    def __init__(self, solver_id: str = "SCIP", time_limit_s: float = 60.0, model=None):
        super().__init__(model)
        self.solver_id = solver_id
        self.time_limit_s = time_limit_s

    def Solve(self, objective: bool = False) -> int:
        lin = _Linearizer(self._proto(), self.solver_id)
        lin.build()
        lin.s.SetTimeLimit(int(self.time_limit_s * 1000))
        params = pywraplp.MPSolverParameters()
        params.SetDoubleParam(params.RELATIVE_MIP_GAP, 0.0)
        t = time.perf_counter()
        status = lin.s.Solve(params)
        self._wall = time.perf_counter() - t
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            self._values = [int(round(v.solution_value())) for v in lin.x]
            self._objective_value = lin.s.Objective().Value() * lin.report
            self._status = (
                cp_model.OPTIMAL
                if status == pywraplp.Solver.OPTIMAL
                else cp_model.FEASIBLE
            )
        elif status == pywraplp.Solver.INFEASIBLE:
            self._status = cp_model.INFEASIBLE
        else:
            self._status = cp_model.UNKNOWN
        return self._status


class BruteForce(_SolvedRecording):
    """Exhaustive search over the recorded model: an oracle for tiny models."""

    def __init__(self, max_nodes: int = 3_000_000, model=None):
        super().__init__(model)
        self.max_nodes = max_nodes

    def Solve(self, objective: bool = False) -> int:
        search = _BruteForceSearch(self._proto(), self.max_nodes)
        t = time.perf_counter()
        search.solve()
        self._wall = time.perf_counter() - t
        if search.best is None:
            self._status = cp_model.INFEASIBLE
            return self._status
        self._values = search.best_a
        self._objective_value = search.objective_value()
        self._status = cp_model.OPTIMAL
        return self._status


# =============================================================================
# Engines (prototype): the MIP linearization and the exhaustive search.
# =============================================================================
class MipUnsupported(Exception):
    pass


def _vals(dom):
    out = []
    for lo, hi in zip(dom[::2], dom[1::2]):
        out.extend(range(lo, hi + 1))
    return out


class _Linearizer:
    def __init__(self, proto, solver_id):
        self.p = proto
        self.s = pywraplp.Solver.CreateSolver(solver_id)
        if self.s is None:
            raise MipUnsupported(solver_id)
        self.x = []
        self.bounds = []
        self.onehot_cache = {}
        for i, v in enumerate(proto.variables):
            dom = list(v.domain)
            lo, hi = dom[0], dom[-1]
            var = self.s.IntVar(lo, hi, f"v{i}")
            self.x.append(var)
            self.bounds.append((lo, hi))
            if len(dom) > 2:  # holes: pin to the listed values
                self._onehot(i)
        self.intervals = {}

    # ---- expressions ----
    def lit(self, r):
        return self.x[r] if r >= 0 else 1 - self.x[-r - 1]

    def lin(self, e):
        return sum(c * self.lit_int(v) for v, c in zip(e.vars, e.coeffs)) + e.offset

    def lit_int(self, r):
        return self.x[r] if r >= 0 else -self.x[-r - 1]

    def bnd(self, e):
        lo = hi = getattr(e, "offset", 0)
        for v, c in zip(e.vars, e.coeffs):
            a, b = self.bounds[v if v >= 0 else -v - 1]
            if v < 0:
                a, b = -b, -a
            lo += min(c * a, c * b)
            hi += max(c * a, c * b)
        return lo, hi

    def slack(self, enf):
        """0 when every enforcement literal is true, >= 1 otherwise."""
        return sum(1 - self.lit(lit) for lit in enf)

    def _onehot(self, v):
        if v not in self.onehot_cache:
            vals = _vals(list(self.p.variables[v].domain))
            if len(vals) > 512:
                raise MipUnsupported(f"domain of v{v} has {len(vals)} values")
            z = {val: self.s.BoolVar(f"oh{v}_{val}") for val in vals}
            self.s.Add(sum(z.values()) == 1)
            self.s.Add(self.x[v] == sum(val * zv for val, zv in z.items()))
            self.onehot_cache[v] = z
        return self.onehot_cache[v]

    def _single_var(self, e):
        if len(e.vars) == 1 and e.vars[0] >= 0:
            return e.vars[0], e.coeffs[0], e.offset
        return None

    # ---- constraints ----
    def add_range(self, expr, lo, hi, enf, ebnd):
        k = self.slack(enf) if enf else 0
        elo, ehi = ebnd
        if lo > elo:
            self.s.Add(expr >= lo - (lo - elo) * k)
        if hi < ehi:
            self.s.Add(expr <= hi + (ehi - hi) * k)

    def linear(self, c):
        lin = c.linear
        expr = sum(cf * self.lit_int(v) for v, cf in zip(lin.vars, lin.coeffs))
        eb = self.bnd(lin)
        dom = list(lin.domain)
        segs = list(zip(dom[::2], dom[1::2]))
        segs = [
            (max(a, eb[0]), min(b, eb[1])) for a, b in segs if b >= eb[0] and a <= eb[1]
        ]
        enf = list(c.enforcement_literal)
        if not segs:
            if enf:
                self.s.Add(self.slack(enf) >= 1)
                return
            self.s.Add(self.s.Sum([]) >= 1)  # infeasible
            return
        if len(segs) == 1:
            self.add_range(expr, *segs[0], enf, eb)
            return
        zs = [self.s.BoolVar("") for _ in segs]
        self.s.Add(sum(zs) == 1)
        for z, (lo, hi) in zip(zs, segs):
            k = (self.slack(enf) if enf else 0) + (1 - z)
            if lo > eb[0]:
                self.s.Add(expr >= lo - (lo - eb[0]) * k)
            if hi < eb[1]:
                self.s.Add(expr <= hi + (eb[1] - hi) * k)

    def element(self, c):
        e = c.element
        idx = self._single_var(e.linear_index)
        if idx is None or idx[1] != 1 or idx[2] != 0:
            raise MipUnsupported("element index")
        consts = []
        for x in e.exprs:
            if len(x.vars):
                raise MipUnsupported("element over vars")
            consts.append(x.offset)
        z = self._onehot(idx[0])
        tgt = self.lin(e.linear_target)
        self.s.Add(
            tgt == sum(consts[v] * zv for v, zv in z.items() if 0 <= v < len(consts))
        )
        for v, zv in z.items():
            if not 0 <= v < len(consts):
                self.s.Add(zv == 0)

    def table(self, c):
        t = c.table
        if t.negated:
            raise MipUnsupported("negated table")
        w = len(t.exprs)
        vals = list(t.values)
        rows = [vals[j : j + w] for j in range(0, len(vals), w)]
        enf = list(c.enforcement_literal)
        r = [self.s.BoolVar("") for _ in rows]
        on = 1 - self.slack(enf) if enf else 1
        self.s.Add(sum(r) <= 1)
        self.s.Add(sum(r) >= on)
        for j, x in enumerate(t.exprs):
            e = self.lin(x)
            lo, hi = self.bnd(x)
            k = self.slack(enf) if enf else 0
            pick = sum(rk * row[j] for rk, row in zip(r, rows))
            big = max(
                hi - min(row[j] for row in rows), max(row[j] for row in rows) - lo, 0
            )
            self.s.Add(e - pick <= big * k)
            self.s.Add(pick - e <= big * k)

    def product(self, c):
        g = c.int_prod
        if len(g.exprs) != 2:
            raise MipUnsupported("int_prod arity")
        a, b = g.exprs
        sa, sb = self._single_var(a), self._single_var(b)
        if sa is None or sb is None:
            raise MipUnsupported("int_prod of sums")
        # one-hot the smaller domain, McCormick the other
        va, vb = sa[0], sb[0]
        if len(_vals(list(self.p.variables[va].domain))) > len(
            _vals(list(self.p.variables[vb].domain))
        ):
            (va, sa), (vb, sb) = (vb, sb), (va, sa)
        z = self._onehot(va)
        lo, hi = self.bnd(b)
        ybexpr = self.lin(b)
        terms = []
        for val, zv in z.items():
            w = self.s.NumVar(min(0, lo), max(0, hi), "")
            self.s.Add(w <= hi * zv)
            self.s.Add(w >= lo * zv)
            self.s.Add(w <= ybexpr - lo * (1 - zv))
            self.s.Add(w >= ybexpr - hi * (1 - zv))
            terms.append((sa[1] * val + sa[2]) * w)
        self.s.Add(self.lin(g.target) == sum(terms))

    def division(self, c):
        g = c.int_div
        n, d = g.exprs
        if len(n.vars) or self._single_var(d) is None:
            raise MipUnsupported("int_div form")
        dv, dc, do = self._single_var(d)
        z = self._onehot(dv)
        num = n.offset
        expr = 0
        for val, zv in z.items():
            den = dc * val + do
            if den == 0:
                self.s.Add(zv == 0)
                continue
            q = abs(num) // abs(den) * (1 if (num >= 0) == (den > 0) else -1)
            expr = expr + q * zv
        self.s.Add(self.lin(g.target) == expr)

    def lin_max(self, c):
        g = c.lin_max
        t = self.lin(g.target)
        tb = self.bnd(g.target)
        bs = [self.s.BoolVar("") for _ in g.exprs]
        self.s.Add(sum(bs) == 1)
        for b, x in zip(bs, g.exprs):
            e = self.lin(x)
            eb = self.bnd(x)
            self.s.Add(t >= e)
            self.s.Add(t <= e + (tb[1] - eb[0]) * (1 - b))

    def interval(self, ci, c):
        iv = c.interval
        enf = list(c.enforcement_literal)
        s, sz, e = self.lin(iv.start), self.lin(iv.size), self.lin(iv.end)
        lo, hi = self.bnd(iv.end)
        ls, hs = self.bnd(iv.start)
        lz, hz = self.bnd(iv.size)
        k = self.slack(enf) if enf else 0
        big = max(abs(hs + hz - lo), abs(hi - ls - lz), 1)
        self.s.Add(e - s - sz <= big * k)
        self.s.Add(s + sz - e <= big * k)
        pres = 1 - k if enf else 1
        self.intervals[ci] = (s, sz, e, (ls, hs), (lo, hi), (lz, hz), pres)

    def no_overlap_2d(self, c):
        no = c.no_overlap_2d
        boxes = [
            (self.intervals[xi], self.intervals[yi])
            for xi, yi in zip(no.x_intervals, no.y_intervals)
        ]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                (xa, ya), (xb, yb) = boxes[i], boxes[j]
                ds = [self.s.BoolVar("") for _ in range(8)]
                # inactive if either box is absent
                self.s.Add(sum(ds) >= xa[6] + ya[6] + xb[6] + yb[6] - 3)
                pairs = [
                    (xa[2], xb[0], xa[4][1] - xb[3][0]),
                    (xb[2], xa[0], xb[4][1] - xa[3][0]),
                    (ya[2], yb[0], ya[4][1] - yb[3][0]),
                    (yb[2], ya[0], yb[4][1] - ya[3][0]),
                ]
                for d, (end, start, big) in zip(ds, pairs):
                    self.s.Add(end - start <= max(big, 0) * (1 - d))
                for d, box in zip(ds[4:], (xa, ya, xb, yb)):
                    self.s.Add(box[1] <= box[5][1] * (1 - d))

    def build(self):
        for ci, c in enumerate(self.p.constraints):
            k = c.WhichOneof("constraint")
            if k == "interval":
                self.interval(ci, c)
        for ci, c in enumerate(self.p.constraints):
            k = c.WhichOneof("constraint")
            enf = list(c.enforcement_literal)
            if k == "linear":
                self.linear(c)
            elif k == "bool_or":
                self.s.Add(
                    sum(self.lit(lit) for lit in c.bool_or.literals)
                    >= 1 - (self.slack(enf) if enf else 0)
                )
            elif k == "bool_and":
                for lit in c.bool_and.literals:
                    self.s.Add(self.lit(lit) >= 1 - (self.slack(enf) if enf else 0))
            elif k == "at_most_one":
                self.s.Add(sum(self.lit(lit) for lit in c.at_most_one.literals) <= 1)
            elif k == "exactly_one":
                self.s.Add(sum(self.lit(lit) for lit in c.exactly_one.literals) == 1)
            elif k == "element":
                self.element(c)
            elif k == "table":
                self.table(c)
            elif k == "int_prod":
                self.product(c)
            elif k == "int_div":
                self.division(c)
            elif k == "lin_max":
                self.lin_max(c)
            elif k == "no_overlap_2d":
                self.no_overlap_2d(c)
            elif k == "interval":
                pass
            else:
                raise MipUnsupported(k)
        if self.p.HasField("objective"):
            o = self.p.objective
            expr = (
                sum(cf * self.lit_int(v) for v, cf in zip(o.vars, o.coeffs)) + o.offset
            )
            self.report = o.scaling_factor or 1.0
            self.s.Minimize(expr)
        elif self.p.HasField("floating_point_objective"):
            o = self.p.floating_point_objective
            expr = (
                sum(cf * self.lit_int(v) for v, cf in zip(o.vars, o.coeffs)) + o.offset
            )
            self.report = 1.0
            (self.s.Maximize if o.maximize else self.s.Minimize)(expr)
        else:
            self.report = 1.0


def _domain_values(dom):
    vals = []
    for lo, hi in zip(dom[::2], dom[1::2]):
        vals.extend(range(lo, hi + 1))
    return vals


class TooBig(Exception):
    pass


class _BruteForceSearch:
    def __init__(self, proto, max_nodes: int = 5_000_000):
        self.p = proto
        self.n = len(proto.variables)
        self.domains = [list(v.domain) for v in proto.variables]
        self.max_nodes = max_nodes
        self.cons = list(proto.constraints)
        self.intervals = {
            i: c.interval for i, c in enumerate(self.cons) if c.HasField("interval")
        }
        # which constraints mention each var
        self.vars_of = [self._vars(c) for c in self.cons]
        self.watch: list[list[int]] = [[] for _ in range(self.n)]
        for ci, vs in enumerate(self.vars_of):
            for v in vs:
                self.watch[v].append(ci)
        self.defs = self._definitions()
        self.objective = self._objective()

    # ---- structure ----
    @staticmethod
    def _ref(r):
        return r if r >= 0 else -r - 1

    def _expr_vars(self, e):
        return {self._ref(v) for v in e.vars}

    def _vars(self, c):
        k = c.WhichOneof("constraint")
        vs = {self._ref(x) for x in c.enforcement_literal}
        if k == "linear":
            vs |= {self._ref(v) for v in c.linear.vars}
        elif k in ("bool_or", "bool_and", "at_most_one", "exactly_one"):
            vs |= {self._ref(v) for v in getattr(c, k).literals}
        elif k == "element":
            e = c.element
            if e.HasField("linear_index"):
                vs |= self._expr_vars(e.linear_index) | self._expr_vars(e.linear_target)
                for x in e.exprs:
                    vs |= self._expr_vars(x)
            else:
                vs |= {e.index, e.target, *e.vars}
        elif k == "table":
            t = c.table
            if len(t.exprs):
                for x in t.exprs:
                    vs |= self._expr_vars(x)
            else:
                vs |= set(t.vars)
        elif k in ("int_prod", "int_div", "lin_max"):
            g = getattr(c, k)
            vs |= self._expr_vars(g.target)
            for x in g.exprs:
                vs |= self._expr_vars(x)
        elif k == "interval":
            iv = c.interval
            vs |= (
                self._expr_vars(iv.start)
                | self._expr_vars(iv.size)
                | self._expr_vars(iv.end)
            )
        elif k == "no_overlap_2d":
            for i in (*c.no_overlap_2d.x_intervals, *c.no_overlap_2d.y_intervals):
                vs |= self._vars(self.cons[i])
        else:
            raise NotImplementedError(k)
        return vs

    def _definitions(self):
        """var -> constraint index that computes it from other vars."""
        defs = {}
        for ci, c in enumerate(self.cons):
            if len(c.enforcement_literal):
                continue
            k = c.WhichOneof("constraint")
            if k == "element" and c.element.HasField("linear_target"):
                t = c.element.linear_target
                if len(t.vars) == 1 and abs(t.coeffs[0]) == 1:
                    defs.setdefault(self._ref(t.vars[0]), ci)
            elif k in ("int_prod", "int_div", "lin_max"):
                t = getattr(c, k).target
                if len(t.vars) == 1 and abs(t.coeffs[0]) == 1:
                    defs.setdefault(self._ref(t.vars[0]), ci)
            elif k == "linear":
                lin = c.linear
                if list(lin.domain) == [lin.domain[0], lin.domain[0]]:
                    for v, co in zip(lin.vars, lin.coeffs):
                        if abs(co) == 1:
                            defs.setdefault(self._ref(v), ci)
        return defs

    def _objective(self):
        """(vars, coeffs, offset, report): minimize offset + sum(c * x); the
        user-facing objective is report * that."""
        if self.p.HasField("objective"):
            o = self.p.objective
            return list(o.vars), list(o.coeffs), o.offset, o.scaling_factor or 1.0
        if self.p.HasField("floating_point_objective"):
            o = self.p.floating_point_objective
            sign = -1.0 if o.maximize else 1.0
            return list(o.vars), [c * sign for c in o.coeffs], o.offset * sign, sign
        return None

    # ---- evaluation ----
    def _val(self, a, r):
        x = a[self._ref(r)]
        if x is None:
            return None
        return x if r >= 0 else 1 - x

    def _lin(self, a, e):
        s = e.offset
        for v, c in zip(e.vars, e.coeffs):
            x = self._val(a, v)
            if x is None:
                return None
            s += c * x
        return s

    def _enforced(self, a, c):
        """True / False / None (unknown)."""
        out = True
        for lit in c.enforcement_literal:
            x = self._val(a, lit)
            if x is None:
                out = None
            elif x == 0:
                return False
        return out

    def _check(self, a, ci) -> bool:
        """False only when constraint ci is certainly violated."""
        c = self.cons[ci]
        k = c.WhichOneof("constraint")
        if k == "interval":
            return True
        enf = self._enforced(a, c)
        if enf is False:
            return True
        if k == "no_overlap_2d":
            return self._no_overlap(a, c.no_overlap_2d)
        vals = [
            a[v]
            for v in self.vars_of[ci]
            if not any(self._ref(lit) == v for lit in c.enforcement_literal)
        ]
        if any(x is None for x in vals) or enf is None:
            return True
        if k == "linear":
            lin = c.linear
            s = sum(co * self._val(a, v) for v, co in zip(lin.vars, lin.coeffs))
            return any(
                lo <= s <= hi for lo, hi in zip(lin.domain[::2], lin.domain[1::2])
            )
        if k == "bool_or":
            return any(self._val(a, x) for x in c.bool_or.literals)
        if k == "bool_and":
            return all(self._val(a, x) for x in c.bool_and.literals)
        if k == "at_most_one":
            return sum(self._val(a, x) for x in c.at_most_one.literals) <= 1
        if k == "exactly_one":
            return sum(self._val(a, x) for x in c.exactly_one.literals) == 1
        if k == "element":
            e = c.element
            i = self._lin(a, e.linear_index)
            if not 0 <= i < len(e.exprs):
                return False
            return self._lin(a, e.exprs[i]) == self._lin(a, e.linear_target)
        if k == "table":
            t = c.table
            row = tuple(self._lin(a, x) for x in t.exprs)
            w = len(row)
            vals = list(t.values)
            hit = any(tuple(vals[j : j + w]) == row for j in range(0, len(vals), w))
            return hit != bool(t.negated)
        if k == "int_prod":
            g = c.int_prod
            return math.prod(self._lin(a, x) for x in g.exprs) == self._lin(a, g.target)
        if k == "int_div":
            g = c.int_div
            n, d = (self._lin(a, x) for x in g.exprs)
            q = abs(n) // abs(d) * (1 if (n >= 0) == (d > 0) else -1)
            return q == self._lin(a, g.target)
        if k == "lin_max":
            g = c.lin_max
            return max(self._lin(a, x) for x in g.exprs) == self._lin(a, g.target)
        raise NotImplementedError(k)

    def _box(self, a, ii):
        c = self.cons[ii]
        pres = self._enforced(a, c)
        iv = c.interval
        s, e = self._lin(a, iv.start), self._lin(a, iv.end)
        return pres, s, e

    def _no_overlap(self, a, no):
        boxes = []
        for xi, yi in zip(no.x_intervals, no.y_intervals):
            px, x0, x1 = self._box(a, xi)
            py, y0, y1 = self._box(a, yi)
            if px is False or py is False:
                continue
            if None in (px, py, x0, x1, y0, y1):
                continue
            if x0 == x1 or y0 == y1:  # zero-area boxes never overlap
                continue
            boxes.append((x0, x1, y0, y1))
        for i in range(len(boxes)):
            a0, a1, b0, b1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                c0, c1, d0, d1 = boxes[j]
                if a0 < c1 and c0 < a1 and b0 < d1 and d0 < b1:
                    return False
        return True

    def _compute(self, a, v):
        """Value of a defined var from its definition, or None."""
        ci = self.defs[v]
        c = self.cons[ci]
        k = c.WhichOneof("constraint")
        if k == "element":
            e = c.element
            i = self._lin(a, e.linear_index)
            if i is None or not 0 <= i < len(e.exprs):
                return None
            val = self._lin(a, e.exprs[i])
            t = e.linear_target
        elif k in ("int_prod", "int_div", "lin_max"):
            g = getattr(c, k)
            xs = [self._lin(a, x) for x in g.exprs]
            if None in xs:
                return None
            if k == "int_prod":
                val = math.prod(xs)
            elif k == "lin_max":
                val = max(xs)
            else:
                n, d = xs
                if d == 0:
                    return None
                val = abs(n) // abs(d) * (1 if (n >= 0) == (d > 0) else -1)
            t = g.target
        else:  # linear equality
            lin = c.linear
            rest = 0
            coef = None
            for r, co in zip(lin.vars, lin.coeffs):
                if self._ref(r) == v and coef is None:
                    coef = co if r >= 0 else -co
                    if r < 0:
                        rest += co
                    continue
                x = self._val(a, r)
                if x is None:
                    return None
                rest += co * x
            num = lin.domain[0] - rest
            if num % coef:
                return None
            return num // coef
        # target = coeff * v + offset
        co, off = t.coeffs[0], t.offset
        if (val - off) % co:
            return None
        return (val - off) // co

    def _in_domain(self, v, x):
        d = self.domains[v]
        return any(lo <= x <= hi for lo, hi in zip(d[::2], d[1::2]))

    # ---- search ----
    def solve(self):
        order = [v for v in range(self.n) if v not in self.defs]
        # Booleans and small domains first, wide (offsets) last; defined vars
        # after everything, branched only if their definition never resolved.
        order.sort(
            key=lambda v: (
                len(_domain_values(self.domains[v])) > 2,
                len(_domain_values(self.domains[v])),
            )
        )
        order += [v for v in range(self.n) if v in self.defs]
        a: list = [None] * self.n
        self.best = None
        self.best_a = None
        self.nodes = 0
        self._dfs(a, order, 0)
        return self.best

    def _propagate(self, a, assigned):
        """Fill defined vars computable now; False on a violation."""
        changed = True
        while changed:
            changed = False
            for v in self.defs:
                if a[v] is None:
                    x = self._compute(a, v)
                    if x is not None:
                        if not self._in_domain(v, x):
                            return False
                        a[v] = x
                        assigned.append(v)
                        changed = True
        return True

    def _consistent(self, a, touched):
        seen = set()
        for v in touched:
            for ci in self.watch[v]:
                if ci not in seen:
                    seen.add(ci)
                    if not self._check(a, ci):
                        return False
        return True

    def _obj(self, a):
        if self.objective is None:
            return 0
        vs, cs, off, _ = self.objective
        return off + sum(c * self._val(a, v) for v, c in zip(vs, cs))

    def objective_value(self):
        return (
            None
            if self.best is None
            else self.best * (self.objective[3] if self.objective else 1)
        )

    def _dfs(self, a, order, i):
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise TooBig(self.nodes)
        if i == len(order):
            assert all(x is not None for x in a)
            val = self._obj(a)
            if self.best is None or val < self.best - 1e-9:
                self.best, self.best_a = val, list(a)
            return
        v = order[i]
        if a[v] is not None:
            self._dfs(a, order, i + 1)
            return
        for x in _domain_values(self.domains[v]):
            a[v] = x
            assigned = [v]
            ok = self._propagate(a, assigned) and self._consistent(a, assigned)
            if ok:
                self._dfs(a, order, i + 1)
            for u in assigned:
                a[u] = None
