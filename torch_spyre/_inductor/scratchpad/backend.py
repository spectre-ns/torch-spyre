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
    def IntVarFromValues(self, values: Sequence[int], name: str) -> Var: ...
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
    def AddDivisionEquality(
        self, target: Var, num: LinearExpr, den: LinearExpr
    ) -> Constraint: ...
    def AddMinEquality(
        self, target: Var, exprs: Sequence[LinearExpr]
    ) -> Constraint: ...
    def AddMultiplicationEquality(
        self, target: Var, factors: Sequence[LinearExpr]
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
    def SetHint(self, vars: Sequence[Var], values: Sequence[int]) -> None: ...
    def HintSolution(self) -> None: ...
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

    def IntVarFromValues(self, values, name):
        return self.model.new_int_var_from_domain(
            cp_model.Domain.from_values(list(values)), name
        )

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

    def AddMultiplicationEquality(self, target, factors):
        return self.model.add_multiplication_equality(target, factors)

    def AddDivisionEquality(self, target, num, den):
        return self.model.add_division_equality(target, num, den)

    def AddMinEquality(self, target, exprs):
        return self.model.add_min_equality(target, exprs)

    def NewOptionalIntervalVar(self, start, size, end, present, name):
        return self.model.new_optional_interval_var(start, size, end, present, name)

    def AddNoOverlap2D(self, x, y):
        return self.model.add_no_overlap_2d(x, y)

    def Minimize(self, expr):
        self.model.minimize(expr)

    def Maximize(self, expr):
        self.model.maximize(expr)

    def SetHint(self, vars, values):
        for var, value in zip(vars, values):
            self.model.add_hint(var, value)

    def HintSolution(self):
        """Hint every variable with the last solution, so a stage that holds
        the previous optimum starts from a solution that meets it."""
        self.model.clear_hints()
        for i in range(len(self.model.proto.variables)):
            var = self.model.get_int_var_from_proto_index(i)
            self.model.add_hint(var, self.Value(var))

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
            a, b = b, a
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


class _DomainValues:
    """A domain's values in order, never listed: offset and cost domains span
    billions of values, and listing them exhausts memory."""

    def __init__(self, dom):
        self.spans = list(zip(dom[::2], dom[1::2]))
        self.size = sum(hi - lo + 1 for lo, hi in self.spans)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if i < 0:
            i += self.size
        for lo, hi in self.spans:
            if i <= hi - lo:
                return lo + i
            i -= hi - lo + 1
        raise IndexError(i)

    def __iter__(self):
        for lo, hi in self.spans:
            yield from range(lo, hi + 1)


def _complementary(d1, d2) -> bool:
    """Whether two domains (flat [lo, hi, ...] lists) partition the int64 range."""
    spans = sorted([*zip(d1[::2], d1[1::2]), *zip(d2[::2], d2[1::2])])
    nxt = -(2**63)
    for lo, hi in spans:
        if lo != nxt:
            return False
        nxt = hi + 1
    return nxt == 2**63


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
        """var -> constraint index that computes it from other vars. One
        constraint defines one var: an equality defining all its vars leaves
        each waiting on the others, and none ever resolves."""
        defs = {}
        self._lookups: dict[int, dict] = {}
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
                    # The newest var: an auxiliary is made after its inputs.
                    unit = [
                        self._ref(v)
                        for v, co in zip(lin.vars, lin.coeffs)
                        if abs(co) == 1 and self._ref(v) not in defs
                    ]
                    if unit:
                        defs[max(unit)] = ci
            elif k == "table":
                lookup = self._functional_table(c.table)
                if lookup is not None and lookup[0] not in defs:
                    defs[lookup[0]] = ci
                    self._lookups[ci] = lookup[1]
        self._reified = self._reifications(defs)
        for v, ci in self._reified.items():
            defs[v] = ci
        return defs

    def _functional_table(self, t):
        """(last var, row -> value) when a table's last column is a var that
        is a function of the other columns, as a lazy ``tie`` posts it (a
        constant column pins it outright)."""
        if t.negated:
            return None
        if len(t.exprs):
            last = t.exprs[-1]
            if (
                len(last.vars) != 1
                or last.coeffs[0] != 1
                or last.offset
                or last.vars[0] < 0
            ):
                return None
            var, w = last.vars[0], len(t.exprs)
        else:
            var, w = t.vars[-1], len(t.vars)
        if w < 2:
            return None
        vals = list(t.values)
        lookup: dict = {}
        for j in range(0, len(vals), w):
            key, out = tuple(vals[j : j + w - 1]), vals[j + w - 1]
            if lookup.setdefault(key, out) != out:
                return None
        return var, lookup

    def _reifications(self, defs):
        """bool var -> the constraint it enforces, for a ``b => lin in D``,
        ``not b => lin in complement(D)`` pair (b is whether lin lies in D),
        or a ``b => and(L)``, ``not b => or(not L)`` pair (b is and(L))."""
        out = {}
        ors = {
            (c.enforcement_literal[0], frozenset(c.bool_or.literals))
            for c in self.cons
            if c.WhichOneof("constraint") == "bool_or"
            and len(c.enforcement_literal) == 1
        }
        for ci, c in enumerate(self.cons):
            if (
                c.WhichOneof("constraint") != "bool_and"
                or len(c.enforcement_literal) != 1
            ):
                continue
            lit = c.enforcement_literal[0]
            if lit < 0 or lit in defs or lit in out:
                continue
            negated = frozenset(-x - 1 for x in c.bool_and.literals)
            if (-lit - 1, negated) in ors:
                out[lit] = ci
        by_expr: dict = {}
        for ci, c in enumerate(self.cons):
            if (
                c.WhichOneof("constraint") != "linear"
                or len(c.enforcement_literal) != 1
            ):
                continue
            key = (tuple(c.linear.vars), tuple(c.linear.coeffs))
            by_expr.setdefault(key, []).append(ci)
        for group in by_expr.values():
            for ci in group:
                lit = self.cons[ci].enforcement_literal[0]
                if lit < 0 or lit in defs or lit in out:
                    continue
                for cj in group:
                    if self.cons[cj].enforcement_literal[0] == -lit - 1 and (
                        _complementary(
                            self.cons[ci].linear.domain, self.cons[cj].linear.domain
                        )
                    ):
                        out[lit] = ci
                        break
        return out

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
        if v in self._reified and k == "bool_and":
            xs = [self._val(a, x) for x in c.bool_and.literals]
            return None if None in xs else int(all(xs))
        if v in self._reified:
            lin = c.linear
            s = 0
            for r, co in zip(lin.vars, lin.coeffs):
                x = self._val(a, r)
                if x is None:
                    return None
                s += co * x
            return int(
                any(lo <= s <= hi for lo, hi in zip(lin.domain[::2], lin.domain[1::2]))
            )
        if k == "table":
            t = c.table
            cols = (
                [self._lin(a, x) for x in t.exprs]
                if len(t.exprs)
                else [a[x] for x in t.vars]
            )
            row = tuple(cols[:-1])
            return None if None in row else self._lookups[ci].get(row)
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
                len(_DomainValues(self.domains[v])) > 2,
                len(_DomainValues(self.domains[v])),
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
        for x in _DomainValues(self.domains[v]):
            a[v] = x
            assigned = [v]
            ok = self._propagate(a, assigned) and self._consistent(a, assigned)
            if ok:
                self._dfs(a, order, i + 1)
            for u in assigned:
                a[u] = None


class _AnnealSearch(_BruteForceSearch):
    """Simulated annealing over the recorded model, no linearization: every
    constraint kind is evaluated natively (the same checks as the exhaustive
    search). Offsets are never searched: each resident box goes, in placement
    order, to the lowest offset clear of the boxes placed before it that
    share its time.

    A move reassigns one free var, or moves one box in the placement order,
    then repairs: while a constraint is violated, it rewrites the free vars
    behind one violated constraint (a whole table row at once), keeping the
    candidate that leaves the fewest violations. Valid plans are not one
    connected region under these moves, so the search may leave them: an
    excursion through plans with at most ``EXCURSION_VIOLATIONS`` violations
    (each costing ``EXCURSION_PENALTY`` times the objective's full scale, so
    leaving the valid plans is rare and short) that finds no valid plan
    within ``EXCURSION_MOVES`` moves is undone back to the last valid one.
    Only valid plans are kept as the best. From an invalid start it first
    descends to a valid plan.

    Moves are differential: only the defined vars downstream of a change are
    recomputed, the packer reruns only when a packing input changed, only the
    constraints watching a changed var are re-checked, and a rejected move is
    undone from a journal. ``SPYRE_ANNEAL_VERIFY=N`` checks the running state
    against a from-scratch evaluation every N moves."""

    REPAIR_ROUNDS = 6
    REPAIR_CANDIDATES = 6
    EXCURSION_VIOLATIONS = 2
    EXCURSION_MOVES = 30
    EXCURSION_PENALTY = 10.0

    def anneal(self, seed: int, budget_s: float, start=None):
        import os
        import random

        rng = random.Random(seed)
        verify = int(os.environ.get("SPYRE_ANNEAL_VERIFY", "0"))
        self._boxes, self._ties = self._placement()
        self._doms = doms = [_DomainValues(self.domains[v]) for v in range(self.n)]
        packed = {b[0] for b in self._boxes if b[0] not in self.defs}
        self._order_definitions()
        free = [
            v
            for v in range(self.n)
            if (v not in self.defs or v in self._floating) and v not in packed
        ]
        self._free = set(free)
        self._sources_of: dict = {}
        self._pack_inputs = self._packing_inputs()
        self._obj_terms: dict = {}
        if self.objective:
            for ref, c in zip(self.objective[0], self.objective[1]):
                self._obj_terms.setdefault(self._ref(ref), []).append((ref, c))
        a = [None] * self.n
        for v in (*free, *packed):
            a[v] = doms[v][0]
        if start is not None:  # the previous ladder stage's plan (a hint)
            for v in (*free, *packed):
                if v < len(start) and start[v] is not None:
                    a[v] = start[v]
        # First fit in the hint's offset order never places a box above its
        # hinted offset, so the hint's plan still packs.
        order = sorted(
            (i for i, b in enumerate(self._boxes) if b[0] in packed),
            key=lambda i: a[self._boxes[i][0]],
        )
        self._evaluate(a, order)
        self._sat = [self._check(a, ci) for ci in range(len(self.cons))]
        self._unsat = {ci for ci, ok in enumerate(self._sat) if not ok}
        self._bad = {d for d in self._topo if not self._in_domain(d, a[d])}
        self._nviol = len(self._unsat) + len(self._bad)
        self._violations(a)  # the start's count, for whoever is watching
        self._objv = self._obj(a)
        temp = (
            max(1.0, sum(abs(c) for c in self.objective[1])) if self.objective else 1.0
        )
        penalty = self.EXCURSION_PENALTY * temp
        best = (self._objv, list(a)) if self._nviol == 0 else None
        excursion: list = []  # the accepted moves since the last valid plan
        t0 = time.perf_counter()
        moves = 0
        while time.perf_counter() - t0 < budget_s and (free or len(order) > 1):
            moves += 1
            journal: list = []
            flips: list = []
            state = (self._nviol, self._objv)
            saved_order = None
            if len(order) > 1 and (not free or rng.random() < 0.2):
                saved_order = list(order)
                i, j = rng.sample(range(len(order)), 2)
                order.insert(j, order.pop(i))
                moved = self._repack(a, order, journal)
                self._apply(a, order, moved, journal, flips, 0, repacked=True)
            else:
                v = rng.choice(free)
                if not self._try(a, order, [(v, rng.choice(doms[v]))], journal, flips):
                    continue
            self._repair(a, order, journal, flips, rng)
            if best is None:  # descending to a first valid plan
                accept = self._nviol <= state[0]
            elif self._nviol > self.EXCURSION_VIOLATIONS:
                accept = False
            else:
                delta = self._objv - state[1] + penalty * (self._nviol - state[0])
                accept = delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9))
            if not accept:
                self._undo_move(a, order, (journal, flips, state, saved_order))
            elif self._nviol == 0:
                excursion.clear()
                if best is None or self._objv < best[0] - 1e-9:
                    best = (self._objv, list(a))
            elif best is not None:
                excursion.append((journal, flips, state, saved_order))
                if len(excursion) > self.EXCURSION_MOVES:
                    for move in reversed(excursion):
                        self._undo_move(a, order, move)
                    excursion.clear()
            if verify and moves % verify == 0:
                self._verify(a, order)
            temp = max(temp * 0.9995, 1e-3)
        self.moves = moves
        if best is not None:
            self.best, self.best_a = best
        return best is not None

    # ---- moves ----

    def _try(self, a, order, assigns, journal, flips) -> bool:
        """Apply ``assigns`` (var, value); False when nothing changes."""
        j0 = len(journal)
        seeds = []
        for v, x in assigns:
            if a[v] != x:
                journal.append((v, a[v]))
                a[v] = x
                seeds.append(v)
        if seeds:
            self._apply(a, order, seeds, journal, flips, j0)
        return bool(seeds)

    def _undo_move(self, a, order, move):
        journal, flips, state, saved_order = move
        self._rollback(a, journal, flips, 0, 0, state)
        if saved_order is not None:
            order[:] = saved_order

    def _rollback(self, a, journal, flips, j0, f0, state):
        for u, old in reversed(journal[j0:]):
            a[u] = old
        for kind, key, old in reversed(flips[f0:]):
            if kind == "sat":
                self._sat[key] = old
                (self._unsat.discard if old else self._unsat.add)(key)
            else:
                (self._bad.add if old else self._bad.discard)(key)
        del journal[j0:], flips[f0:]
        self._nviol, self._objv = state

    def _repair(self, a, order, journal, flips, rng):
        """Greedily rewrite the free vars behind violated constraints."""
        for _ in range(self.REPAIR_ROUNDS):
            if not self._nviol:
                return
            violated = sorted(self._unsat) + [self.defs[d] for d in sorted(self._bad)]
            ci = rng.choice(violated)
            pick = None
            for cand in self._candidates(a, ci, rng):
                j0, f0, state = len(journal), len(flips), (self._nviol, self._objv)
                if self._try(a, order, cand, journal, flips):
                    if pick is None or self._nviol < pick[0]:
                        pick = (self._nviol, cand)
                    self._rollback(a, journal, flips, j0, f0, state)
            if pick is None or pick[0] >= self._nviol:
                return
            self._try(a, order, pick[1], journal, flips)

    def _candidates(self, a, ci, rng):
        """Assignments that might satisfy constraint ci: a row for a table's
        free columns, else one free var behind ci at a new value."""
        c = self.cons[ci]
        k = self.REPAIR_CANDIDATES
        if c.WhichOneof("constraint") == "table" and not c.table.negated:
            t = c.table
            cols = (
                [
                    x.vars[0]
                    if len(x.vars) == 1
                    and x.coeffs[0] == 1
                    and not x.offset
                    and x.vars[0] >= 0
                    else None
                    for x in t.exprs
                ]
                if len(t.exprs)
                else list(t.vars)
            )
            free = [
                (j, v) for j, v in enumerate(cols) if v is not None and v in self._free
            ]
            if free:
                w = len(cols)
                rows = len(t.values) // w
                for r in rng.sample(range(rows), min(k, rows)):
                    yield [(v, t.values[r * w + j]) for j, v in free]
                return
        sources = self._sources(ci)
        for v in rng.sample(sources, min(k, len(sources))):
            yield [(v, rng.choice(self._doms[v]))]

    def _sources(self, ci):
        """The free vars constraint ci depends on, through definitions."""
        if ci not in self._sources_of:
            seen, todo, out = set(), list(self.vars_of[ci]), []
            while todo:
                u = todo.pop()
                if u in seen:
                    continue
                seen.add(u)
                if u in self._free:
                    out.append(u)
                elif u in self._pos:
                    todo.extend(self._deps[u])
            self._sources_of[ci] = sorted(out)
        return self._sources_of[ci]

    # ---- structure ----

    def _order_definitions(self):
        """Each defined var after the vars its definition reads (Kahn). A var
        left in a cycle floats: the search moves it like a free one."""
        deps = {
            d: {u for u in self.vars_of[ci] if u != d} for d, ci in self.defs.items()
        }
        waiting = {d: sum(u in self.defs for u in ds) for d, ds in deps.items()}
        readers: dict = {}
        for d, ds in deps.items():
            for u in ds:
                readers.setdefault(u, []).append(d)
        ready = sorted(d for d, k in waiting.items() if k == 0)
        topo = []
        while ready:
            d = ready.pop()
            topo.append(d)
            for w in readers.get(d, ()):
                waiting[w] -= 1
                if waiting[w] == 0:
                    ready.append(w)
        self._topo = topo
        self._deps = deps
        self._pos = {d: i for i, d in enumerate(topo)}
        self._floating = set(self.defs) - set(topo)
        self._readers = {
            u: [d for d in ds if d in self._pos] for u, ds in readers.items()
        }

    def _packing_inputs(self):
        """Vars the packer reads: presences, lifetimes, sizes, merge
        literals, and offsets it does not choose."""
        out = set()
        packed = {b[0] for b in self._boxes if b[0] not in self.defs}
        for off, lit, xs, xe, ys in self._boxes:
            if lit is not None:
                out.add(self._ref(lit))
            for e in (xs, xe, ys):
                out |= self._expr_vars(e)
            if off not in packed:
                out.add(off)
        for ties in self._ties.values():
            out |= {self._ref(lit) for lit, _ in ties}
        return out

    # ---- evaluation ----

    def _define(self, a, d):
        x = self._compute(a, d)
        return self._doms[d][0] if x is None else x

    def _evaluate(self, a, order):
        """Every defined var and offset, from scratch."""
        for d in self._topo:
            a[d] = self._define(a, d)
        self._pack(a, order)
        for d in self._topo:
            a[d] = self._define(a, d)

    def _forward(self, a, seeds, journal):
        """Recompute, in definition order, the defined vars downstream of
        ``seeds``; return every var that changed."""
        import heapq

        heap: list = []
        queued: set = set()
        for u in seeds:
            for d in self._readers.get(u, ()):
                if d not in queued:
                    queued.add(d)
                    heapq.heappush(heap, (self._pos[d], d))
        changed = list(seeds)
        while heap:
            _, d = heapq.heappop(heap)
            x = self._define(a, d)
            if x == a[d]:
                continue
            journal.append((d, a[d]))
            a[d] = x
            changed.append(d)
            for w in self._readers.get(d, ()):
                if w not in queued:
                    queued.add(w)
                    heapq.heappush(heap, (self._pos[w], w))
        return changed

    def _repack(self, a, order, journal):
        offs = [self._boxes[i][0] for i in order]
        old = [a[o] for o in offs]
        self._pack(a, order)
        moved = [(o, x) for o, x in zip(offs, old) if a[o] != x]
        journal.extend(moved)
        return [o for o, _ in moved]

    def _apply(self, a, order, seeds, journal, flips, j0, repacked=False):
        """Settle what ``seeds`` (changed since journal[j0]) reach and update
        the violation count and objective."""
        changed = self._forward(a, seeds, journal)
        while not repacked and self._pack_inputs.intersection(changed):
            moved = self._repack(a, order, journal)
            changed += self._forward(a, moved, journal)
            # An offset never feeds the packer back, so one pass settles it.
            repacked = True
        first: dict = {}
        for u, old in journal[j0:]:
            first.setdefault(u, old)
        touched: set = set()
        for u, old in first.items():
            if a[u] == old:
                continue
            for ref, c in self._obj_terms.get(u, ()):
                if ref >= 0:
                    self._objv += c * (a[u] - old)
                else:
                    self._objv -= c * (a[u] - old)
            if u in self._pos:
                bad = not self._in_domain(u, a[u])
                if bad != (u in self._bad):
                    flips.append(("bad", u, u in self._bad))
                    (self._bad.add if bad else self._bad.discard)(u)
                    self._nviol += 1 if bad else -1
            touched.update(self.watch[u])
        for ci in touched:
            ok = self._check(a, ci)
            if ok != self._sat[ci]:
                flips.append(("sat", ci, self._sat[ci]))
                self._sat[ci] = ok
                (self._unsat.discard if ok else self._unsat.add)(ci)
                self._nviol += -1 if ok else 1

    def _verify(self, a, order):
        """The running state equals a from-scratch evaluation."""
        ref = list(a)
        self._evaluate(ref, order)
        nviol = self._violations(ref) + sum(
            not self._in_domain(d, ref[d]) for d in self._topo
        )
        assert ref == a, [
            (self.p.variables[v].name, a[v], ref[v])
            for v in range(self.n)
            if a[v] != ref[v]
        ][:10]
        assert nviol == self._nviol, (nviol, self._nviol)
        assert self._unsat == {
            ci for ci in range(len(self.cons)) if not self._check(ref, ci)
        }
        assert abs(self._obj(ref) - self._objv) < 1e-6, (self._obj(ref), self._objv)

    def _placement(self):
        """The no-overlap boxes as (offset var, presence lit, x start, x end,
        y size), and each offset's merge ties: (lit, other offset) for a
        ``lit => off == other``."""
        boxes = []
        for c in self.cons:
            if c.WhichOneof("constraint") != "no_overlap_2d":
                continue
            no = c.no_overlap_2d
            for xi, yi in zip(no.x_intervals, no.y_intervals):
                x, y = self.cons[xi], self.cons[yi]
                s = y.interval.start
                if len(s.vars) != 1 or s.coeffs[0] != 1 or s.offset or s.vars[0] < 0:
                    continue
                lit = y.enforcement_literal[0] if len(y.enforcement_literal) else None
                boxes.append(
                    (s.vars[0], lit, x.interval.start, x.interval.end, y.interval.size)
                )
        offsets = {b[0] for b in boxes}
        ties: dict = {}
        for c in self.cons:
            if (
                c.WhichOneof("constraint") != "linear"
                or len(c.enforcement_literal) != 1
            ):
                continue
            lin = c.linear
            if (
                len(lin.vars) == 2
                and sorted(lin.coeffs) == [-1, 1]
                and list(lin.domain) == [0, 0]
                and all(v >= 0 and v in offsets for v in lin.vars)
            ):
                p, q = lin.vars
                lit = c.enforcement_literal[0]
                ties.setdefault(p, []).append((lit, q))
                ties.setdefault(q, []).append((lit, p))
        return boxes, ties

    def _pack(self, a, order):
        """Offsets for the boxes in ``order`` (``_C.pack_boxes``); a present
        box outside it keeps its offset as an obstacle."""
        from torch_spyre import _C

        packed = set(order)
        index = {b[0]: i for i, b in enumerate(self._boxes)}
        x0, x1, size, present, pinned, ties = [], [], [], [], [], []
        for i, (off, lit, xs, xe, ys) in enumerate(self._boxes):
            dims = (self._lin(a, xs), self._lin(a, xe), self._lin(a, ys))
            on = None not in dims and (lit is None or bool(self._val(a, lit)))
            x0.append(dims[0] if on else 0)
            x1.append(dims[1] if on else 0)
            size.append(dims[2] if on else 0)
            present.append(on)
            pinned.append(a[off] if i not in packed and a[off] is not None else -1)
            ties.append(
                [index[q] for lit, q in self._ties.get(off, ()) if self._val(a, lit)]
            )
        offsets = _C.pack_boxes(order, x0, x1, size, present, pinned, ties)
        for i in order:
            a[self._boxes[i][0]] = offsets[i]

    def _violations(self, a):
        return sum(0 if self._check(a, ci) else 1 for ci in range(len(self.cons)))


class Annealing(_SolvedRecording):
    """Simulated annealing over the recorded model, scored natively."""

    def __init__(self, seed: int = 0, budget_s: float = 10.0, model=None):
        super().__init__(model)
        self.seed = seed
        self.budget_s = budget_s

    def Solve(self, objective: bool = False) -> int:
        search = _AnnealSearch(self._proto(), max_nodes=0)
        t = time.perf_counter()
        found = search.anneal(self.seed, self.budget_s, start=self._values)
        self._wall = time.perf_counter() - t
        if not found:
            self._status = cp_model.UNKNOWN
            return self._status
        self._values = search.best_a
        self._objective_value = search.objective_value()
        self._status = cp_model.FEASIBLE
        return self._status


def from_env() -> CpSat:
    """The backend ``SPYRE_LX_BACKEND`` names: ``cpsat`` (default),
    ``mip:<SCIP|HIGHS|CBC>``, ``bruteforce`` or ``anneal[:<budget s>]``."""
    import os

    choice = os.environ.get("SPYRE_LX_BACKEND", "cpsat")
    if choice.startswith("mip"):
        return Mip(choice.partition(":")[2] or "SCIP")
    if choice == "bruteforce":
        return BruteForce()
    if choice.startswith("anneal"):
        return Annealing(budget_s=float(choice.partition(":")[2] or 10))
    return CpSat()
