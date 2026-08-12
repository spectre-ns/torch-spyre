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

"""Lowering of a sympy cost expression onto a CP-SAT model (R3.3).

A solver builds the decision variables -- per-buffer residency bools,
``AddElement``-selected per-config scalars -- and mints a sympy symbol for each.
An objective is then written in those symbols and lowered here into CP-SAT
expressions and constraints, so the objective is *injected* rather than
hardcoded (M2).

:func:`lower` is the whole of that conversion and the only entry point a solver
needs. It takes a model, an expression and an environment binding every symbol
the expression names to the variable that decides it -- nothing about who built
those variables, which is what lets the same lowering serve the joint
core-division solve, the placement-only solve and a test's hand-built model
alike. Two steps: rescale the coefficients to the integers CP-SAT works in
(:func:`_rescale_to_integers` -- a cost model speaks in ns and GB/s), then
interpret the result onto the model (:func:`interp`).

The supported grammar is exactly the method set of :class:`CpSatAnalysis`:
``Add`` / ``Mul`` (reified when both factors are non-constant) / ``Pow`` with a
small non-negative integer exponent / ``Min`` / ``Max``, over integer
constants. Anything else -- transcendentals, division by a variable, unbound
symbols -- raises :class:`CostExpressionError` naming what it was. Silently
approximating an objective is worse than a compile error, so nothing here
rewrites an expression it does not understand. The one rewrite is the rescale,
which is a rounding rather than an approximation of the objective's *shape*,
and it refuses (rather than drops) any term that rounding would erase.

Every reified node gets its own ``IntVar``, and CP-SAT needs an explicit domain
for each one. Those domains are derived by interpreting the expression a second
time under torch's own :class:`ValueRanges` arithmetic, which rides along in
:class:`Val`. A blanket-wide domain is not a valid shortcut -- see
``TestCostExprWideDomainsAreUnsafe`` in ``tests/inductor/test_scratchpad_solver.py``,
which pins ortools returning ``OPTIMAL`` with an objective that does not match
the solution it hands back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy
import torch
from ortools.sat.python import cp_model
from torch.utils._sympy.interp import sympy_interp
from torch.utils._sympy.value_ranges import SymPyValueRangeAnalysis as VRA
from torch.utils._sympy.value_ranges import ValueRanges

# CP-SAT rejects domains outside [-kint64max/2, +kint64max/2], and misbehaves
# well below that (TestCostExprWideDomainsAreUnsafe pins the observed failure at
# 2**55). 2**40 is comfortably inside both and far above any real cost magnitude.
_MAX_MAGNITUDE = 2**40

# What ``_rescale_to_integers`` aims the scaled objective's magnitude at. Four
# bits below the ceiling above: enough headroom for the interval arithmetic's own
# over-approximation of a subexpression, while spending the rest of the budget on
# coefficient precision.
_COST_TARGET_MAGNITUDE = 2**36


class CostExpressionError(Exception):
    """A cost expression outside the supported grammar, or a symbol we cannot
    bind. Always names the offending node or symbol."""

    pass


@dataclass
class Val:
    """An interpreted node: the CP-SAT object plus the range it lives in.

    Every reified node needs a *new* IntVar and CP-SAT needs an explicit domain
    for it. A blanket-wide domain is not a shortcut: at +/-2**55 the solver
    returns OPTIMAL with an objective that does not match the solution it hands
    back. So the range rides along, and the arithmetic for it is torch's, not
    ours.
    """

    e: Any  # int | cp_model.LinearExpr | cp_model.IntVar
    vr: ValueRanges

    @property
    def is_const(self) -> bool:
        return isinstance(self.e, int)


def _is_concrete(bound: Any) -> bool:
    """Whether a ValueRanges endpoint can become a CP-SAT domain edge.

    Test integrality, not finiteness: torch's ``int_oo`` reports
    ``is_finite == True``, so an ``is_finite`` guard lets infinity straight
    through and the failure surfaces much later, from inside the range
    arithmetic.
    """
    return bool(getattr(bound, "is_Integer", False))


def _bounds(vr: ValueRanges) -> tuple[int, int]:
    """ValueRanges -> a domain CP-SAT will accept, or a clean rejection."""
    lo, hi = vr.lower, vr.upper

    if not (_is_concrete(lo) and _is_concrete(hi)):
        raise CostExpressionError(
            f"unbounded subexpression {vr}: cannot size a CP-SAT variable; "
            "bound the leaf symbols"
        )

    lo, hi = int(lo), int(hi)

    if max(abs(lo), abs(hi)) > _MAX_MAGNITUDE:
        raise CostExpressionError(
            f"subexpression range {vr} exceeds the safe magnitude "
            f"{_MAX_MAGNITUDE}; rescale (COST_SCALE) before lowering"
        )

    return lo, hi


class CpSatAnalysis:
    """One method per supported sympy node. The supported grammar *is* this
    method set: an unimplemented node raises through :func:`lower` rather than
    failing somewhere inside ortools."""

    def __init__(self, model: cp_model.CpModel) -> None:
        self.m = model
        self.aux: list[cp_model.IntVar] = []
        # Multiplier ``lower`` applied to the coefficients before interpreting;
        # 1 for an expression that was already integral. The objective it builds
        # is in those scaled units, so a caller reading an objective *value* back
        # off the solver divides by it.
        self.scale = 1

    def _reify(self, vr: ValueRanges, tag: str) -> cp_model.IntVar:
        lo, hi = _bounds(vr)
        v = self.m.new_int_var(lo, hi, f"{tag}{len(self.aux)}")
        self.aux.append(v)
        return v

    # -- leaves --------------------------------------------------------------
    def constant(self, c: Any, dtype: torch.dtype) -> Val:
        if not isinstance(c, sympy.Integer):
            raise CostExpressionError(
                f"non-integer constant {c!r}: apply COST_SCALE before lowering"
            )
        return Val(int(c), VRA.constant(c, torch.int64))

    # -- linear: no variable, no constraint ----------------------------------
    def sym_sum(self, args: list[Val]) -> Val:
        """n-ary integer Add fastpath: one flat sum, not a fold chain."""
        vr = args[0].vr
        for a in args[1:]:
            vr = VRA.add(vr, a.vr)
        return Val(sum(a.e for a in args), vr)

    def add(self, a: Val, b: Val) -> Val:
        return Val(a.e + b.e, VRA.add(a.vr, b.vr))

    def mul(self, a: Val, b: Val) -> Val:
        vr = VRA.mul(a.vr, b.vr)
        if a.is_const or b.is_const:  # at most one non-constant factor
            return Val(a.e * b.e, vr)
        t = self._reify(vr, "mul")
        self.m.add_multiplication_equality(t, [a.e, b.e])
        return Val(t, vr)

    def pow_by_natural(self, a: Val, b: Val) -> Val:
        if b.is_const and b.e < 0:
            # sympy spells ``x / y`` as ``x * y**-1``, so this is where division
            # by a variable lands.
            raise CostExpressionError(
                f"unsupported division by a variable (Pow exponent {b.e})"
            )
        if not b.is_const:
            raise CostExpressionError(f"unsupported non-constant exponent {b.e}")

        acc = Val(1, VRA.constant(sympy.Integer(1), torch.int64))
        for _ in range(b.e):
            acc = self.mul(acc, a)
        return acc

    # -- reified: one variable + one constraint each -------------------------
    def minimum(self, a: Val, b: Val) -> Val:
        vr = VRA.minimum(a.vr, b.vr)
        t = self._reify(vr, "min")
        self.m.add_min_equality(t, [a.e, b.e])
        return Val(t, vr)

    def maximum(self, a: Val, b: Val) -> Val:
        vr = VRA.maximum(a.vr, b.vr)
        t = self._reify(vr, "max")
        self.m.add_max_equality(t, [a.e, b.e])
        return Val(t, vr)


def _check_env(expr: sympy.Expr, env: dict[sympy.Symbol, Val]) -> None:
    """Reject an environment that cannot carry ``expr``, naming the symbol.

    Both checks have to happen before anything interprets the expression: an
    unbound symbol has no range for the rescale to bound the objective by, and
    an infinite range blows up inside the range arithmetic before the domain
    check in :func:`_bounds` could ever see it.
    """
    unbound = sorted(str(s) for s in expr.free_symbols - set(env))
    if unbound:
        known = ", ".join(sorted(str(s) for s in env))
        raise CostExpressionError(
            f"unbound symbol(s): {', '.join(unbound)} (bound: {known})"
        )

    for sym, val in sorted(env.items(), key=lambda kv: str(kv[0])):
        if not (_is_concrete(val.vr.lower) and _is_concrete(val.vr.upper)):
            raise CostExpressionError(f"unbounded leaf symbol {sym}: {val.vr}")


def interp(
    model: cp_model.CpModel,
    expr: sympy.Expr,
    env: dict[sympy.Symbol, Val],
) -> tuple[Val, CpSatAnalysis]:
    """Interpret ``expr`` -- integer constants only -- against ``env``
    (symbol -> already-built CP-SAT var), building the vars and constraints its
    non-affine nodes need on ``model``.

    The grammar step of :func:`lower`, which is what a solver calls; this is
    separate so the accept/reject boundary can be exercised on its own terms,
    without the rescale that stands in front of it.

    Returns the interpreted root and the analysis, so callers can inspect the
    aux vars the lowering created.
    """

    def unbound(sym: sympy.Symbol) -> Val:
        known = ", ".join(sorted(str(s) for s in env))
        raise CostExpressionError(f"unbound symbol: {sym} (bound: {known})")

    _check_env(expr, env)

    analysis = CpSatAnalysis(model)

    try:
        out = sympy_interp(analysis, env, expr, missing_handler=unbound)
    except KeyError as exc:  # sympy node torch's dispatch table does not know
        raise CostExpressionError(f"unsupported sympy node: {exc}") from exc
    except AttributeError as exc:  # node we deliberately do not support
        raise CostExpressionError(f"unsupported operation: {exc}") from exc

    return out, analysis


def bind(model: cp_model.CpModel, name: str, lo: int, hi: int) -> Val:
    """A leaf: the CP-SAT var plus the range it already carries."""
    return Val(model.new_int_var(lo, hi, name), ValueRanges(lo, hi))


def dump(
    model: cp_model.CpModel,
    analysis: CpSatAnalysis | None = None,
    solver: cp_model.CpSolver | None = None,
    root: Val | None = None,
    status: Any = None,
) -> None:
    """Print the model, and the solution if there is one. Call it from a
    breakpoint when a lowered objective does not read the way it should."""

    print("--- variables ---")
    for idx, v in enumerate(model.proto.variables):
        val = ""
        if solver is not None:
            val = f" = {solver.value(model.get_int_var_from_proto_index(idx))}"
        print(f"  {v.name or '(anon)':>12} domain={list(v.domain)}{val}")

    if analysis is not None:
        print(f"--- aux vars: {len(analysis.aux)} ---")

    if root is not None:
        print(f"--- root expr: {root.e} range={root.vr} ---")

    if solver is not None:
        name = solver.status_name(status) if status is not None else "?"
        print(f"--- status={name} objective={solver.objective_value}")


def _round_coeff(coeff: sympy.Expr, node: sympy.Expr) -> sympy.Integer:
    """Round one scaled coefficient to the integer ``lower`` requires.

    Rounding is the whole approximation in the rescale, and it is only sound
    while it stays a rounding: a non-zero coefficient that lands on zero has had
    its term *deleted*, which is the silent-approximation failure the cost
    lowering exists to avoid, so it raises instead.
    """
    rounded = int(sympy.Integer(round(float(coeff))))
    if rounded == 0 and coeff != 0:
        raise CostExpressionError(
            f"coefficient {float(coeff):g} of {node} underflows to 0 at the "
            "objective's integer scale; the term would be silently dropped"
        )
    return sympy.Integer(rounded)


def _integer_coeffs(expr: sympy.Expr, scale: sympy.Expr) -> sympy.Expr:
    """Rewrite ``scale * expr`` (``scale != 0``) with every coefficient integral.

    ``lower`` accepts integer constants only, while the cost model speaks in
    nanoseconds and GB/s and hands over floats throughout. Multiplying an
    objective by a positive constant leaves its argmin alone, so the scale is
    pushed *through* the expression rather than applied at the root: ``Add`` and
    ``Min`` / ``Max`` are positively homogeneous, so it distributes over their
    arguments, and only at a coefficient does it round (:func:`_round_coeff`).
    A negative multiplier is homogeneous too, with the order reversed -- so it
    distributes just the same, swapping ``Min`` for ``Max``.

    Pushing it down rather than rounding at the root is what keeps a nested
    coefficient meaningful: ``s * (2.5 * Min(0.5*x, y))`` becomes
    ``Min(1.25s*x, 2.5s*y)``, where multiplying out at the root would have had
    to round 0.5 and 2.5 on their own.
    """
    if expr.is_number:
        return _round_coeff(scale * expr, expr)
    if expr.is_Symbol:
        return _round_coeff(scale, expr) * expr
    if expr.is_Add:
        return sympy.Add(*(_integer_coeffs(a, scale) for a in expr.args))
    if isinstance(expr, (sympy.Min, sympy.Max)):
        flipped = {sympy.Min: sympy.Max, sympy.Max: sympy.Min}
        func = expr.func if scale > 0 else flipped[expr.func]
        return func(*(_integer_coeffs(a, scale) for a in expr.args))
    if expr.is_Mul:
        # The scale rides on exactly one factor (with the numeric coefficient
        # folded into it); every other factor is rewritten at scale 1, which
        # leaves it alone unless it carries a coefficient of its own.
        coeff, rest = expr.as_coeff_Mul()
        head, *tail = sympy.Mul.make_args(rest)
        return sympy.Mul(
            _integer_coeffs(head, scale * coeff),
            *(_integer_coeffs(f, sympy.Integer(1)) for f in tail),
        )
    if expr.is_Pow:
        # Exponent legality (small, non-negative, integral) is ``lower``'s call,
        # so a bad one reaches it and is named there rather than here.
        base, exponent = expr.args
        return _round_coeff(scale, expr) * sympy.Pow(
            _integer_coeffs(base, sympy.Integer(1)), exponent
        )
    raise CostExpressionError(
        f"cannot rescale {type(expr).__name__} node {expr} to integer coefficients"
    )


def _rescale_to_integers(
    expr: sympy.Expr, ranges: dict[sympy.Symbol, ValueRanges]
) -> tuple[sympy.Expr, int]:
    """Scale ``expr`` up until its coefficients are integers, but no further
    than CP-SAT can safely represent.

    The two ends pull against each other: a coefficient like ``0.006 ns/byte``
    needs a large scale to survive rounding, while the objective's own magnitude
    must stay inside the domain width ``cost_expr`` will accept. So the scale is
    read off the expression itself -- the largest one that keeps the objective
    under :data:`_COST_TARGET_MAGNITUDE`, bounded by interpreting the expression
    under torch's interval arithmetic over the leaf ranges (the same ranges the
    lowering will derive its variable domains from). That spends the whole
    magnitude budget on precision, and spends it where the expression actually
    is rather than where a fixed constant guessed it would be.

    An expression whose constants are already integers is left alone. There is
    no precision to buy there, and scaling it up anyway would hand CP-SAT a much
    wider objective domain to reason over for nothing -- which matters because
    the CP-SAT solver's own default objective
    (``ilp_solver_ortools.default_cost_expr``) is exactly such an expression, and
    it runs on every solve that does not supply its own.
    """
    if all(n.is_Integer for n in expr.atoms(sympy.Number)):
        return expr, 1

    try:
        magnitude = sympy_interp(VRA, ranges, expr)
    except Exception as exc:  # noqa: BLE001 - any failure here is a bad objective
        # The interval arithmetic runs over the same dispatch table the lowering
        # does, so a node it cannot handle is a node ``lower`` would reject too;
        # report it the same way rather than letting a KeyError out of a helper.
        raise CostExpressionError(
            f"cannot bound the cost expression {expr}: {exc}"
        ) from exc

    bound = max(abs(float(magnitude.lower)), abs(float(magnitude.upper)))
    scale = max(1, int(_COST_TARGET_MAGNITUDE // bound)) if bound > 0 else 1
    return _integer_coeffs(expr, sympy.Integer(scale)), scale


def lower(
    model: cp_model.CpModel,
    expr: sympy.Expr,
    env: dict[sympy.Symbol, Val],
) -> tuple[Val, CpSatAnalysis]:
    """Convert ``expr`` into a CP-SAT expression on ``model``, in the decisions
    ``env`` binds its symbols to.

    The whole conversion, and all a solver needs to call: ``env`` maps each
    symbol the expression may name to the variable that decides it (and the
    range that variable lives in), and nothing here knows or asks where those
    variables came from. Two steps, each of which can only fail loudly:
    :func:`_rescale_to_integers`, since a cost model's coefficients are floats
    and CP-SAT is integer-only, then :func:`interp`, which raises
    ``CostExpressionError`` on anything outside the supported grammar rather
    than approximating it.

    Returns the interpreted root -- ``root.e`` is the expression to minimize --
    and the analysis, whose ``aux`` lists the variables the lowering created and
    whose ``scale`` is the multiplier the objective now carries.
    """
    _check_env(expr, env)

    scaled, scale = _rescale_to_integers(
        expr, {sym: val.vr for sym, val in env.items()}
    )
    root, analysis = interp(model, scaled, env)
    analysis.scale = scale
    return root, analysis
