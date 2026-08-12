# Implementation Plan — Coarse Tiling Optimization

| Field | Value |
|---|---|
| Status | Draft |
| Implements | [`draft-unified-tiling-cpsat.md`](draft-unified-tiling-cpsat.md) (collateral doc 1, Phase 1) |
| Branch | `course-chunking-planning` |
| Scope | 59 numbered requirements (R1.1–R10.6) across `wsr`, `scratchpad`, `padding.py` |

This is the execution plan for the RFC, not a restatement of it. It cites
requirements by number and specifies **what to build, in what order, and which
test pins each requirement**. Every RFC requirement appears exactly once in the
traceability matrix at the end.

The plan follows the five steps as scoped, with one prerequisite added ahead of
them. Steps 1 and 2 write tests that are expected to fail, and an expected
failure only means something if the names it touches already exist — so Step 0
lands those names first.

Three points where the plan departs from a literal reading of the five steps are
argued where they occur rather than up front: prediction lands in step 3 rather
than step 4 (§3b), the objective collapse is split out ahead of the tiling axis
(§3a), and the reduction-axis cut pin is a step-3 obligation even though
reductions are otherwise step 5 (§3b).

## Step 0 — Scaffolding and the de-risking spike

Lands green: no xfails, no behaviour change. Two jobs — confirm the premise the
phase rests on, and make steps 1–2's expected failures mean something.

### 0.1 The spike (do this first, before writing any code)

The RFC's entire motivation rests on one claim: tiling buys LX residency through
the run's *interior*, because `_propagate_tiled_op` sets `output_tiled_dims = []`
so the per-tile scratch stays LX-eligible.

Static reading confirms it — `_is_tiled_advancing`'s docstring
(`scratchpad/utils.py:218-234`) states the rule verbatim: "A loop-internal
buffer (e.g. drained by a copy op every iteration) can be tiled yet have its own
write pinned at a fixed address; such a buffer is LX-eligible."

Confirm it empirically anyway, on today's code, before building anything on it:
take a hint-tiled two-group graph, run the allocator, and assert the
*Background* table row by row — interior scratch and read-side tile copies carry
a `None` `residency_reason`; `full_buf` and the write-side copy op do not. This
is RFC testing item 7's positive half, and it costs a day. If it does not hold,
the phase needs rescoping before step 1.

### 0.2 The expected-failure mechanism

An xfail marker records *that* a test failed, never *why*. Both standard
mechanisms are too broad for a phase this long:

- `pyproject.toml` sets no `xfail_strict`, so `pytest.mark.xfail` does not even
  fail on unexpected pass.
- `unittest.expectedFailure` is strict about that, but absorbs **any**
  exception. A step-1 test opening with
  `ts_inductor_config.patch(unified_tiling=True)` before that gate exists raises
  `AttributeError`, is recorded as a satisfied expectation, and *keeps* being
  satisfied after the feature lands — including when it lands wrong. Same for
  step 2 importing `PartitionConfig`.

So the expectation is narrowed to one declared cause: **a marked test may fail
only by reaching an unbuilt stub.** Every Step 0 stub raises the built-in
`NotImplementedError` under a fixed message prefix —
`raise NotImplementedError("unified-tiling: enumerate_tile_options")` — and both
test steps use one shared decorator in place of `unittest.expectedFailure`:

```python
def expected_unimplemented(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            fn(self, *args, **kwargs)
        except NotImplementedError as exc:
            pytest.xfail(f"not built yet: {exc}")
        else:
            self.fail(f"{fn.__name__} passed — remove @expected_unimplemented")
    return wrapper
```

It cannot wrap `unittest.expectedFailure`: that absorbs whatever reaches it, so
a non-`NotImplementedError` failure could never be surfaced through it.
Imperative `pytest.xfail()` is the repo's existing idiom for a runtime-decided
expected failure (`indirect_access_common.py:450`, `:474`, called from
`unittest.TestCase` subclasses).

| Outcome | Plain `expectedFailure` | `expected_unimplemented` |
|---|---|---|
| Reached an unbuilt stub | xfail | xfail, reason names the stub |
| Passed cleanly | fail (strict) | fail — "remove the marker" |
| Real assertion failure | **xfail — hidden** | fail |
| Typo, `AttributeError`, `ImportError` | **xfail — hidden** | fail |

Two properties follow. As implementation advances, a test that xfailed on stub A
starts running further and xfails on stub B — the reason string tracks the
frontier with no edits. And because the marker is imperative rather than a
pytest mark, `-m 'not xfail'` does not deselect these tests; they still execute
and still xfail at runtime, which is the behaviour we want.

**Residual risk, accepted.** `_inductor` already raises bare
`NotImplementedError` in five places these tests can traverse —
`graph_editor.py:247` (inside `push_allocation_with_clone`, so every
boundary-clone test), `propagate_named_dims.py:650` (a pass at `passes.py:426`),
`spyre_kernel.py:491`, `codegen/compute_ops.py:745`, `codegen/bundle.py:546`.
If one fires during a marked test it will read as "not built yet" rather than
failing red, masking a real backend gap. The message prefix is the tell: an
xfail whose reason does not start `not built yet: unified-tiling:` came from one
of those five. Worth a glance whenever the xfail set changes shape unexpectedly.
A dedicated exception subclass would make this structural rather than
conventional; the built-in was chosen deliberately over that.

**Phase completion criterion.** `grep -rn "unified-tiling:" torch_spyre/`
returning nothing means every scaffold is built. Pair it with
`grep -rn "expected_unimplemented" tests/` returning nothing: the two must empty
together, and either one alone emptying is a bug.

### 0.3 Scaffolding

| Deliverable | Where | Requirement |
|---|---|---|
| `expected_unimplemented` decorator (§0.2) | `tests/inductor/utils_inductor.py` | — |
| `unified_tiling`, `auto_coarse_tiling` gates, default off | `config.py` (match the `os.environ.get(...) == "1"` style at `:22-25`) | R8.1, R5.8 |
| `TileOption`, `PartitionConfig` dataclasses with derived properties | `scratchpad/plan_solver.py` (beside `CoreDivision`, `:93`) | R2.1 |
| `enumerate_tile_options` signature; body raises `NotImplementedError("unified-tiling: ...")` | new `wsr/enumerate_tilings.py` | R1.1 |
| `PredictedFrame` / `PredictedBufferSet` types real, builders raise (§3b) | new `wsr/tile_prediction.py` | R7.1 |
| `CostSpec`, `CostExpressionError`; `validate()` and `lower()` raise | new `scratchpad/cost_expr.py` | R3.3 |
| `sym_is_lx`, `sym_inv_cores` symbol properties, real (§3a) | `plan_solver.py:137` (`CoreDivisionBuffer`) | R3.7 |
| `objective` keyword-only param, typed `CostSpec` / `sympy.Expr` / `None`, on both ABCs and all four concrete overrides | `plan_solver.py:260,297`; `ilp_solver_ortools.py:348`, `greedy_solver.py:134`, `firstfit_bestfit_solver.py:186`, `simulated_annealing.py:122` | R3.1 |
| Placement solvers ignore a non-`None` objective, warn once | the four solvers above | R3.6 |
| `_UNIFIED_TILING_HINT_ID = 20000`, a reserved `hint_id` base (§4) | `wsr/enumerate_tilings.py` | R4.5 |
| CI config yamls for the two new test files | `tests/configs/torch_spyre_tests/inductor/` | — |

R3.1 is source-compatible: all three in-tree callers already pass
`log_lx_usage` by keyword (`allocator.py:184`, `:1348`, `:1468`).

The split between *real* and *stubbed* is what makes §0.2 work. Anything a test
**constructs or reads** must be real here — the config gates (so `config.patch`
resolves), the dataclasses and their derived scalars (so fixtures build), the
signatures (so calls bind). Only the bodies that *compute* anything raise. A
stubbed dataclass would put the failure back in the `AttributeError` bucket
§0.2 exists to drain.

## Step 1 — Operation-level tests, marked xfail

Style and models follow `test_scratchpad_use.py`. Two mechanisms already in that
file do most of the work:

- `_ParameterizedScratchpadMeta` (`:251`) expands `parameter_models` ×
  `parameter_axes` into one method per combo.
- `case_decorators(params)` (`:1024`) applies per-combo decorators — already used
  to mark the `cpsat` combos `expectedFailure`.

So step 1.1 is not hand-marking tests. Add `unified_tiling` and
`auto_coarse_tiling` to `parameter_axes`, and return `[expected_unimplemented]`
(§0.2) from `case_decorators` for the combos where either is `True`.

Those returns are then **never edited again**. Each combo stops xfailing on its
own, at the moment the last stub on its path is built, because the decorator
keys on the exception rather than on a hand-maintained list — and the run turns
red the day a combo passes, which is the signal to delete its entry. Steps 3–5
retire them by making them fire, not by editing them.

Models: reuse `_softmax_case` (dim 0 and −1), `_mlp_case`, `_swiglu_case`, and
the pointwise chains from `TestCloneAtGraphBoundaries`. Add one **span-pressure**
model — a shape that forces tiling today — since that is the only way to observe
R2.3's over-tiling fix; `_E2E_SHAPE = (1, 8195, 256, 64)` from
`test_span_overflow_hint_analysis.py:216` is the known-overflowing shape.

New classes in `test_scratchpad_use.py`:

| Class | Covers | RFC test item |
|---|---|---|
| `TestUnifiedTilingGates` | R8.1, R5.8 — the on/off matrix; gate off reproduces today | 1, 2 |
| `TestUnifiedTilingParity` | R3.5 — spill-parity, no core regression at equal spill | 2 |
| `TestOverTilingFix` | R2.3 — strictly smaller tile count than the `core_split_estimate = 1` path | 9 |
| `TestBoundaryBufferLxStatus` | R4.8–R4.10, R7.5 — the *Background* table; the 1 / 2 / 0 rule; the untiled→tiled row | 7 |
| `TestPerCoreViewPrediction` | R2.6 — predicted view equals `_prepare_per_core_view` recomputed after `coarse_tile` | 8 |
| `TestUnifiedTilingHints` | R5.1–R5.8 — pins honoured, never re-tiled, `IGNORE_HINTS` drops pins | — |
| `TestGroupRoundTrip` | R4.5 — `(groups, dim_hint_assignments)` round-trips; no `loop_group_id` **or** `hint_id` collision (§4) | 4 |
| `TestUnifiedTilingFallback` | R8.3 — `INFEASIBLE` → span-overflow path, IR untouched; timeout → incumbent applied | 11 |
| `TestUnifiedTilingDeterminism` | R8.4 — identical plans across runs | 10 |
| `TestOpOrderInvariant` | R4.4 — op order unchanged across the solve | — |

Extend `test_padding.py` for R10.1 (derived pad reaches sizing) and R10.4
(shared-operand emission), and `test_coarse_tile_e2e.py` for R1.6 (every
enumerated option applies **and** matches CPU — applicability alone would pass on
the known wrong-numerics shapes).

**Note.** RFC testing item 1 states `test_coarse_tiling.py` has no CI config
yaml. That is stale — `test_coarse_tiling_config.yaml` exists, with
`labels: [core, full, device_critical]` and
`unlisted_test_mode: mandatory_success`. All target suites carry that mode, so
new tests must pass to land; xfail satisfies it, unexpected pass does not.

## Step 2 — Solver tests, marked xfail

Device-free, in `test_scratchpad_solver.py`, marked with the same
`expected_unimplemented` decorator (§0.2) — these have no metaclass, so it is
applied per method. The file's existing shape is the right one to extend:
`BaseLayoutSolverTests` is the shared contract with a `make_buffer` hook, and
`JointDivisionSolverTests` (`:776`) overrides that hook plus
`solve`/`check_result` to drive the joint entry point.

Add `TilingSolverTests` as a sibling mixin whose `make_buffer` emits
`PartitionConfig`-carrying buffers with the unity tile, so the whole shared
contract runs against the tiling path unchanged. Then the axis-specific classes:

| Class | Covers | RFC test item |
|---|---|---|
| `TestCostExprLowering` | R3.3 accept/reject table (one `CostExpressionError` case per rejected construct, raised by `validate` **before** any ortools error can surface), unbound-symbol containment, `Min`/`Max` flattening preserves the arg set, R3.4 one scale for the whole expression + overflow, R3.7 binding bounds, `SumOverEdges`/`relayout_bytes` **absent** | 3 |
| `TestSinglePhaseObjective` | R3.2 — one `Minimize`, no lock inequality, no second solve | 3 |
| `TestCutTables` | R4.1 totality, R4.2 untileable pins + structural contiguity, R4.7 directionality and fail-closed, claim orientation after the equality sweep | 4 |
| `TestCutConsequences` | R4.8 (no `boundary_op ⟹ ¬in_buffer` constraint; row-3 eviction only), R4.9 optional rectangles, R4.10 `full_size`/`boundary_view` | 7 |
| `TestReductionPinning` | R4.6 — both boundaries pinned, singleton group | 4 |
| `TestConfigEnumeration` | R2.2 per-tiling division sets, R2.4 seed pair retained + dedup, R2.5 `Unsupported` | 5 |
| `TestTilingAwareViews` | R2.6 — `prep_cache` key includes the tile; **negative**: two configs differing only in tiling must not share an entry | 8 |
| `TestWarmStart` | R8.2 — `AddHint` from the heuristic plan | 11 |

Enumerator tests go in a new `tests/inductor/test_enumerate_tilings.py` (with its
config yaml), not in `test_scratchpad_solver.py` — the enumerator is a `wsr`
module and its tests need no solver. It covers R1.1–R1.5, R1.7, R1.9, and later
R1.8.

Two cases in that file carry more weight than the rest:

- **R1.9's silent-failure guard.** For an op under *no* span pressure but with a
  legally splittable host dim, assert the enumerator returns more than the
  untiled option. Built on `_candidate_host_dims` (`:911`) alone it returns one
  option, the solver never tiles that op, and the LX-residency motivation
  quietly does nothing — no error, no warning.
- **R1.1 completeness.** Brute-force reference on small shapes; the enumerator's
  set must equal it.

## Step 3 — Solver implementation

### 3a — The M2 objective

**Why this is split out.** R3.2 replaces `_run`'s two-phase lexicographic solve
(`ilp_solver_ortools.py:446-518` — minimize `sum(spill_cost)`, lock it with
`model.add(sum(hbm_terms) <= round(solver.ObjectiveValue()))` at `:479`, then
maximize `sum(cores)`) with one weighted `Minimize`. That changes **every
existing CP-SAT plan with tiling switched off entirely**. R3.5 concedes
bit-identity is not required, only spill-parity — but
`CoOptAllocatorIntegrationTests`' prescribed fingerprints
(`test_scratchpad_use.py:883-1021`) are exact-match dicts. Landing the collapse
alongside the tiling axis would conflate "did tiling break this?" with "did the
reweighting break this?" on the one suite that catches either.

Land and prove it before any tiling variable exists.

#### How the cost function is injected

`isuruf/torch-spyre@cost` is a working end-to-end injection and is the reference
for this step. It threads a cost function through four layers:

| Layer | On the reference branch | Adopt as |
|---|---|---|
| Producer | `_inductor/cost_model.py::predict_ops(op_features) -> sympy.Expr`, fed per op by `allocator._extract_op_features` (`:2068` there) | **not this phase** — see *Producer* below |
| Symbol namespace | `sym_is_lx` / `sym_inv_cores` / `sym_core_divs` properties on `CoreDivisionBuffer`, minting `sympy.Symbol(f"is_lx_{self.name}")` | adopt; §3b adds the tiling symbols |
| Transport | `plan_layout_and_core_divisions(buffers, cost_expr)` → `_plan_layout_generic` → `_run` | adopt, renamed to R3.1's keyword-only `objective` |
| Lowering | sympy rewrite → `lambdify(syms, expr, modules=[{"min", "max"}, "math"])` applied to the CP-SAT vars → one `model.minimize` | adopt, with a validator in front |

Three of its decisions are worth taking as-is.

1. **Symbols live on the buffer, not in a table passed alongside it.** Producer
   and solver agree through the buffer name alone, so nothing has to carry a
   `symbol -> var` map across the allocator/solver boundary and the solver's
   binding step is a five-line loop over `tensors.values()`. It also keeps the
   namespace open: §3b's tiling symbols are new properties, not a new parameter.
2. **`lambdify` replaces a hand-written lowerer.** The tree walk this plan
   previously specified (`lower(expr, bindings) -> cp_model.LinearExpr`) is what
   `lambdify` already does — it prints the expression to Python source and
   evaluates it against the CP-SAT vars, so `Add` and constant `Mul` lower
   through the vars' own operator overloads and only the nodes with no operator
   form (`Min`, `Max`) need entries in the custom module. `cost_expr.py` stays,
   owning validation and scaling rather than tree-walking.
3. **Non-linearity in the division axis is a lookup table, not a constraint.**
   The cost model wants time, which goes as `1/cores`; a reciprocal of a decision
   variable is not expressible, so the branch precomputes
   `[32 // cd.cores_used for cd in b.core_divisions]` and ties it to the division
   index with one `AddElement` — the same shape `eff_size` already uses. Any
   per-candidate scalar, however non-linear in the choice, therefore costs
   exactly what R3.7 budgets. This is the pattern §3b's `tile_count` and
   `full_size` follow.

The branch's `if cost_expr is not None: ... else: <two-phase>` split is
transitional and does **not** come across: R3.2 replaces the two-phase block
rather than sitting beside it, and `TestSinglePhaseObjective` pins one
`Minimize` and one `Solve` on the default path too.

#### What must change on adoption

Nine items, found by reading the branch against this tree. Items 1–5 and 7 are
defects on the branch as it stands, not porting friction; 6, 8 and 9 are gaps to
close on the way in.

| # | On the branch | Consequence | Fix |
|---|---|---|---|
| 1 | `unnest_min` appends nothing for an arg that is neither `c*Min(...)` nor `Add` | It is silently dropped. Verified: `Min(4, 5*Min(x, y))` — the function's own docstring example — rewrites to `Min(5x, 5y)`, losing the constant bound. Wrong cost, no error | add the missing `else: result.append(arg)`; pin with a rewrite test asserting the flattened arg set, not just the node count |
| 2 | `sym_map` binds `t.buffer.sym_inv_cores -> t.inv_cores` for **every** wrapper | `_LifetimeBufferWithCpVars` has no `inv_cores` and plain `LifetimeBoundBuffer` has no `sym_inv_cores` (it is a `CoreDivisionBuffer` property) → `AttributeError` on any `plan_layout` solve | bind through a `bind_symbols()` wrapper hook, empty on the placement-only wrapper — the same hooks-on-the-wrapper shape §3b relies on |
| 3 | `core_terms` is read by the DEBUG log block but bound only in the `else` branch | `UnboundLocalError` whenever an injected objective runs under DEBUG logging | branch the log line with the objective |
| 4 | the joint wrapper renames `cores` → `inv_cores` and never sets `self.cores` | `core_terms` is then always empty, so the legacy path's phase 2 silently stops running and every no-objective plan loses its parallelism phase | moot once R3.2 lands; until then keep both attributes |
| 5 | two ad-hoc float scales: `truncate_floats_min` (×10⁴, then `/m` back out) and `my_min` (×10³, returning `min_var * 1e-3`) | Verified: CP-SAT accepts float coefficients in `minimize` but **rejects** them in `AddMinEquality` (`ValueError: Failed to convert integer linear expression`). A float-scaled `Min` nested in another `Min`/`Max` raises | one documented `COST_SCALE` applied once to the whole expression (R3.4); never a per-node scale |
| 6 | `modules=[custom, "math"]` — an unsupported node falls through to `math` or to the vars' own operators | the failure surfaces from inside ortools naming neither the sympy node nor the buffer. Verified: `sqrt`/`log` → `TypeError: must be real number`; `x*y` → `TypeError: __mul__(): incompatible function arguments`; `x**2`, `x/y`, `Piecewise` → `NotImplementedError` from `cp_model` | validate before lowering (R3.3), so every rejection carries its own message |
| 7 | nothing checks `cost_expr.free_symbols` against what the solver binds | an unbound symbol reaches `lambdify` as a free variable → `NameError` at call time. Live today: `sym_core_divs` is minted and written to `op.op_it_space_splits` but never bound | assert containment in `validate`; raise `CostExpressionError` naming the unbound symbols in sorted order (`free_symbols` is a `set` — R8.4) |
| 8 | `32 // cd.cores_used` | hardcodes 32 cores rather than reading `SENCORES`, and floor division is lossy: `32 // 7 = 4` against a true 4.57, a 12% error that can mis-rank two divisions | scale the reciprocal table (`SCALE // cores_used`, `SCALE` large) and take the core count from `get_ncores` |
| 9 | three `print()` calls, a commented-out `NewIntVar`, a function-local `import math` | debug residue | delete |

**Producer, and what this step owes it.** `predict_ops` and its
`dump_cost_model` feature extractors live only on the reference branch — neither
exists in this tree. That is not a blocker: R3.5 has `objective=None` select a
default objective built from today's terms, so M2 lands, is testable, and is
useful with no producer at all, and `TestCostExprLowering` authors its
expressions directly, which is what keeps it device-free. Porting the analytic
cost model is separate work. What this step owes it is a surface it can bind to
unchanged — which is why the namespace is extensible (`_extract_op_features`
wants symbols for splits as well as for residency and cores) rather than a
closed set of scalars.

**Expression build cost, measured.** `lambdify` is not the bottleneck (0.77 s to
lower a 3000-symbol sum). Building the sympy expression is: accumulating terms
with `sum()` or `+=` is superlinear — 0.34 s at 500 terms, **37.7 s at 1500** —
against 0.09 s for `sympy.Add(*terms)` at 1500. Any producer, and the default
objective here, must assemble with `Add(*terms)`. Worth an explicit line in
`cost_expr.py`, since the natural way to write it is the slow way.

**Needs folding back into R3.7.** R3.7 bounds binding at one `AddElement` per
symbol and says nothing about the aux vars `Min`/`Max` reification creates — one
`IntVar` plus one `AddMinEquality`/`AddMaxEquality` per surviving node, which is
exactly what `unnest_min`'s flattening exists to keep down. The bound wanted is
"one `AddElement` per symbol **and** one reified var per `Min`/`Max` node
surviving flattening"; `TestCostExprLowering` pins the second half.

#### Work items

1. `scratchpad/cost_expr.py`: `CostSpec`; `CostExpressionError`;
   `validate(expr, bound)` — the R3.3 accept/reject walk plus item 7's
   containment check, raising with the offending node named; `COST_SCALE`
   derivation and the int64 overflow check (R3.4); `lower(expr, bindings)` =
   validate, flatten nested `Min`/`Max`, scale once, then `lambdify` against the
   `{"min", "max"}` module. Model-level symbols with no buffer to hang from stay
   in this module's namespace — `peak_lx_bytes` is one `AddMaxEquality` over the
   existing `top[b]` vars (`_add_no_overlap_2d`, `:568`), never a time-indexed
   sum (R3.7).
2. `plan_solver.py`: `sym_is_lx` and `sym_inv_cores` on `CoreDivisionBuffer`
   (`:137`), scaffolded at §0.3 so step-2 tests can construct them.
3. `ilp_solver_ortools.py`: `bind_symbols()` on both wrappers (item 2); the
   scaled reciprocal-cores table (item 8); and replace `_run`'s two-phase block
   (`:467-492`) with a single `Minimize`, default objective
   `SumOverBuffers(spill_cost) - SumOverBuffers(cores)` with the spill term
   weighted to dominate (R3.5). Phase 2 is currently skipped by testing
   `sb.cores is None`, which only the placement-only wrapper sets (`:193`);
   collapsing to one phase removes that flag's meaning, so the placement-only
   path needs an explicit skip in its place.

Adopt the injection mechanism only. The branch also carries an unrelated
divergence in this file — its `_add_no_overlap_2d` shortens the **parent's**
lifetime on an in-place merge where this tree shortens the **child's**, and its
`_justify` drops the capacity check — and it subclasses a
`LifetimeBoundBufferWithSolverVars` base that does not exist here. None of that
is part of the cost function; leave this tree's versions alone.

**Gate to proceed:** RFC testing item 2 green — spill-parity against today's
CP-SAT output, no core regression at equal spill, on
`CoOptAllocatorIntegrationTests`' four prescribed models. Expect fingerprint
churn; re-baseline deliberately, one model at a time.

### 3b — The tiling axis, output ranges only

The solver loop is already written against a wrapper hook surface — `_run`,
`_add_inplace_relaxation`, `_add_no_overlap_2d`, `_get_children`,
`_add_core_division`, `_extract` touch only `eff_size`, `cores`, `in_buffer`,
`offset`, `merge_vars`, `parents`, `match_pairs()`, `spill_cost()`,
`constrain_residency()`, `constrain_merge()`, `footprint()`,
`record_division()`. So this is a **wrapper plus dispatch** change, not a solver
rewrite.

#### Where prediction lives, and why it lands here

`GraphEditor` (`scratchpad/graph_editor.py`) is the wrong place to put it: it
does one thing — insert a Pointwise `clone` for LX boundary cloning
(`push_allocation_with_clone`, `:103`) — has no generic insert-op or
allocate-buffer primitive, rejects anything that is not `Pointwise`/`Reduction`
(`is_rewritable_consumer`, `:271`), and knows nothing about loop groups.

Nor does it need to be. Every mutation this phase performs already exists in
`coarse_tile.py`: `_allocate_full_buffer` (`:1519`), `_insert_copy_op`
(`:1657`), `_insert_read_copy_ops` (`:1907`), `_insert_combine_op` (`:2307`),
`_insert_reduction_copy_op` (`:2404`). R4.5 is explicit that "no new application
path is introduced", and M6 forbids reimplementing a predicate that exists. What
is missing is the **inverse** of a mutation: a pure predictor reporting what
`_apply_plan` *would* insert without inserting it. That is its own concern and
gets its own module.

**`wsr/tile_prediction.py`** is the sole owner of "what would this config
assignment produce", exporting two things built by composing existing
zero-mutation helpers:

- `PredictedFrame` — divided ranges plus resized device layout, per
  `(op, TileOption)`. Composes `_planned_tile_extents_per_level`
  (`coarse_tile.py:309`) and `_post_tile_layout_for_splits`
  (`span_overflow_hint_analysis.py:175`). Consumed by R2.6's per-core views.
- `PredictedBufferSet` — the buffers a config plus cut assignment materializes
  (`full_buf`, write copy, read copies; in step 5 the accumulator, identity fill
  and combine op), at R7.5's interstitial tick coordinates. Consumed by §2's
  optional rectangles.

The RFC requires R2.6 and R7.1 to "share one predictor; they must not drift
apart" — one module makes that structural rather than a promise, which is the
argument against putting the frame beside the enumerator and the buffer set
beside the solver.

**It lands in step 3, not step 4, and lands whole.** R2.6 gates *residency* on
`config_matches`, computed from per-core views taken on the predicted frame; a
wrong view grants residency on a slicing agreement that does not hold — wrong
data, not a mispredicted size, and explicitly outside R7.4's degrade-to-spill
safety. The RFC calls it "the highest-risk area in the design", and this repo
has already shipped one bug of that shape (coeff-keyed signature conflation in
the co-optimizing path). §2's optional read-copy rectangles need the buffer set
for the same reason. Step 4 keeps only the apply wiring and the
predicted-vs-realized comparison (R7.2, R7.5).

Two boundaries hold this together:

- **`coarse_tile.py` gets exactly one edit** — `_propagate_tiled_op`'s branch
  condition (`:1280`, the `_find_outside_consumers` test and
  `_full_buffer_read_deps` at `:1425`) extracted to a module-level pure
  predicate `boundary_role(op, ...) -> BoundaryRole`, which `_apply_plan` calls
  where it currently inlines the condition. No behaviour change, one copy of the
  rule. Dependencies stay one-way (`tile_prediction` → `coarse_tile`,
  `tile_prediction` → `span_overflow_hint_analysis`); putting the predicate in
  the new module instead would create a cycle.
- **The solver must not import `tile_prediction`.** `ilp_solver_ortools.py` sees
  only buffers and tables; the allocator calls the predictor and hands results
  across. That is what keeps `test_scratchpad_solver.py` device- and IR-free,
  and it is easy to breach by accident once the module exists.

| Work item | Where | Requirement |
|---|---|---|
| `enumerate_tile_options`, output ranges only | `wsr/enumerate_tilings.py` | R1.1–R1.5, R1.7, R1.9 |
| Expose the R1.4 predicates as reusable helpers; `_search_min_cost_tile_plan` becomes a thin ranked wrapper | `wsr/span_overflow_hint_analysis.py` | R1.4, R1.2 |
| Extract `_propagate_tiled_op`'s branch condition into `boundary_role()`; `_apply_plan` calls it | `wsr/coarse_tile.py:1280` | — |
| `PredictedFrame` — divided ranges + resized layout | new `wsr/tile_prediction.py` | R7.1 (frame half) |
| `PredictedBufferSet` — boundary copies at interstitial ticks | `wsr/tile_prediction.py` | R7.1 (buffer-set half) |
| `_enumerate_core_divisions` → config enumeration, once per tiling option | `allocator.py:1558` | R2.2, R2.3, R2.4, R2.5 |
| `_cd_parent_matches` → `_config_matches` on tiling-aware views; `_views_for_divs` `prep_cache` key gains the tile | `allocator.py:1973`, `:2078` (key at `:2092`, coeff at `:2095`) | R2.6, R6.3 |
| `_prepare_per_core_view` / `_per_core_view_on_buf` accept a predicted frame | `pass_utils.py:1467`, `:1696` | R2.6 |
| `_TilingBufferWithCpVars` extending `_CoreDivisionBufferWithCpVars` (`:243`): two-level `tile`/`div`, `AddAllowedAssignments` for the flat pair, `AddElement` for `eff_size`/`cores`/`tile_count` (per-candidate tables — §3a, "non-linearity is a lookup table"), `bind_symbols()` extended with the tiling symbols | `ilp_solver_ortools.py` | R4.8 |
| `_wrap` dispatch to the new wrapper | `ilp_solver_ortools.py:379` | — |
| Direction-indexed `cut_parents`/`cut_children` claim dicts; `_add_cut_equalities` sweep in `_run` | `ilp_solver_ortools.py` | R4.1, R4.2, R4.7 |
| Read copies as optional rectangles; row-3 eviction implication | `_add_no_overlap_2d`, `:568` | R4.9, R4.8 |
| `full_size`/`boundary_view` tables | new wrapper | R4.10 |
| Pin `cut = 1` on both boundaries of **hint-driven** reduction-tiled ops | new wrapper | R4.6 (partial) |
| Warm start via `AddHint`; `num_search_workers = 1`, `random_seed = 0`, `max_deterministic_time` | `_run`, `:452-463` | R8.2, R8.4 |

**"Omit reductions" does not omit R4.6.** Hint-driven reduction-axis tiling
exists today and applies **pre**-stickification at `passes.py:430`. The step-3
solve runs post-stickification, so a graph reaching it can already contain
reduction-tiled ops whether or not the enumerator emits any. R4.6's pin is
therefore a step-3 obligation; step 5 only widens it from "hint-driven" to
"hint-driven or solver-chosen". Leaving it out here lets the solver fuse a
reduction-tiled op into a group, which
`_plan_is_loop_invariant_at_reduction_levels` (`coarse_tile.py:559`) then
rejects at apply time — an illegal emission, which §5 of the RFC classes as a
hard failure rather than an R8.3 fallback.

R6.1/R6.2 need no code — R6.2 is accepted pessimism, guarded by a regression
check that restickify count does not grow, not by an assertion that it shrinks.

## Step 4 — Apply

No new mutation machinery — apply is `coarse_tile` as today (§3b). This step
emits artefacts, wires the pass, and proves prediction matched reality.

**Two `hint_id` namespaces have to stay disjoint, not one.** R4.5's
`group_idx_offset` covers `loop_group_id`; `hint_id` needs the same treatment
separately and does not get it for free. `span_overflow_groups` mints from a
reserved base, `_SPAN_OVERFLOW_HINT_ID = 10000`
(`coarse_tile_span_overflow.py:45`), precisely so its ids cannot collide with
the user `spyre_hint` ids `assign_dim_hints` stamps. The solver is a **third**
source and needs its own base above both (§0.3). Without one,
`validate_coarse_tile_groups` (`coarse_tile.py:112`) sees a single `hint_id` in
two groups and raises — during apply, after the solve reported success, so it
surfaces as an illegal emission rather than anything the model could have ruled
out.

| Work item | Where | Requirement |
|---|---|---|
| `_commit_divisions` also emits `(groups, dim_hint_assignments)`, mirroring `span_overflow_groups`' return shape | `allocator.py:1605` | R4.5 |
| Mint `hint_id`s from the solver's own block; derive `group_idx_offset` from existing `loop_group_id[0]` as `_maybe_coarse_tile_span_overflow` does (`passes.py:343-379`) | `allocator.py` | R4.5 |
| `unified_partition_solve` pass + its own `SolveError` handler at the new slot — the existing try/except (`allocator.py:2211`) wraps only pass 455 and never sees this | `passes.py` (between `:443` and `:448`) | R8.3 |
| Skip `_maybe_coarse_tile_span_overflow` on success; retain it verbatim as the fallback tiler | `passes.py:448` | R8.3 |
| Force `cut = 0` inside a hint scope | solver, driven from `op.dim_hints` | R4.3 |
| Hint pins: hinted op enters as a single-config buffer; `SPYRE_INDUCTOR_IGNORE_HINTS` drops pins | enumeration | R5.1–R5.8 |
| Record predicted sizes, lifetimes, views under H5; interstitial tick coordinates | solver + allocator | R7.2, R7.5 |
| Placement re-solve warm-started from residency intent; degrade-to-spill carries a distinct `residency_reason` | `allocator.py`, `plan_solver.py:68` | R7.3, R7.4 |
| Derived legality pad reaches predicted sizing; lift `padding.py`'s fixed policies; shared-operand emission in `lower_pad_sequence` | `padding.py:73`, `:163`, `:183`; `pass_utils.py:1191` | R10.1, R10.4 |
| Post-apply legality assertion — a hard failure, never a degrade-to-greedy | `passes.py` | §5 |

R10.2 / R10.3 / R10.5 (discretionary pad) stay closed until the R10.3
measurement says otherwise — see *Parallel work* below.

## Step 5 — Reductions

| Work item | Where | Requirement |
|---|---|---|
| Emit single-level reduction options; never nested or multi-level | `wsr/enumerate_tilings.py` | R1.8 |
| Reuse `_seed_buffer_for_carry` (`:575`) to reject carry-propagating recurrences. Do **not** gate on `_validate_reduction_tiling` (`:1233`) — it over-approves the known wrong-numerics shapes | enumerator | R1.4, R1.8 |
| Widen the R4.6 pin from hint-driven to hint-driven **or** solver-chosen | solver wrapper | R4.6 |
| Predict the accumulator, identity fill, and combine op (`_propagate_tiled_reduction_op`, `:2501`) — not the nested second accumulator, which is never emitted | `wsr/tile_prediction.py` | R7.1 |
| `enable_reduction_tiling` keeps its default and meaning | `config.py:82` | R5.6 |

Test extension: assert **no** nested output+reduction or multi-level reduction
option is emitted, anchored on the exact shapes that are known-wrong today in
`test_coarse_tile_e2e.py`:

- `correctness=False`, "nested tiling + reduction correctness bug" —
  `test_min_2d_512x256_reduce_dim0_A4_B4` (`:706`),
  `test_min_2d_512x256_reduce_dim1_A4_B4` (`:747`),
  `test_min_3d_512x256x256_reduce_dim2_A4_B2_C4` (`:802`).
- `@pytest.mark.skip`, "inconsistent loop_count across reduction fill/combine
  nodes" — `test_min_3d_..._reduce_dim0_A4_B2_C4` (`:765`), `..._dim1_...`
  (`:784`), `test_add_min_3d_..._reduce_dim0/1/2_A4_B2_C4` (`:950`, `:976`,
  `:1002`).

Every admitted option must apply **and** match CPU (R1.6) — applicability alone
would pass on all eight, because their failure mode is a silent wrong answer.

## Requirement traceability

Legend — **Step**: 0/1/2/3a/3b/4/5. **Level**: E2E (`test_scratchpad_use.py`,
`test_padding.py`, `test_coarse_tile_e2e.py`), SOL (`test_scratchpad_solver.py`),
ENU (`test_enumerate_tilings.py`), BENCH (measurement, not a test).

| Req | Step | Level | Pinned by |
|---|---|---|---|
| R1.1 | 3b | ENU | brute-force reference equality on small shapes |
| R1.2 | 3b | ENU | tiered deterministic order; truncation drops from tail; mandatory keeps never dropped; `_combo_cost` absent from the path |
| R1.3 | 3b | ENU | untiled option present in every returned set |
| R1.4 | 3b | ENU | patch each reused predicate, assert called |
| R1.5 | 3b | ENU | extents equal `_planned_tile_extents_per_level` output |
| R1.6 | 3b/5 | E2E | apply each option **and** compare to CPU |
| R1.7 | 3b | ENU | cap defaults; assert not migrated to `config.py` |
| R1.8 | 5 | ENU+E2E | no nested/multi-level option, on the eight known-wrong shapes |
| R1.9 | 3b | ENU | no-span-pressure op still yields > 1 option |
| R2.1 | 0 | SOL | `PartitionConfig` fields and derived scalars |
| R2.2 | 3b | SOL | division enumeration runs once per tiling option |
| R2.3 | 3b | SOL+E2E | joint span check; over-tiling fix picks a smaller tile count |
| R2.4 | 3b | SOL | seed pair always retained; signature dedup; no cap |
| R2.5 | 3b | SOL | empty config set raises `Unsupported` at today's point |
| R2.6 | 3b | SOL+E2E | `prep_cache` key includes tile; negative cache-sharing test; predicted view == recomputed post-`coarse_tile` |
| R3.1 | 0 | SOL | `objective` keyword-only on all five solvers; accepts a `sympy.Expr` |
| R3.2 | 3a | SOL | one `Minimize`; no lock inequality; no second `Solve`; no cost-expr/legacy fork |
| R3.3 | 3a | SOL | accept case per node; `CostExpressionError` per rejected construct, raised by `validate` before lowering and naming the node; unbound symbols rejected |
| R3.4 | 3a | SOL | one `COST_SCALE` for the whole expression; overflow raises; no float coefficient reaches `AddMinEquality` |
| R3.5 | 3a | E2E | spill-parity, no core regression at equal spill |
| R3.6 | 0 | SOL | four placement solvers ignore objective, warn once |
| R3.7 | 3a | SOL | ≤ 1 `AddElement` per symbol **and** ≤ 1 reified var per `Min`/`Max` surviving flattening; model size linear; `SumOverEdges`/`relayout_bytes` absent |
| R4.1 | 3b | SOL | triple table total; `cut` determined not merely constrained |
| R4.2 | 3b | SOL | untileable boundary pinned; every cut-free run is a contiguous slice |
| R4.3 | 4 | SOL+E2E | hint scope never split |
| R4.4 | 3b | E2E | op order unchanged across the solve |
| R4.5 | 4 | E2E | groups round-trip; no `loop_group_id` **or** `hint_id` collision |
| R4.6 | 3b/5 | SOL+E2E | both boundaries pinned; singleton group; hint- and solver-chosen |
| R4.7 | 3b | SOL | directional admission; unverifiable pair fails closed; claim orientation |
| R4.8 | 3b | SOL | no `boundary_op ⟹ ¬in_buffer` constraint; row-3 eviction only |
| R4.9 | 3b | SOL | cut-free reserves no read copy; cut creates one |
| R4.10 | 3b/4 | SOL+E2E | `full_size`/`boundary_view` equal realized values |
| R5.1 | 4 | E2E | hinted op enters as a single-config buffer |
| R5.2 | 4 | E2E | solver never re-tiles or un-tiles a hinted op |
| R5.3 | 4 | E2E | unhinted ops are tiled automatically |
| R5.4 | 4 | E2E | `SPYRE_INDUCTOR_IGNORE_HINTS=1` drops pins |
| R5.5 | 4 | E2E | **negative**: hint group is not grown with solver neighbours |
| R5.6 | 3b/5 | E2E | reduction hint applies; `enable_reduction_tiling=0` raises `Unsupported` |
| R5.7 | 4 | SOL+E2E | H3 levels 1/2/3 name the offending key |
| R5.8 | 0/3b | E2E | `AUTO_COARSE_TILING` off ⇒ every unhinted op `tile=None` |
| R6.1 | 3b | E2E | restickifies still inserted, unchanged |
| R6.2 | 3b | E2E | **guard only**: restickify count does not regress |
| R6.3 | 3b | SOL | views land (== R2.6); relayout symbols absent (== R3.7) |
| R7.1 | 3b | SOL | predictor mutates no IR (deep-compare before/after) |
| R7.2 | 4 | E2E | records exist; offline fixture comparison holds |
| R7.3 | 4 | E2E | decide → `coarse_tile` → commit → re-solve |
| R7.4 | 4 | E2E | re-solve warm-started; degrade carries a distinct reason |
| R7.5 | 4 | E2E | predicted vs realized lifetimes under rank-order normalization |
| R8.1 | 0 | E2E | gates default off; warn and no-op without `cpsat` + co-opt |
| R8.2 | 3b | SOL | `AddHint` seeded from the heuristic plan |
| R8.3 | 4 | E2E | `INFEASIBLE` → span-overflow path, IR untouched; timeout → incumbent applied |
| R8.4 | 3b | E2E | identical plans across two runs |
| R8.5 | 0 | E2E | covered by existing `TestCpSatAllocatorFallback` |
| R10.1 | 4 | E2E | predicted sizes from padded `device_size` |
| R10.2 | 4* | SOL | pad row unlocks an otherwise-blocked split |
| R10.3 | 0 | BENCH | unaligned-`K` matmul vs hand-pre-padded |
| R10.4 | 4 | E2E | two matmuls sharing an operand emit one padded buffer |
| R10.5 | 4* | E2E | pad pin lowers to a pin; empty set named at H3 level 2 |
| R10.6 | — | — | **absence**: issue #1756 restriction untouched (Phase 3) |

`4*` = conditional on R10.3's measurement.

## Sequencing, risk, parallel work

```text
0.1 spike ──▶ 0.3 scaffold ──┬──▶ 1 (e2e xfail) ──┐
                             └──▶ 2 (solver xfail) ┴──▶ 3a ──▶ 3b ──▶ 4 ──▶ 5
R10.3 bench ─────────────────────────────────────────────────▶ (opens R10.2/.5)
```

Steps 1 and 2 are independent of each other and can run in parallel. The R10.3
measurement is independent of everything and should start at step 0, because its
answer decides whether R10.2/R10.5 are in scope at all.

**Highest risks, in order.**

1. **R2.6 tiling-aware per-core views.** A wrong view grants residency on a
   slicing agreement that does not hold — wrong data, and outside R7.4's
   degrade-to-spill safety. The negative cache-sharing test is the cheapest
   guard, and §0.2's mechanism means it exists before the code does.
2. **The objective collapse (R3.2/R3.5).** Changes every existing plan with
   tiling off. Isolated as step 3a so its blast radius is measurable.
3. **R4.6 in step 3.** Omitting the hint-driven pin produces plans
   `coarse_tile` will reject at apply — an illegal emission, not a fallback.
4. **Name-keyed symbol binding (§3a).** Producer and solver agree only through
   the buffer name, and every failure mode of that agreement is quiet: an
   unbound symbol is a `NameError` from generated code, a dropped `Min` arg is a
   wrong cost with no error at all (both live on the reference branch today).
   Nothing downstream can detect it — the plan is legal, just worse. `validate`'s
   containment check and the flattening test are the only guards, which is why
   both are step-2 obligations rather than step-3 ones.
5. **Enumeration cost.** `enumerate_work_division_candidates` runs per tiling
   option (R2.2) and `_views_for_divs`' sympy prep is no longer
   candidate-invariant (R2.6). Both were built assuming they are paid once per
   op. Measure compile time at step 3b, not at step 5.

## RFC corrections — applied

Found while grounding this plan against the tree, and folded back into
`draft-unified-tiling-cpsat.md`.

| Location | Correction |
|---|---|
| Testing item 1 | Claimed `test_coarse_tiling.py` has no CI config yaml. It does — as do all five named suites, each `unlisted_test_mode: mandatory_success`. Replaced with what that mode implies for landing an xfail. |
| R1.4, R1.9 | `_host_dim_has_legal_nontrivial_split` cited at `:935`; it is at `:936`. |
| R1.4 | Reuse list omitted `_remaining_span_candidates_after_tile` (`:1236`) — the span-*sufficiency* check both public entry points compose (`:1344`, `:1471`), and the one R2.3's joint span feasibility should extend rather than restate. Added with that rationale. |
| R4.5 | Handled `loop_group_id` collision via `group_idx_offset` but not `hint_id`. Added the second namespace: span-overflow mints from `_SPAN_OVERFLOW_HINT_ID = 10000`, so the solver needs its own reserved base or `validate_coarse_tile_groups` raises during apply — an illegal emission, not something the model could rule out. |

**One pending fold-back, not yet applied.** R3.7 bounds symbol binding at one
`AddElement` per symbol but is silent on the aux vars `Min`/`Max` reification
creates (§3a). The plan carries the tighter bound and pins it; R3.7 should gain
the second clause.

**Two earlier flags retracted.** Both came from a misreading of the RFC, not
from the RFC:

- §1 and R1.4 cite `op_out_coords` at `pass_utils.py:363` and describe
  `host_dim` as the frame it indexes. That is exactly right — `op_out_coords`
  is at `:363`, and `host_dim` is a positional index into its return. No change.
- *Background* line 163 writes `{id(op): CoarseTileInfo}` without claiming where
  `CoarseTileInfo` is defined (it is `loop_info.py:33`, imported at
  `coarse_tile.py:93`). Nothing to fix.

**Not a correction — a known forward reference.** The parent link
`draft-compiler-optimization.md` does not resolve on this branch; the roadmap
lives on `optimization-roadmap-draft`, and commit `048e558` removed the local
`draft-compiler-optimization-roadmap.md` copy. It resolves when the branches
converge. Left as-is deliberately.

**Verified sound, not a correction.** The *Background* claim that interior
per-tile scratch stays LX-eligible is stated verbatim in `_is_tiled_advancing`'s
docstring (`scratchpad/utils.py:218-234`). The motivation holds as written; step
0.1 confirms it end-to-end anyway.
