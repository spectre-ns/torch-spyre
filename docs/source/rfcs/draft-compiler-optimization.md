# RFC (draft) — Compiler Optimization Roadmap

| Field | Value |
|---|---|
| Status | Draft — pending issue number |
| Area | Compiler |
| Target | `torch_spyre/_inductor` — `wsr`, `scratchpad`, layout and padding passes |
| Depends on | RFC 0047 (Tensors with Device-Specific Layouts), RFC 1358 (Coarse Tiling) |

> **Draft note.** RFC sources live in
> [`torch-spyre/rfcs`](https://github.com/torch-spyre/rfcs) as
> `NNNN-PascalCase/NNNN-PascalCaseRFC.md`, where `NNNN` is the GitHub issue
> number. This file is the working draft; on filing the issue it moves to that
> repository and gains a row plus a summary paragraph in
> `docs/source/rfcs/index.md`.

## Summary

This is the **parent document** for a sequence of compiler optimizations. It is
not itself an implementation proposal: it states the model those optimizations
share, the order they land in, and why that order is the one that works. Each
individual optimization gets its own RFC, and each of those is an instantiation
of this document rather than an input to it.

The compiler makes a sequence of decisions about the same underlying —
how a buffer is partitioned, shaped, laid out, placed, and when it is live — at
different points of one pass list, each with its own cost model and each blind
to the others. This document folds those decisions into a single model with a
single objective, and specifies the contract that makes doing so tractable:
> Every axis for each buffer is either **pinned** by a hint or **optimized**. A
> validator proves the hint set is mutually consistent before the solve. The
> solver determines the optimized axes — a pin collapses its axis to the hinted
> value before the model is built, and the result is optimal over what remains.

The two-mode framing is load-bearing, not a footnote. **An axis that is not being
optimized still has to be valid, and validity checks are cheap.** A pin is proved
consistent before the solve rather than priced during it, so a hinted axis costs
neither variables nor search.

## Motivation

The decisions are made at different points of `CustomPreSchedulingPasses`
(`passes.py:416-455`), by cost models that cannot see each other:

| Decision | Where | Cost model |
|---|---|---|
| Coarse tiling | `wsr/`, passes 6 and 17 | `_combo_cost` structural proxy |
| Core division | `work_division.py`, passes 18-19 | µs model for matmul; heuristic otherwise |
| LX residency | `scratchpad/`, pass 20 | CP-SAT, two-phase lexicographic |
| Padding | `padding.py`, pass 15 | none |
| Restickification | `optimize_restickify.py`, pass 10 | element count |
| Op order | upstream DFS, repaired in `scheduler.py` | none |
| LX placement | `scratchpad/` solvers | per-solver: greedy heuristic, or CP-SAT with no fragmentation term |

Coarse tiling ranks split combinations by `_combo_cost` — total tile count, then
tiled-dim count, then largest split (`wsr/span_overflow_hint_analysis.py:1181-1192`)
— and takes the cheapest feasible one from a sorted product
(`_iter_split_combos`, `:1205`). Core division prices matmuls in microseconds
(`_matmul_split_cost`, `work_division.py:1206`) and everything else by a
dimension-priority heuristic (`prioritize_dimensions`, `:670`). LX residency runs
a two-phase lexicographic CP-SAT solve — minimize spill, lock it, then maximize
cores (`scratchpad/ilp_solver_ortools.py:446-518`). The remaining three have no
cost model at all.

Six specific consequences follow, each verifiable in the tree.

### 1. The span budget is spent twice

`core_split_estimate` is hardcoded to `1` at both `ChunkingInfo` construction
sites (`wsr/span_overflow_hint_analysis.py:666`, `:827`), so the tiling planner
does span arithmetic as if the operation ran on a single core. Work division then
splits the same dimensions again. `MAX_SPAN_BYTES` (`work_division.py:73`) ends
up satisfied twice over and the emitted tile counts are larger than necessary —
loop overhead, plus HBM traffic wherever the surplus tiling forces boundary
copies.

The code says so directly. The planner's docstring (`:1495-1502`) notes that
`max_cores` is threaded through "but it has no effect on the emitted split today:
every candidate's `core_split_estimate` is hardcoded to 1, so span math always
assumes work division gives no help". `:1504-1506` carries the TODO: *"make a
common planner for Work Division and Working Set Reduction together, so this pass
can get a proper core_split_estimate instead of the hardcoded 1."*

Stated in the model's terms: tiles are **sequential** loop iterations and cores
are **parallel**, but both draw down the same divisibility budget on a dimension.
Today each decision spends that budget as if it were alone.

### 2. Tiling buys LX residency only by accident

Shrinking a chain's working set until it fits the LX scratchpad is the stated
purpose of `wsr/`, and it does sometimes achieve that — but with no feedback
edge. The tiling planner cannot see LX occupancy and the LX solver cannot choose
a tiling, so nothing verifies that a chosen tiling improved residency, nothing
rejects one that made it worse, and nothing establishes that the selected
configuration is the best of the feasible ones. Residency is an uninstrumented
side effect of a decision taken for span reasons.

`docs/source/compiler/scratchpad_planning.md` records this among the remaining
gaps under "Co-optimization is still limited", listing "**No coarse-tiling
integration** when that pass also drives split decisions" alongside "**No
performance model**".

Consequences 1 and 2 are the same missing edge read from two directions: tiling
cannot see what division has done, and neither can see what residency would cost.

### 3. Padding cannot influence the layouts it must satisfy

`insert_bmm_padding` runs at pass 15 (`passes.py:441`); `finalize_layouts`
commits layouts at pass 11. Padding is therefore applied to a layout it had no
say in.

The reverse direction is blocked too. Three sites in `propagate_layouts.py`
(`:271`, `:455`, `:1078`) each skip a candidate stick dimension whose size is not
stick-divisible, under the same TODO referencing issue #1756 — because padding is
not available to them as a legalization tool. The layout search is smaller than
it needs to be, and the reason is a pass-ordering artifact.

Within its own scope, padding is a fixed policy: `compute_padding`
(`padding.py:73-76`) rounds up to one stick, and the pass pads the y operand only,
on the reduction dimension only, at the right end only, with zero fill only. For
the amount needed to reach legality a fixed policy is the right answer — there is
nothing to decide (phase 1). The defects are the ones the policy forecloses: no
sharing, so the loop at `:183` emits a separate four-op pad sequence per matmul
and two matmuls reading the same operand each pay for their own; and no way to
pad past legality, which is what would let a dimension become splittable or a
stick dimension become admissible.

In summary, padding costs memory (Both HBM and LX) and this cost is not accounted
for inside the current padding strategy as it falls out of upstream choices.

### 4. Restickification is priced blind to what it costs

The cost function is `prod(in_stl.device_size)` for a needed relayout, `0` when
compatible, and infinity when infeasible (`optimize_restickify.py:83-106`) —
element count and nothing else. It omits that a restickify also bars its input
from LX entirely: `_restickify_barrier` (`scratchpad/allocator.py:475-497`)
observes that the operation's per-core read and write frames are transposes of
each other, so a per-core slice of the output can need bytes from another core's
slice of the input.

The pass named `optimize_restickify_locations` also does not choose locations.
It selects layouts; the restickify node is always spliced immediately before its
consumer (`insert_restickify.py:215-216`), with no hoisting, no sinking, and no
sharing between two consumers that need the same relayout. Its search is a beam
of hardcoded width 200 (`optimize_restickify.py:467`), and graph-input layouts —
which are free at DMA time — are pinned to a single candidate
(`propagate_layouts.py:1515`) and never explored.

### 5. Op order is chosen by a DFS and repaired after the damage

Upstream Inductor's topological sort guarantees only *a* valid order; it can
interleave unrelated nodes into the middle of what coarse tiling built as one
contiguous loop group. `_regroup_by_outer_loop_key` (`scheduler.py:128-215`)
restores that contiguity — but it runs after LX planning has already committed
against the old order, and `demote_incoherent_lx_buffers` (`:402-475`) then
un-pins buffers whose per-core views stopped agreeing. Order is thus already
being changed; it is simply being changed reactively, downstream of the decisions
that depend on it.

### 6. LX placement cannot move a buffer that is already live

The greedy solver — today's `LAYOUT_SOLVER` default, and a development artifact
rather than a chosen one (M1) — processes operations in
topological order making irrevocable placement decisions with no lookahead, and
`_find_free_block` can locate holes between allocations but cannot compact them —
both recorded in `scratchpad_planning.md` under "Greedy single-pass, no lookahead"
and "No defragmentation". First-fit and best-fit mitigate by sorting buffers up
front (`scratchpad/firstfit_bestfit_solver.py:186-245`), and `CpSatLayoutSolver`
removes the ordering question altogether, placing every buffer in one global
`AddNoOverlap2D` over time × address (`ilp_solver_ortools.py:31`, `:568`) and
squeezing out the residual gaps in `_justify` (`:747`).

What no solver can do is move a buffer once it is live. Every one of them assigns
a single address for a whole lifetime (`LifetimeBoundBuffer.address`,
`plan_solver.py:64`), so allocate/deallocate cycles fragment the address space and
a plan whose aggregate free space is more than sufficient can still fail to place
a buffer for want of one large enough hole.

### 7. (Adjacent) Caching key insufficiency
One adjacent bug is recorded here because the roadmap makes it more expensive,
not because the roadmap fixes it. torch-spyre participates in Inductor's FX graph
cache and extends its key to cover input `SpyreTensorLayout`s
(`_patch_fx_graph_hash`, `torch_spyre/_monkey_patch.py:291-335`), while the pass
pipeline separately hashes only pass **source files** (`passes.py:144-151`). No
torch-spyre config value reaches either key, so a `SENCORES=8` run and a
`SENCORES=32` run share an entry. That is a wrong-artifact bug today,
independent of this document, and the fix is an extension to a hook that already
exists; it should land immediately rather than being scheduled against any phase
here. Every phase that follows widens the gap between two runs that share a key,
which is the only reason it appears in this list.

## The model

These requirements are stated once here so that no phase re-derives them. Each
collateral RFC satisfies the ones its axis touches and cites them by number
rather than restating them.

- **M1 — One solve, one objective, and the solver is CP-SAT.** All *optimized*
  axes resolve in a single CP-SAT instance minimizing a single expression. This
  is the target the whole document is written against: `CpSatLayoutSolver`
  (`ilp_solver_ortools.py:321`) is the solver the phases extend, and every
  encoding decision here — M4's tables, M5's joint feasibility, H2's
  enumeration-time pins — is chosen for it.

  The heuristic placement solvers are **not** targets. Greedy, first-fit, and
  best-fit exist as the M9 fallback, and nothing in this document optimizes them
  or treats their internal policies as decisions.
  Simulated annealing (`SimulatedAnnealingLayoutSolver`, registered in
  `_PLACEMENT_SOLVERS` at `scratchpad/allocator.py:2108-2113`) is the anticipated
  *second* backend rather than a fallback: it searches a different space — a
  buffer permutation — so it can accept work CP-SAT finds expensive. No phase
  here delivers it, but no phase may preclude it, which is a constraint on M2 and
  M4 rather than a deliverable. The existing `LAYOUT_SOLVER` plug point
  (`config.py:111-113`) is the template both for selecting a solver and for
  degrading when `ortools` is absent.

  That `LAYOUT_SOLVER` currently defaults to `greedy` is a development artifact,
  not a design position — the standing TODO already moves it to `firstfit`
  (`config.py:110`), and a solver-backed default is the expected end state. This
  document assumes it: the phases are written for the configuration the compiler
  is heading toward, and the heuristics persist for fallback and for the parity
  testing M3's candidate invariant is built on, not as the destination.

- **M2 — The objective is injected, not hardcoded.** The cost function is
  caller-supplied, so cost experiments do not require editing the solver. Today's
  objective is inlined in `_run` (`ilp_solver_ortools.py:446-518`). The
  replacement is a symbol namespace bound to model variables plus a lowering to
  CP-SAT expressions over an explicitly bounded grammar; a construct outside that
  grammar is a compile error naming the offending node. Silently approximating an
  objective is worse than failing to build one. A pinned axis contributes no
  symbols.

  **The objective is a cost, and the model minimizes it.** One expression in one
  unit, not a ranking. Today's two-phase lexicographic structure — minimize spill,
  lock that value, then maximize cores (`ilp_solver_ortools.py:446-518`) —
  collapses into it, which is a real change and not a restatement: lexicographic
  order makes a spill infinitely worse than any amount of lost parallelism, while
  a cost prices the two against each other and takes the cheaper. That is the
  intended behaviour. A trade that a correct cost model says is worth making is
  worth making, and a trade it gets wrong is a C1 defect to be fixed in the
  numbers rather than fenced off with a guard in the solver.

  The grammar must additionally be **numerically evaluable on a fully assigned
  candidate**, not only lowerable to CP-SAT expressions. A search-based backend
  scores concrete states rather than building an expression tree, so an objective
  that exists only as a lowering locks the model to CP-SAT for good. Two
  consumers of one grammar is also the cheapest available check that the lowering
  is faithful: the evaluated cost of a returned solution must equal the objective
  value the solver reports.

- **M3 — Two modes per axis: pinned and optimized.** The central tractability
  mechanism. The two differ along one dimension — what determines the value.

  - *Pinned* — a hint pre-selects the value. The domain is collapsed at
    enumeration time, so the axis contributes no variables and no search (H4).
  - *Optimized* — the objective prices the value. The axis contributes variables
    and objective terms and is ranked.

- **M4 — Enumerate-and-table encoding.** Nonlinearity — divisibility, stick
  alignment, span limits, the core budget — is absorbed by precomputing a
  feasible candidate set per operation and binding derived scalars with
  `AddElement`, exactly as `CoreDivision` is consumed today
  (`ilp_solver_ortools.py:255-273`). The rule that follows is the model's only
  extension mechanism, and later phases depend on it: **any
  `f(candidate) -> int` becomes a table entry plus one `AddElement`.** A
  nonlinear property of a candidate becomes a table lookup rather than a
  constraint. The cost is enumeration, which M5 and C2 bound and phase 5 reduces
  if measurement demands it.

- **M5 — Feasibility is evaluated on the combination, not per subsystem.** A
  candidate is feasible iff it satisfies every subsystem's guard
  jointly. Reuse `enumerate_work_division_candidates`'s `valid_split` guards
  verbatim (`work_division.py:809-823`): the core budget, at most one reduction
  dimension split, a per-core span within `MAX_SPAN_BYTES` on every tensor
  dependency, and no coordinate-masked dimension split. Evaluating them on the
  combination rather than one axis at a time is what discharges consequence 1.

  The table is **per operation**. The joint (tiling, division) space is
  enumerated eagerly for one operation at a time — which is what
  `enumerate_work_division_candidates` already does for division alone
  (`work_division.py:825-830`) — and the combination *across* operations is what
  the solver decides, never what enumeration materializes. Eager is the
  deliberate choice for current problem sizes; phase 5 owns shrinking the
  per-operation table when measurement says it must.

- **M6 — Predicates are reused, never reimplemented.** Every legality predicate
  the model needs already exists, in `wsr/span_overflow_hint_analysis.py`,
  `pass_utils.py`, and `work_division.py`. Reimplementing one creates a second
  source of truth that will drift. An optimized axis is constrained by the model
  itself, so the obligation binds the M9 fallback path: the
  legality check the existing heuristic applies must be the *same* predicate the
  model constrains against, or the two disagree about what is legal and the
  fallback stops being a fallback.

- **M7 — Pure prediction, then apply.** A decision is scored against a
  *predicted* buffer set with no IR mutation, then applied. State the hazard once
  rather than per phase: liveness is index-based (`calculate_liveness`,
  `scratchpad/utils.py:84-103`, with `start_time` / `end_time` derived at
  `scratchpad/plan_solver.py:75-81`), so any pass that *inserts* operations shifts
  the lifetime ticks of everything downstream. That is a systematic offset, not
  noise, so any comparison of predicted to realized values is made under
  rank-order normalization.

  Whether the predictor is faithful is settled **offline**, against recorded
  plans, not by an assertion pass inside the compiler: no phase owes a verify
  mode, and none adds one. What the phases owe instead is that the inputs to that
  offline check exist — the predicted values a decision was scored against are
  recorded alongside the decision itself (H5), so predicted and realized can be
  compared after the fact without re-deriving either.

- **M8 — Addresses come from the final solve.** Whatever the joint model
  predicts, physical placement is recomputed over the real post-transform
  buffers. A misprediction therefore degrades to a spill, never to a wrong
  address.

- **M9 — Failure never discards a pin.** A pin is an enumeration-time domain
  restriction (H2), not a solver-level preference, so it survives any fallback by
  construction: the fallback runs over the *already-pinned* candidate sets. A
  hint is therefore always respected, or the compile fails with an error naming
  the hints that conflict. It is never silently dropped. Three failure causes,
  three outcomes:

  1. *Solver unavailable, timed out, or errored.* Every axis leaves the model over
     the pinned candidate sets: the existing heuristic runs and only validity is
     enforced, using the same predicates the model constrains against (M6). This
     is a degraded *choice* over the same candidate sets, not a phase that stops
     running: enumeration and legality are unaffected, and only ranking is lost.
     M3's invariant is what makes it available — today's value is in the table.
     This generalizes the current `SolveError` fallback to the greedy allocator
     (`scratchpad/allocator.py:2193-2219`) and the ortools-missing fallback
     (`_make_cpsat_solver`, `:2116-2136`). The existing heuristic already honours
     hints on the one axis that has them — `work_distribution_pass` returns early
     on a resolved hint (`work_division.py:1109-1141`) — so this is the current
     behaviour generalized, not a new obligation.
  2. *Infeasible with the pins, feasible without them.* A hint conflict. Hard
     error naming the conflicting subset (H3.3). Never a fallback: falling back
     here would silently produce a plan the user's hints forbid.
  3. *Infeasible without the pins.* A compiler defect or a genuinely infeasible
     graph. Error, not fallback.

  Distinguishing (2) from (3) costs one extra pin-free solve, on the failure path
  only, and it is what gives H3.3's diagnostic pass its entry condition. Note
  that H3.2 catches most pin conflicts before any solve; M9's error path exists
  for conflicts that are only visible across operations.

  The graph must be unmutated when the solve fails, so the fallback starts from
  clean IR — which means the solve precedes the transform it decides.

## Hints, pins, and the validator

The spine. Collateral document 0 owns it in full; this section states the
contract every phase registers against.
- **H1 — A hint registry.** A table mapping each hint key to its value schema and
  its scope kind (operation, dimension, edge, buffer, subgraph), consulted at
  validation time. It is **not a new data path**: hints continue to be attached
  and read exactly as they are today, per operation.

  The *transport* needs no redesign. `spyre_hint(**kwargs)`
  (`propagate_hints.py:85-93`) is a `torch.fx.traceback.annotate` context manager,
  so every FX node traced in scope carries the hint id — `decompositions.py:447-455`
  nests four of them — and `get_op_hints` (`:96-113`) reads them back per
  operation off `op.origins`. AOT re-tracing is survived by `collect_spyre_hints` /
  `recover_spyre_hints` (`:138-208`). Hint scope is therefore already
  multi-operation and already propagated: `coarse_tile_hints.py` groups operations
  by hint id today. That is why scope kind is a *declared* field rather than an
  implied one — a pin has to know what it collapses.

  What is missing is the schema. Consumers each pull their own key out of the
  untyped dict — `work_div` at `work_division.py:835-843`, and
  `tiles` / `slices` / `num_tiles_per_dim` at
  `wsr/propagate_named_dims.py:636-644` — so a key no consumer matches is silently
  a no-op: `spyre_hint(tile={"A": 2})`, singular, compiles and does nothing.
  Well-formedness is ad hoc for the same reason: the one-of-`tiles`/`slices`/
  `num_tiles_per_dim` rule is an inline generator expression at `:636-644`, while
  `work_div`'s shape checks live in `_apply_user_hint`
  (`work_division.py:866-932`). Registration turns an unknown key into an error
  naming the nearest registered one, and gives H3.1 one place to check shape and
  scope instead of one per consumer.

  What a registry does **not** buy is agreement between two hint families on the
  same operation. That needs feasibility predicates, not a schema, and it is
  H3.2's job.

- **H2 — A hint lowers to a pin.** Each hint becomes a domain restriction applied
  at *enumeration* time: the candidate list is filtered before the model is
  built, not overridden after a choice is made. On a hinted axis the filter leaves
  **exactly one option**, so the solver is not choosing under a preference — it is
  handed the value and searches only what is left.

  This generalizes a mechanism that already exists. `_division_map` and
  `_fixed_core_division` (`scratchpad/allocator.py:1527-1556`) already collapse an
  operation's candidate list to a single fixed `CoreDivision` for ops in an
  offset-mutation component. Hint-as-pin is that same path, driven by a hint
  rather than a legality rule.

  The current gap is that `enumerate_work_division_candidates` is hint-blind, so
  hints today take ownership *after* selection: `work_distribution_pass`
  (`work_division.py:1110-1142`) returns early on a resolved hint, and
  `_cost_model_divide_op` (`:1642-1644`) declines to override one. Correct
  behaviour, wrong layer.

- **H3 — The validator.** Three levels, each anchored to a point in the pipeline:
  level 1 runs before enumeration, level 2 during enumeration, and level 3 only
  after the solver returns `INFEASIBLE`. Levels 1 and 2 are the entire normal
  path — a hint that is malformed, or that no candidate satisfies, is named before
  any model is built, so the solver only ever sees pins that are individually
  legal.

  **The design target is that conflict detection is linear, not NP.** A solve is
  exponential in the worst case and answers a question most hint conflicts do not
  need asked: two pins that disagree, or a pin that violates a budget, are local
  facts about the hint set. Levels 1 and 2 are therefore built to catch as much as
  a pass over the hints and the operations they scope can catch — O(hints) plus
  O(operations), no search — and level 3 exists only for what provably cannot be
  decided that way. Every conflict pushed down into the linear check is both
  faster and better attributed, since it names the rule that failed rather than a
  subset of assumptions.

  1. *Well-formedness — before enumeration, on the hint set alone.* The key is
     registered, the value's shape and type are
     correct, and the scope resolves to a live operation, dimension, edge, or
     buffer. This subsumes the checks scattered through `_apply_user_hint`
     (`work_division.py:866-932`) and the one-of-`tiles`/`slices` rule at
     `propagate_named_dims.py:636-644`.
  2. *Realizability — during enumeration, one operation at a time.* The pinned
     value survives the phase's own feasibility
     predicates (M5, M6). This closes a real hole: a `work_div` hint that reduces
     splits `span_reduction` already committed proceeds today with only a warning
     — *"Applying strict user hint; this may violate the hardware span limit"*
     (`work_division.py:1110-1142`) — and `_apply_user_hint` silently skips, at
     `logger.info` level, a dimension whose split would exceed `SENCORES`. Both
     become named errors.

     This is also where two hint families on one operation are reconciled, which
     no consumer does today: because M5 evaluates the combination rather than one
     axis at a time, a `tiles` pin and a `work_div` pin on the same dimension are
     checked against the same `valid_split` guards, and a pair that is
     individually legal but jointly infeasible is named here rather than
     surviving to the solve. A pin that empties an operation's candidate list is
     caught here too: enumeration returns nothing, and the error names the hint
     and the predicate that rejected it.

     All of this stays linear because the predicates are evaluated per candidate
     row and per operation, never across the graph: the pinned combination either
     appears in the operation's enumerated set or it does not. The collateral
     document owns the list of conflicts reachable this way, and the pins whose
     product is checkable against a fixed budget — `SENCORES`, `MAX_SPAN_BYTES`,
     one reduction split — belong on it.
  3. *Conflict diagnosis — only after the solve returns `INFEASIBLE`.* Levels 1
     and 2 each look at one operation, so what survives them and still fails is
     exactly one thing: a set of pins that are individually realizable but
     jointly infeasible *across* operations. That is the whole of level 3's
     domain, and nothing else routes to it — in particular `UNKNOWN` from the
     wall-clock limit is not `INFEASIBLE` and goes to M9 case 1.

     The normal solve carries no diagnostic apparatus: pins are collapsed
     domains, so a satisfiable model pays nothing for them. On infeasibility a
     **separate diagnostic pass** re-expresses the pins as retractable
     assumptions and reports a conflicting subset, so the error names specific
     hints instead of a bare UNSAT. The cost is paid only on the failure path,
     which is the point.

     Two properties of the underlying API, recorded so the phase that builds this
     does not assume them away. OR-Tools returns a subset *sufficient* for
     infeasibility, not a minimal one — with an objective attached it will happily
     return every assumption — so the diagnostic model is built objective-free and
     deletion-refined if a true minimal subset is wanted. And re-expressing pins
     as assumptions means re-enumerating without pin filtering, which is the real
     cost of the level.

     **Level 3 is therefore staged behind levels 1 and 2**, which land first and
     cover every conflict visible within one operation. It is also the only part
     of the validator that pays solver-order cost, which is the argument for
     keeping it small: the residue after the linear checks, not the mechanism the
     validator is built around.

  One trap to record so no phase assumes it away: a predicate quantified over a
  *set* of operations does not follow from checking adjacent pairs. Where a
  phase's legality is not pairwise, it must pin conservatively rather than admit
  a merge it cannot verify.

- **H4 — Hints shrink the model; they are accepted, not re-derived.** A pinned
  decision is removed from the decision space at enumeration time (H2), so it
  contributes no variables and no search. The solver does **not** compute what an
  unpinned solve would have chosen, and does not report an optimality gap —
  paying to explore the space a hint exists to avoid would defeat the purpose of
  supplying one.

  Hints are therefore a first-class tractability lever, and with no unpriced mode
  they are the *only* way to keep an axis out of the search once its phase is on.
  The honest claim is
  that the solve is optimal *over what remains*, and this document makes that
  claim rather than a claim of global optimality.

- **H5 — Every decision explains itself.** A per-axis `decision_reason`,
  extending the reason-string pattern already used for residency —
  `residency_reason` (`plan_solver.py:68`), `spill_reasons` (`:219`),
  `reject_reasons` (`allocator.py:137`), `_SOLVER_CHOSE_SPILL`
  (`ilp_solver_ortools.py:113`), and `excluded()` / `record_exclusions()`
  (`plan_solver.py:221-258`). Every committed decision records whether it came
  from a user hint, a compiler-generated hint, a fallback, or the solver. Without
  this a plan cannot be debugged, and the two modes of M3 cannot be told apart
  after the fact: a value a pin fixed is indistinguishable in the output from one
  the solver chose, and both from one the fallback heuristic produced.

## Phases

A phase that adds an axis follows the same template: the defect it closes, the
decision variables it adds, the objective terms that price them, the baseline
candidate M3's invariant requires in its table, the hint form it registers under
H1, what the validator must newly check, what it depends on, and its exit
criteria. One phase adds no axis and does not follow it — phase 5 is
enumeration mechanism — and it says so in its own first line. Padding likewise
adds no axis, so it gets no phase: it is policy and legalization work, and it
lands inside the phases that consume it, 1 and 2.

### Phase 1 — Coarse tiling, for sizing and temporal op splitting

**Vocabulary.** This document calls a loop tile — sequential iterations on one
core — a **temporal split**, and a core division — parallel across cores — a
**spatial split**. The repository says "coarse tiling" and "work division"; the
paired terms are used here only where the contrast matters. Both splits draw down
the same divisibility and `MAX_SPAN_BYTES` budget on a dimension, which is
exactly why spending it twice over-tiles (consequence 1).

**Sizing** is the axis's second role, and the one later phases depend on: the
chosen tile determines the predicted buffer sizes and lifetimes that every
subsequent phase is defined over, and it is the only lever that can shrink a
working set enough to make a buffer LX-resident (consequence 2).

**Variables.** A per-operation candidate index over the joint (tiling, core
division) space under M4, plus one boolean per adjacent pair of operations
deciding whether a loop nest breaks there. Grouping then falls out of the
objective — a tiling group is a maximal run with no break — rather than being
precomputed by a grouping heuristic that a wrong guess would make unrecoverable.

**Padding, which is not an axis.** Padding gets no phase of its own and no index
space; it lands here because this is the first phase that needs it, and it needs
it for sizing. The pad required to reach legality is *derived* — `compute_padding`
(`padding.py:73-76`) rounds up to one stick, and given the layout and tiling
there is no freedom and therefore nothing to decide. It is a scalar on each
candidate row, so buffer sizes are computed from padded `device_size` rather than
from the unpadded shape. Padding *beyond* legality is the one part with a genuine
choice — it can unlock a divisibility `valid_split` requires
(`work_division.py:809-823`), turning an illegal core split into a legal one —
and it enters as additional rows in the tables this phase already builds. **No
new index space**, precisely the case M4's rule was written for.

Whether those extra rows are worth having is an empirical question this document
does not presume the answer to. Padding is not free today: `lower_pad_sequence`
(`pass_utils.py:1191-1227`) emits a four-op sequence — allocate, fill constant,
fill pad region, copy — per matmul, with no sharing between two matmuls reading
the same operand (`padding.py:183`); y's buffer grows and competes for LX; and
K→K_padded widens the SDSC iteration space at codegen
(`_extend_matmul_k_to_padded`). Measuring that — an unaligned-K matmul against
the same model pre-padded by hand — gates whether discretionary pad values are
enumerated at all. If the delta is noise, the derived pad stays purely in the
apply step and no pad row is ever added.

Either way this phase lifts the fixed policies in `padding.py`, since the derived
amount cannot be expressed without them: pad operands other than y, dimensions
other than K, ends other than the right, and multiples other than one stick; and
share a padded buffer between matmuls that read the same operand rather than
emitting a pad sequence per matmul (`:183`). The remaining half — padding as a
*layout legalization* tool, removing the issue #1756 restriction at
`propagate_layouts.py:271`, `:455`, and `:1078` — lands in phase 3 with the
layout search that consumes it.

**Baseline candidate.** Today's behaviour — the `_combo_cost`-ranked first
feasible tiling, checked for span legality only, and padding's unconditional
round-up to the next stick — must remain an enumerated candidate (M3), and is
what the M9 fallback selects.

**Hint form.** The existing `tiles` / `slices` / `num_tiles_per_dim` and
`work_div` keys, registered under H1 and lowered to pins under H2, plus a pin on
a buffer's pad amount.

**Validator.** The hinted value survives enumeration; a hint scope must not
straddle a loop-nest break, the invariant `validate_coarse_tile_groups`
(`wsr/coarse_tile.py:112-137`) enforces today; group contiguity
(`_validate_contiguous`, `:785-814`) should hold structurally, by pinning a break
wherever an operation cannot be tiled, rather than being checked after the fact;
and stick alignment must hold jointly across pad, tile, and division, reusing
`_post_tile_stick_alignment_error` (`span_overflow_hint_analysis.py:263`) and
`_combined_tile_stick_alignment_error` (`:1215`).

**Owed to later phases.** The M4 candidate encoding, the M2 objective namespace,
the M7 predictor, and per-core views evaluated against a *predicted*
post-transform frame — `_prepare_per_core_view` (`pass_utils.py:1467`) and
`_per_core_view_on_buf` (`:1696`) taking a predicted frame rather than reading the
operation's current layout. Phase 3 cannot start without that last one.

**Exit.** Today's tiling is in the enumerated table, asserted per operation. A
case where span overflow
forces tiling *and* work division splits the same dimension provably picks a
smaller tile count than the `core_split_estimate = 1` path. The padding cost
measurement reported either way, existing bmm tests
unchanged, and `tests/inductor/test_padding.py` extended with a shared-operand
sharing case — plus a chosen-pad case if the measurement justifies one.

### Phase 2 — LX relocation and compaction

Paired with phase 4 on the same time axis, and deliberately not adjacent to it:
phase 4 decides *when operations run*, this phase decides *where in LX their
buffers sit* and whether that placement may change while a buffer is live.

It comes second because its defect is the one that is independent of every other
axis. Fragmentation costs spills today, under a program order and a buffer set
that no later phase has to settle first, so it can be measured and closed before
anything else is a variable. Phase 4 later moves lifetimes, so the relocation plan
is re-solved and phase 2's numbers re-measured rather than inherited; nothing is
invalidated, since relocation is defined over whatever lifetimes hold.

**Allocation order is not the axis, and this phase does not model it as one.**
Under a solver, addresses are not handed out in an order at all: `CpSatLayoutSolver`
places every buffer with one global `AddNoOverlap2D` over time × address
rectangles (`ilp_solver_ortools.py:31`, `:568`) and `_justify` (`:747`) then
slides each unit down to the lowest free address to squeeze out gaps the search
left. Order exists only in the heuristics — greedy walking topological order with
irrevocable decisions, first-fit and best-fit sorting buffers up front
(`firstfit_bestfit_solver.py:186-245`) — which is to say it exists only on the
M9 fallback path, where it is a property of the heuristic rather than a decision
anyone makes. Optimizing it would be optimizing the fallback.

**Variables.** Relocation: a buffer's address becomes a function of time rather
than a single value, so a live buffer can be moved and a hole closed. This is a
genuine extension to the model even under CP-SAT, where each buffer is one
rectangle with one address today. Two consequences the collateral document owns.
The plan representation changes — `LifetimeBoundBuffer.address` is a single
`Optional[int]` (`plan_solver.py:64`), and consumers after a move must bind the
new address. And relocation is not free the way an ordering permutation would
have been: it emits an LX→LX copy, so it needs a cost term in the M2 objective or
the solver will relocate whenever it closes a byte.

The second half is objective, not variables: fragmentation and peak occupancy
have no term today — the existing solve minimizes spill, locks it, then maximizes
cores (`ilp_solver_ortools.py:446-518`) — so a plan that fits with a fragmented
address space and one that fits compactly score identically until C1 supplies the
term.

**Baseline candidate.** Today's placement, whichever solver `LAYOUT_SOLVER`
selects, with no relocation: buffers keep one address for their whole lifetime.
The no-relocation plan must stay reachable so M3's invariant holds and the M9
fallback has something to select. The relocation cost term is a precondition for
landing this phase, not a later refinement — without it the variable is unpriced,
which M3 forbids.

**Hint form.** Pin a buffer's residency, or pin it against relocation. Residency
hands users a direct lever over the outcome they most often want to control,
which no current hint family reaches.

**Validator.** A pinned residency must fit the packing across its lifetime; a
buffer pinned against relocation must still admit a single-address placement.

**Exit.** A fragmentation case that no current solver can pack — aggregate free
space sufficient, no single hole large enough — is packed by relocating. No
regression in spill count on the tracked benchmarks, and no relocation emitted
when the fragmentation-free plan already fits.

### Phase 3 — Restickification

Two separable efforts, and the document says so because the existing pass name
conceals the split (consequence 4).

**3a — pricing.** A per-edge relayout variable over *dataflow* edges — the first
**new index space** in the model, everything before it being per-buffer or
per-adjacent-pair — and a relayout-bytes term in the M2 objective, precomputed
per edge from the tiling-aware per-core views phase 1 delivers. The variable is
*determined* rather than chosen: an edge needs a relayout, or it does not, given
the two endpoint candidates. What the solver gains is the ability to prefer
candidate pairs that agree.

**3b — placement.** Hoist, sink, and share restickify operations; unpin
graph-input layouts, currently fixed to one candidate
(`propagate_layouts.py:1515`, consumed at `optimize_restickify.py:388` and
`:616`) though they are free at DMA time; relax `FixedInOutNode`'s
single-required-STL limit (`propagate_layouts.py:678-681`); and retire the
hardcoded `BEAM_WIDTH = 200` (`optimize_restickify.py:467`). 3b touches passes
10-14 and does not depend on the solver at all, so it is independently landable
and should not be gated behind 3a.

**Baseline candidate.** The layout today's beam search would select, with
placement fixed at the consumer, must remain in the candidate set (M3).

**Hint form.** Pin an edge to no-relayout, or pin a buffer's `SpyreTensorLayout`.

**Validator.** A pinned layout must be reachable from the endpoint candidates; an
edge pinned to no-relayout must admit a common stick dimension
(`stick_compatible`, `pass_utils.py:1060-1077`).

**Exit.** A measurable reduction in relayout bytes on the two tracked benchmarks
against the baselines recorded in `scratchpad_planning.md` — `mlp-linear-kn.t` at
roughly 79% process-engine utilization after pointwise seeding, and `mha_4h`
converging on `B/4·M/8` with the scores matrix pinned.

### Phase 4 — Op re-ordering

The hard one, and the document is explicit about why: order is a *given* that
three mechanisms depend on, and making it a variable invalidates all three at
once.

- Loop-nest-break adjacency is program order. Phase 1's per-pair booleans can be
  built up front only because that adjacency is static and fully known when the
  model is constructed. If order is a decision, adjacency is itself variable and
  the topology cannot be baked in.
- Liveness *is* the index into `graph.operations` (`scratchpad/utils.py:84-103`),
  so every packing rectangle's time axis moves when an operation moves. Phase 2
  sharpens this rather than softening it: a relocation is scheduled *at a tick*,
  so moving an operation moves the point at which a buffer changes address, and
  the hole the relocation was closing may not be there any more.
- In-place merging requires **exact** tick adjacency —
  `parent.end_time == child.start_time + 1`, asserted at
  `scratchpad/plan_solver.py:169-194` and relied on at
  `scratchpad/allocator.py:610-611` and `:706-725`. Separating a producer and its
  consumer by one operation silently kills the merge. This is the chief
  regression risk in the whole roadmap, and it fails quietly.

This axis is the worked example for M3's rule that an axis enters the model only
once the objective prices it, and it is the sharpest case in the roadmap because
the axis cannot be added tentatively. Landing it without pricing adjacency would
be exactly the failure M3 describes: every permutation that separates a producer
from its consumer is dataflow-legal, so an objective blind to adjacency has no
reason not to pick one, and the merge dies quietly. With no off switch, that
regression ships. Adjacency must be a constraint or an objective term before this
phase lands, not after.

**Variables.** Position variables constrained by the dataflow DAG, with liveness
derived from them. Scope the optimized mode as a **bounded** reordering — a
generalization of `reorder_unhinted_interlopers`
(`wsr/coarse_tile_hints.py:166-283`, whose legality check `_no_dep_conflict` at
`:110-131` is already the right predicate) over a window — rather than
unrestricted permutation of the graph.

Note that reordering already happens, reactively and too late (consequence 5).
Phase 4 makes the decision proactive rather than introducing one.

**Hint form.** Pin a relative order, or an operation's position.

**Validator.** Order pins must be acyclic and dataflow-legal.

**Exit.** No in-place-merge regression, asserted directly against the exact-tick
invariant. Today's topological order is a candidate the model can return, and the
M9 fallback returns it exactly.

### Phase 5 — Enumeration scaling

Deferred by choice, and the document records it as a choice rather than an
oversight. M4 buys model simplicity with enumeration, and until measured problem
sizes make that trade bad, the **eager cross product stays**: the joint
(tiling, division) table is formed per operation as the product of the two
enumerations, extending what `enumerate_work_division_candidates` already does
for division alone (`work_division.py:825-830`) within the caps coarse tiling
already applies (`_MAX_AUTO_TILE_SPLIT_COUNT = 64` per dimension,
`_MAX_TILE_COMBOS = 512` combinations, `span_overflow_hint_analysis.py:144-149`).

**Trigger.** Not a position in the order. C2 reports candidate rows per operation
alongside enumeration and solve time; this phase lands when either exceeds the
compile-time budget. Since the joint table is phase 1's, that may well be
immediately after it and before phases 2 through 4 rather than after them.
Nothing earlier depends on it, and no interface changes when it lands — the table
is built differently, not consumed differently.

**What it does, cheapest first.**

1. *Table over distinct derived-scalar signatures.* M4 consumes only
   `f(candidate) -> int`, so two candidates producing the same row — per-core
   span, buffer bytes, tile count, cores, cost — are indistinguishable to the
   solver. Deduplicating by signature bounds the row count by the number of
   distinct signatures rather than by the combinatorial count.
2. *Per-dimension tables with channeling constraints.* Keep per-dimension tile
   and core variables whose **domain is the divisor set**, so divisibility is
   declared rather than enumerated; express joint per-dimension legality as one
   allowed-assignment table per dimension; and keep the cross-dimension couplings
   as constraints — the core budget as a chained bounded product, the
   `MAX_SPAN_BYTES` limit likewise, at most one reduction split as a reified sum.
   This replaces a product over dimensions with a sum over dimensions.
3. *The quotient, posed rather than tabulated.* The reason sizes are table
   entries today is that per-core size is a **quotient** of the dimension by the
   split, and a quotient of two variables is not linear. Restricting the split
   variable's domain to the divisor set removes the rounding, and what is left is
   a product: `split × per_core == dim` states the relation exactly, with
   `per_core` a variable the objective and the span constraints can read
   directly. Padding makes it a three-way relation over `dim_padded`, and
   `AddMultiplicationEquality` / `AddDivisionEquality` are the constructs in
   question. Whether that formulation propagates well enough to be worth having —
   a table gives the solver perfect propagation, a multiplication gives it much
   less — is the open part, and it is an experiment this phase runs rather than a
   design it assumes.
4. *Hybrid and lazy.* The flat combination table below a threshold, the channeled
   encoding above it; enumerate per operation in parallel and cache, extending
   the `_views_for_divs` prep cache (`scratchpad/allocator.py:2079`).

**The limit to record.** (2) pays only where every derived scalar is separable
per dimension. Per-core span is a product over dimensions and separates; anything
needing the full per-core view (`_prepare_per_core_view`, `pass_utils.py:1467`)
does not — and that is exactly what phase 3 consumes. So (2) shrinks the
feasibility encoding while view-dependent objective terms may still require a
combination table, which is what makes (1) the primary lever rather than the
fallback.

**Exit.** Identical solutions to the eager encoding on the tracked benchmarks,
with a measured reduction in enumeration time and table size.

## Cross-cutting tracks

These run alongside the phases rather than between them.

- **C1 — Cost model.** M2 makes the objective injectable; it does not make it
  good. Every phase contributes terms — pad bytes (1), fragmentation and peak
  effects (2), relayout bytes (3), lifetime effects (4) — so this track owns
  the per-phase term requirements and the calibration procedure against measured
  process-engine utilization and fused kernel time on the tracked benchmarks.
  Without it, every phase adds precision to a proxy, and phase-level benchmark
  results should not be read as a verdict on the approach.

- **C2 — Determinism, tractability, and fallback.** Every phase enlarges one
  model, and nothing about that model divides by axis.

  - *Budget.* Solve time does not decompose: CP-SAT couples every variable through
    propagation, so an axis can make a solve harder or easier and no per-axis
    share exists. A phase measures the *configuration* it creates — whole-model
    solve time against one wall-clock budget — and those deltas do not compose,
    so a pair of axes is measured as a pair. Only shipping configurations are
    measured, not the 2^n cross product. With no off switch there is no
    within-binary A/B: the comparison is against the commit before the phase
    landed, so a phase that does not measure before merging cannot be measured
    afterward without a revert.
  - *Enumeration.* The exception, and why phase 5's trigger is defined over it:
    rows come from a per-operation table built before the solver runs, so they are
    countable per phase in the way solve time is not. Report rows per operation
    and enumeration time alongside solve time.
  - *Fallback.* A timeout is equally unattributable, so M9 case 1 degrades **all**
    axes to their baseline candidates at once; no phase may assume its own is the
    one that keeps its solved value.
  - *Determinism.* Conditional, not automatic.
    `torch.are_deterministic_algorithms_enabled()` forces
    `num_search_workers = 1` (`ilp_solver_ortools.py:459-463`) which, with
    `random_seed = 0`, makes a *completed* solve reproducible; the multi-worker
    default is not. Any run compared against another sets the flag — and its solve
    time is then not the shipping configuration's, so the budget above is measured
    on both or on neither. A solve terminated by the wall-clock limit (`:328-333`)
    stays machine-dependent even at one worker, absent a `max_deterministic_time`
    limit.

## Collateral documents

Each optimization gets its own RFC, with its own issue number. A phase document
may not restate the model; it cites M, H, and C requirements by number and
specifies only what its axis adds.

| # | Document | Owns |
|---|---|---|
| 0 | Optimization and Hinting Strategy | The spine: registry, pin lowering, axis modes, validator, conflict diagnosis, explainability |
| 1 | Coarse Tiling Optimization | Tiling for sizing and temporal op splitting, jointly with core division; pad amount as a candidate scalar, pad policy and sharing |
| 2 | LX Relocation and Compaction | Time-varying addresses, defragmentation, the fragmentation objective term |
| 3 | Restickification Optimization | Relayout pricing; restickify placement, sharing, input-layout choice; padding as layout legalization |
| 4 | Op Re-ordering Optimization | Program order as a bounded decision |
| 5 | Enumeration Scaling | Signature dedup, per-dimension channeling, lazy enumeration |

Padding has no row: it adds no axis, so documents 1 and 3 carry the halves each
one consumes rather than a document owning the subject.

Document 0 is drafted first: every other depends on it, and a phase that invents
its own hint channel before the registry exists will have to be unwound.
Documents 1 through 5 follow the phase order.

Per document, each must state: the defect it closes, the decision variables it
adds and their encoding under M4, the objective terms that price them, what the
baseline candidate M3's invariant requires in its table, the hint keys it
registers under H1, the validator rules it adds under H3, what it consumes from
earlier phases, and testable exit criteria.

## Sequencing rationale

Phase 1 first and phase 4 last are dependency facts; the two in between are
ordered by which defect is closable soonest, and this section says which is
which.

**1 → everything.** Tiling is the only axis that changes the *buffer set itself*.
Every later phase is defined over the buffers and lifetimes it produces — the
rectangles phase 2 packs, the per-core views phase 3 prices relayouts from, the
positions phase 4 permutes — so nothing can precede it.

**1 → 2.** Relocation needs only a buffer set and lifetimes, both of which phase 1
delivers, and its defect is independent of every remaining axis: fragmentation
costs spills today under an unchanged program order and an unchanged layout
assignment. That independence is why it can be closed and measured early, and it
carries one consequence rather than one caveat: phase 4 will move the lifetimes
the relocation plan was built against, so that plan is re-solved and its numbers
re-measured once order is a variable.

**2 → 3.** No dependency runs between them; restickification depends on phase 1,
not on phase 2. The order reflects that relayout pricing needs a C1 cost term
before its exit criterion means anything, while fragmentation is already
measurable in spill counts. If the relayout term lands first, the two swap
without consequence.

**3 → 4.** Every phase before 4 indexes on a static program order — per-buffer,
per-adjacent-pair, per-dataflow-edge, and now the time axis of phase 2's
relocation rectangles. Making order variable invalidates all of them at once, so
it comes after the phases that rely on it, not before.

**4 → 5, conditionally.** Phase 5 is placed last because nothing depends on it,
not because it must come last. Its trigger is a measurement — C2's reported
enumeration cost and table size — so it is pulled forward the moment the eager
cross product stops fitting the compile-time budget, whenever that happens.

