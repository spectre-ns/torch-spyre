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

The compiler makes a sequence of decisions about the same underlying object —
how a buffer is partitioned, shaped, laid out, placed, and when it is live — at
different points of one pass list, each with its own cost model and each blind
to the others. This document folds those decisions into a single model with a
single objective, and specifies the contract that makes doing so tractable:

> Every axis can be **pinned** by a hint, left **free**, or **optimized**. A
> validator proves the hint set is mutually consistent before the solve. The
> solver determines only the free variables, so the result is valid, honours the
> hints, and is optimal over what remains.

The three-mode framing is load-bearing, not a footnote. **An axis that is not
being optimized still has to be valid, and validity is cheap.** Op ordering left
free means the existing topological order is used and merely checked for
legality — it contributes no decision variables, no objective terms, and no
search. That is what lets the model scale: each phase adds an axis that can be
switched *on*, and switching it off returns to a validity-only treatment that
reproduces today's behaviour by construction.

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
| LX allocation order | `scratchpad/` solvers | fixed per-solver heuristic |
| Plan reuse | — | none |

Coarse tiling ranks split combinations by `_combo_cost` — total tile count, then
tiled-dim count, then largest split (`wsr/span_overflow_hint_analysis.py:1181-1192`)
— and takes the cheapest feasible one from a sorted product
(`_iter_split_combos`, `:1205`). Core division prices matmuls in microseconds
(`_matmul_split_cost`, `work_division.py:1206`) and everything else by a
dimension-priority heuristic (`prioritize_dimensions`, `:670`). LX residency runs
a two-phase lexicographic CP-SAT solve — minimize spill, lock it, then maximize
cores (`scratchpad/ilp_solver_ortools.py:446-518`). The remaining four have no
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

### 2. Tiling can never buy LX residency

Shrinking a chain's working set until it fits the LX scratchpad is the stated
purpose of `wsr/`, yet the tiling planner cannot see LX occupancy and the LX
solver cannot choose a tiling. `docs/source/compiler/scratchpad_planning.md`
records this among the remaining gaps under "Co-optimization is still limited",
listing "**No coarse-tiling integration** when that pass also drives split
decisions" alongside "**No performance model**".

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

Within its own scope, padding is a fixed policy rather than a decision:
`compute_padding` (`padding.py:73-76`) rounds up to one stick, and the pass pads
the y operand only, on the reduction dimension only, at the right end only, with
zero fill only. It has no cost model, and no sharing — the loop at `:183` emits a
separate four-op pad sequence per matmul, so two matmuls reading the same operand
each pay for their own.

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

### 6. LX cannot compact, and nothing is reused between compiles

The greedy solver processes operations in topological order making irrevocable
placement decisions with no lookahead, and `_find_free_block` can locate holes
between allocations but cannot compact them, so allocate/deallocate cycles
fragment the address space — both recorded in `scratchpad_planning.md` under
"Greedy single-pass, no lookahead" and "No defragmentation". First-fit and
best-fit mitigate by sorting buffers up front
(`scratchpad/firstfit_bestfit_solver.py:186-245`), which is itself a fixed
heuristic over an axis that could be decided.

Separately, no compiler decision is serialized anywhere. Compiled SDSC bundles
are written to `uuid4()`-named temporary directories and recompiled on every run
(`execution/async_compile.py:38-45`); the plan that produced them is discarded.

## The model

These requirements are stated once here so that no phase re-derives them. Each
collateral RFC satisfies the ones its axis touches and cites them by number
rather than restating them.

- **M1 — One solve, one objective.** All *optimized* axes resolve in a single
  CP-SAT instance minimizing a single expression. The existing `LAYOUT_SOLVER`
  plug point (`config.py:111-113`, `_PLACEMENT_SOLVERS` at
  `scratchpad/allocator.py:2108-2113`) is the template both for selecting a
  solver and for degrading when `ortools` is absent.

- **M2 — The objective is injected, not hardcoded.** The cost function is
  caller-supplied, so cost experiments do not require editing the solver. Today's
  objective is inlined in `_run` (`ilp_solver_ortools.py:446-518`). The
  replacement is a symbol namespace bound to model variables plus a lowering to
  CP-SAT expressions over an explicitly bounded grammar; a construct outside that
  grammar is a compile error naming the offending node. Silently approximating an
  objective is worse than failing to build one. An axis in free mode contributes
  no symbols.

- **M3 — Three modes per axis: pinned, free, optimized.** The central
  tractability mechanism.

  - *Pinned* — a hint fixes the value. The domain is collapsed at enumeration
    time, so the axis contributes no variables and no search (H4).
  - *Free* — any valid value is acceptable. The axis contributes no variables and
    no objective terms; the existing heuristic or identity value is used and
    checked only for legality.
  - *Optimized* — the axis contributes variables and objective terms and is
    solved.

  Free is the default for every axis a phase has not enabled, and it is what
  makes "never worse than today" structural rather than aspirational: with every
  axis free, the pipeline is today's pipeline. Modes are per axis and, where it
  makes sense, per scope — an operation or a subgraph — so a graph can optimize
  tiling while leaving order free. Gating follows the existing `LX_PLANNING` /
  `CO_OPTIMIZING_LX_PLANNING` / `LAYOUT_SOLVER` family (`config.py:22-25`,
  `:111`): a config entry plus an environment variable, defaulting off.

- **M4 — Enumerate-and-table encoding.** Nonlinearity — divisibility, stick
  alignment, span limits, the core budget — is absorbed by precomputing a
  feasible candidate set per operation and binding derived scalars with
  `AddElement`, exactly as `CoreDivision` is consumed today
  (`ilp_solver_ortools.py:255-273`). The rule that follows is the model's only
  extension mechanism, and later phases depend on it: **any
  `f(candidate) -> int` becomes a table entry plus one `AddElement`.** A
  nonlinear property of a candidate becomes a table lookup rather than a
  constraint. The cost is enumeration, which M5 and C3 bound.

- **M5 — Feasibility is evaluated on the combination, not per subsystem.** A
  candidate is feasible if and only if it satisfies every subsystem's guard
  jointly. Reuse `enumerate_work_division_candidates`'s `valid_split` guards
  verbatim (`work_division.py:809-823`): the core budget, at most one reduction
  dimension split, a per-core span within `MAX_SPAN_BYTES` on every tensor
  dependency, and no coordinate-masked dimension split. Evaluating them on the
  combination rather than one axis at a time is what discharges consequence 1.

- **M6 — Predicates are reused, never reimplemented.** Every legality predicate
  the model needs already exists, in `wsr/span_overflow_hint_analysis.py`,
  `pass_utils.py`, and `work_division.py`. Reimplementing one creates a second
  source of truth that will drift. This binds free mode too: the validity check
  in free mode must call the *same* predicate optimized mode calls, or the two
  modes disagree about what is legal and the fallback stops being a fallback.

- **M7 — Pure prediction, then apply.** A decision is scored against a
  *predicted* buffer set with no IR mutation, then applied. State the hazard once
  rather than per phase: liveness is index-based (`calculate_liveness`,
  `scratchpad/utils.py:84-103`, with `start_time` / `end_time` derived at
  `scratchpad/plan_solver.py:75-81`), so any pass that *inserts* operations shifts
  the lifetime ticks of everything downstream. That is a systematic offset, not
  noise, so predictions are compared to realized values under rank-order
  normalization. Each phase owes a verify mode (C2).

- **M8 — Addresses come from the final solve.** Whatever the joint model
  predicts, physical placement is recomputed over the real post-transform
  buffers. A misprediction therefore degrades to a spill, never to a wrong
  address.

- **M9 — Failure reverts to free mode.** On solver failure, timeout, or a missing
  optional dependency, the affected axes drop to free mode: the existing
  heuristic runs and only validity is enforced. This generalizes the current
  `SolveError` fallback to the greedy allocator
  (`scratchpad/allocator.py:2193-2219`) and the ortools-missing fallback
  (`_make_cpsat_solver`, `:2116-2136`). The graph must be unmutated when the
  solve fails, so the fallback starts from clean IR — which means the solve
  precedes the transform it decides.

## Hints, pins, and the validator

The spine. Collateral document 0 owns it in full; this section states the
contract every phase registers against.

- **H1 — A hint registry.** A declared schema per axis: key name, value shape,
  scope kind (operation, dimension, edge, buffer, graph), and owning phase.

  The *transport* needs no redesign. `spyre_hint(**kwargs)`
  (`propagate_hints.py:85-93`) already attaches an arbitrary kwargs dict to every
  FX node in scope, keyed by hint id, and survives AOT re-tracing through
  `collect_spyre_hints` / `recover_spyre_hints` (`:138-208`). It is already
  axis-agnostic.

  What is missing is the registry. Consumers each pull their own key out of the
  untyped dict — `work_div` at `work_division.py:835-843`, and
  `tiles` / `slices` / `num_tiles_per_dim` at `propagate_named_dims.py:636-644` —
  so an unregistered or misspelled key is silently dropped, and two hint families
  on one operation never see each other. Registration turns an unknown key into
  an error naming the nearest registered one.

- **H2 — A hint lowers to a pin.** Each hint becomes a domain restriction applied
  at *enumeration* time: the candidate list is filtered before the model is
  built, not overridden after a choice is made.

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

- **H3 — The validator.** Two levels run always; a third runs only on failure.

  1. *Well-formedness.* The key is registered, the value's shape and type are
     correct, and the scope resolves to a live operation, dimension, edge, or
     buffer. This subsumes the checks scattered through `_apply_user_hint`
     (`work_division.py:866-932`) and the one-of-`tiles`/`slices` rule at
     `propagate_named_dims.py:636-644`.
  2. *Realizability.* The pinned value survives the phase's own feasibility
     predicates (M5, M6). This closes a real hole: a `work_div` hint that reduces
     splits `span_reduction` already committed proceeds today with only a warning
     — *"Applying strict user hint; this may violate the hardware span limit"*
     (`work_division.py:1110-1142`) — and `_apply_user_hint` silently skips, at
     `logger.info` level, a dimension whose split would exceed `SENCORES`. Both
     become named errors.
  3. *Conflict diagnosis, on infeasibility only.* The normal solve carries no
     diagnostic apparatus: pins are collapsed domains, so a satisfiable model pays
     nothing for them. When the solve reports infeasible, a **separate diagnostic
     pass** re-expresses the pins as retractable assumptions and reports the
     minimal conflicting subset, so the error names the specific hints that
     conflict instead of reporting a bare UNSAT. The cost is paid only on the
     failure path, which is the point.

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

  Hints are therefore a first-class tractability lever, on equal footing with
  free mode: both are ways of not spending search on an axis. The honest claim is
  that the solve is optimal *over what remains*, and this document makes that
  claim rather than a claim of global optimality.

- **H5 — Every decision explains itself.** A per-axis `decision_reason`,
  extending the reason-string pattern already used for residency —
  `residency_reason` (`plan_solver.py:68`), `spill_reasons` (`:219`),
  `reject_reasons` (`allocator.py:137`), `_SOLVER_CHOSE_SPILL`
  (`ilp_solver_ortools.py:113`), and `excluded()` / `record_exclusions()`
  (`plan_solver.py:221-258`). Every committed decision records whether it came
  from a user hint, a compiler-generated hint, a cache replay, free mode, a
  fallback, or the solver. Without this, a plan cannot be debugged and a cache
  replay cannot be distinguished from a fresh solve.

- **H6 — A warm start is not a pin.** OR-Tools' `AddHint` is a soft search seed
  with no semantic guarantee; a torch-spyre hint is a hard pin validated by H3.
  They must not share vocabulary in code or in documentation, or the two will be
  conflated in review.

## Phases

Each phase follows the same template: the defect it closes, the decision
variables it adds, what free mode means for that axis, the hint form it registers
under H1, what the validator must newly check, what it depends on, and its exit
criteria.

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

**Free mode.** Today's behaviour: the `_combo_cost`-ranked first feasible tiling,
checked for span legality only.

**Hint form.** The existing `tiles` / `slices` / `num_tiles_per_dim` and
`work_div` keys, registered under H1 and lowered to pins under H2.

**Validator.** The hinted value survives enumeration; a hint scope must not
straddle a loop-nest break, the invariant `validate_coarse_tile_groups`
(`wsr/coarse_tile.py:112-137`) enforces today; and group contiguity
(`_validate_contiguous`, `:785-814`) should hold structurally, by pinning a break
wherever an operation cannot be tiled, rather than being checked after the fact.

**Owed to later phases.** The M4 candidate encoding, the M2 objective namespace,
the M7 predictor, and per-core views evaluated against a *predicted*
post-transform frame — `_prepare_per_core_view` (`pass_utils.py:1467`) and
`_per_core_view_on_buf` (`:1696`) taking a predicted frame rather than reading the
operation's current layout. Phase 3 cannot start without that last one.

**Exit.** Parity with today when the axis is free. A case where span overflow
forces tiling *and* work division splits the same dimension provably picks a
smaller tile count than the `core_split_estimate = 1` path. Prediction verified
under C2.

### Phase 2 — Padding

The cheapest addition, and it sharpens phase 1 retroactively: padded
`device_size` is what buffer sizes are computed from, so phase 1 otherwise
optimizes against numbers a fixed policy chose.

**Variables.** A pad choice per padded dimension over a small enumerated set.
**No new index space** — a pad choice is a per-candidate scalar folded into the
existing tables under M4, which is precisely the case M4's rule was written for.

**Policy work, not solver work.** The second half of the phase lifts the fixed
policies in `padding.py`: pad operands other than y, dimensions other than K,
ends other than the right, and multiples other than one stick; share a padded
buffer between matmuls that read the same operand rather than emitting a pad
sequence per matmul (`:183`); and — the part phase 3 consumes — make padding
available as a *layout legalization* tool, removing the issue #1756 restriction
at `propagate_layouts.py:271`, `:455`, and `:1078`.

**Free mode.** Today's unconditional round-up to the next stick.

**Hint form.** Pin a buffer's pad amount.

**Validator.** Joint stick alignment across pad, tile, and division, reusing
`_post_tile_stick_alignment_error` (`span_overflow_hint_analysis.py:263`) and
`_combined_tile_stick_alignment_error` (`:1215`).

**Exit.** Existing bmm tests unchanged. `tests/inductor/test_padding.py` extended
with a shared-operand sharing case and a chosen-pad case. Padded candidates
visible to `_candidate_output_stls` (`propagate_layouts.py:249-278`), which is
the observable that phase 3's search space grew.

### Phase 3 — Restickification

Two separable efforts, and the document says so because the existing pass name
conceals the split (consequence 4).

**3a — pricing.** A per-edge relayout variable over *dataflow* edges — the first
**new index space** in the model, everything before phase 3 being per-buffer or
per-adjacent-pair — and a relayout-bytes term in the M2 objective, precomputed
per edge from the tiling-aware per-core views phase 1 delivers. The variable is
*determined* rather than free: an edge needs a relayout, or it does not, given
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

**Free mode.** Today's beam search over layouts, with placement fixed at the
consumer.

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
  so every packing rectangle's time axis moves when an operation moves.
- In-place merging requires **exact** tick adjacency —
  `parent.end_time == child.start_time + 1`, asserted at
  `scratchpad/plan_solver.py:169-194` and relied on at
  `scratchpad/allocator.py:610-611` and `:706-725`. Separating a producer and its
  consumer by one operation silently kills the merge. This is the chief
  regression risk in the whole roadmap, and it fails quietly.

This axis is the worked example for M3's free mode. Left free, the existing
topological order is used and only checked for dataflow legality — which is what
happens today, and costs nothing. Optimizing it is opt-in.

**Variables.** Position variables constrained by the dataflow DAG, with liveness
derived from them. Scope the optimized mode as a **bounded** reordering — a
generalization of `reorder_unhinted_interlopers`
(`wsr/coarse_tile_hints.py:166-283`, whose legality check `_no_dep_conflict` at
`:110-131` is already the right predicate) over a window — rather than free
permutation of the graph.

Note that reordering already happens, reactively and too late (consequence 5).
Phase 4 makes the decision proactive rather than introducing one.

**Hint form.** Pin a relative order, or an operation's position.

**Validator.** Order pins must be acyclic and dataflow-legal.

**Exit.** No in-place-merge regression, asserted directly against the exact-tick
invariant. Free mode reproduces today's order exactly.

### Phase 5 — LX allocation re-ordering

Sibling of phase 4 on the same time axis: phase 4 decides *when operations run*,
phase 5 decides *in what order addresses are assigned* to the buffers they
produce, and whether the address space can be compacted.

**Variables.** Allocation order or priority over the placeable set, and — the
harder half — relocation, permitting a buffer's address to change during its
lifetime so holes can be closed. The permutation machinery exists as a foundation
(`scratchpad/permutation_layout.py`, and `SolverToPermutation` in
`scratchpad/simulated_annealing.py`) and should be reused rather than reinvented
(M6).

**Free mode.** Today's solver-specific order: greedy topological, or the
first-fit / best-fit up-front sort.

**Hint form.** Pin a buffer's residency, or its relative allocation priority.
This also hands users a direct lever over the outcome they most often want to
control, which no current hint family reaches.

**Validator.** A pinned residency must fit the packing across its lifetime;
pinned priorities must form a consistent partial order.

**Exit.** A fragmentation case the greedy solver cannot pack today is packed. No
regression in spill count on the tracked benchmarks.

### Phase 6 — Optimization caching

The closing argument for the whole spine: **a cached plan is a complete hint
set.**

Replaying it through the H3 validator means a stale entry — the graph changed,
the config changed, enumeration changed — fails validation and falls back to a
fresh solve instead of being silently mis-applied. Invalidation therefore needs
no logic beyond the key. A *partially* valid entry degrades naturally: pins on
the decisions that still validate, with the rest left free (M3) or warm-started
(H6). This is also why caching is last — it needs the decision space closed and
the validator total.

**Requirements.** A whole-graph signature, which does not exist today.
`provenance.py`'s `_stable_id` (`:52-80`) explicitly disclaims cross-run
stability at `:58-63`: *"a within-compile linking key, not a cross-run
fingerprint"*. The key must cover config values, solver identity, and the
existing pass-source uuids (`passes.py:144-151`). Plan serialization is new.
Enumeration caching belongs here too — candidate tables and the `_views_for_divs`
prep cache (`scratchpad/allocator.py:2079`) — since enumeration cost grows with
every phase (C3).

**Two hard prerequisites**, both named rather than assumed:

1. *Config values are absent from the cache key.* Only pass **source files** are
   hashed (`passes.py:144-151`), so a `LAYOUT_SOLVER=cpsat` run and a
   `LAYOUT_SOLVER=greedy` run produce the same key. This is a wrong-artifact risk
   today, independent of this roadmap, and should be fixed early rather than
   waiting for phase 6.
2. *CP-SAT is not reproducible on the default path.* It runs
   `num_search_workers = os.cpu_count()` under a 600-second wall-clock limit
   (`ilp_solver_ortools.py:328-333`, `:456-463`); `random_seed = 0` guarantees
   reproducibility only for a fixed worker configuration, so a
   timeout-terminated multi-worker solve is not reproducible. The intent is
   already recorded elsewhere in the tree —
   `scratchpad/simulated_annealing.py:205-212` states that "compilation caching
   depends on" deterministic layout planning.

**Exit.** A warm second compile of an unchanged graph replays the plan and
produces an identical result. A mutated graph misses cleanly. A config change
misses cleanly.

## Cross-cutting tracks

These run alongside the phases rather than between them.

- **C1 — Cost model.** M2 makes the objective injectable; it does not make it
  good. Every phase contributes terms — pad bytes (2), relayout bytes (3),
  lifetime effects (4), peak and fragmentation effects (5) — so this track owns
  the per-phase term requirements and the calibration procedure against measured
  process-engine utilization and fused kernel time on the tracked benchmarks.
  Without it, every phase adds precision to a proxy, and phase-level benchmark
  results should not be read as a verdict on the approach.

- **C2 — Prediction fidelity.** A per-axis predict-versus-realize verify mode
  under an environment flag, asserting that predicted sizes, lifetimes, and
  per-core views match the realized ones after the transform applies. A
  mispredicted *view* is a wrong-residency bug and does not enjoy M8's
  degrade-to-spill safety, so it gets the strictest assertion of the three.

- **C3 — Determinism, tractability, and fallback.** Each phase enlarges the
  model, so each owes a solve-time budget, a warm start, and the M9 fallback.
  Enumeration time must be reported alongside solve time — M4 trades enumeration
  for model simplicity, and the two have different mitigations. Determinism must
  hold before phase 6 can cache anything.

## Collateral documents

Each optimization gets its own RFC, with its own issue number. A phase document
may not restate the model; it cites M, H, and C requirements by number and
specifies only what its axis adds.

| # | Document | Owns |
|---|---|---|
| 0 | Optimization and Hinting Strategy | The spine: registry, pin lowering, axis modes, validator, conflict diagnosis, explainability |
| 1 | Coarse Tiling Optimization | Tiling for sizing and temporal op splitting, jointly with core division |
| 2 | Padding Optimization | Pad amount, placement, sharing; padding as layout legalization |
| 3 | Restickification Optimization | Relayout pricing; restickify placement, sharing, input-layout choice |
| 4 | Op Re-ordering Optimization | Program order as a bounded decision |
| 5 | LX Allocation Re-ordering | Address-assignment order, compaction, defragmentation |
| 6 | Optimization Caching | Graph signature, plan persistence, replay-as-hints |

Document 0 is drafted first: every other depends on it, and a phase that invents
its own hint channel before the registry exists will have to be unwound.
Documents 1 through 6 follow the phase order.

Per document, each must state: the defect it closes, the decision variables it
adds and their encoding under M4, what free mode falls back to, the hint keys it
registers under H1, the validator rules it adds under H3, what it consumes from
earlier phases, and testable exit criteria.

## Sequencing rationale

The order is a dependency chain, not a preference.

**1 → 2.** Tiling is the only axis that changes the *buffer set itself*;
everything downstream is defined over the buffers it produces. Padding then adds
no new index space (M4) and corrects the sizes phase 1 optimizes against, so it
is both the cheapest next step and a retroactive improvement to the previous one.

**2 → 3.** Padding as a legalization tool removes the stick-divisibility
restriction on layout candidates, enlarging the search space restickification
works over. Pricing relayout before that restriction lifts would optimize over an
artificially small set.

**3 → 4.** Every phase before 4 indexes on a static program order — per-buffer,
per-adjacent-pair, and per-dataflow-edge variables all assume it. Making order
variable invalidates that assumption, so it comes after the phases that rely on
it, not before.

**4 → 5.** Allocation order is optimized against lifetimes, so lifetimes must
settle first. This ordering is a judgment call and is recorded as such: the
fragmentation and lookahead defects phase 5 closes are independent of op order
and could be pulled earlier if they hurt sooner.

**5 → 6.** The cache key cannot be defined until the decision space is closed. A
key that omits an axis added later silently returns plans made under different
assumptions.

## Non-goals

- **No ring transfers.** The `core_div_mismatch` hard wall stays. Dissolving it
  requires a data ring or reduce-sum ring emitted in the SuperDSC schedule, which
  is separate work.
- **No new backend**, and no change to the DeepTools interface.
- **No recalibration of `_matmul_split_cost`** as part of the mechanism work.
  Improving the numbers is C1's concern; this roadmap delivers the structure that
  lets them be improved.
- **No claim of global optimality.** Under H4 the solve is optimal over the free
  variables only, and under M3 an axis left free is not optimized at all. Both
  are deliberate.

## Open questions

- **Objective weighting.** The default objective reproduces today's terms.
  Once the mechanism is trusted, what weighting and what additional terms should
  it carry — and does the move from a two-phase lexicographic solve to a single
  weighted one need a guard so that parallelism can never buy a spill?
- **Where hints apply.** Hints currently apply pre-stickification, which is what
  keeps them authoritative but also prevents the solver from *growing* a hinted
  group with neighbours it would like to add. Moving hint application later would
  lift that, at the cost of a contract the current ordering exists to protect.
- **Phase 5's position.** Recorded above as a judgment call; worth revisiting
  once phase 3 lands and the fragmentation picture is measured rather than
  inferred.
- **Enumeration growth.** M4 trades enumeration cost for model simplicity, and
  each phase multiplies the candidate set. At what point does pruning become
  mandatory rather than optional, and what prunes safely?
