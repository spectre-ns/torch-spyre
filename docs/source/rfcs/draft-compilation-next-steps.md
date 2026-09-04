# Compilation: next steps

A working plan for the compiler, written against `enable-default-cooptimization`
at `dcd8a184`. Everything below is grounded in the tree as it stands today; line
numbers are from that commit and will drift.

The seven areas this covers, in the order they matter to us:
<!-- This is the execution order management expects the first 2 in that order -->
1. Coarse tiling
2. Encapsulating op and graph attributes so the scratchpad stops using reflection
3. Hints for core division
4. Improved core-division pruning
5. Improved coarse-tiling pruning
6. Restickification optimization
7. Op reordering

Priority is not execution order. Section 3 explains where the two differ and why.

## 1. Where the tree stands

### 1.1 The pass spine

`CustomPreSchedulingPasses.passes` (`passes.py:451-506`) runs, in order:

```text
deadcode_elimination -> decompositions -> propagate_named_dims -> assign_dim_hints
  -> _maybe_reorder_unhinted_interlopers
  -> _maybe_coarse_tile_hints            (pre-stickify tiling)
  -> insert_bmm_padding -> split_multi_ops
  -> propagate_spyre_tensor_layouts -> optimize_restickify_locations
  -> finalize_layouts -> insert_restickify -> enforce_indirect_access_layout
  -> insert_post_mutation_restickify -> insert_restickify_padding
  -> dedup_and_promote_constants
  -> _maybe_coarse_tile_span_overflow    (post-stickify tiling)
  -> span_reduction
  -> _distribute_work                    (work division committed here)
  -> _maybe_scratchpad_planning          (LX planning; co-opt re-decides division)
  -> elide_proven_read_copies
```

Two facts about that spine drive most of this plan.

**Work division is decided twice.** `_distribute_work` commits a division per op;
`_maybe_scratchpad_planning` then re-opens it, because `co_optimizing_lx_planning`
defaults on. `CoOptimizingAllocator._commit_divisions` (`allocator.py:1964-1990`)
overwrites `iteration_space_ownership` for *every* buffer, resident or spilled.
Anything the first pass established and the joint solve does not know about is
silently discarded. That is why user work-division hints do not survive, and why
the pin ladder exists.

**Coarse tiling is applied before either decision.** Both live tiling paths mutate
the graph in the pre-scheduling pipeline, so by the time the solver runs, a tiled
op simply presents a smaller footprint. Tiling is not something the solve chooses.

### 1.2 Defaults now in force

From `config.py`: `lx_planning=1`, `co_optimizing_lx_planning=1`,
`layout_solver="cpsat"`, `hbm_pool_planning=True`, `enable_reduction_tiling=1`,
`sencores=32`. `ignore_span_overflow_hints=1`, so automatic span-overflow tiling
is opt-in. The docs have not caught up:
`docs/source/compiler/work_division_planning.md:579-580` and
`docs/source/compiler/scratchpad_planning.md:16-17` both still describe
co-optimization as off by default.

### 1.3 The coarse-tiling stack is split across branches

Stages 1-3 landed on `main` (PRs #3971, #3969). Stages 4, 5 and 6 exist only as
branches (`coarse-tiling-stage4/5/6`, tip `66ef6e60`), and they are what "partial
coarse tiling discovery with hints respected by the solver" refers to. None of it
is on this branch: nothing populates `CoreDivision.tiling`,
`ilp_solver_ortools.py` contains no reference to tiling, `CoarseTilingPass`
(`scratchpad/coarse_tiling.py:181`) is never instantiated because
`select_allocator` passes no `pre_optimization_passes`, and `enumerate_tile_options`
(`wsr/enumerate_tilings.py:216`) has no production caller.

**The stage6 branch is stale and must not be merged.** Its merge base with HEAD is
`fea0c4be`; `git diff HEAD coarse-tiling-stage6` spans 288 files and ~117k
deletions, because stages 1-3 landed on main afterwards and the branch's own copies
of `enumerate_tilings.py`, `scratchpad/coarse_tiling.py` and the `TileSpec` model
are now *older* than HEAD's. At least three of its hunks no longer apply as
written: HEAD's `_enumerate_core_divisions` keys splits by iteration symbol rather
than index coefficient, HEAD's `_solve` takes `(solver, graph)`, and HEAD's
`enumerate_tile_options` dropped the `max_cores` positional the branch passes.
The genuinely new content is small and isolable — a tiled-frame predictor, config
gates, `cd.tiling` in the model's per-core sizing, a `tiling=` argument on the
work-division enumerator, `_tiling_candidates`, `dim_hints_to_tile_spec`, and a
read-distance filter. Cherry-pick by hunk, re-derived against HEAD, and retire the
branches afterwards so nobody rebases the withdrawn `cost_expr` prototype or the
three-valued `BoundaryRole` back in.

### 1.4 A live regression

`_is_coarse_tiled` (`allocator.py:1263-1281`) has **zero call sites repo-wide** —
grep returns only its own definition. Commit `05276cf0` commented its
`_division_map` entry out; `dcd8a184` deleted the comment. Its docstring documents
what it prevents: the joint solver hands a coarse-tile loop group's members
incompatible divisions, driving compute ops to a single core while sibling read
copies stay split, which corrupts the shared loop nest's per-core addressing —
"silently wrong output (~9% of a hinted two-GEMM MLP tiled S x Dout)".

The classic work-division path is not a substitute.
`coarse_tile_local_dim_split_domains` (`work_division_constraints.py:388`) pins
only tile-local dims and explicitly exempts generated `coarse_tile_read_copy_*`
ops (`:356`) — exactly the compute-op-versus-read-copy divergence the docstring
names. There is a second, independent gap in the same file:
`_determine_in_place_division_invariant` (`allocator.py:1993`) has no coarse-tile
guard at all, so a coarse-tiled buffer can be handed an in-place LX slot.
`_safe_in_place_parents` (`:142`) and the `lifetime_end_overrides` skip (`:2032`)
are both parent-only, leaving the child side uncovered.

### 1.5 The scratchpad asks questions it has no way to name

Across `scratchpad/*.py`: 40 occurrences of `getattr(`, 20 of `hasattr(`, 4 of
`setattr(`, 102 of `isinstance(`. `allocator.py` alone accounts for roughly a
third of each. The cause is structural —
torch-spyre owns no op-level type, so its entire per-op extension surface is 20
attributes monkey-attached to Inductor buffers, declared only as a tuple of
strings in `_SPYRE_METADATA_ATTRS` (`loop_info.py:375-391`) and copied by a
`hasattr`/`getattr`/`setattr` loop (`:393-403`). 52 assignment sites carry
`# type: ignore[attr-defined]`, 37 of them in `wsr/coarse_tile.py`.

Because the questions have no names, they get re-answered inconsistently:

- **"Do this buffer's users agree on its per-core slicing?"** exists twice with
  different guard sets — `get_ncores_for_buffers` (`scratchpad/utils.py:525-616`)
  applies a broadcast-read guard the other lacks; `_cd_parent_matches`
  (`allocator.py:2342-2444`) applies matmul and frame-changing-clone guards the
  other lacks. The comment at `allocator.py:2407-2408` claims parity with a matmul
  guard in `get_ncores_for_buffers`; `grep -i matmul scratchpad/utils.py` returns
  nothing.
- **"Is this an avgpool?"** is asked in two disjoint string namespaces inside one
  file: the aten name `avg_pool2d` in `OP_OUTPUT_NOT_GOOD_FOR_LX_REUSE`
  (`utils.py:47-53`) and the SDSC `reduction_type in POOL_OPS`
  (`allocator.py:1256-1261`). `conv2d` means different things in each.
- **"When is this buffer live?"** has five implementations on two time bases with
  two end conventions, and only one — `hbm_pool_planning._compute_live_ranges`
  (`:124-177`) — models aliasing at all.
- **Buffer byte size** is written out longhand in five places with four different
  failure semantics; **"is this LX-resident?"** is a bare `"lx" in
  layout.allocation` at roughly twenty sites with no accessor.

Two decisions are `elif` ladders rather than registries. `_buffer_residency_reason`
(`:449-535`) is twelve checks whose docstring says "Order matters" because later
checks dereference `device_layout` unguarded. `_division_map` (`:1854-1922`) is
seven pin guards, each one a distinct silent-wrong-code or DeepTools-abort
incident. An `elif` chain cannot be tested for exhaustiveness, which is exactly how
`_is_coarse_tiled` fell out of it unnoticed.

The ladder also does not do what its own comment says. `allocator.py:1882-1883`
states the graph-level group sets are built once rather than rescanned per op;
lines 1890, 1894 and 1896 call `ops_in_offset_mutation_component(graph)` and
`_fused_layout_group_ops(graph, ...)` *inside* the per-op loop. That is O(N²) on
every compile, on the now-default path.

### 1.6 The solver is slower and less deterministic than it needs to be

Measured this session on the in-tree capture corpus, no hardware required:

| Finding | Measurement |
| --- | --- |
| `num_search_workers = os.cpu_count()` (`ilp_solver_ortools.py:812-814`) | flash_big 3627 ms at 192 workers vs 1105 ms at 8 — **3.3x slower**; corpus total 6497 ms -> 2824 ms with the fix alone |
| Run-to-run spread | 2174-4062 ms at 192 workers, 1094-1098 ms at 8 |
| Plan stability across worker counts | 5 of 11 captures produce different LX **addresses** (residency, division and cores identical) — and addresses are baked into emitted SDSC |
| Candidates, not placement, dominate | collapsing every menu to one candidate: 1244 ms -> 58 ms (21x). Shrinking LX capacity 16x: only 1.6x |
| Residency saturation | flash_big holds 60/79 resident by k=8; candidates 9-32 cost ~25% of runtime for zero residency gain |

The 30 s limit (`ilp_solver_ortools.py:643`) is set once on a `CpSolver` reused
across the lexicographic ladder, so the real worst case is 3x30 s (the `cost_expr`
path and the three-phase ladder are mutually exclusive, so it is 1x30 or 3x30, not
4x30). When it is exceeded the solve raises `SolveError`, `scratchpad_planning`
falls back to placement-only greedy for the entire graph, and that is logged at
DEBUG (`allocator.py:2590-2600`).

### 1.7 Nothing measures whether any of this helps

- `tests/inductor/conftest.py:55-75` installs an autouse **session** fixture that
  allocates ~2.8 GB of `device="spyre"` tensors with no availability guard. With no
  free card every inductor test — including pure-sympy unit tests — reports ERROR
  at setup, not SKIP. Neutralizing that one fixture, 41 device-free files
  (1,669 tests) ran 1,635 passed / 5 failed in 3m51s with no card.
- Eight config files carry labels (`[core, full]`) that name no recognized CI tier,
  so **240 tests run under no trigger at all**. Four of the eight are the
  co-optimizer solver suites. `tests/scripts/oot_checker/checks.py:23-66` checks
  only that a `labels` key exists, never that its values name a tier.
- There is no e2e perf gate on any automatic trigger. The perf job
  (`_test_matrix.yaml:539-698`) is fully implemented and is `workflow_dispatch`
  only; no cron in this repo dispatches `test_type=perf`.
- `cost_model_pass` returns a `CostReport.total_us` on the final IR — documented as
  the "compare plan A vs plan B without compiling either" API — and nothing in
  `tests/` asserts on it for a real model.
- There are **three** capture corpora in the tree, not two:
  `cooptimization_captures.json`, `_large.json`, and `_regen.json` (committed in
  `23819620`, ~2.8x larger menus: flash_big 2299 candidates / 1614 pairs versus
  815 / 581). Any solver-tuning claim measured on only the first two is untrustworthy —
  see §7.

### 1.8 The obvious acceptance gate is blind to the failure we actually have

Four workstreams here would naturally be graded on "residency and spilled bytes
unchanged". That gate cannot see the thing that breaks. `_commit_divisions` writes
the solver's chosen division into `iteration_space_ownership` for *every* buffer,
resident or spilled — that is real per-core kernel slicing. A prototype of the
CP-SAT table-constraint rewrite held residency and spill bytes **identical on all
11 captured graphs while changing the committed division on all 11**.

Every silent-wrong-output incident in this branch's history is a model-legal
division change with unremarkable residency: the avgpool window split (~19%), the
coarse-tile group (~9%), the offset-slice read (~44%), flash `tile_B_H` (~45.5%).
So: **every baseline this plan adds records per-op `chosen_division`, and every
acceptance criterion diffs it.** Without that, four workstreams ship behind an
instrument that cannot detect their characteristic failure.

### 1.9 The suite that would grade coarse tiling is partly vacuous

Neither `config.unified_tiling` nor `config.auto_coarse_tiling` exists.
`test_solver_auto_coarse_tiling.py:351` raises
`NotImplementedError("unified-tiling: config.auto_coarse_tiling")` *before
compiling anything* for every auto row, and `:371` is a bare
`# TODO: Patch coarse tiling config here` where the config patch belongs. So the
hinted rows currently exercise the old pre-stickification hint path regardless of
what a solver-side change does, and the auto rows never compile. Fixing that
harness is a prerequisite for the solver-integration work, not a validation detail.

Related: `dim_hints_to_tile_spec` — the pin-becomes-constraint inverse the whole
"hints respected by the solver" claim turns on — **does not exist**.
`scratchpad/coarse_tiling.py` defines only the forward direction
(`tile_spec_to_dim_hints`). That step is new code, not a port, and should be sized
as such.

## 2. Do these now

Small, independently landable, and each closes something that is wrong today.

**N1 — Restore the coarse-tile division pin, and close the in-place gate.**
Add `elif _is_coarse_tiled(op): reason = "coarse tile group"` to the `_division_map`
ladder, and skip in-place candidacy in `_determine_in_place_division_invariant`
when either the child or the candidate parent carries `loop_info`. Put the pin
behind a named config flag rather than a bare `elif`, so the next investigation
does not have to edit source and so its removal is greppable.

Two things to know before landing it. `_is_coarse_tiled` is wide —
`getattr(op, "loop_info", None) is not None` — and `loop_info` is also stamped
outside coarse tiling by `insert_restickify.py:338/345`, `padding.py:410` and
`read_copy_elision.py:261/395`. And the pin path can *hard-fail* a compile, not
merely deoptimize: `_legal_fixed_division` raises
`Unsupported: fixed split violates hard domain` when the committed division fails
re-validation (`allocator.py:1388-1398`). So device validation must look for new
`Unsupported` raises, not only numerical deltas. `work_division_constraints.py:340-355`
names `test_copy_running_max_4d_H4_Lq4` as the test that broke when read-copy
tile-local dims were pinned; run it by name.

**N2 — Stop scaling CP-SAT search workers with `os.cpu_count()`.** Add
`cpsat_search_workers` to `config.py` defaulting to 8, wire it through
`_make_cpsat_solver` (`allocator.py:2489`) rather than reading config inside the
solver — `allocator.py`'s own docstring says the factory is the single place that
maps config to a solver. Keep the
`torch.are_deterministic_algorithms_enabled() -> 1` branch. This is the single
largest measured win in the plan and it also stops the emitted SDSC depending on
the build machine's core count. Budget for one round of golden-address test
regeneration; inventorying which tests those are is part of the change.

**N3 — Make the time limit and the fallback honest.** Convert
`max_time_in_seconds` from per-Solve to a shared wall-clock deadline. Log the
solve at INFO (buffer count, candidate total, per-phase elapsed, status) instead
of DEBUG-only, and raise the `SolveError` -> greedy fallback log to WARNING. Note
the trade the deadline makes: a starved phase returns `UNKNOWN`, which raises
`SolveError`, which silently drops co-optimization for the whole graph. Without
the WARNING this converts "slow but co-optimized" into "fast and quietly not".

**N4 — Floor the matmul cost divisor.** `_matmul_axes_for_split_cost`
(`cost_model.py:1365-1373`) computes `M = matmul_rows_per_core * m_split` and then
divides by `M` and by `M * N`. `dump_cost_model` initializes both per-core figures
to `0.0` and resets them to `0.0` on any extraction failure, so a degenerate matmul
raises `ZeroDivisionError`. The `cost_expr` guard at `allocator.py:1812` catches
only `(ValueError, RuntimeError, TypeError)`, so it escapes and kills the compile
instead of falling back to the lexicographic objective. Floor `M` and `N` at 1.0
and add `ArithmeticError` to the except clause.

**N5 — Hoist the loop-invariant group sets out of `_division_map`'s per-op loop**,
making the comment at `allocator.py:1882-1883` true. Two hoisted locals, since
`_fused_layout_group_ops` is called with two different seed sets.

**N6 — Record per-op `chosen_division` in every baseline and every diff**, per
§1.8. This is a property of the harnesses, not of any one workstream, and it has to
exist before the first search-quality change lands or that change is unfalsifiable.

**N7 — Fix the stale docstring at `allocator.py:674`**, which cites
`CoOptimizingAllocator._residency_reason` — a method that does not exist; the class
uses the inherited `_residency_reasons`. It is the second instance of the same
defect catalogued at `:2407-2408` (a comment asserting parity with a guard that is
not there), and that pattern is one of the strongest arguments in §5.

**N8 — Correct the docs that a hint user reads first.**
`work_division_planning.md:579-580` and `scratchpad_planning.md:16-17` say
co-optimization is opt-in; `config.py:23-25` defaults it on. While there,
`work_division_planning.md:616-618` claims a dropped hint logs a warning; it logs
at INFO (`work_division.py:1030`), and `:627-632` describes the divisibility gate
as the span floor.

## 3. Execution order

The priority order is a statement about value and it is the shape of this plan.
Three things force a different execution order, and each is checkable in the tree
rather than a matter of taste.

**Safety is not a phase.** The most urgent item on this branch is not in any
workstream's critical path — it is one `elif` plus one guard (N1), closing a
documented ~9%-wrong hole with zero call sites. It goes in front of everything.

**Three workstreams are proposing to fork the same 45 lines.** The pin ladder
receives an `elif` from coarse tiling, another from division hints, and a wholesale
replacement from encapsulation; `_enumerate_core_divisions` receives a seed from
division pruning, a tiling loop from coarse tiling, and an absorption from
encapsulation. Left alone that is three merge conflicts and three chances to drop a
guard silently — which is exactly how `_is_coarse_tiled` died. So the registry —
priority 2 — is promoted to a genuine prerequisite: it turns contested `elif`
insertions into independent table entries, and its exhaustiveness test is the
mechanism that catches the next dead guard.

**Coarse tiling's integration step is gated on three things that do not exist yet**
— the restored pin, the shared footprint function, and a test harness that actually
compiles (§1.9). Landing it before the harness is fixed ships the headline behavior
untested. Division hints and division pruning are the smaller, better-measured half
of the same problem — "a decision must survive into the joint solve" — so they are
ready first in practice.

This costs the headline feature less than it looks, because coarse tiling's first
three steps do not touch `_division_map` at all and run concurrently with the
encapsulation work.

| Phase | Contents | Gate to leave it |
| --- | --- | --- |
| 0 | N1-N8, plus coarse tiling A (§4) — inert today, a silent LX over-commit later | Hinted coarse-tiled graphs reach the solver pinned; corpus solve deterministic and ~2.5-3x faster; 240 orphaned tests running |
| 1 | Substrate (§9): device-free inner loop, solver bench over all three corpora with division diffs, pass-order contract, off-device graph-builder fixture | A planner change is evaluable in ~30 s with no card, reporting time, spilled traffic, and a per-op `chosen_division` diff |
| 2 | Encapsulation A-D (§5): layout accessors, `SpyreGraph`+lifetimes, `SpyreOp`, `legal_core_divisions` registry | Exhaustiveness test passes; O(N²) rescan gone; LX pin set byte-identical |
| 2' | Coarse tiling B-C (§4) in parallel: fail-closed guards, tiled-frame predictor | Guards fail closed with a measured e2e diff; predictor round-trips against the applicator |
| 3 | Division hints (§6) and division pruning (§7) | Hint survives co-optimization with the two test workarounds deleted; one table encoding exists; budget regression-free at every k tested or shipped off |
| 4 | Coarse tiling D-G (§4): harness fix, solver integration, tiling pruning, logical grouping, then the group constraint | `hinted` and `partial` rows green on device at `SENCORES` 32 and 2; both gates off proven bit-identical, not assumed |
| 5 | Restickification (§8) and op reordering (§10), each gated on its own measurement | Abandonment is an acceptable outcome for either |

Encapsulation's behavior-*changing* steps (§5, E-G) are deliberately not on this
critical path. They land whenever there is device time, one at a time.

Phase 5 is last on evidence, not preference. Both workstreams had a headline claim
disproven while this plan was being written — restickify's "measurement only" step
is a term in the default CP-SAT objective, and op reordering's bundle-count target
is arithmetically impossible — so their sizing is not trustworthy even though their
underlying findings are real.

## 4. Coarse tiling

Goal: coarse tiling becomes a decision the joint solve makes rather than a mutation
applied before it. Every op reaches the solve with a menu of `(CoreDivision,
TileSpec)` candidates; a user `spyre_hint` reaches it as a *constraint* on that menu
rather than an already-applied tiling, so a pin and a discovered level compose in
one solve. Discovery ships default-off; with the gates off the tree is bit-identical
to today.

### A. One per-core footprint function

`CoreDivisionBuffer.min_footprint` (`plan_solver.py:325-329`) already divides by
`cd.output_partition * cd.tiling.output_tile_count` and drives the pre-solve
capacity exclusion (`:477-482`). The model's `per_core` (`ilp_solver_ortools.py:284`)
and post-solve `footprint` (`:374`) divide by `output_partition` alone. They agree
today only because every `TileSpec` is empty. The moment a non-empty spec exists, a
buffer can pass the capacity gate on its tiled footprint and then be packed at its
untiled one — an over-committed LX plan with no error.

Add `per_core_footprint(size, cd)` to `plan_solver.py` and route the sizing sites
through it. There are **seven**, not three: the two above plus
`exhaustive_search.py:129/137/147` and `sa_cooptimizer.py:381`, both reachable
under co-optimization. Keep `min_footprint`'s empty-candidates guard
(`if not self.core_divisions: return self.size`) — replacing the body with a bare
`min(...)` raises on an empty sequence. Leave `mem_usage_by_buf`
(`scratchpad/utils.py:236-239`) alone; it correctly reports the committed layout.

Write down the sizing invariant while you are there: `CoreDivisionBuffer.size` must
be the *untiled* device footprint, so a caller planning over already-tiled IR must
hand every buffer an empty `TileSpec`. The second solve depends on it.

Device-free. Any diff in `test_scratchpad_solver.py` or `test_scratchpad_use.py` is
a bug in the refactor, since `output_tile_count` is 1 everywhere today.

### B. Fail-closed guards for the four shapes tiling gets wrong

These must exist before discovery can ever be enabled, and each also improves the
two paths live today.

**Stick-dim upscale.** `enumerate_tilings.py:38-49` states it plainly:
`_split_candidates_for_host_dim` admits a whole-stick split of the stick-carrying
dim, applying one produces an undersized boundary `full_buf` and silently wrong
results, and the enumerator therefore drops the stick dim entirely rather than
trusting the alignment predicate. But that enumerator is dead, and the predicate it
distrusts (`_post_tile_stick_alignment_error`,
`span_overflow_hint_analysis.py:275`, satisfied as soon as
`tile_size % stick_elems == 0`) is what both live paths use — while the hint path
consults nothing at all. Reject the stick host dim outright in
`_split_candidates_for_host_dim`, and raise `Unsupported` in
`plan_coarse_tile_groups` when a hint lands on it. The check must also fire
wherever `_stick_host_dim` returns `None`, because `_resize_device_layout`
(`ir.py:119-124`) silently takes a default path in that case — that is how the
wrong-axis resize actually happens.

Hard-reject first, then measure against `test_span_overflow_hint_analysis.py` and
narrow to "whole-stick multiple with a full boundary tile" only if a real case
regresses. "Probably the true rule" is not the standard for a silent-wrong-code
guard, and the measurement is one suite run.

**Unit output tiles.** A dim tiled to per-tile extent 1 is squeezed out of the
index. This is the strict xfail at `test_coarse_tile_e2e.py:5564` (a `squeeze_pos`
KeyError) and it is also a misclassification hazard for the solver integration,
because `_core_division` (`allocator.py:1219-1225`) sorts a symbol into
output-versus-reduction splits on `write.index.coeff(s) != 0`, and a squeezed dim's
coefficient is 0. Do **not** make this a blanket refusal: the tree deliberately
allows Pointwise unit tiles and rejects them only for reductions
(`span_overflow_hint_analysis.py:1367-1385`, whose comment says so). Scope the guard
to reductions, or to a unit-tiled dim a read copy must index, and diff the e2e
pass/fail set before landing.

**Reduction-axis discovery.** `enable_reduction_tiling` defaults True and must stay
that way — the hint path exercises reduction tiling in passing tests. But that one
flag currently means both "a user may ask for this" and "the compiler may choose
this", and the second is the source of the worst numerics: the flash-attention Lk
xfail at ~90% element mismatch (`test_coarse_tile_e2e.py:4236-4249`) and the
softmax reduction case that "does compile and is *numerically wrong* today"
(`test_solver_auto_coarse_tiling.py:655-664`). Add
`auto_tile_reduction_axes: bool = False`, read only by the candidate builder, so
the policy has a name and relaxing it is deliberate.

**Chained tile groups.** Two chained groups produce a group-2 read copy that reads
group-1's per-tile scratch while its `full_buf` goes unread. Put the check where
all three producers converge — `plan_coarse_tile_groups` (`coarse_tile.py:432`) or
beside `validate_coarse_tile_groups` (`:358`) — *not* in `CoarseTilingPass`, which
is never instantiated and so would protect neither live path. Sharing with the hint
path is the point: the hazard is live there today.

### C. The tiled-frame predictor

The solve must price a tiling it has not applied, which means it needs the tiled
frame — divided ranges, resized output layout, rescaled indices, divided iteration
space — as a pure function of `(op, TileSpec)`. Port `predict_frame` and its
helpers into a new `wsr/tile_prediction.py` (~190 lines of stage4's 314).

Drop `BoundaryRole` and its three helpers. Main has moved past them:
`plan_coarse_tile_groups` + `_plan_tiling_propagation` (`coarse_tile.py:432/747`)
is already a complete zero-mutation planner emitting four `PropagationPlan` kinds
(`loop_info.py:170`) including `mutation_write_back`, which a three-valued
`BoundaryRole` cannot express — porting it would silently reclassify every
mutation-write-back op, which includes flash attention's running-max carry. Where a
boundary role is genuinely needed, call main's planner and read
`info.propagation.kind`.

One hazard the branch's own version does not close: the `buf_layout` override on
`_prepare_per_core_view` replaces the device layout but not the `dep`, and
`try_device_coordinates` (`pass_utils.py:2821`) and the `dep_coeff` computation
(`:2820`) both still use the untiled `dep`. That yields a per-core view computed
from an untiled index against a tiled layout — the silent-wrong-`PerCoreView`
class, feeding LX residency and in-place merge decisions. Predict the dep as well,
or give `_prepare_per_core_view` a `dep_override`, and make the round-trip test
assert the *prep*, not just the frame.

The load-bearing test is an application round-trip: for each mock op,
`predict_frame(op, spec)` must equal the post-`_apply_plan` state — ranges, layout
size/stride, `device_layout.device_size`, write index — after actually running
`coarse_tile_post_stickify` with the equivalent hint. Cover a matmul and a
broadcast, because `_rescale_index(..., strict=False)` deliberately passes those
through unchanged. Device-free.

### D. Tiling candidates enter the joint solve

This is the step that delivers "hints respected by the solver". Behind
`unified_tiling` (default off), the pre-stickification hint pass stands down, a
user hint survives on `op.dim_hints` as data, `dim_hints_to_tile_spec` lifts it
into a mandatory `TileSpec` that restricts the op's menu, the solve picks a
`(division, tiling)` pair, `CoarseTilingPass` materializes exactly those tilings,
and the graph is re-planned once with tiling suppressed.

Extract `ScratchpadOptimizationPass` from `allocator.py:199` into a new
`scratchpad/passes.py` first. `scratchpad/coarse_tiling.py:47` imports the ABC from
`.allocator`, so the allocator cannot import `CoarseTilingPass` at module scope
without a cycle. It is a 19-line move and it removes the cycle for good.

Two prerequisites before any of that is worth writing. `dim_hints_to_tile_spec`
does not exist — only the forward `tile_spec_to_dim_hints` does — so the
pin-as-constraint direction is new code, not a port. And the harness must actually
compile: `test_solver_auto_coarse_tiling.py:351` raises before compiling anything
for every auto row, and `:371` is a bare `# TODO: Patch coarse tiling config here`.
Fix that first, or this step lands untested.

Four things this step must settle, none of which the branch version does:

**Work division is stale for solver-tiled ops.** `_distribute_work` runs before
scratchpad planning, so a tiling minted inside the solve never gets a
work-division pass. `_fixed_core_division` reads an `iteration_space_ownership`
computed on *untiled* ranges, and `_legal_fixed_division` then re-validates it and
raises when a per-tile range no longer admits that split. Worse,
`coarse_tile.py:4063` calls `copy_op_metadata(sizing_op, copy_buf)` on every
inserted read copy, and `iteration_space_ownership` is in `_SPYRE_METADATA_ATTRS` —
so a copy op created by the second solve inherits a *foreign* op's committed
division, and the restored pin then pins it there. That is precisely the
compute-op-versus-read-copy shape from §1.4. Pick one: re-run work division over
the tiled graph between materialization and the second solve, move materialization
to a pass slot before `_distribute_work`, or recompute each tiled op's fixed
division from the tiled ranges. Whichever, generated copy ops must have their
inherited ownership cleared — `coarse_tile.py:4067-4069` already does this for
`work_div_loop_info` and is the precedent.

**Applicator-inserted ops have no division at all, and restoring the pin does not
fix that.** `_commit_divisions` skips any op without `iteration_space_ownership`
(`allocator.py:1965`), and the only pass that stamps it is `_distribute_work`,
which ran before scratchpad planning. So every read copy, full buffer and drain op
`CoarseTilingPass` inserts gets `_fixed_core_division(op)` — a bare one-core
`CoreDivision()` — and is then skipped by the commit, while its sibling compute ops
keep a multi-core division. That is verbatim the divergence `_is_coarse_tiled`'s
docstring names, arriving by a different route. Whichever answer is chosen for the
stale-division question above has to cover the inserted ops too.

**The second solve must distinguish span-overflow `loop_info` from solver
`loop_info`.** After materialization both carry `loop_info` and both hit the pin,
but the span-overflow ops' fixed division was computed on tiled ranges (valid) and
the solver ops' on untiled ranges (stale).

**Rewriting only `it_space` in the enumerator mixes frames.**
`enumerate_work_division_candidates` builds `input_tds`/`output_td` from the
committed untiled layouts and then adjusts a tiled `it_space` against them. A
shrunken iteration space over a full-size input layout *under*-estimates the
per-core span, which admits an illegal split — the opposite direction from safe.
Either hand `collect_tensor_deps` the predicted layouts too, or restrict the tiled
path to output-axis tilings and prove the error direction per tensor, with a test
that constructs a split legal under the tiled space and illegal under the real span.

Also: `derive_tiling_groups` looks up `choices.get(op.get_operation_name())`
(`scratchpad/coarse_tiling.py:141`) while `_division_map` keys on `op.name`
(`allocator.py:1904`). These are different namespaces. Assert every non-empty
choice resolved to an op — the failure mode is "nothing was tiled", with no error.

Two gates, two validations. Both flags off: prove the whole tree is unchanged
against the device-free capture corpus, do not assume it. `unified_tiling` on: the
`hinted` rows of `test_solver_auto_coarse_tiling.py` must still pass — that is the
pin-as-constraint round trip and the point of the step — and
`TestCoarseTilingPassEquivalence` (`test_coarse_tiling.py:8357`) must still assert
byte-identical `loop_info` and ranges against the hint path.

Do not flip `unified_tiling` on inside this workstream. It stands the
pre-stickification hint pass down entirely and reroutes every `spyre_hint` in the
tree through a path that did not exist before; the pre-stickification path inserts
read copy-ins the post-stickification applicator deliberately skips, and the
equivalence test covers `loop_info` and ranges, not the inserted copy machinery.
Earn the flip with the full `hinted` + `partial` matrix on device at both
`SENCORES=32` and `SENCORES=2`.

### E. Prune and rank the tiling option set

This is priority 5, and it is small once D exists.

`_finalize_options` (`enumerate_tilings.py:199-213`) does exact-duplicate dedup and
nothing else, then sorts by `_canonical_key` — `(depth, axes_tuple)`, whose own
docstring says it is "explicitly NOT `_combo_cost`" — and truncates at 64. Because
every depth-1 spec sorts before every depth-2 spec, three tileable dims at
`_MAX_SPLITS_PER_DIM = 16` give 48 depth-1 specs and leave 15 depth-2 slots; with
five tileable dims, nothing nested survives at all. Meanwhile the sibling
span-overflow search *does* rank by cost (`_combo_cost`,
`span_overflow_hint_analysis.py:1433-1444`). The two engines disagree.

- Replace `_canonical_key` with a `_combo_cost`-shaped ranking — total tile count,
  tiled-dim count, max split, axes — keeping the untiled spec pinned first and never
  truncated.
- Add two structural filters inside the enumerator: sub-stick (per-tile extent below
  one stick is pure loop overhead) and dominance (same axis set, counts an
  elementwise multiple of a surviving spec's).
- Put the **LX-fit** filter in the caller, not the enumerator: drop a spec whose
  predicted per-tile footprint still exceeds the LX budget. This keeps the LX budget
  out of a module that is currently a pure function of the op.
- Port the read-distance filter as a third caller-side filter, memoized per
  `(op, spec)` — it does a real layout rebuild plus span re-analysis per call.
- Unify the four caps into config with today's values preserved *per path*:
  `_MAX_TILE_DIMS` is legitimately 2 in the enumerator and 3 in the span search,
  because the span search is a satisfier that must reach a fitting combination while
  the enumerator is a menu builder. Note that `enumerate_tilings.py:74-76` records a
  deliberate decision *not* to migrate `_MAX_AUTO_TILE_SPLIT_COUNT` to config;
  reversing it is fine but should be argued.

Dominance must be defined structurally, never on a guessed cost. Coarse tiling is
deliberately unquantified in the CP-SAT objective — tilings compete only through
the LX-residency footprint and the work-division terms — so a cost-derived
dominance rule would be encoding a preference the objective does not hold.

### F. Logical rather than positional tile-group derivation

`TileAxis.host_dim` is a positional index and `derive_tiling_groups` breaks a run on
exact positional spec equality. The hint path has no such problem: it groups on
`frozenset(hint_id)`, and a `hint_id` is bound to a *named* dim, so a matmul and a
pointwise op with different output-dim numbering still share a group. A solver
emitting per-op `TileSpec`s therefore fragments groups at exactly the
matmul-to-pointwise boundary tiling most wants to span — which is the MLP and SwiGLU
shape, and why the `partial` rows cannot pass on positional grouping.

Keep `TileAxis` positional (it is the applicator's contract and the round-trip
depends on it) and change the *join rule*: a candidate op joins a run when, for each
level, its tiled loop var indexes the run's tiled coordinate through its read of a
run member. `_consumer_shares_group_tiled_dim`
(`coarse_tile_span_overflow.py:71`) already does the harder version of this check;
factor the per-level predicate out so there is one implementation. Fail closed —
any failure to establish correspondence breaks the run, which is today's behavior.

The `@expected_unimplemented` decorator is applied by `hint_mode`, not per model
(`test_solver_auto_coarse_tiling.py:694-702`), and it fails on a clean pass. So all
three `partial` rows must go green in the same commit, or `case_decorators` must be
restructured to key on model name as well. Softmax additionally needs a
"no discovery expected" case, because its only other axis is the reduction one.

### G. Later: replace the pin with a group-consistency constraint

The pin from N1 freezes every op carrying `loop_info` at its committed division.
That is correct and cheap while only hinted graphs are tiled; once discovery is on,
it is most of a tiled graph and forfeits exactly the joint optimization this branch
exists to enable. The right end state is a constraint: every op sharing a
`loop_info.loop_group_id` must select divisions that agree on the shared loop frame,
leaving the solver free to choose which consistent set.

This is genuinely new machinery, not a reuse. `cd_parent_matches` is a
producer-to-consumer *edge* relation and a loop group is a *set* — two sibling read
copies feeding one compute op are not each other's parent, and `_views_for_divs`
requires a `(dep, buf_name)` pair the op actually touches. The group relation has to
be stated on per-core ownership of the loop's tile index, which is what
`commit_iteration_space_ownership` already models. Size it accordingly.

Encode it as one `AddAllowedAssignments` table per member against a representative,
not per-pair reified literals — and coordinate with §7 so there is one table
encoding, not two. Keep the pin as the fallback when a group's compatible set is
empty: `_gate_divisions` currently forces non-residency in that case, which is the
wrong outcome for a group.

Land this only after discovery, and only if it beats the pin on a measured case. A
constraint that reproduces the pin's answer everywhere has bought nothing.

## 5. Encapsulating op and graph attributes

The framing in §1.5 is the whole argument: the scratchpad is procedural because the
questions it asks have no names. Give them names, one at a time, and the reflection
goes away as a side effect rather than as a goal.

Four entry points, and they are the only sanctioned way to ask:

```python
SpyreOp(op).requires_hbm_inputs(spyre_graph)
SpyreOp(op).requires_hbm_write()
SpyreOp(op).legal_core_divisions(spyre_graph, max_cores=...)
SpyreGraph(graph).get_topological_sort_lifetimes()
```

Two of those signatures are not what they look like, and the plan is better for
saying so up front.

`legal_core_divisions` **must** take a graph handle. Two of the seven pin guards are
graph-scoped: `_fused_layout_group_ops` and `ops_in_offset_mutation_component` both
scan every op to answer a question about one op. A purely per-op signature cannot
express them, and `SpyreGraph`'s memoized sets are what keep the call O(1). It also
needs two things `(op, graph)` cannot reach: `self.prune`, a `CoOptimizingAllocator`
constructor argument, and the `(profiles, matmul_roles)` pair computed once per
`_division_map` call. Put `prune` in the signature and the matmul profiles on
`SpyreGraph` as a cached property, or the absorbed path is not reproducible.

`requires_hbm_write()` is honestly only the op-intrinsic subset. Of the twelve
residency branches, only two are genuinely per-op — the unsized/no-device-layout
check and `_is_tiled_advancing`. The rest need `planned_lx_buffers` (solve-scoped),
`mutated_buffers` (graph-scoped), `graph_output_names`, `uses`, or the whole-graph
buffer-user-dep map. Giving `requires_hbm_write` a graph parameter to cover them
makes it indistinguishable from `residency_reason` in nicer syntax. Two honest APIs
beat one dishonest one; the rest lives on `SpyreGraph.residency_reason(name)`.

One piece of hidden work has to be scoped up front. Several of these steps want
cheap CPU-only equivalence tables, and the fixtures that look like the vehicle are
not: `cooptimization_captures*.json` and `synthetic_cooptimization_graphs.py`
contain solver records — `CoreDivisionBuffer` and `CoreDivision` — and no
`Operation`, `GraphLowering`, layout or `MemoryDep`, while all four APIs above take
exactly those. That validation is not weak, it is impossible as written. Budget an
off-device graph-builder fixture — hand-built `ComputedBuffer`s with real
`FixedTiledLayout`s inside a `V.set_graph_handler` context — at roughly the size of
one step, and land it in phase 1 so the rest is not blocked on it. It is feasible
without a card: importing `torch_spyre` and constructing a `FixedTiledLayout` needs
no device.

### A. Accessors on the two types we own

`FixedTiledLayout` (`ir.py:88-116`) is torch-spyre's, is constructed only by
torch-spyre code, and already holds `device_layout`, `allocation` and `lx_view`.
Add read-only properties — `num_sticks`, `size_bytes`, `is_lx`, `is_hbm_pool`,
`lx_address` — and share them with `TensorArg`, whose `allocation` is documented as
mirroring the layout's. Keep `allocation` a plain dict as the storage;
`op_spec_validation.py:379` enforces an exactly-one-key invariant on it.

This collapses the five open-coded `math.prod(device_size[:-1]) * 128` sites and
most of the twenty `"lx" in allocation` tests. Two of those sites are deliberately
duck-typed over layouts that are *not* `FixedTiledLayout` and must be left alone:
`scheduler.py:357-358` and `dump_cost_model.py:107-108` both guard with `getattr`
precisely because they see `NoneLayout` and plain `FixedLayout`. Also note the
degenerate-case question this raises and answer it explicitly:
`allocator.py:875` returns 0 when there is no `device_layout` while `:2140`
dereferences it unguarded, and a property on `FixedTiledLayout` cannot be called on
the plain `FixedLayout` the first guard exists for — so that guard moves, it does
not stay.

Independent of everything else. Land it first.

### B. `SpyreGraph` and a typed `Lifetimes`

Five liveness implementations disagree on index base, end convention and aliasing.
This step does not merge them; it names the one the LX planner uses and states its
invariants as fields rather than prose: `frame` (pre-scheduler operations versus
post-fusion nodes), `end_is_exclusive`, `alias_model`. The two legitimate
specializations — counted-loop end overrides, and the LX-relayout half-tick rescale
at `allocator.py:1016` — become methods instead of caller arithmetic.

Put `spyre_graph.py` at the top level, not under `scratchpad/`. `cost_model.py`,
`work_division.py` and `hbm_pool_planning.py` are all prospective callers, and
`allocator.py` already reaches across module boundaries for four private helpers —
including a function-local import from a *solver* — precisely because these
predicates have no declared home.

The one real hazard is caching. The scratchpad pass mutates the graph mid-solve, so
a cached `Lifetimes` can go stale where four independent recomputes could not — and
the obvious invalidation contract does not work: `commit_iteration_space_ownership`
rewrites ownership in place at `allocator.py:1991` without changing
`len(graph.operations)`, and `lx_relayout.py:412-413` mutates `layout.allocation`
in place. So: cache only structurally-derived facts on `SpyreGraph` — `op_by_name`,
the buffer-user-dep map, the two group sets — and key anything ownership-derived on
a generation counter bumped inside `commit_iteration_space_ownership`. Also note
that `mem_usage_by_buf` and `get_ncores_for_buffers` are today called *with* a cache
at some sites and *without* at others; hoisting one shared cache makes previously
uncached sites cached, which is a behavior change on top of the staleness question.

Consolidating the four `calculate_liveness` calls is safe and is the part to keep.

### C. `SpyreOp`: op-kind predicates and the two HBM questions

A per-call frozen slots dataclass holding only the op — no cache. Passes stamp
attributes throughout the pipeline, so a cached facade would need an invalidation
hook at every stamp site; the expensive work already sits behind `op_read_writes`'s
`op.__dict__` memo, which the facade must delegate to and must never shadow.

Expose the three op-identity namespaces under *distinct* names — `.short_name`
(aten), `.reduction_type` (SDSC), and the derived predicates — rather than picking a
winner. Choosing which is authoritative for LX eligibility changes which buffers are
eligible and is its own change; making the mismatch greppable is this step's job.

`requires_hbm_inputs` inverts today's buffer-first shape: `_restickify_barrier` walks
a buffer's uses asking "is any consumer a restickify?", and the facade asks "does
this op require its operands in HBM?". One thing must survive the inversion: the
`!= name` clause at `allocator.py:679-681` that exempts the buffer's own producing
op. A restickify's own output takes the ordinary residency path, and both candidate
user maps include the write dep, so the exemption has to be explicit in the inverted
call rather than implied by the map's shape.

This is also where the O(N²) fix (N5) lands properly, because `SpyreGraph` now owns
the memoized group sets.

The visible contract is the reason strings: `_log_lx_pinning` surfaces them and
tests name them literally. Make them enums carrying `.message` with the current
string verbatim, so every rule gets a stable identifier for registry-membership
tests without breaking either consumer.

### D. `legal_core_divisions` and a pin registry with an exhaustiveness test

The registry shape already exists in this codebase and works:
`work_division_constraints.py:109-125` is a tuple of named constraint callables with
per-rule attribution. Model `_PIN_RULES` on it, preserving today's evaluation order —
order only decides which reason is reported, since all seven share one remedy.

Return a `CoreDivisionDomain(candidates, pinned_by, constraint)`. Surfacing the
`ConstraintResult` alongside the pin reason makes visible, without changing it, the
asymmetry between the two layers: `pool_window_blocked_vars` explicitly permits
output-spatial splits while `_is_windowed_pool` pins the whole op;
`indirect_access_split_domains` pins only the shared data dims while
`_is_indirect_access_op` pins the whole op. Name both in one return value; do not
act on it here.

The test that justifies the whole step: walk the module for callables matching
`_is_*` / `_reads_*` whose first parameter is an `Operation`, and assert each appears
in `_PIN_RULES` or in an explicit, comment-justified allowlist. **It must fail on the
tree as it stands** — that failure is the proof of value, and N1 is what makes it
pass.

Two things not to do here. Do not substitute `pass_utils.is_keep_by_index` into
`_fused_layout_group_ops`: that helper is parametric and is called with two different
seed sets, and `is_keep_by_index` hardcodes one of them. And note that the
`self.prune` branch is dead on every production path at HEAD (`dcd8a184` removed
`prune=True` from the cpsat factory), so moving it is bookkeeping, not "the bulk".

### E-G. The behavior-changing steps

These are separated deliberately, each individually revertable, each landing when
there is device time.

**E. `SpyreOpMeta`.** One typed dataclass under `op._spyre`, replacing the ten
declared attributes; `copy_op_metadata` becomes a one-line `dataclasses.replace`.
The risk is entirely in one pattern: `hasattr` used as a control-flow predicate,
which silently becomes always-true once the attribute always exists. There are
**14** such sites in `torch_spyre/` at HEAD, not the handful one remembers, and one
of them — `insert_restickify.py:675` — is a whole-graph filter
(`[op for op in operations if hasattr(op, "_restickify_plan")]`) that would become
universal. Enumerate them mechanically before writing the commit. Seven existing
tests assert the absence semantics this abolishes, including one in `tests/tensor/`,
outside the suite this step's validation would otherwise run.

Two attributes cannot come along: `layouts` and `committed_stl` live on `TensorBox`
and `InputBuffer` objects, which a `ComputedBuffer`-keyed dataclass and copier can
never reach. Fold in only the ComputedBuffer-attached ones and leave those two as
raw attributes with their teardown intact.

Also: the `type: ignore` count is a fine metric but its floor is ~43, not 0 — six
sites are `_dim_prop_info` and three are `_read_copy_elision_record`, which must stay
a raw attribute for the `hasattr` at `read_copy_elision.py:399` to keep working.

**F. Make the residency rules total.** Today the ladder's order is load-bearing for
*safety*: `_would_produce_lx_back_gap` dereferences `device_layout` with no guard.
Give each rule its own precondition guard so order becomes a reporting preference.
Note the rationale is weaker than the docstring implies — on the graph-input path
there is no preceding device-layout check anywhere in that ladder, and the deref is
safe by a pipeline invariant instead (`insert_restickify.py:453` converts every
graph input's layout). The conclusion still holds; the argument should be stated
correctly. The registry needs a rule type returning `Reason | None`, not `bool`, so
the graph-output branch (two reasons, or fall through) can be expressed.

**G. Unify the two per-core-slicing predicates.** Extract the four guards —
partial-reduction writer, broadcast read, matmul multi-dim split, frame-changing
clone — as separately named predicates over the shared view machinery, and have each
path opt in to exactly the set it applies today. Correct the false parity comment at
`allocator.py:2407-2408` rather than tidying the sets. Then, and only then, enable
the two missing guards **one commit each**, with the LX pin-set delta in each commit
message. Both guards move buffers *out* of LX, so a shrinking pin set is expected and
a growing one is a bug. If the matmul guard's addition shows the placement path has
been shipping a wrong-code exposure, that needs its own issue.

## 6. Core-division hints

The prediction in the ask — "this would likely result in evictions at the perimeter
of the kernels" — is exactly right, and the mechanism is already in the model.
`constrain_residency` gates a buffer's residency on the `cd_parent_matches` pair
list, and `_gate_divisions` forces non-residency when that list is empty. Narrowing
a hinted op's candidate list shrinks its neighbours' pair lists, and whatever no
longer matches is spilled at `(reads_served + write) * size` bytes.

Today the question is moot, because hints do not reach the solver at all:
`work_distribution_pass` commits the hinted splits and `_commit_divisions`
overwrites them one pass later. Every end-to-end hint test hides this by patching
`co_optimizing_lx_planning=False`, with a TODO saying so.

**Shape: partial pin by default, full pin as opt-in.** A partial pin — fix the
hinted symbols, leave the rest to the solver — is a strict generalization of the
existing contract, because `_apply_user_hint` only ever validates and returns the
symbols the user named; the unhinted dims sitting at 1 is an artifact of
`work_distribution_pass` returning early, not a user statement. It also keeps the
perimeter eviction *bounded and optimizable*: several candidates that all agree on
the hinted dims still let a neighbour find a matching pair, whereas one candidate
makes the eviction structural. But "lock down critical kernels" is a real
requirement, so `work_div_exact=True` escalates to the whole-op pin.

Reject the third option — a hint term in the objective. The CP-SAT objective is
strictly lexicographic (residency, then parallelism, then balance), so a preference
has to become a new *level*, and where it goes is the entire semantics: above
residency it is a hard pin with extra steps, below it the hint loses to any spill.
Level 1 is denominated in HBM bytes; a hint has no byte value.

**Mechanism: the `allowed_splits` channel, not a new parameter.** The enumerator
already accepts a partial assignment, and `carried_reduction_pinned_row`
(`work_division_constraints.py:165-190`) is a working precedent that pins a split to
`frozenset({n})`. A `pinned=` parameter would constrain only the candidate list at
one call site and leave the *legality predicate* untouched, so
`_split_option_is_legal`, `_legal_fixed_division` and `_commit_divisions` would all
still accept an unhinted division. Going through `allowed_splits` fixes all six
sites with one rule, and it fails safe: `_legal_split_factors` *intersects* rather
than overrides, so a pin can only shrink the candidate set, never manufacture an
illegal division.

**Persist the hint name-keyed, not symbol-keyed.** This is the one place the obvious
design is wrong. Iteration-space symbols are op-local positional names (`d0`, `d1`)
built by `sympy_index_symbol(f"d{squeezed}")`; `_enumerate_core_divisions`'s own
docstring says deduplication uses local symbol names *only within this operation*.
`copy_op_metadata` has three cross-op call sites, and one of them —
`copy_op_metadata(sizing_op, copy_buf)` at `coarse_tile.py:4063` — has already
produced a documented wrong-dim bug from exactly this mechanism
(`work_division_constraints.py:424-431`: "the copy's own D axis at position 1
misread as the sizing op's H axis"). A membership filter on the destination's
iteration space is no protection, since it passes for any op of equal or greater
rank. Persist `{dim_name: value}` and resolve to symbols at rule time through
`work_div_loop_info`, which is already carried in the metadata and keeps resolution
correct on the destination op.

Persist the **post-prune** map: `_apply_user_hint` silently drops a hinted dim that
would exceed `SENCORES`, and pinning a dim the pass deliberately abandoned would be
a bug.

**An infeasible hint is currently silent, and the obvious diagnostic targets dead
code.** The `Unsupported: no legal core-division candidates` raise at
`allocator.py:1925` is unreachable from all three `_division_map` branches — each
falls back to `_legal_fixed_division`, and the existing test at
`test_work_division.py:1240-1268` proves it by patching the enumerator to return
`[]` and asserting the result is `[fixed]`. So when a hint pin empties the
enumeration, the real behavior is a silent fall back to the hinted division with
every unhinted dim at 1, logged at DEBUG — the serial behavior the partial pin
exists to avoid. Make that path raise with the op name, the pinned pairs and which
filter emptied the set. Note the enumerator's filters are strictly stronger than
`_apply_user_hint`'s: it also enforces the 256 MB per-core span and
`prod(splits) <= max_cores`, neither of which the hint validator checks. Add a
fourth cause the naive taxonomy misses: a pin whose value divides the concretized
size but not the *granularity* yields an empty factor list for symbolic dims.

**Report the perimeter eviction rather than absorbing it.** Compute
`cd_parent_matches` twice for a hinted op's edges — pinned and unpinned — and record
every buffer whose pair list goes from non-empty to empty. Two implementation
notes: once the constraint rule is registered there is no unpinned enumeration to
compare against without threading a bypass, and `spill_cost` is a pure function of
fields the allocator already holds, so compute it allocator-side rather than reading
it back from solver locals that do not survive the solve.

One case is structural, not heuristic, and belongs at hint time: a hinted
*reduction* split makes `has_partial_reduction` true, which maps the producer's view
to `None` in `cd_parent_matches`, so the hinted op's output can never be LX-resident.
`test_matmul_work_div_hint_maps_by_name` hints `K: 4`, so the flagship hint test is
precisely this case. Warn loudly at `_apply_user_hint`; do not refuse — a K-split is
often the right call for a critical matmul.

**Precedence.** The `work_div_exact` entry goes last in the ladder, after the seven
correctness pins. Each of those encodes a cited wrong-code or scheduler-abort
incident and a user hint must not override one. The interaction is milder than it
looks — all eight branches share the same remedy and `_fixed_core_division` returns
the division `_apply_user_hint` committed, so when a safety pin fires on a hinted op
it pins to the hinted division anyway and only the logged reason differs. What does
matter is that a *partially* hinted op tripping a safety guard gets the full pin and
loses its unhinted freedom; that must show up in the eviction report.

Finally, note the existing tests cannot demonstrate the headline claim.
`test_pointwise_work_div_hint_applied` is M=128 x N=64 fp16, and N is exactly one
stick, so nothing is left for the solver to split;
`test_matmul_work_div_hint_maps_by_name` hints K=4 and M=2 under `sencores=8`,
consuming the whole budget. A new case with a free, splittable, non-stick dim and a
hint consuming only part of the budget is required.

## 7. Core-division pruning

The measurements in §1.6 say what to do: the worker-count fix carries the whole
reliable win, and candidate truncation is a real but unproven trade.

**Where the cap goes.** Three attempts to bound the search inside
`enumerate_work_division_candidates` were added and reverted in a single afternoon —
`0eb73a2a` sliced `candidates[:64]` off an unordered `itertools.product`,
`34d3e30d` truncated the axis list itself, `7a431af5` changed the budget test to
`!= max_cores` and deleted the all-ones and seed divisions. The common defect is
that the enumerator has none of the information that makes truncation safe: it sees
one op's symbols, not the producer/consumer per-core-view pairs that are the model's
only cross-op coupling and the sole thing a dropped candidate can destroy. That
information exists exactly once, after `_cd_parent_matches` has run.

So: keep the enumerator a complete, unit-testable oracle — its only production
caller is the allocator and tests assert its exact complete output — and apply any
budget in `_build_cd_bound_buffers`, after the match graph exists.

**Ranking by match degree is necessary and not sufficient.** At the same k and the
same candidate count, ranking by weighted match-pair degree keeps 60/79 resident on
flash_big and 51/56 on block_x4 with byte-identical spill, while truncating by
enumeration order keeps 51/79 and 36/56. That part reproduces on the large graphs —
but on the *small* graphs the same ranking measured **worse** than plain enumeration
order on 4 of 7 at k=4, so the budget's apparent safety comes from self-disabling
below the threshold, not from the ranking being right. And the proposed rank
key has **no footprint term**, so truncation removes the candidate with
the largest `output_partition` — the one that makes a buffer fit in LX at all. On
the regen corpus, four flash_big buffers' `min_footprint` grew 4x, and
`min_footprint` gates `MemoryPlanSolver.excluded()` outright, not merely by
preference. Forcing the max-`output_partition` candidate into the keep set makes it
*worse* (44 resident). Same graph, same k, three sane keep rules, residency 44 / 51
/ 57.

**And the default fails on the third corpus.** At budget 450 the algorithm
reproduces on `cooptimization_captures.json` + `_large.json` — 11/11 byte-identical,
flash_big 1097 ms -> 433 ms — and regresses on `cooptimization_captures_regen.json`,
which is already in the tree: flash_big 57 resident -> 51, spill 13.6 MB -> 18.9 MB.
Raising the budget is not monotone (600 -> 51, 800 -> 49, 1200 -> 53, 1600 -> 53), so
the sweep-and-record-the-boundary procedure terminates with no answer.

**Therefore:** ship the budget **opt-in, defaulting off**, and land the worker fix
and the bench regardless. Do not present ranked truncation as a safe truncation; it
is a quality/time trade with no proof. If a default is ever set, it must be gated on
all three corpora and on a post-truncation assertion that no buffer's `min_footprint`
increased — with the ability to abandon the budget for a graph rather than proceed.

Also drop the seed-feasibility safety argument. "Pinning the seed makes the all-seed
assignment feasible, so a truncation can never score worse than the
non-co-optimized baseline" is true and irrelevant: it bounds the truncated optimum
against the plan co-optimization exists to beat, while the acceptance criterion is
"no worse than today's co-optimized plan". Seed-pinning is worth doing for a stable
index-0 convention, not as a proof.

**Two real fixes to make while here.**

`work_division_splits_are_legal` never checks `prod(splits) <= max_cores`, and it is
the sole predicate behind `_split_option_is_legal`, `_legal_fixed_division` and
`_commit_divisions`. Add the clause — but document the raise-path consequence: if it
ever fires on a *committed* division it produces `Unsupported: fixed split violates
hard domain` at menu-build time, a compile failure on a graph that compiles today.

Do **not** add a redundant per-core span re-check, and do not claim the existing one
is redundant either. The tempting argument — that a split dominating the
`_span_min_splits` floors is automatically span-legal — cannot be right, because the
enumerator applies those same floors and then still runs an explicit span filter; if
domination implied legality, that filter would be dead code. Two of its supports are
also false: `must_split_vars` commits `best_above` when nothing fits
(`work_division.py:658`, `668-702`), so the recorded floor vector can itself exceed
the limit, and there is no raise at the cited line. Keep a property test of
`get_per_core_span`'s per-axis monotonicity and the fall-through bound, and state the
honest conclusion: the enumerator's span filter is load-bearing, and any path that
builds candidates without it is where the hole actually is.

**Solver-side: measured, and the table is the wrong replacement.** The idea of
collapsing `_gate_divisions`' per-pair half-reified literals into one
`AddAllowedAssignments` per edge is mechanically viable but measured 1.2-1.35x on
some graphs and 0.67x on block_x4. It keeps the relation as a *table*, and that is
why it is mixed. Re-measured this session with the worker fix in place, on all
three corpora:

| encoding | corpus total (15 graphs) | regresses on |
| --- | ---: | --- |
| landed (`division` index + `add_element` + pair literals) | 5780 ms | — |
| menu one-hot + support-clause gate | **3528 ms (1.64x)** | nothing above noise |
| menu one-hot + per-class indicator equality | 3577 ms (1.62x) | nothing above noise |
| one linear class-id equality per edge (no pair literals) | 4746 ms (1.22x) | 5 of 15 graphs |

Median of 3 runs each.

All three lexicographic objective values *and* the resident set are identical to
the landed encoding on all 15 graphs. The argmin moves on 4 of them (1-7 buffers,
equal balance cost) — ties, plus the offset symmetry Section 1.6 already records —
so the gate is objective-and-residency equality, not a byte-identical plan.

Two things that licenses. First, the pair table is provably redundant:
`ResidencyEdge.compatible` is `parent_view(p) == consumer_view(c)`, each side a
function of one candidate, so the relation is a key equality — verified
rectangular on all 607 captured edges. Second, the *maximally* compact form is the
weakest and the only unstable one: 1.22x overall but slower than the landed
encoding on 5 of 15 graphs, and it inverts to 0.74x if measured at the current
`os.cpu_count()` worker default. Replace a table with Boolean structure, not with
integer arithmetic; constraint count is not the objective.

Prefer the support-clause variant: it and the class-indicator form are within
noise, and it is simpler. Keep the empty-pairs branch either way — no compatible
pairs must force `in_buffer == 0`, not be vacuously true.

This is a different kind of change from the truncation budget above: same optimum,
found faster, so it needs no quality argument. `draft-constraint-core-division.md`
carries the encoding, the measurements, and the structural (menu-free) model it
opens the door to — including the finding that solve time is *sublinear* in menu
size, which is why de-enumeration is worth doing for precompute and guard
expressiveness rather than for wall-clock.

The offset dimension — ~12.7k values per buffer, zero objective weight, no
canonicalization — is the remaining structural symmetry and the reason the plan
depends on worker count. It is a restructuring of the solve into a relaxation plus a
packing pass, with its own spill-repair failure mode. Not now.

## 8. Restickification

Measured share of HBM bytes from the in-tree op-features fixture: 0% for
mlp/softmax/swiglu/rms_norm, ~2.9% for block_x2/x3/x4, 6.2% simple_attn, and
roughly 13-15% for the attention graphs — and that *understates* it, because the
pass-inserted `restickify_default*` buffers are invisible to the runtime model's
detector. Adding them puts total stick-reorganization traffic near 28-30% of HBM
bytes on the flash graphs. (Two independent derivations of that number disagreed at
the decimal, so pin the methodology before it becomes a landing gate.)

Order the work so measurement comes before pricing, and pricing before anything that
moves the beam's preferences.

**A. Make it visible.** The runtime model's `_hbm_pattern` decides "restickify" from
*host* index coefficients, but a pass-inserted restickify is an identity copy in
host index space — `lower_restickify` builds `inner_fn(index) = loader(index)` and
the stick swap lives entirely in the two device layouts. So every pass-inserted
restickify is untagged. Swap the detector to the authoritative device-coordinate
predicate `is_restickify_coords`, exactly as `padding.py:517-528` already does.

Three corrections to carry into that change. First, **this is not
behavior-preserving.** `hbm_pattern` flows into `OpFeatures` -> `predict_by_bundle`
-> `cost_expr` -> the CP-SAT objective, and co-optimization is on by default. Either
compute the corrected pattern on a report-only path first, or accept it as a
behavior change and gate on a device run with a before/after of chosen residency and
divisions. Second, the tag makes an op **~10% cheaper**, not more expensive
(2B/116 GB/s versus 2B/150 GB/s plus turnaround), so any acceptance criterion
phrased as "the restickify share must not rise" has to be re-derived. Third, the
swap will also *untag* ops currently tagged — the expands, which
`is_restickify_coords` early-outs on as broadcasts — so validate against the ops the
old predicate tags, not only the additions.

Alongside it: a per-compile summary in `CostReport` (count, bytes, attributed
microseconds, share, split plan-inserted versus op-intrinsic), the beam's max-states
versus `BEAM_WIDTH`, the buffers `_restickify_barrier` bars, and how many
restickifies `restickify_padding_blocked_vars` pinned to one core. Distinguish
plan-inserted restickifies by the predicate `_get_op_name(op) == "restickify"`, not
by a name prefix. Keep the summary inside `CostReport`: `cost_model_pass` is
deliberately outside `self.passes` so hashing it cannot invalidate the Inductor
cache for a report that cannot change the compiled result.

**B. CSE identical restickifies.** There is provably no dedup today: the plan is
keyed per consumer op and a node is created per plan entry, while the restickify
buffer is a pure function of the resulting `(source, target layout)` pair. Key the
cache on the result and reuse the buffer.

Refuse to merge in three cases, not one. When either consumer carries `loop_info` —
the restickify takes over the consumer's per-read tile advance and two consumers
cannot both hand over their advance to one node (this is issue #4008 one level up).
When the consumers are in different loop groups. And when any candidate consumer is
an **indirect-access op**: `enforce_indirect_access_layout` runs *after*
`insert_restickify` and rewrites a producer's committed device layout in place for
all its consumers, with a gate that explicitly says multiple consumers are fine —
so merging would let a gather requirement on one consumer silently retarget the
layout another was committed to and indexes with.

Ship behind a kill switch. The exposure is not correctness but lifetime: a shared
restickify's live range stretches to its *last* consumer, which can delete an LX
in-place edge (which requires exact tick adjacency) and lengthens intervals both
memory planners price.

**C. Price the edge in bytes.** `cost = float(math.prod(in_stl.device_size))` is
dtype-blind, so a 128x128 fp32 restickify and an fp16 one cost the same. Compute
bytes as `math.prod(device_size) * 128 // stl.elems_per_stick()` — **not** via
`device_dtype.itemsize`, which does not exist: `device_dtype` is a
`torch_spyre._C.DataFormats` pybind enum, and `elems_per_stick` is a method, not a
property. The stick formula is also the answer to the fp8 worry, since `device_size`
and `elems_per_stick` are consistent by construction.

Keep the beam's accumulator in **integer** units and apply the bandwidth divisor only
where a number is reported. Dividing by 116.0 inside the search turns exact-integer
edge costs into non-representable doubles that the beam then sums, so two plans tied
as integers can differ by one ULP — which flips the first-minimum tie-break the LX
in-place promotion depends on.

The migration is cheaper than it looks and the plan should say so: the 61
`optimal_cost=` assertions sum host elements over the captured *plan*, not the beam's
internal cost, and every tensor in that file is fp16 — so on a uniform-dtype graph
the new cost is a positive constant multiple and the argmin is unchanged. Make
bit-identical plans the acceptance test rather than rescaling anything, and add the
first mixed-dtype restickify tests, which do not exist today.

**D. Price the clone that is physically a restickify.** `AnyInNode.cost` returns 0.0
unconditionally, and in the captured graphs the only restickify-tagged ops *are*
clones. Charge the clone's own output move when the input and output device
coordinates say it is a stick swap, and install the new node only on the clone
branch — leave `AnyInNode` at its ten other sites, which are fills, constants,
no-ops, `MultiOutput`, `DeviceCopy` and collective fallbacks.

The subtle part must land atomically with the cost: both the `downstream` map and
`_compute_last_use` are built only from `edge_costs`, so an `AnyInNode` clone appears
in neither, and the beam's liveness merge nulls any slot whose last use is absent.
Give a clone a nonzero cost without also giving it edge_costs and the merge collapses
two states that differ in a slot the clone still reads — a wrong-layout bug, not a
slow one. Note also that `_reorder_any_in_nodes` gates on `isinstance(..., AnyInNode)`,
so a *sibling* class silently stops relocating clones while a *subclass* keeps it;
pick one and state the consequence.

**E. Free layouts for single-consumer constant fills.** A constant is layout-free by
construction, yet the main loop hands it one generic candidate, making it a
restickify source for no reason. The scoped fix is named in the code's own docstring.
Gate on the beam max-states number from step A. Two cautions: `test_fill_plus_xt`
uses the `SpyreEmptyFallback` path, which *already* calls `_all_constant_layouts`, so
it cannot be affected by this change and its current cost should be checked before
assuming the premise; and the single-consumer gate does not actually address the
stated blow-up, since an unrolled graph has one constant with one consumer per
iteration.

**What not to do.** Do not couple layout selection with work division. The alignment
preference a penalty would buy already exists in a stronger form: candidate
enumeration is lexicographic on alignment at all three sites, breaking out of the
aligned group as soon as it yields anything, so unaligned stick dims are offered only
when no aligned dim produces a valid layout. A cost penalty could only discriminate
among already-degenerate candidates. If the one-core pin later proves expensive, the
right answer is the double-restickify capability — restickify into a stick-multiple
temp, then slice — which removes the unaligned case entirely.

Do not narrow `_restickify_barrier` here either. Its docstring invites a narrowing
keyed on core count, but with co-optimization on the core division is a decision
*variable*, not a fact, so "bar unless this buffer is on one core" becomes a joint
decision made from a guess. Measure it in step A; if it is expensive, the safe
formulation is a solver-enforced implication, not an allocator-side prediction.

## 9. Measurement and validation substrate

Not on the priority list, but every other workstream here is a performance or
search-quality change and none of them is currently falsifiable. Keep this minimal.

**A. Guard the HBM-poison fixture on device *presence*.** This one needs care,
because the obvious version is dangerous. Probe whether a device exists at all —
not whether an allocation succeeds — and skip the poison only when there is none.
**Keep `DeviceOpenFail` fatal.** The device is single-tenant, so catching that
signature would mean a run that merely lost a race for a busy card silently
disables the virgin-zero guard for the whole session and then executes on real
hardware. (Worth stating plainly: a card is free on this machine right now —
`tests/inductor/test_work_division.py` is 55 passed in 7.17 s — so the "every
inductor test ERRORs" observation was contention, not a permanent state. The
structural problem is unchanged: the suite is hardware-mandatory by construction.)

Do not `pytest.skip()`: it is an autouse session fixture, so a skip would silently
skip the entire inductor suite. Add two meta-tests, one proving the guard cannot
swallow a real fault.

This turns ~1,635 planner/solver/IR tests into a four-minute inner loop with no card.
It also settles where device-free tests live: **in `tests/inductor/`**, matching
existing practice. Do not add a second test directory; guarding the fixture removes
the reason one was wanted.

**B. Make an unrecognized CI label a config-integrity failure**, then relabel the
eight orphaned configs to the tiers their sibling planner configs use. This restores
240 tests, 81 of them the co-optimizer solver suites this plan leans on. They were
measured at 237 green / 3 red, all three in one class; xfail those three through the
per-test config mechanism, file an issue, and land the relabel. Land the checker as a
warning first, then fatal.

One practical trap: adding a bare three-name xfail block registers that config's
basename with the checker and turns every *other* collectable name in the file into
an uncovered hard problem, failing `make check-all-configs` — the very gate being
extended. Include a catch-all `- names: [".*"]` entry alongside the xfails.

**C. A device-free solver bench** over **all three** capture corpora, in `tools/`
(outside the conftest's reach), with a checked-in quality baseline and a `__main__`
regeneration entry point. Pin the solver's time limit high inside the harness so a
loaded runner degrades wall time — caught by a loose timing ceiling — rather than
degrading plan quality and reddening the quality assertion for environmental
reasons. Add a `--cost-expr` mode: production passes a `cost_expr` and, when it
linearizes, the entire lexicographic ladder is skipped, so a bench that never passes
one exercises a path production may not take.

Putting the regeneration command *in the module* rather than in a docstring is the
direct lesson of `test_op_features.py:31`, which names a capture script that does not
exist anywhere in the tree, leaving a 4.4 MB fixture unrefreshable.

**D. A relative plan-quality assertion** on `CostReport.total_us` for the models the
HBM-byte sweep already compiles — co-opt off versus on, LX off versus on, relative
only, never absolute microseconds. Include a meta-test proving the assertion bites,
because a silently-`None` report would otherwise make the whole class vacuous.
`cost_model_pass` is outside `self.passes` and so is not in the Inductor cache key,
which means a cache hit leaves the stored report stale — reset the compiler between
measurements.

**E. An executable pass-order contract** for `CustomPreSchedulingPasses`, asserting
relational pairs only (never a full expected list), one per ordering intent that
`passes.py` already states in a comment. Plus one negative assertion that
`cost_model_pass` is *not* in the list. The post-fusion pipeline already has exactly
this test; this is the same test for the pipeline that matters here.

**F. A weekly perf cron.** The perf job is fully implemented — runner selection, AIU
topology refresh, report check, ClickHouse ingest — and its own comment calls it
weekly-only. Only the trigger is missing. One new caller workflow with a schedule, a
concurrency group, and a fan-in gate job.

Out of scope: anything requiring changes to the external perf suite, anything in an
orchestrator repo, per-PR perf gating, a cardless CI job, and absolute thresholds
anywhere.

## 10. Op reordering

Start by correcting the premise this workstream is usually motivated by.

Upstream's `reorder_for_peak_memory` **is** enabled (`torch/_inductor/config.py:429`;
torch-spyre patches only `reorder_for_locality`) and it **does** run at
`torch/_inductor/scheduler.py:4173`, after every Spyre plan is committed. But it was
measured inert: by the time it is called, `spyre_fuse_nodes` has already collapsed
every maximal contiguous Spyre run into one `FusedSchedulerNode`, so it saw 2-4 nodes
on four probe graphs and returned unchanged every time. Its sizing is also
layout-blind (`get_numel() * dtype_size`, with no notion of tiled padding, LX
residency or bundle-scoped pools).

Disable it anyway — one line next to the existing `reorder_for_locality: False`, plus
a test asserting the patched value so a torch bump cannot silently re-enable it. The
asymmetry is the reverse of the obvious one: the risk is not losing an HBM-peak win
on four small probes, it is that on a model-sized graph it reorders *after*
`hbm_pool_planning` has committed bundle-scoped live ranges. "Inert on four 15-op
graphs" is not evidence about a model graph.

Similarly, the pre-scheduling order does survive to fusion, and not only by probe:
`topological_sort_schedule` emits a node's deps before the node, so on an
already-topologically-valid list it reproduces list order exactly. Which means
`_regroup_by_outer_loop_key`'s docstring blaming Inductor's DFS for coarse-tile group
fragmentation is probably wrong — though it is a cheap defensive invariant and
`build_loop_scheduler_nodes` raises without it, so leave it.

Half the LX argument for reordering also fails, and it is worth killing early:
`_restickify_barrier` iterates `for u in uses` (`allocator.py:678-682`), so it
inspects only the ops that touch the buffer — a set invariant under any permutation.
No reorder can lift or plant a restickify barrier. Only
`_extern_kernel_in_live_range` is genuinely positional
(`range(min(uses), max(uses) + 1)`, `:172-174`), and `SpyreEmptyFallback` — the one
freely movable extern — is already exempt from it, so moving that lifts no bar
either.

**Then do a census before writing any mover.** The design case for reordering is that
one interposed non-fusable op fractures a run into three bundles, shrinking HBM-pool
reuse scope and breaking the cost model's per-bundle dedup (a recorded 45%
under-prediction on a five-op softmax priced as five kernels). But
`group_contiguous_fusable` emits a bundle for *every* non-fusable entry wherever it
sits, so two boundary ops give a floor of three bundles no matter how they are
arranged — a "3 -> 1" target is arithmetically impossible. And
`dedup_and_promote_constants` unconditionally front-loads every constant before the
proposed pass would run, so no `SpyreConstantFallback` can be interior. The realizable
interior-and-movable set is only `SpyreEmptyFallback` and CPU-device
`ComputedBuffer`s, because `_is_movable_interloper` refuses general `FallbackKernel`s
whose read/mutation accessors are known-incomplete. Establish that this set is
non-empty across `tests/inductor/` before sizing anything.

**Extract the legality core.** `_no_dep_conflict`, `_can_move_before`,
`_can_move_after`, `_is_movable_interloper`, `_index_by_identity` are battle-tested,
private to `wsr/coarse_tile_hints.py`, and imported nowhere else. Move them to a
shared `op_order.py` with their own device-free tests; leave
`_unhinted_predecessor_closure` behind, since it is genuinely hint-specific. This is
a pure move, and `TestReorderUnhintedInterlopers` is the regression net.

**One pin predicate, memoized graph-scoped, not per-op.** Coarse-tile loop groups;
mutation copy-in/copy-back triples; offset-mutation components. Two categories that
look right and are not: in-place handoff partners are *derived* from the order at
planning time (`_determine_in_place_division_invariant` recomputes liveness itself),
so a reorder cannot break that invariant — the real effect is silent opportunity
loss, which argues for scoring it, not pinning it. And counted-loop end overrides are
a pure function of `graph.operations` recomputed at planning time, keyed by buffer
name, covering values born *outside* the loop — neither the domain nor the range the
pin would assume.

Cost matters here: `ops_in_offset_mutation_component` rebuilds the full adjacency and
walks the component on every call, so a per-op predicate inside a hill-climb is
O(n³)+. Build the pin set once.

**Position and objective.** The pass goes between `_distribute_work` and
`_maybe_scratchpad_planning`: work division is per-op and order-insensitive, while
`calculate_liveness` and the CP-SAT objective are exactly what must see the new
order. Use a pass-list entry rather than the `ScratchpadOptimizationPass` slot — a
pass reached only through `select_allocator` appears in no `_pass_sources` list and
would silently not invalidate the Inductor cache.

Score bundle count first (exact, cheap, computed by the same grouping function the
real pass uses). Do **not** score LX pressure by summing raw buffer `size` over
overlapping intervals: under co-optimization the LX footprint is
`min(ceil_div(size, output_partition * output_tile_count))` over candidate divisions
the solver has not chosen yet, so raw size over-counts by a different factor per
buffer — up to 32x — and the proxy's ordering of two candidate orders is not a noisy
version of the real objective, it is unrelated to it. Score with `min_footprint`, or
gate acceptance on an actual solve via the plan-quality harness from §9.

Both movers ship behind default-off flags. A reorder perturbs liveness for every
buffer at once, which is a wider blast radius than any of the co-optimization
defaults that produced this branch's string of silent-wrong-output regressions.

One more thing the order probe must see: `elide_proven_read_copies` runs *after*
scratchpad planning, is on by default, and removes ops from the list — shifting every
liveness index after the removed copy relative to what the LX plan was built on.
Sample the order immediately before `_maybe_scratchpad_planning`, not only at the end
of the pipeline.

## 11. Decisions needed

1. **Restore the coarse-tile pin, or delete the predicate?** Recommendation: restore
   it (N1), behind a named flag, and decide the companion in-place guard in the same
   commit. A dead predicate that still reads as a live safety guard is the worst of
   the three states. If a device run shows the ~9% case passing without it, delete the
   predicate too and record which run proved it. Note it has one owner, not two: it
   is a *division* pin. If op reordering later wants "carries `loop_info`" as an
   *order* pin, that is a second predicate with a different justification and no
   written failure signature — it must not be framed as reviving this one.

2. **Does the registry land before coarse-tiling's solver integration?**
   Recommendation: yes (§3). The alternative is doing the integration twice.

3. **Partial pin or full pin as the default hint semantics?** Recommendation: partial
   by default, `work_div_exact=True` for the lock-down case. Both are reachable; only
   the default is in question.

4. **Do we reject unknown `spyre_hint` kwargs?** `spyre_hint(**kwargs)` does no key
   validation, so `work_div_strict=True` is silently dropped and the user gets the
   partial pin believing they got the full one. Rejecting unknown keys fixes that but
   could break any downstream consumer of an unlisted key.

5. **Does the candidate budget ship at all?** Recommendation: opt-in, default off,
   until a keep rule survives all three corpora. The worker fix is unconditional.

6. **Is `_hbm_pattern`'s correction allowed to change compiled output in one step,
   or does it land report-only first?** It feeds the CP-SAT objective either way.

7. **Do the non-co-optimized and co-optimized hint paths need to agree?** Today a
   one-dim hint uses 4 of 32 cores with co-opt off and fills the rest with it on.
   Converging them changes long-standing tested behavior on a path several tests
   explicitly select.

## 12. Risks

- **Two stacks, one file.** `scratchpad/allocator.py` is edited by the coarse-tiling
  stack, the encapsulation registry, the hint pin, the pruning budget and the
  restickify instrumentation — and by an unlanded three-branch stack that rewrites
  roughly 29% of it. §3's ordering is what keeps that bounded. Make the
  retire-versus-merge call on those branches *before* phase 2 refactors the file
  further, or the port cost roughly doubles.
- **The default acceptance gate is blind** (§1.8). Residency and spilled bytes can
  be identical while every committed division changes, and the division is real
  per-core slicing. Four workstreams would otherwise ship behind that gate.
- **Two headline claims in this space were disproven while the plan was written.**
  A restickify step described as measurement-only turns out to be a term in the
  default CP-SAT objective, and an op-reordering target turns out to be
  arithmetically impossible. Both underlying findings are real; treat effort
  estimates in those two areas as unvalidated until their own measurements land.
- **Device time is the real constraint.** Every correctness gate here needs a card;
  the device is single-tenant and concurrent runs fail with a misleading
  resource-busy error, so these suites must be serialized. §9A is what keeps the
  inner loop off that critical path.
- **The corpora are small.** The largest capture is 79 buffers (2299 candidates on
  the regen set). Whether a production decode graph lands on the linear or the
  superlinear side of the observed scaling decides whether 30 s is generous or already
  binding, and no in-tree measurement answers it.
- **Silent wrong output is this area's characteristic failure**, not crashes: ~19% on
  an avgpool window split, ~9% on a coarse-tile group, ~44% on an offset-slice read,
  ~90% on reduction-tiled flash attention. Every behavior-changing step here should
  land alone, with numerical comparison against CPU, and with the measured delta in
  the commit message so a bisect can attribute a later regression.
- **Fixtures that cannot be regenerated.** The op-features capture script named in
  `test_op_features.py:31` does not exist. Any new baseline this plan adds must ship
  its regeneration entry point in the same module.
