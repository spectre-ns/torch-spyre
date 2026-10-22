# Scratchpad refactor: open questions

Running list while the plan (`~/spyre-roadmap/plans/cpsat-graph-buffer-refactor.md`)
is implemented end to end on `scratchpad-refactor-experimental` as an exploration.
Each entry says what I assumed in the meantime, so work did not stop on it. Newest
last. Feasibility results so far: `~/spyre-roadmap/plans/cpsat-graph-buffer-refactor-feasibility.md`.

Status (2026-10-22): stopped at Phase 2 plus 3a (Q11). The zoo benchmark is done:
every new mode (derived, Lazy with points, digits, point menus) reaches the same
objective as today's tables on 64/64 solves, and the agreement checks pass (746/746
derived edges, 1,011/1,011 Lazy edges, 582/582 Lazy spaces). Planning time over the
zoo, against today's tables with relayouts off: tiling compiles 161 s -> 123 s
(derived) and 113 s (Lazy + point menus); default compiles 47 s -> 57-70 s, all of
it CP-SAT time (Q12). The report has the full tables and an implementation guide.

## Progress: the whole plan (requested 2026-10-22, "implement the whole plan with green CI")

| step | state |
|---|---|
| Phase 1 plan text: `Transform` / `TransformKind`, `encode_transform` / `encode_transform_use`, `encode_lifetime`, `NodeVars.start/end`, `edge.merged` / `edge.transform`, `chosen_transforms()` | done; same model (smoke 37/37, unit-test solves 99/99) |
| 2c input clones as constraints (`encode_clone`, `CloneRead`) | done under `LX_VIEW_RELATION=derived`; clone consumers can be Lazy |
| 2d relayouts as constraints (`LX_RELAYOUT_ENCODING=lazy`, `lazy_relayout.py`) | done; 8 models x 2 configs, no errors, objectives far below no-relayout (attn 31455 -> 21199, transformer 83385 -> 58349), residency within 1 buffer of the table relayouts |
| 2e in-place as constraints | covered: an in-place edge between Lazy nodes is the derived view relation plus footprint equality, each footprint a table over the axes it varies with (Q13) |
| 3a `set_solver`, `GraphSolver`, `MarkTableAxis`, `Decomposed` (CP-SAT leaves) | done |
| 3b node absorbs `lx_view`, relayout plans, paired members (pinned transforms) | done |
| 3c graph from the IR (`ir_graph.py`, `LX_IR_GRAPH_CHECK`) | done; agreement run in progress |
| 3d placement solvers read placement views (`PlacementSolver`); greedy drops its isinstance | done |
| 3e exhaustive search as `DecomposedPlacement`; annealer charges `spill_cost` | done; annealer adjacency stays `op_position` (Q14) |
| 3f tests moved with the API | done for 3a-3g |
| 3g one allocator (`division_mode`), `CoOptimizingAllocator` deleted, every solve through `graph.solve` | done; same-plans check in progress; the buffer construction stays (Q15) |
| Final cleanup: delete redundant code (superseded encodings/arms, the merged allocator subclass, dead helpers) | requested 2026-10-22; last step |

## Q1. Stage 9: extraction or generation from the op?

Stage 9 is done as an extraction: the candidate derivations moved verbatim from
`CoOptimizingAllocator` into `scratchpad/table_candidates.py`, and the solver still
reads candidates from the buffers. Generating them inside the strategy from
`node.op` / `node.space` needs the graph built from the IR (3c), and until then
the cost expression and the SA / exhaustive solvers read the same buffers.

- Assumed: extraction is acceptable for Phase 1; true generation comes with 3c.
- Answer needed: is that the split you want, or should Stage 9 wait for 3c?

## Q2. Apply the plan's later model changes (Transform, lifetime vars) to the code?

The plan changed after Phase 1 was implemented (`RelayoutLayout` -> `Transform`,
`encode_relayout` -> `encode_transform_use`, `NodeVars.start/end`,
`encode_lifetime`). The spike shows free lifetime ends cost nothing (79 captures,
same optimum, median 1.01x). The code still uses the old names.

- Assumed: rename later, once the Phase 2 prototype settles; it is mechanical.
- Answer needed: rename now, or keep the code at the plan revision it was built
  from (22966fe)?

## Q3. Phase 2 "lazy": stop at L1 (per-node tables) or go to L2 (no menu)?

The first lazy strategy follows the plan's 2.0 L1 arm: menus stay as per-node
tables, pair tables go, and every edge's view relation becomes equality of
per-node view ids, tile-view ids and core counts derived from the edge's
`ResidencyEdge`. Overridden edges (input clones, loop carries) keep pair tables.
L2 (no menu) needs views derived in closed form from index maps (2c proper), which
is a re-derivation of `_per_core_view_from_prep` (311 lines with rescues).

- Assumed: build L1 end to end first (`LX_VIEW_RELATION=derived`), benchmark it,
  then attempt L2 for the common digit-form views with a per-node fallback.
- Answer needed: is L1 enough for the decision you want to make?

## Q4. Relayouts under the derived relation

The relayout candidates are priced per view pair (`_cd_parent_relayouts`), so the
first derived-relation strategy turns relayouts off (no candidates generated).

- Assumed: off is acceptable for the first benchmark; the plan's 2d (lazy relayouts:
  one optional layout per edge, shared by reified view equality, closed-form
  legality) comes after L1 works.
- Answer needed: should the benchmark compare against table-with-relayouts or
  table-without-relayouts? (I will report both.)

## Q5. Pushing exploration commits to PR #5676

Every commit on `scratchpad-refactor-experimental` updates PR #5676 (draft, "DO NOT
MERGE") and runs CI. The Phase 1 commits are green.

- Assumed: keep pushing; CI on the exploration is useful signal.
- Answer needed: or should the Phase 2/3 exploration go to a separate branch so
  #5676 stays at Phase 1?

## Q6. MIP backend: one stage still INFEASIBLE

After fixing the product linearization, SCIP still reports `sdpa_1x8x256x64`
stage 0 INFEASIBLE (43k MIP vars) where CP-SAT is optimal: a remaining linearizer
defect or big-M numerics.

- Assumed: not worth chasing for the feasibility question (MIP is a second opinion
  on small models only).
- Answer needed: do you want the MIP backend kept at all?

## Q7. What the Lazy work split tabulates over

`LX_DIVISION_ENCODING=lazy` (with `LX_VIEW_RELATION=derived`) gives a node no
candidate index. Its split is one var per loop axis plus a tiling choice; legality is
posted per tiling frame from its `WorkDivisionContext` (`lazy_division.py`, ported
from the 2f spike). Every per-division quantity (footprint, core cost, tile counts,
view ids, cost-expression terms) is a table over only the axes it varies with,
tabulated over the space's *points*: each frame's factor domains within the core
budget, a superset of the legal divisions that is cheap to list. No legality check
and no menu are involved in those tables.

- Assumed (recommended): tabulate over points for now. Views need per-point view
  computation (`_view_for_div`), which is the cost 2c's closed-form views remove.
- Answer needed: is per-(node, dep) view-id tables over points an acceptable
  fallback, with 2c's digit form as the follow-up that removes the view computation?

## Q8. The allocator still builds menus

The Lazy strategy ignores the menu in the model, but the allocator still enumerates
it: the carry and clone pair tables, residency gating and relayout pricing read it,
and the solve writes its choice back as a menu index (`b.chosen_division`). So the
planning-time saving of "menus gone" (the plan's ~78 s zoo-tiling column) does not
show yet; the benchmark will show model size and solve time only.

Update: point menus (`LX_LAZY_MENUS=1`) skip the enumeration for nodes that will be
Lazy, which takes the zoo-tiling allocator time from 99.6 s to 61.4 s (134.8 s with
today's tables). The carry, clone and restickify-pricing consumers still get full
menus.

- Assumed (recommended): keep the menus until Phase 3 (3c/3g) moves those consumers
  onto the graph; report allocator time separately.
- Answer needed: should removing the menu from the allocator be part of this
  exploration, or is the model-side result enough?

## Q9. Which nodes stay Table under Lazy

A node stays `Table` if its split is pinned or committed, it has no space (not
enumerable), it is an end of an edge whose relation is still a pair table over menu
indices (loop-carry update/read edges, input clones), or its space has more than
4096 points. This is the plan's mixed enumeration (rule 5).

- Assumed (recommended): as above; clones and carries move with 2c's clone encoding.
- Answer needed: none unless you want a different cut-off.

## Q10. Relayouts under Lazy (2d): worth building now?

The derived relation and Lazy run without relayout candidates (Q4). Measured on the
zoo (table with vs without relayouts, same code, separate processes): relayouts make
20 more real buffers resident (610 vs 590, about 3%), plans are identical on 40 of
64 solves, and planning takes about 70% longer (prepare 239.7 s vs 141.5 s, solve
105.2 s vs 66.3 s in total).

- Assumed (recommended): 2d is not built in this exploration; its closed form for
  `movement_supported` is already verified offline (Phase 2.0), and the remaining
  per-pair gates and the price are the work.
- Answer needed: is a ~3% residency gain worth 2d before the rest, or should
  relayouts become an opt-in refinement?

## Q11. How far into Phase 3

3a (backend protocol) is built. 3b-3g are restructurings of the allocator: 3c builds
the graph from the IR beside the buffers and must agree with it, and 3g deletes
`CoOptimizingAllocator`. Their feasibility was judged from the code and the 3c spike
(762/762 `produced_at`), and the plan says Phase 3 needs its own plan.

- Assumed (recommended): stop implementation at Phase 2 (L1 derived relation, Lazy
  work split, digit views, point menus) plus 3a, finish the benchmark, and write the
  implementation guide and results into the feasibility report.
- Answer needed: do you want a 3c prototype (IR builder plus agreement check) too?

## Q12. CP-SAT is slower on every new encoding

Over the zoo's default compiles, CP-SAT time goes from 12.2 s (pair tables) to
21.7 s (derived) and 26.9-31.8 s (Lazy). Default compiles have small menus, so the
allocator has little to save and the total rises 21-48%. On tiling compiles the
allocator saving outweighs it (-24% derived, -30% Lazy with point menus). Equality
of id vars propagates worse than a pair table, and Lazy adds product chains and
reified span terms.

- Assumed (recommended): not investigated further here. The guide lists the
  candidates (keep pair tables where both menus are small, solution hints from a
  table solve, a table instead of the product chain for the core count) and says to
  measure them before making either encoding the default.
- Answer needed: is a per-compile choice (derived / Lazy only when coarse tiling is
  on) acceptable, or must the new encoding be no slower everywhere?

## Q13. 2e as a product equality

The plan writes in-place footprint equality as `out_partition * out_tiles`, a
product equality. Under the Lazy encoding a node's footprint is already a table over
the axes it varies with (the output splits and the tiling option), and an in-place
edge posts footprint equality plus the derived view relation, which is the same
constraint.

- Assumed (recommended): no separate product form; it would add a product chain
  per node for the same feasible set.
- Answer needed: none unless you want the literal product form.

## Q14. The annealer's op adjacency

The plan moves `SaCoOptimizingSolver`'s op adjacency onto execution edges. On real
graphs `op_position` equals each op output's `produced_at`. But 21 annealer tests
build synthetic buffers whose `uses` all start at tick 0 and rely on `op_position`
alone, so deriving positions from the chain makes them collide.

- Assumed (recommended): keep `op_position` for the annealer; its spill cost now is
  `buffer_graph.spill_cost`.
- Answer needed: rewrite those fixtures (distinct ticks) and drop `op_position`?

## Q15. How far "delete the allocator's buffer construction" goes

3g says the allocator's buffer construction is deleted and the graph is built from the
IR, with the buffer form of `add_node` kept for tests and replay. Today every
consumer still reads `CoreDivisionBuffer`s: the CP-SAT strategy (`from_buffers`),
the annealer (about 2000 lines), and the post-solve commit path (`commit_divisions`,
`commit_lx_views`, `_apply_chosen_tilings`, `_push_allocation`). The IR graph agrees
with the buffer graph (3c), so the switch is possible. But it rewrites the annealer
and the commit path onto nodes, and the plan's own 3e makes the annealer rewrite a
precondition.

Why it is not one more step: `_build_cd_bound_buffers` derives the solve's incoming
data (lifetimes and overrides, drain plans, residency reasons, in-place pairs,
clones, carries) in the same pass as the candidates (menus, match tables,
relayouts). The commit path needs each buffer's chosen `CoreDivision` (reduction
symbols, tile splits), which a `WorkSplit` does not carry and only the menu has.
Splitting that pass is the Stage 9 "generate from the op" work, and the plan
already lists it as open.

- Assumed (recommended): stop 3g at one allocator class with every solve going
  through `graph.solve`. The buffers stay the allocator's own structures, which the
  plan's Interfaces section allows ("writes the decisions it finds on the nodes back
  to its own structures"). The IR graph (3c) is checked against them. The split is
  written up in the feasibility report as the next stage, in this order:
  1. `WorkSplit` carries the chosen `CoreDivision`.
  2. The commit path reads nodes.
  3. Incoming data moves into `add_node(op)`.
  4. Candidates come from the strategy.
  5. The annealer runs over nodes and the strategy's tables.
  6. The buffer construction is deleted.
- Answer needed: should that split be done on this branch before you re-implement?
