# Scratchpad refactor: open questions

Running list while the plan (`~/spyre-roadmap/plans/cpsat-graph-buffer-refactor.md`)
is implemented end to end on `scratchpad-refactor-experimental` as an exploration.
Each entry says what I assumed in the meantime, so work did not stop on it. Newest
last. Feasibility results so far: `~/spyre-roadmap/plans/cpsat-graph-buffer-refactor-feasibility.md`.

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
