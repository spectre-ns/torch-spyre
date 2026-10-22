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
