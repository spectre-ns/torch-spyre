# Unified z3 Planning Solver for the Spyre Pre-Scheduling Pipeline

## Context

The torch-spyre pre-scheduling pipeline (`CustomPreSchedulingPasses` in
`torch_spyre/_inductor/passes.py`) makes a chain of coupled decisions in
**separate, greedy, irreversible passes**: tensor layout / stick-axis choice,
restickify placement, padding, constant dedup, working-set reduction
(chunking + coarse tiling), core division (work splits), and finally LX
(scratchpad) placement.

Because each pass commits before the next runs, an early decision can force a
bad outcome that only surfaces at the very end, with no feedback path to fix
it. The canonical example, verified in code:

- The **stick axis** is chosen first (`propagate_layouts.py:130-139, 220-234`).
- A poor choice removes the good output-parallel split, leaving **core
  division** to fall back on a **reduction-axis (K) split**.
- A K-split produces partial sums on most cores; `get_ncores_for_buffers`
  flags this as a core-division mismatch (`scratchpad/utils.py:160-193`,
  returns `-1`).
- Mismatch buffers are filtered out of **LX** (`core_div_mismatch`,
  `utils.py:112`), so they spill to HBM.

The first transform and the last pass are tightly coupled, but the pipeline
cannot represent the relationship. The fix is to model these decisions as
**constrained variables in a single z3 solve** so coupled choices are made
jointly.

**Decisions locked with the user:**

- **Scope:** monolithic — subsume the whole pipeline (all passes except
  `deadcode_elimination`).
- **Solver:** extend the existing z3 prototype (`planning-sandbox.py`).
- **Objective:** minimize HBM traffic (maximize LX residency / minimize
  spilled bytes).
- **Delivery:** phased, behind a config flag, with legacy passes as the
  fallback until parity is proven.

## Goal

Replace the sequential pre-scheduling passes with one z3 model that chooses
layout/stick-axis, restickify points, padding, chunking, coarse-tile loop
counts, core splits, in-place merges, and LX placement **together**, then
realizes the solution by driving the existing graph-transform code as
mechanical appliers. Ship it behind `UNIFIED_PLANNING` so the legacy pipeline
stays default until the new path reaches parity.

## Current state (what we build on)

- **`planning-sandbox.py`** — standalone z3 prototype. Already models a coupled
  subset: per-buffer offset (`off_{n}`), LX residency (`in_buf_{n}`),
  in-place merge-group activation (`group_{names}`), and **core-division
  selection** (`div_{n}` indexing `list[CoreDivision]`, with effective size
  `size_{n}` coupled to the chosen division). Key methods: `_add_buffer_vars`,
  `_add_merge_groups`, `_apply_no_overlap_constraint`, `_add_core_division`,
  `_search` (spill-budget relaxation, **feasibility-only, no objective**),
  `_justify` (bottom-justify), `_extract`. `Box` dataclass holds a merge
  group's `(start_time, end_time, goff, gsize, var)`.
- **Production scratchpad** — `torch_spyre/_inductor/scratchpad/`:
  `scratchpad_planning(graph, allocator)` (`allocator.py`), heuristic solvers
  (`plan_solver.py`, `firstfit_bestfit_solver.py`), `_push_allocation` writes
  `buffer.layout.allocation["lx"] = address`, read by `codegen/superdsc.py`.
  `StrategyBCoOptimizingAllocator` already does a bounded DFS over split
  variants — the seed of joint optimization.
- **Decision representations downstream codegen expects** (must be preserved):
  `op.op_it_space_splits` (`work_division.py` `apply_splits` /
  `splits_by_index_coeff`), `op.loop_info` = `CoarseTileInfo` (`loop_info.py`),
  `op.data.ranges` (divided in place), `op.layout` (`FixedTiledLayout` +
  `allocation` dict).
- **z3 is NOT a declared dependency** — sandbox only.

## Unified model design

### Variable set

Grouped by the decision each represents. Free variables vs. derived
expressions called out.

| Decision | Variable | z3 type / domain | Notes |
|---|---|---|---|
| Stick axis per tensor | `stick_{n}` | Int, `0..rank-1` | Free. Drives layout realization. Often constrained equal to a producer's stick (propagation) — see constraints. |
| Layout/order compatibility | `relayout_{p,c}` | Bool | Derived: true iff producer `p` and consumer `c` stick/order differ → a restickify is required on that edge. |
| Padding amount | `pad_{n}` | Int, `0..stick_elems-1` | Derived from `stick_{n}` and shape (stick-alignment); appears in size/HBM terms. |
| Chunk count | `nchunk_{n}` | Int, `1..max_cores` | Free where pointwise + oversized; `1` means no chunk. |
| Coarse-tile loop count | `tile_{n,d}` | Int, divisors of dim `d` | Free. Replaces hint-driven `loop_count`. |
| Core split (output/reduction) | `div_{n}` | Int, index into `list[CoreDivision]` | **Already in sandbox.** Enumerate candidate `CoreDivision`s per op. |
| Effective per-core size | `size_{n}` | Int (derived) | **Already in sandbox.** Couples to `div_{n}`, `tile_{n,d}`, `pad_{n}`. |
| LX residency | `in_buf_{n}` | Bool | **Already in sandbox.** |
| In-place merge group active | `group_{names}` | Bool | **Already in sandbox.** |
| Group offset / peak | `goff_{names}`, `peak` | Int | **Already in sandbox.** |

The structural decisions (chunk, tile, restickify, pad, stick axis) become
**decision variables**; their solved values feed a deterministic realization
step. z3 never "creates an op" — it picks the numbers the appliers consume.

### Constraint families

1. **Memory non-overlap** — present (`_apply_no_overlap_constraint`): active
   groups overlapping in time must not overlap in address.
2. **LX capacity / peak** — present: `Implies(group, goff + gsize <= peak)`,
   `peak <= M`.
3. **Group membership** — present:
   `Sum(group vars containing n) == in_buf_{n}`.
4. **Core-division matching** — present (`_add_core_division`): a child may be
   on LX only if its division matches a parent's chosen division.
5. **Stick propagation** — new: for an edge where the stick dim survives,
   `stick_{c}` follows `stick_{p}` unless `relayout_{p,c}` is true (a
   restickify is inserted). Ties `relayout` to layout difference.
6. **K-split ⇒ not-LX (the motivating constraint)** — new: if `div_{n}`
   selects a reduction (K) split **and** the buffer has >1 consumer with no
   broadcast op, then `Not(in_buf_{n})`. This is exactly the
   `get_ncores_for_buffers == -1` rule lifted into the model, now linking
   `stick → div → in_buf` in one solve.
7. **Tiling ↔ split interaction** — new: splits operate on the *reduced*
   (post-tile) iteration space; `div_{n}` candidates derived from
   `ranges / tile_{n,d}`, mirroring why `coarse_tile` must run before
   `work_distribution` (`coarse_tiling_loops.md:548-557`).
8. **Stick-alignment / padding** — new: `size_{n}` rounds the stick dim up to
   `stick_elems` (64 fp16); `pad_{n}` captures the waste, feeding the HBM term.
9. **Span limit** — new: per-core span (`prod(device_size[:-1]) * 128 /
   ncores`) `<= 256 MB`, the constraint `span_reduction` enforces today.

### Objective: minimize HBM traffic

Move from the sandbox's feasibility-only `_search` to a cost objective:

- Switch the solver to **`z3.Optimize`** and minimize
  `Sum(device_bytes_{n} * If(in_buf_{n}, 0, 1))` — spilled bytes — matching
  `StrategyBCoOptimizingAllocator._score_layout`.
- Add HBM terms for the structural decisions: each active `relayout_{p,c}`
  (a restickify round-trip), each `pad_{n}` (wasted stick bytes), and chunk
  boundary copies contribute weighted byte costs.
- **Keep the spill-budget relaxation as a fallback**: if `Optimize` is too
  slow on a given graph, fall back to the existing `Solver` +
  monotonically-relaxed `spill_count <= budget` loop (seeded at the forced
  set) for a feasible-but-unoptimized plan. Surface which path ran via
  `log`/INFO.

### Post-solve realization (reuse existing appliers)

The solver outputs numbers; existing pass code performs the graph edits, in
this order:

1. **Layout / stick** — apply `stick_{n}` by driving the layout assignment in
   `propagate_layouts.py` / `finalize_layouts` with the chosen order.
2. **Restickify** — for each active `relayout_{p,c}`, call the insertion logic
   in `insert_restickify.py` (`compute_restickify_target_layout`).
3. **Padding** — `insert_bmm_padding` / `padding.py`, parameterized by
   `pad_{n}`.
4. **Chunk** — `chunk_large_tensors.py` `_chunk_op`, given `nchunk_{n}`.
5. **Coarse tile** — `coarse_tile.py` `insert_tiling_propagation` + range
   division, given `tile_{n,d}`.
6. **Splits** — `work_division.py` `apply_splits` / `splits_by_index_coeff`,
   writing `op.op_it_space_splits` from the chosen `CoreDivision`.
7. **LX placement** — `scratchpad/allocator.py` `_push_allocation` /
   `GraphEditor`, writing `buffer.layout.allocation["lx"]` from `off_{n}`.

Each applier already exists; the work is parameterizing it with solver output
rather than letting it decide.

## Phasing plan (behind `UNIFIED_PLANNING`)

Each phase disables the listed legacy passes and adds the listed variables.
The hybrid stays coherent because realization always emits the same op
metadata the remaining legacy passes expect.

- **Phase 0 — scaffolding.** Add `z3` dependency, `UNIFIED_PLANNING` flag, new
  module `torch_spyre/_inductor/unified_planning.py`, and the
  graph→`LifetimeBoundBuffer` adapter (reuse `calculate_liveness`,
  `mem_usage_by_buf`, `_determine_in_place`). Solver still only does what
  `scratchpad_planning` does; prove parity vs. legacy LX planning.
- **Phase 1 — core division + LX + in-place.** Absorb `span_reduction`,
  `_distribute_work`, `_maybe_scratchpad_planning`. This is the sandbox's
  current model (`div_{n}`, `in_buf_{n}`, `group_{names}`) wired to real
  graphs, plus the HBM objective. Legacy layout/WSR passes still run upstream.
- **Phase 2 — coarse tiling.** Add `tile_{n,d}` and constraint family 7;
  disable `_maybe_coarse_tile` / `assign_dim_hints` (hints become solver
  seeds, not commitments). Realize via `insert_tiling_propagation`.
- **Phase 3 — layout + restickify (closes the stick↔LX gap).** Add
  `stick_{n}`, `relayout_{p,c}`, constraints 5 + 6. Disable
  `propagate_spyre_tensor_layouts`, `optimize_restickify_locations`,
  `finalize_layouts`, `insert_restickify`. **This is the phase that actually
  fixes the motivating pessimization** — pull it forward if that gap is the
  priority.
- **Phase 4 — chunk + padding + dedup.** Add `nchunk_{n}`, `pad_{n}`; disable
  `_maybe_chunk_large_tensors`, `insert_bmm_padding`,
  `dedup_and_promote_constants`. Full monolithic solve.

## Integration seam

- **Flag** — in `config.py`, alongside `lx_planning`:
  `unified_planning = os.environ.get("UNIFIED_PLANNING", "0") == "1"`.
- **Pipeline branch** — in `CustomPreSchedulingPasses.__init__`, build
  `self.passes` conditionally: when the flag is set, the subsumed passes (per
  current phase) are replaced by a single `unified_planning` entry; otherwise
  the legacy list is unchanged. `deadcode_elimination` always runs first.
- **Cache keying** — `uuid()` calls `_uuid(self.passes)`; since the pass set
  differs by flag, the cache key changes automatically. Verify
  `unified_planning`'s source is included so edits invalidate the cache.
- **Entry point** — `unified_planning(graph: GraphLowering) -> None` in the new
  module: builds the model from the graph, solves, then runs the realization
  appliers in order, mutating `graph.operations` in place (same contract as
  every other pass).

## Files

**New:**

- `torch_spyre/_inductor/unified_planning.py` — model build, solve, realize.
  Port and generalize the `planning-sandbox.py` solver classes here (Apache
  header, `import regex`, 88-col).
- `tests/inductor/test_unified_planning.py` — unit tests for the model.

**Modified:**

- `torch_spyre/_inductor/passes.py` — conditional pipeline + flag branch.
- `torch_spyre/_inductor/config.py` — `unified_planning` flag.
- `pyproject.toml` / `setup.py` — add `z3-solver` dependency.
- Appliers parameterized to accept solver output (no behavior change when flag
  off): `insert_restickify.py`, `padding.py`, `chunk_large_tensors.py`,
  `coarse_tile.py`, `work_division.py` (`apply_splits`), `scratchpad/allocator.py`
  (`_push_allocation`).

## Dependencies & conventions

- Add **`z3-solver`** to project deps; it was sandbox-only.
- Apache-2.0 14-line header on every new file (per CLAUDE.md).
- `import regex` (never `import re`); 88-char lines; Google style.
- `git commit -s` (DCO); run `pre-commit run --all-files` before pushing.

## Risks

- **z3 nondeterminism** breaking reproducible compiles — fix the solver seed /
  set deterministic config; assert stable output in tests.
- **Scalability** — a monolithic solve over a large graph may be intractable.
  Mitigation: partition into independent subgraphs (by liveness-disjoint
  regions) and solve per region; keep the spill-budget fallback. Flag any
  graph exceeding a model-size threshold and fall back to legacy.
- **Partial-pipeline incoherence** during phasing — each phase's realization
  must emit exactly the metadata the still-active legacy passes consume;
  guard with parity tests per phase.
- **Cache invalidation** correctness when toggling the flag.

## Verification

1. **Unit** — `tests/inductor/test_unified_planning.py`: feed crafted
   `LifetimeBoundBuffer` graphs, assert solved offsets respect non-overlap,
   capacity, K-split⇒not-LX, and that the HBM objective beats or ties the
   greedy allocator on known cases.
2. **Parity** — run `tests/inductor/test_scratchpad_solver.py`,
   `test_scratchpad_use.py`, `test_scratchpad_patterns.py` and the op suites
   (`tests/inductor/test_inductor_ops.py`, `tests/test_spyre.py`) under both
   `UNIFIED_PLANNING=0` and `=1`; compare `op_it_space_splits`, LX addresses,
   and spilled-byte totals. Unified must spill ≤ legacy.
3. **Motivating case** — construct a graph where the legacy stick-axis choice
   forces a K-split that spills; assert the unified solver (Phase 3+) keeps it
   on LX.
4. **Full suite** — `python3 -m pytest tests/` green with the flag off
   (no regression) and the targeted suites green with it on.
5. `pre-commit run --all-files` clean.
