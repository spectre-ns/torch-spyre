# RFC (draft) — Unified Tiling, Work Division, and LX Residency Co-optimization

| Field | Value |
|---|---|
| Status | Draft — pending issue number |
| Area | Compiler |
| Target | `torch_spyre/_inductor/scratchpad`, `torch_spyre/_inductor/wsr` |
| Depends on | RFC 1358 (Coarse Tiling), RFC 0047 (Tensors with Device-Specific Layouts) |

> **Draft note.** RFC sources live in
> [`torch-spyre/rfcs`](https://github.com/torch-spyre/rfcs) as
> `NNNN-PascalCase/NNNN-PascalCaseRFC.md`, where `NNNN` is the GitHub issue
> number. This file is the working draft; on filing the issue it moves to that
> repository and gains a row plus a summary paragraph in
> `docs/source/rfcs/index.md`.

## Summary

Fold coarse tiling into the CP-SAT model that already chooses core divisions and
LX residency together, so all three partitioning decisions are made against a
single objective. The objective becomes caller-supplied: `plan_layout` and
`plan_layout_and_core_divisions` accept a sympy expression that is lowered to
CP-SAT expressions over the model's own optimization variables. The solver
enumerates every valid tiling option per operation and lets tiling *groups*
emerge from minimizing that objective rather than being pre-computed from hint
scopes.

## Motivation

torch-spyre makes three independent decisions about the same underlying
variable — how a buffer is partitioned — using three different cost models at
three points in the `CustomPreSchedulingPasses` pass list (`passes.py:416-456`):

| Decision | Where | Cost model | Blind to |
|---|---|---|---|
| Coarse tiling (loop nest / WSR) | `wsr/coarse_tile.py`, applied at `passes.py:430` (hints, pre-stickify) and `passes.py:448` (span overflow, post-stickify) | `_combo_cost` = `(total tiles, #tiled dims, max split, combo)`, first feasible wins | core division, LX |
| Core division | `work_division.py`, two pass-list entries at `passes.py:451-452` expanding to three passes (`_distribute_work` is `@_runs(cost_model_matmul_division, work_distribution)`, `:382`) | `_matmul_split_cost` (µs) for matmuls; priority heuristic otherwise | LX, tiling |
| LX residency + placement | `scratchpad/ilp_solver_ortools.py`, `passes.py:455` | CP-SAT, two-phase lexicographic: `spill_cost()` then `sum(cores)` | tiling |

Two concrete defects follow directly.

**Over-tiling.** `plan_span_overflow_tile`
(`wsr/span_overflow_hint_analysis.py:1479`) is handed `core_split_estimate = 1`,
hardcoded at both `ChunkingInfo` construction sites (`:666`, `:827`), and carries
the TODO at `:1504-1506`: *"make a common planner for Work Division and Working
Set Reduction together, so this pass can get a proper `core_split_estimate`
instead of the hardcoded 1."* It chooses a tiling as if the op ran on one core; work
division then splits the same dimensions again. The `MAX_SPAN_BYTES` constraint
is satisfied twice over and the emitted tile counts are larger than necessary,
costing loop overhead and, where tiling forces boundary copies, HBM traffic.

**Tiling never buys LX residency.** Shrinking a chain's working set so it fits
the 2 MB LX is the purpose of working-set reduction, yet the tiling planner
cannot see LX occupancy and the LX solver cannot choose a tiling.
`docs/source/compiler/scratchpad_planning.md:578` lists "**No coarse-tiling
integration** when that pass also drives split decisions" among the remaining
gaps under "Co-optimization is still limited" (`:555`). It is logged there against
`StrategyBCoOptimizingAllocator` (`:557`), but the limitation is shared verbatim
by the CP-SAT `CoOptimizingAllocator` (`:534`) that this RFC extends.

The joint pattern is already proven for two of the three axes.
`CoOptimizingAllocator` (`scratchpad/allocator.py:1424`) hands enumerated
`CoreDivision` candidates plus producer/consumer slicing-match tables to
`CpSatLayoutSolver`, which picks division and placement in one model. This RFC
adds tiling as a third axis of that same model.

A secondary motivation is that the objective is currently hardcoded. Recent work
on the `tighten-spill-cost` branch encapsulated the per-buffer spill term into
`_LifetimeBufferWithCpVars.spill_cost()`, which now returns a
`cp_model.LinearExpr` already gated on `1 - in_buffer`. That makes the objective
a single overridable hook — the natural point at which to let callers supply the
cost function instead of editing the solver.

## Background: what exists today

### Coarse tiling

`coarse_tile(graph, groups, group_idx_offset=0)` (`wsr/coarse_tile.py:1392`)
annotates `groups` as a bare `list[tuple]`; the documented contract is
`(ops, levels)` where `levels` is `[(hint_id, count), ...]` outermost-first. It
runs in two phases:

- `plan_coarse_tile_groups(operations, groups)` (`:187`) — **zero mutation**, and
  it raises `Unsupported` before any transformation rather than half-applying.
  Produces `{id(op): CoarseTileInfo}` using `_planned_tile_extents_per_level`
  (`:310`), which reads pre-mutation `op.data.ranges` / `reduction_ranges`, and
  `_tiled_dims_for_dep` (`:422`), which filters those extents by
  `dep.index.free_symbols`.
- `_apply_plan` (`:996`) — real IR mutation: `_divide_ranges` (`:846`),
  `_divide_reduction_ranges` (`:953`), layout resize via `_resize_device_layout`
  (`_inductor/ir.py:113`), then buffer propagation. That propagation inserts
  *tile-sized* copies into separately allocated full-size buffers —
  `_allocate_full_buffer` (`:1851`) then `_insert_copy_op` (`:1989`) on the write
  side, `_insert_read_copy_ops` (`:2239`) on the read side — plus reduction
  accumulation (`_insert_combine_op` (`:2639`), `_insert_reduction_copy_op`
  (`:2700`)).

Groups are produced by two mutually exclusive sources:
`hints_to_coarse_tile_groups` (`wsr/coarse_tile_hints.py:269`), which collects
contiguous runs of ops sharing a `frozenset` of hint IDs; and
`span_overflow_groups` (`wsr/coarse_tile_span_overflow.py:209`), which groups on
`_auto_span_plan_signature` (`:48`), lets a consumer *adopt* a run's split via
`can_conform_pointwise_tile` (`:448`), lets a `Reduction` join an open run via
`_reduction_shares_group_tiled_dim` (`:68`), and returns a
`(groups, dim_hint_assignments)` pair without applying the hints. The latter is
the existing "common tiling across ops" primitive, and the direct ancestor of
what this RFC generalizes.

`validate_coarse_tile_groups` (`:113`) forbids a hint scope spanning two groups;
`_validate_contiguous` (`:814`) requires a group to be a contiguous slice of
`graph.operations`.

### The CP-SAT solver

`CpSatLayoutSolver` (`scratchpad/ilp_solver_ortools.py:324`) works in alignment
units, wrapping each buffer in `_LifetimeBufferWithCpVars` (`:145`) or
`_CoreDivisionBufferWithCpVars` (`:247`). The joint wrapper creates:

```python
self.division = m.new_int_var(0, len(b.core_divisions) - 1, f"div_{b.name}")
self.eff_size = m.new_int_var(0, max(per_core), f"eff_size_{b.name}")
self.cores    = m.new_int_var(0, max(cores_used), f"occ_{b.name}")
m.add_element(self.division, per_core,   self.eff_size)
m.add_element(self.division, cores_used, self.cores)
```

Constraints: residency gated on pairwise division compatibility with every
consumer — `constrain_residency` (`:285`) loops the consumers, reads their pairs
from the precomputed `cd_parent_matches` via `match_pairs()` (`:282`), and
delegates each edge to the generic `_gate_divisions` helper (`:128`) with
`in_buffer` as the enforce literal; in-place reuse as shared-offset relaxation
(`_add_inplace_relaxation`, `:523`); and, called from that relaxation, global 2D
no-overlap over optional rectangles
`[start_time, end_time) × [offset, offset + eff_size)` with presence literal
`in_buffer` (`_add_no_overlap_2d`, `:571`).

Objective (`_run`, `:449`) is two-phase lexicographic: minimize
`sum(spill_cost)`, lock it with a rounded inequality
(`model.add(sum(hbm_terms) <= round(solver.ObjectiveValue()))`, `:482`), then
maximize `sum(cores)`. Phase 2 is skipped entirely when no buffer has a division
to choose, which is the placement-only path.

### Why the three cannot simply be reordered

The tiling decision needs exact padded byte sizes and stick counts, so it wants
to run *after* stickification. The LX solver needs the buffer set that tiling
produces, so it wants to run *after* tiling. Work division needs stick counts
and is constrained by the same span limit as tiling. Any purely sequential
ordering makes one of the three guess about another — which is precisely the
`core_split_estimate = 1` guess that exists today.

## Proposed design

### 1. Config-as-unit

Collapse tiling and division into a single per-operation **config** index.
Precomputing the feasible cross product absorbs every nonlinearity —
divisibility, stick alignment, `MAX_SPAN_BYTES`, core budget — into a table
lookup, which is exactly how `CoreDivision` is already consumed via
`AddElement`.

`CoreDivision` (`scratchpad/plan_solver.py:94`) is generalized rather than
forked. It carries just two stored fields — `output_splits` and
`reduction_splits`, both the coeff-keyed encoding from
`pass_utils.splits_by_index_coeff` — with `cores_used`, `output_partition`,
`is_clean`, and `signature_key()` all derived, so wrapping it costs nothing:

```python
@dataclass
class TileOption:
    """One candidate coarse tiling of a single op."""
    levels: tuple[tuple[int, int], ...]   # (level_id, split_count), outermost-first
    signature: Hashable                   # compatibility / grouping key

    @property
    def tile_count(self) -> int: ...      # product of split counts


@dataclass
class PartitionConfig:
    """One jointly-feasible (tiling, core division) pair for an op."""
    division: CoreDivision
    tile: TileOption | None = None        # None == untiled (today's behaviour)
    per_core_bytes: int = 0               # precomputed
    cores_used: int = 0
    tile_count: int = 1
```

`CoreDivisionBuffer.core_divisions` becomes `configs: list[PartitionConfig]`,
`chosen_division` becomes `chosen_config`, and `cd_parent_matches` becomes
`config_matches`. Today's behaviour is exactly the `tile=None` slice of the new
space, so the migration is mechanical and parity is directly checkable.

### 2. Model variables

Per buffer, extending `_CoreDivisionBufferWithCpVars`:

- `config[b] ∈ [0, |C_b|)` — replaces `division`.
- `eff_size[b]`, `cores[b]`, `tile_count[b]` — each via
  `AddElement(config, <table>, var)`, generalizing the two existing
  `add_element` calls.
- `in_buffer[b]`, `offset[b]`, `top[b]`, `merge_vars[b][parent]` — unchanged.

**Every per-config scalar reaches the model the same way.** Any function
`f(PartitionConfig) -> int` can be precomputed into a per-buffer table and bound
to a symbol by one more `AddElement`. That is the only extension mechanism the
model needs, and it is what the enumeration cost buys: a nonlinear property of a
tiling becomes a table entry rather than a constraint. It also fixes the limit —
a *scalar* derived from the chosen tile is available to the objective, but the
tile **shape** is not a decision variable. An objective term sensitive to *which*
dimension was split must fold that into a precomputed per-config number (§4).

New, at graph level:

- **`cut[i]`** — one boolean per adjacent pair in `graph.operations`.
  `cut[i] == 0` means the two ops share a loop nest.
- **`relayout[e]`** — one boolean per producer→consumer edge, set when the chosen
  configs force a restickify. A cost term only, never a feasibility gate.

Both are *determined*, not merely constrained, by the same mechanism: a
precomputed table of `(config_src, config_dst, flag)` triples installed with
`AddAllowedAssignments`. For `cut[i]` the table admits `flag == 0` only on
tiling-compatible config pairs, so an incompatible pair forces a cut; for
`relayout[e]` the table sets `flag == 1` on exactly those pairs whose physical
per-core views differ (R6.1). This is the same shape the per-edge helper
`_gate_divisions` (`:128`) already uses for divisions, widened by one column.

The two share a mechanism but **not an index set**, and conflating them would be
a modelling error: `cut[i]` is indexed over *program-order adjacency* in
`graph.operations`, because that is what a loop nest is, while `relayout[e]` is
indexed over *dataflow* producer→consumer edges, because that is what a
restickify sits on. A pair can be in both, one, or neither. Both are also
distinct from `config_matches` (§1, today's `cd_parent_matches`), which stays a
plain pair set rather than a triple table because `constrain_residency` applies
it *conditionally*, under `in_buffer`, whereas cut and relayout hold
unconditionally.

Crucially, `cut[i]` is indexed over **all** adjacent pairs in `graph.operations`,
not just tileable ones, and is **pinned to 1** at any boundary where either side
cannot be tiled. That is what makes §3's contiguity guarantee structural rather
than aspirational: an untileable op can never end up inside a cut-free run.

Boundary buffers materialized by a cut (`_allocate_full_buffer` /
`_insert_copy_op`, `_insert_read_copy_ops`, `_insert_combine_op`) enter
`_add_no_overlap_2d` as optional rectangles with presence `cut[i] ∧ in_buffer`.
The solver already models optional intervals, so this is a presence-literal
change rather than new machinery.

### 3. Tiling groups fall out of the cut variables

**A tiling group is a maximal cut-free run of operations.** This is the central
simplification, and it buys three things at once:

1. `_validate_contiguous` is satisfied *structurally*. Because `cut[i]` ranges
   over adjacent pairs of `graph.operations` and is pinned to 1 wherever either
   side is untileable (§2), a maximal cut-free run is by construction a
   contiguous slice of `graph.operations` — exactly what `_validate_contiguous`
   (`:814`) checks against the `op_to_position` map `coarse_tile` builds.
   Contiguity never has to be expressed as a constraint. An untileable op
   between two ops that would otherwise merge simply forces a cut; hoisting such
   interlopers out of the way stays `reorder_unhinted_interlopers`'s job (R4.4),
   not the solver's.
2. Grouping is driven purely by the objective, as required: the solver merges
   two ops into a group exactly when agreeing on a tiling scores better than
   paying for the boundary copies a cut would materialize.
3. A cut is never priced *directly*. It is priced through the consequences it
   materializes — the boundary copy buffers, their HBM bytes, and their LX
   occupancy — all of which are already in the model. There is deliberately no
   `n_groups` penalty term: the performance profile is read off the real
   outcomes (tile shape, LX pinning status), not off a proxy for them.

`validate_coarse_tile_groups`'s invariant (a hint scope must not split across
groups) becomes a constraint: `cut[i] == 0` is forced for every `i` interior to
a hint scope.

### 4. Injected sympy objective

A new module `torch_spyre/_inductor/scratchpad/cost_expr.py` provides:

- **A symbol namespace** the solver binds to model variables. Per-buffer:
  `size`, `read_count`, `in_lx`, `spilled`, `cores`, `tile_count`,
  `is_intermediate`. Aggregators: `SumOverBuffers`, `SumOverEdges`. Globals:
  `total_hbm_bytes`, `peak_lx_bytes`, `idle_cores`, `relayout_bytes`.
  `peak_lx_bytes` is defined as the packing high-water mark — one
  `AddMaxEquality` over the `top[b]` vars `_add_no_overlap_2d` already creates
  (`:571`) — *not* a time-indexed occupancy sum, which would cost a constraint
  per timestep and is why the naive reading of "peak" is rejected.
  `tile_count` and `in_lx` are the primitives from which a performance profile
  is derived; there is no group-count symbol.

  Anything beyond this list must arrive as a per-config scalar, precomputed into
  the §2 table and bound by one more `AddElement`. The tile *shape* is
  deliberately not a symbol (R3.8).
- **A lowering** `lower(expr, bindings) -> cp_model.LinearExpr` over an
  explicitly bounded sympy subset (R3.3). Anything outside that subset raises
  `CostExpressionError` naming the offending node. Silently approximating an
  objective is worse than a compile error.
- **A default spec** that reproduces today's objective exactly:
  `[SumOverBuffers(spill_cost), -SumOverBuffers(cores)]`.

`CostSpec` holds a *sequence* of expressions minimized lexicographically, each
locked as a hard bound before the next — generalizing the existing phase-1 lock
rather than discarding the guarantee that parallelism can never buy a spill. A
bare expression is accepted and promoted to a one-element sequence.

### 5. Pipeline placement — decide once, apply in dependency order

The solve needs device layouts, so it sits post-stickification. Manual hints keep
applying pre-stickification exactly as today, preserving the rationale spelled
out at `passes.py:419-425` (running before stickification means `_divide_ranges`
never calls `_resize_device_layout`, which is what dissolved the
`insert_restickify`→hint cross-phase contract, issue #3135). The solver sees
hint-tiled ops as **pinned single-config buffers** and optimizes the rest of the
graph around them.

```text
  propagate_named_dims, validate_named_dims          # 426-427
  assign_dim_hints                                   # 428
  _maybe_reorder_unhinted_interlopers                # 429
  _maybe_coarse_tile_hints                           # 430 — unchanged, hints
                                                     #       stay authoritative
  # --- Tensor Layout (Stickification), passes.py:433-441 ---
  split_multi_ops                                    # 433
  propagate_spyre_tensor_layouts                     # 434
  validate_ops, optimize_restickify_locations        # 435-436
  finalize_layouts                                   # 437
  insert_restickify                                  # 438
  enforce_indirect_access_layout,
  insert_post_mutation_restickify, insert_bmm_padding # 439-441
  dedup_and_promote_constants                        # 443
+ unified_partition_solve                            # NEW: one CP-SAT model
+     -> per-op config, cuts, residency intent
+ coarse_tile(graph, groups, group_idx_offset)       # apply chosen groups, at the
                                                     # 448 slot span-overflow holds
  span_reduction, _distribute_work                   # 451-452 — commit divisions
  _maybe_scratchpad_planning                         # 455 — placement-only
                                                     # re-solve, divisions fixed,
                                                     # warm-started from intent
```

`_maybe_coarse_tile_span_overflow` is subsumed: span-forced tiling becomes a
feasibility constraint on configs rather than a separate pass.

The final placement solve remains **authoritative for addresses**. The joint
model decides on a *predicted* buffer set; addresses are then computed over the
real post-tiling buffers. A misprediction therefore degrades to a spill, never
to a wrong address.

### 6. Tractability

- **Segment at matmul boundaries.** Matmul operands always come from HBM, so
  those edges are hard cuts; the model decomposes into one CP-SAT instance per
  segment, keeping each small.
- **Cap configs per op**, ranked by the existing `_combo_cost` ordering.
- **Warm-start** via `AddHint` with the current heuristic's plan. Combined with
  the existing time limit → `SolveError` → greedy fallback
  (`scratchpad/allocator.py:2159`), this yields an anytime property and a
  never-worse-than-today floor.

## Requirements

### R1 — Tiling enumeration

- **R1.1** Add `enumerate_tile_options(op, *, max_dims, max_splits_per_dim,
  max_options) -> list[TileOption]`, returning **all** feasible options within
  bounds. This is the behavioural change from `_search_min_cost_tile_plan`
  (`:1269`), which returns the *first* combo in `_combo_cost` order. Both of its
  failure modes are preserved: it *raises* `Unsupported` when no combo passes
  (`:1395`, `:1399`), and returns `None` only when there are no candidate host
  dims at all (`:1303`).
- **R1.2** `_combo_cost` is retained solely as the ranking used to truncate to
  `max_options`, and as the deterministic tie-break.
- **R1.3** The untiled option `TileOption(levels=())` is always present, so
  feasibility is never worse than today.
- **R1.4** Validity predicates must be **reused, not reimplemented**. From
  `wsr/span_overflow_hint_analysis.py`: `_within_stick_host_dim` (`:240`),
  `_post_tile_stick_alignment_error` (`:263`), `_candidate_host_dims` (`:911`),
  `_cap_split_candidates` (`:979`), `_input_stick_alignment_error` (`:1042`),
  `_split_candidates_for_host_dim` (`:1100`), `_iter_split_combos` (`:1195`),
  `_combined_tile_stick_alignment_error` (`:1215`). From `wsr/coarse_tile.py`:
  `_validate_reduction_tiling` (`:1560`), `_seed_buffer_for_carry` (`:604`,
  carry-propagating recurrences). From `pass_utils.py`: `coeff_through_floor`
  (`:848`, sub-stick guard).
- **R1.5** Derived quantities come from the existing zero-mutation planner —
  `_planned_tile_extents_per_level` (`:310`) for the extents themselves, and
  `_tiled_dims_for_dep` (`:422`) to filter them per dep. No new extent arithmetic
  is written.
- **R1.6** Every returned option must be applicable: a test applies each one via
  `_apply_plan` and asserts no `Unsupported` is raised.
- **R1.7** Existing caps are the defaults, and stay where they are defined today
  in `wsr/span_overflow_hint_analysis.py` rather than migrating to `config.py`:
  `_MAX_TILE_DIMS = 3`, `_MAX_TILE_COMBOS = 512`, `_MAX_SPLITS_PER_DIM = 16`
  (`:143-145`), `_MAX_AUTO_TILE_SPLIT_COUNT = 64` (`:149`).

### R2 — Config construction and joint feasibility

- **R2.1** `PartitionConfig` pairs a `CoreDivision` with a `TileOption | None`
  and precomputes `per_core_bytes`, `cores_used`, `tile_count`.
- **R2.2** Divisions continue to come from `enumerate_work_division_candidates`
  (`work_division.py:685`) unchanged, including all five of its guards: at most
  one reduction dim split (`:744`), no coordinate-masked dim split (`:751`),
  TOPK left unsplit (`:706`), the core budget `prod(splits) <= max_cores`
  (`:742`), and a per-core span within `MAX_SPAN_BYTES` on every tensor dep
  (`:746-750`).
- **R2.3** **Span feasibility is evaluated on the pair, not per subsystem.** A
  config is feasible iff its per-core, per-tile span is within `MAX_SPAN_BYTES`
  (`work_division.py:72`, `65535 * 4096` ≈ 256 MiB). This generalizes the
  per-core-only check R2.2 already performs at `:746-750`, and discharges the
  `core_split_estimate = 1` TODO.
- **R2.4** Configs are capped per op and deduped by signature. The pair
  (work-division seed, untiled) is always retained, so the model's feasible set
  always contains today's answer.
- **R2.5** An op with no feasible config raises `Unsupported` at the same
  pipeline point it does today.

### R3 — Injected cost function

- **R3.1** Signature change. Today (`scratchpad/plan_solver.py:261`, `:298`)
  neither ABC is keyword-only, and only one takes `log_lx_usage`:

  ```python
  # today
  def plan_layout(
      self, buffers: Sequence[LifetimeBoundBuffer], log_lx_usage: bool = False
  ) -> list[LifetimeBoundBuffer]: ...

  def plan_layout_and_core_divisions(
      self, buffers: Sequence[CoreDivisionBuffer]
  ) -> list[CoreDivisionBuffer]: ...
  ```

  ```python
  # proposed — `objective` added, trailing arguments made keyword-only
  def plan_layout(
      self,
      buffers: Sequence[LifetimeBoundBuffer],
      *,
      objective: CostSpec | sympy.Expr | None = None,
      log_lx_usage: bool = False,
  ) -> list[LifetimeBoundBuffer]: ...

  def plan_layout_and_core_divisions(
      self,
      buffers: Sequence[CoreDivisionBuffer],
      *,
      objective: CostSpec | sympy.Expr | None = None,
  ) -> list[CoreDivisionBuffer]: ...
  ```

  Introducing `*` is source-compatible: every in-tree caller already passes
  `log_lx_usage` by keyword (`allocator.py:180`, `:1296`). The four concrete
  overrides move in lockstep — `ilp_solver_ortools.py:351`, `greedy_solver.py:107`,
  `firstfit_bestfit_solver.py:187`, `simulated_annealing.py:123`.

- **R3.2** `CostSpec.objectives` is a tuple of expressions minimized
  lexicographically; each is locked with `model.Add(expr <= round(value))`
  before the next is minimized, generalizing the existing phase-1 lock
  (`ilp_solver_ortools.py:482`), which already has exactly this rounded-inequality
  form.
- **R3.3** Supported sympy subset, stated exhaustively:
  - `Add`; `Mul` with at most one non-constant factor per term (otherwise
    reified with `AddMultiplicationEquality`); `Pow` with a small non-negative
    integer exponent (expanded); `Integer` / `Rational` / `Float` coefficients.
  - `Min` / `Max` → `AddMinEquality` / `AddMaxEquality` over reified int vars.
  - `Piecewise` whose conditions are boolean model vars, reified via
    `OnlyEnforceIf`.
  - **Rejected**, with `CostExpressionError` naming the node: transcendentals
    (`log`, `exp`, `sqrt`), division by a variable, unbound free symbols,
    symbolic shapes.
- **R3.4** Rational and float coefficients are scaled to integers by a
  documented `COST_SCALE` (lcm of denominators, capped). Raise if the scaled
  coefficients would risk int64 overflow rather than silently wrapping.
- **R3.5** `objective=None` selects the default spec, which must produce a plan
  **bit-identical** to today's two-phase objective.
- **R3.6** The four placement-only solvers (`greedy`, `firstfit`, `bestfit`,
  `simulated_annealing`; registry at `allocator.py:2056-2061`) accept the
  parameter for ABC conformance, ignore a non-`None` objective, and log a warning
  once. The ABC docstring states this explicitly — the contract must not imply
  support these solvers lack. Note `LAYOUT_SOLVER` has a fifth value, `cpsat`,
  which is handled ahead of that registry (`allocator.py:2107`) and is the one
  solver for which `objective` is honoured.
- **R3.7** `SPYRE_COST_EXPR` parses a string against the exported namespace, so
  objectives can be explored without a code change.
- **R3.8** **Symbol binding is bounded.** Every symbol resolves to either an
  existing model variable or a single `AddElement` lookup over a per-config table
  computed at enumeration time. No symbol may add constraints scaling with
  anything but the buffer and edge counts — which is why `peak_lx_bytes` is the
  packing high-water mark over the existing `top[b]` vars rather than a
  time-indexed occupancy sum, and why the tile *shape* is not a symbol. A term
  needing shape sensitivity precomputes a per-config scalar instead. Adding a
  symbol that violates this is a design error, not a performance trade-off.

### R4 — Tiling groups

- **R4.1** Groups are **not** pre-computed. `cut[i]` booleans over adjacent pairs
  of `graph.operations` define them, fixed by `AddAllowedAssignments` over the
  per-edge `(config_src, config_dst, cut)` triple table, which admits `cut == 0`
  only on tiling-compatible config pairs.
- **R4.2** `cut[i]` is pinned to 1 at every boundary where either side is
  untileable. Contiguity is therefore structural and `_validate_contiguous`
  (`coarse_tile.py:814`) passes by construction — a maximal cut-free run of a
  list whose untileable positions are all cut is a contiguous slice of that
  list. A test asserts this directly rather than relying on the argument.
- **R4.3** `cut[i] == 0` is forced for every `i` interior to a hint scope,
  preserving `validate_coarse_tile_groups`'s invariant.
- **R4.4** `reorder_unhinted_interlopers` continues to run as a pre-step. The
  solver does not reorder operations.
- **R4.5** The solver emits `groups` in exactly the shape `coarse_tile()` already
  consumes — the documented `(ops, levels)` tuples, passed with the existing
  `group_idx_offset` parameter so emitted `loop_group_id`s do not collide with
  those the hint pass already stamped. No new application path is introduced.

### R5 — Hints

- **R5.1** Manual `spyre_hint` tiling applies pre-stickification exactly as
  today, and remains authoritative. Hinted ops enter the model as pinned
  single-config buffers.
- **R5.2** The solver never re-tiles or un-tiles a hinted op.
- **R5.3** Where no hint is present, the solver tiles automatically.
- **R5.4** `SPYRE_INDUCTOR_IGNORE_HINTS=1` disables hints, handing those ops to
  the solver as ordinary un-hinted ops.
- **R5.5** *Deferred (phase 2).* Growing an existing hint group with
  solver-chosen neighbours requires either moving hint application
  post-stickification or emitting nested groups with a matching
  `loop_group_id` prefix. Out of scope for phase 1; the constraint is recorded
  here so the limitation is understood, not discovered.

### R6 — Relayout avoidance

- **R6.1** `relayout[e]` is *determined* by the §2 per-edge triple table:
  `AddAllowedAssignments` over `(config[p], config[c], relayout[e])`, with the
  flag column set on exactly those config pairs whose physical per-core views
  differ. The comparison reuses `_prepare_per_core_view` (`pass_utils.py:1467`)
  and `_per_core_view_on_buf` (`:1696`), extended to account for tiling, and runs
  once per edge **at enumeration time** — never inside the solve.
- **R6.2** Relayout is a **cost term only**, never a feasibility gate — the
  compiler can always insert a restickify. Concretely: the triple table always
  admits both flag values for some config pair on every edge, so no edge can be
  made infeasible by relayout alone.
- **R6.3** `relayout_bytes` is exposed in the objective namespace as
  `SumOverEdges(relayout[e] * bytes[e])`, where `bytes[e]` is a per-edge
  constant, keeping it inside R3.8's linear-binding rule.

### R7 — Prediction fidelity and application

- **R7.1** A pure predictor maps a candidate config set to the predicted buffer
  set (sizes, lifetimes, boundary copies) with **no IR mutation**.
- **R7.2** `SPYRE_VERIFY_TILE_PREDICTION=1` applies the plan and asserts
  predicted per-buffer sizes and lifetimes match the realized ones. This is the
  highest-risk area in the design and gets its own test suite.
- **R7.3** Application order is: decide → `coarse_tile` → commit divisions →
  placement-only re-solve. Addresses always come from the final solve over real
  buffers.
- **R7.4** The placement re-solve is warm-started from the joint solve's
  residency intent. A buffer that no longer fits degrades to a spill with a
  distinct per-buffer `residency_reason` (`plan_solver.py:68`) — which surfaces
  through the solver-level `spill_reasons` map (`plan_solver.py:219`) and the
  allocator's `reject_reasons` mirror (`allocator.py:137`) — so the mispredict is
  visible rather than silent.

### R8 — Robustness, gating, determinism

- **R8.1** New gate `UNIFIED_TILING` / `config.unified_tiling`, **default off**.
  The bare `UPPER_SNAKE` form matches the LX-planning family this gate composes
  with (`LX_PLANNING`, `CO_OPTIMIZING_LX_PLANNING`, `LAYOUT_SOLVER`;
  `config.py:22-26`, `:111`), while the diagnostic flags below keep the newer
  `SPYRE_`-prefixed style of `SPYRE_INDUCTOR_*` — the split is deliberate, not
  accidental. Requires `LAYOUT_SOLVER=cpsat` and `CO_OPTIMIZING_LX_PLANNING=1`,
  the latter itself default-off today (`config.py:23-25`); warn and no-op
  otherwise.
- **R8.2** Warm-start the model via `AddHint` with the current heuristic's plan,
  so hitting the time limit yields today's answer rather than a worse one.
- **R8.3** Time limit → `SolveError` (`plan_solver.py:27`) → the existing fallback
  at `allocator.py:2159`, unchanged: it discards the configured allocator and
  re-plans with placement-only greedy, relying on the graph being unmutated when
  the solve raises.
- **R8.4** Determinism: keep `num_search_workers = 1` under
  `torch.are_deterministic_algorithms_enabled()` and `random_seed = 0`. Tiling
  adds symmetry, so add symmetry-breaking over equal-cost configs and document
  the tie-break, so plans are reproducible across runs.
- **R8.5** `ortools` remains the optional extra named `cpsat`
  (`pyproject.toml:36-38`, `ortools>=9.0`); the import stays guarded
  (`ilp_solver_ortools.py:83-89`) and `_make_cpsat_solver` (`allocator.py:2064`)
  keeps translating a missing `ortools` into the greedy fallback. `sympy` needs no
  new dependency: it is not declared directly but arrives transitively via torch
  (`sympy>=1.13.3`), and Inductor already depends on it.

### R9 — Non-goals

- **No ring transfers.** The `core_div_mismatch` hard wall stays. Dissolving it
  needs a data ring or reduce-sum ring emitted in the SuperDSC schedule, which
  is separate work.
- **No new performance model in microseconds.** The objective is caller-supplied;
  supplying a *good* one is follow-on work. This RFC delivers the mechanism.
- **No operation reordering** beyond the existing
  `reorder_unhinted_interlopers`.
- **No change** to `_matmul_split_cost` in `work_division.py`.

## Files

**New**

- `torch_spyre/_inductor/scratchpad/cost_expr.py` — symbol namespace,
  `CostSpec`, sympy→CP-SAT lowering, `CostExpressionError`.
- `torch_spyre/_inductor/wsr/enumerate_tilings.py` — `enumerate_tile_options`,
  built on the R1.4 predicates.

**Modified**

- `scratchpad/plan_solver.py` — `TileOption`, `PartitionConfig`; `CoreDivision`
  retained as a config field; the R3.1 signatures.
- `scratchpad/ilp_solver_ortools.py` — `config` replaces `division`; `cut[i]`
  and `relayout[e]`; objective driven by `CostSpec`; `_extract` writes
  `chosen_config`.
- `scratchpad/allocator.py` — `_enumerate_core_divisions` (`:1506`) becomes config
  enumeration; `_cd_parent_matches` (`:1921`) becomes `_config_matches`;
  `_commit_divisions` (`:1553`) also emits `groups` for `coarse_tile`.
- `passes.py` — insert `unified_partition_solve` and the apply step; subsume
  `_maybe_coarse_tile_span_overflow`.
- `config.py` — `unified_tiling`, `cost_expr`, `verify_tile_prediction`.
- `wsr/span_overflow_hint_analysis.py` — expose the predicates as reusable
  helpers; `_search_min_cost_tile_plan` becomes a thin ranked wrapper over the
  enumerator.

**Docs updated on landing**

`docs/source/compiler/work_division_planning.md`,
`docs/source/compiler/scratchpad_planning.md` (the "no coarse-tiling
integration" gap), `docs/source/compiler/coarse_tiling_loops.md`, and
`docs/source/rfcs/index.md` (row plus summary).

## Testing and verification

1. **Parity, gate off.** `tests/inductor/test_scratchpad_solver.py` (CP-SAT
   coverage lives in `JointDivisionSolverTests:617` and
   `TestCpSatPlacementOnly:960`), `test_scratchpad_use.py`, `test_coarse_tiling.py`,
   `test_coarse_tile_e2e.py`, `test_span_overflow_hint_analysis.py` all pass
   unchanged. Note `test_coarse_tiling.py` has no CI config yaml under
   `tests/configs/torch_spyre_tests/inductor/`, unlike its siblings, so it must be
   run explicitly rather than assumed covered.
2. **Parity, gate on with tiling disabled.** Plans must be bit-identical to
   today's CP-SAT output. This is the regression guard for the
   `CoreDivision` → `PartitionConfig` migration.
3. **Cost lowering.** Unit tests for the R3.3 accept/reject table, R3.4 scaling,
   and lexicographic locking, with a `CostExpressionError` case per rejected
   construct. Plus R3.8: every namespace symbol adds at most one `AddElement` (or
   the single `AddMaxEquality` for `peak_lx_bytes`), and model size grows
   linearly in buffer and edge count as the graph scales.
4. **Edge tables and structural contiguity.** The `cut[i]` and `relayout[e]`
   triple tables are total — every `(config_src, config_dst)` pair appears
   exactly once — `cut[i]` is pinned to 1 at every untileable boundary, and no
   edge's table admits only `relayout == 1` (R6.2). Then the property that
   §3 rests on: for any solution, every maximal cut-free run is a contiguous
   slice of `graph.operations` (R4.2), asserted directly rather than argued.
5. **Enumerator completeness.** Brute-force reference on small shapes; the
   enumerator's set must equal the reference's, and every option must apply
   without `Unsupported` (R1.6).
6. **Prediction fidelity.** Run the coarse-tiling and scratchpad suites under
   `SPYRE_VERIFY_TILE_PREDICTION=1`.
7. **Over-tiling fix.** A case where span overflow forces tiling *and* work
   division splits the same dim: assert the joint model picks a strictly smaller
   tile count than the `core_split_estimate = 1` path.
8. **End-to-end performance.** `mlp-linear-kn.t` and `mha_4h` at `SENCORES=32`,
   the two benchmarks already tracked in `scratchpad_planning.md`, measured against
   the baselines recorded there: `mlp-linear-kn.t` at ~79% process-engine
   utilization after pointwise seeding, ~17% below its pre-seeding fused kernel
   time (`:519`); `mha_4h` converging on `B/4·M/8` with the scores matrix pinned
   (`:508`), but with the reduction option pushing search into tens of seconds
   (`:566`). Report PE utilization, fused kernel time, *and* solve time against
   the gate-off baseline — the last matters because tiling enlarges the model.
9. **Determinism.** Identical plans across two runs under
   `torch.use_deterministic_algorithms(True)`.

## Alternatives considered

**A separate CP-SAT tiling stage ahead of layout planning.** Cleaner to land and
test, but it reproduces the current defect in a new place: a tiling chosen
without seeing LX occupancy or the core division still has to guess. Rejected in
favour of one joint model.

**Keeping the hardcoded two-phase objective and adding tiling terms to it.**
Requires editing the solver for every cost experiment, and the interesting
question — how to trade HBM traffic against parallelism against loop overhead —
is exactly the one that needs iteration. Rejected in favour of injection.

**Pre-computing tiling groups from producer/consumer connectivity, then having
CP-SAT pick one tiling per group.** A smaller model, but the grouping heuristic
becomes a second place where a wrong guess is unrecoverable, and grouping is
precisely what the objective should decide. Rejected in favour of cut variables.

**Modelling tiling and division as independent variables** rather than a
precomputed config cross product. Keeps the model smaller in variable count, but
reintroduces the products and divisibility conditions as nonlinear constraints.
The config encoding absorbs them into `AddElement` table lookups at the cost of
enumeration, which is bounded by the caps in R1.7.

## Open questions

- **Segmentation granularity.** One CP-SAT instance per matmul-bounded segment
  is proposed for tractability. Is per-segment optimality acceptable, or is a
  whole-graph model wanted despite the size?
- **How cuts should be priced.** §3 deliberately exposes no group-count symbol: a
  cut is priced only through the consequences it materializes. The alternatives
  are a derived `n_groups = 1 + sum(cut)` term, or a direct per-cut byte cost. Is
  pricing purely through consequences the right stance, or does loop overhead need
  a term of its own?
- **Config cap per op.** What ceiling is acceptable before the model gets too
  large, and should it adapt to graph size?
- **Default objective after phase 1.** The default reproduces today's behaviour
  for safety. What should it become once the mechanism is trusted?
