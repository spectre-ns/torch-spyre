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
enumerates every valid output-range tiling option per operation and lets tiling
*groups* emerge from minimizing that objective rather than being pre-computed
from hint scopes. Reduction-axis tiling stays hint-only and out of the solver's
decision space (R9).

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

The channel through which tiling would buy residency is specific, and naming it
sharpens what the joint model is for. Residency accrues to a cut-free run's
**interior** — the per-tile scratch that never materializes and the read-side
tile copies, both tile-sized and fixed-address, hence LX-eligible. It never
accrues to anything crossing a group boundary, which is HBM-only by construction
(*Background*). So the decision the objective must make is not "tile more" but
"where to place cuts such that the buffers left in the interior are the ones
worth pinning" — a question neither of today's two planners can even pose.

The joint pattern is already proven for two of the three axes.
`CoOptimizingAllocator` (`scratchpad/allocator.py:1476`) hands enumerated
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

`coarse_tile(graph, groups, group_idx_offset=0)` (`wsr/coarse_tile.py:1072`)
annotates `groups` as a bare `list[tuple]`; the documented contract is
`(ops, levels)` where `levels` is `[(hint_id, count), ...]` outermost-first. It
runs in two phases:

- `plan_coarse_tile_groups(operations, groups)` (`:186`) — **zero mutation**, and
  it raises `Unsupported` before any transformation rather than half-applying.
  Produces `{id(op): CoarseTileInfo}` using `_planned_tile_extents_per_level`
  (`:309`), which reads pre-mutation `op.data.ranges` / `reduction_ranges`, and
  `_tiled_dims_for_dep` (`:421`), which filters those extents by
  `dep.index.free_symbols`.
- `_apply_plan` (`:967`) — real IR mutation: `_divide_ranges` (`:817`),
  `_divide_reduction_ranges` (`:924`), layout resize via `_resize_device_layout`
  (`_inductor/ir.py:112`), then buffer propagation. That propagation inserts
  *tile-sized* copies into separately allocated full-size buffers —
  `_allocate_full_buffer` (`:1519`) then `_insert_copy_op` (`:1657`) on the write
  side, `_insert_read_copy_ops` (`:1907`) on the read side — plus reduction
  accumulation (`_insert_combine_op` (`:2307`), `_insert_reduction_copy_op`
  (`:2404`)).

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

`validate_coarse_tile_groups` (`:112`) forbids a hint scope spanning two groups;
`_validate_contiguous` (`:785`) requires a group to be a contiguous slice of
`graph.operations`.

### What a group boundary materializes

Buffer propagation is **per buffer at the boundary**, not per group.
`_propagate_tiled_op` (`:1322`) asks `_find_outside_consumers` whether anything
reads the buffer from a different outer `loop_group_id`, or whether it is a graph
output, and branches:

- **Neither** — the buffer is per-tile scratch reused every iteration. Its
  `output_tiled_dims` is set to `[]` and **no full-size buffer is allocated at
  all**. Values interior to a cut-free run never round-trip.
- **Either** — `_allocate_full_buffer` (`:1519`) splices a `SpyreEmptyFallback`
  into `graph.operations` *before the first op of the loop group*,
  `_insert_copy_op` (`:1657`) appends a copy op carrying
  `MutationLayoutSHOULDREMOVE(full_buf)` after the tiled op, and outside
  consumers are patched to read the full buffer. Symmetrically on the read side,
  `_full_buffer_read_deps` (`:1425`) / `_insert_read_copy_ops` (`:1907`) insert a
  tile-sized copy before a consumer that reads a buffer produced outside its own
  group.

The LX consequences are asymmetric, and they are what give a cut its price. A
tiled → tiled cut expands one producer → consumer edge into a four-node chain,
and each node lands differently:

```text
  ╭─ group A loop body ─────────╮                ╭─ group B loop body ────╮
  │ tile scratch ──▶ copy op    ├──▶ full_buf ──▶┤ read copy ──▶ consumer │
  │ Eligible         HBM only   │    HBM only    │ Eligible               │
  │ tile alloc       no storage │    full alloc  │ tile alloc             │
  ╰─────────────────────────────╯                ╰────────────────────────╯
```

`full_buf` is drawn outside both boxes because it is in neither loop body:
`_allocate_full_buffer` splices it in *before the first op of the loop group*,
and only the copy op — which does run in group A's body — touches it on the
write side. Everything inside a box executes once per tile; `full_buf` is
allocated once and is the sole node that outlives an iteration.

Note the write side is **two** nodes and the read side is **one**. The
write-side copy owns no storage — it aliases into a separately allocated
`full_buf` — whereas the read-side copy allocates the tile buffer it writes.
The two write-side rows below are therefore one relationship read from both
ends: the copy op points at `full_buf`; `full_buf` never points back.

| Node | Where | LX status | Why |
|---|---|---|---|
| Interior per-tile scratch | inside a group, no cut | **Eligible** | its own write address is fixed. `_is_tiled_advancing` (`scratchpad/utils.py:218`) keys on `output_tiled_dims` — *does this op's own write advance* — not `loop_tiled_dims`, which only says the op sits in a tiled loop. Sitting in the loop is not itself disqualifying |
| Write-side copy op (`coarse_tile_copy_*`) | producer, in the loop body | **HBM only** | it owns no storage: its layout **is** `MutationLayoutSHOULDREMOVE(full_buf)`, so its stores land in `full_buf`. `_op_output_good_for_lx_reuse` rejects that layout outright (`allocator.py:218`) — there is no allocation to place |
| Boundary `full_buf` | spans the cut | **HBM only** | stamped `"op not allowed"` (`allocator.py:308`): its producer is a `SpyreEmptyFallback`, i.e. an `ExternKernel`, and `_op_output_good_for_lx_reuse` requires a `ComputedBuffer`. Two later gates would reject it anyway, but never run — it is the mutation *target*, so its name is in `mutated_buffers` → `"mutation target"` (`:315`); and in the tiled → tiled case group B's read of it advances → `"tiled (advancing)"` via `_is_read_advancing_anywhere` (`:324`) |
| Read-side tile copy | consumer, in the loop body | **Eligible** | unlike the write-side copy this one *owns* its buffer — a physically smaller allocation with fresh contiguous tile-local strides, not an aliased view of `full_buf`. Its own write is fixed |

The interior scratch survives for a reason worth stating explicitly, because it
is not automatic: the write-side copy op derives its read and write tiled-dim
decisions *separately* (`_fixed_level_extents` vs. real per-level extents). Its
read of the scratch is deliberately non-advancing — the scratch is reused in
place — so `_is_read_advancing_anywhere`, which walks a buffer's *readers*, does
not flag it. Had the copy's read advanced, the scratch would be HBM too and
tiling would buy no residency at all.

Two facts follow that the design sections depend on. First, **a boundary buffer
never occupies LX**, so a cut has no LX-occupancy term to price (§3) — though a
cut is not otherwise free on the read side: where the consumer is tiled, the
read copy's advancing read evicts an untiled producer from LX candidacy (the
row-3 discussion below). Second, **tiling buys residency through the
interior**: what becomes pinnable is the loop-internal scratch and the read-side
tile copies, both tile-sized and fixed-address. `_is_tiled_advancing`'s docstring
states the rule directly — "a loop-internal buffer (e.g. drained by a copy op
every iteration) can be tiled yet have its own write pinned at a fixed address;
such a buffer is LX-eligible."

Both the full buffer and the copy ops are **inserted into `graph.operations`**,
not merely allocated. Liveness is index-based (`calculate_liveness`), so each
insertion shifts lifetime ticks for everything downstream of it (R7.1).

#### Not every cut materializes a buffer

`_apply_plan`'s propagation loop (`:1158-1164`) iterates
`for group_ops, _ in groups` — it visits **only ops inside a group**. An untiled
producer is therefore never rewritten: it gets no `full_buf`, acquires no
`MutationLayoutSHOULDREMOVE`, and keeps whatever LX eligibility it had. Only the
tiled consumer's own `_full_buffer_read_deps` / `_insert_read_copy_ops` fires.
What a cut costs therefore depends on which side carries a tile:

| Cut | Write side | Read side | Producer LX |
|---|---|---|---|
| tiled → tiled | `full_buf` + mutation copy op | tile-sized read copy | HBM only |
| tiled → untiled | `full_buf` + mutation copy op | consumer reads `full_buf` directly | HBM only |
| untiled → tiled | **nothing** — producer never visited | tile-sized read copy | **evicted** — advancing read (below) |
| untiled → untiled | nothing | nothing | unchanged |

Only the first two rows allocate. The third is the most common in practice — a
forced cut whose *producer* side is untileable (§2) is a row 3 or row 4, since
an untileable op is by definition never in a group. The converse is row 2: a
tiled chain feeding an untileable consumer still materializes `full_buf` plus
the mutation copy.

Row 3 allocates nothing on the write side, but it is not free. An untiled →
tiled cut requires a copy of a tile into the narrowed-span op: the read copy
reads the *full* producer buffer, advancing one tile per iteration
(`_insert_read_copy_ops` builds real per-level read extents,
`coarse_tile.py:2169-2213`, and says so at `:2131-2137`). That advancing read
is exactly what `_is_read_advancing_anywhere` walks, so the allocator stamps
the producer `"tiled (advancing)"` (`allocator.py:324`) — the same rejection
the first table applies to `full_buf`. The producer keeps its layout and is
never rewritten, but it loses LX candidacy. The exception is a read invariant
along every tiled dim of the consumer (`_tiled_dims_for_dep` filters it to
nothing): such a read stays fixed and the producer's eligibility survives.
Row 3's price is therefore one tile-sized copy plus, in the advancing case,
the producer's residency.

### Reduction-axis tiling

Coarse tiling can divide a *reduction* range, not only an output range.
`_divide_reduction_ranges` (`wsr/coarse_tile.py:924`) shrinks `K` to `K/T`,
leaving the output ranges untouched — the reduction axis is tiled in the
iteration space, and no buffer changes shape. Each iteration therefore writes a
full-shaped **partial** result, and `_propagate_tiled_reduction_op` (`:2501`)
folds those partials: a full-size HBM accumulator (`_allocate_full_buffer`,
`:1519`) seeded with `_reduction_identity_value` (`:764`), then a per-iteration
`_insert_combine_op` (`:2307`) mutating it in place through the reduction's
monoid operator, plus — in the nested case — a second, tile-sized accumulator
drained outward by `_insert_reduction_copy_op` (`:2404`).

It is gated by `enable_reduction_tiling` (`config.py:82`, default on) and is
reachable only through `spyre_hint`; the automatic planner never emits it
(`SpanOverflowTileLevel.is_reduction` is hardcoded `False`,
`wsr/span_overflow_hint_analysis.py:85-99`, because reduction-range tiling
"would require partial-result accumulation").

Four properties of that machinery bear directly on this RFC, and together they
are why R9 places reduction-axis tiling out of scope for phase 1:

- **The accumulator is loop-carried.** Tile `t` reads what tile `t-1` wrote, so
  the op cannot share a loop nest with peers that tile at the same level.
  `_plan_is_loop_invariant_at_reduction_levels` (`:559`) admits only peers that
  are loop-invariant at every level some group member tiles a reduction dim at
  (`_group_reduction_tiled_levels_in_group`, `:485`), and `_seed_buffer_for_carry`
  (`:575`) rejects carry-propagating recurrences outright.
- **That invariant is not pairwise**, so §3's cut model cannot express it
  (R4.6).
- **It is not a pure working-set reduction.** The input span shrinks by the tile
  count, but an extra output-shaped HBM buffer appears and is
  read-modify-written once per tile — a trade whose sign depends on the
  `K`-to-output ratio, not a strict win.
- **`_validate_reduction_tiling` (`:1233`) over-approves.** Its docstring lists
  nested "outer output dims + innermost reduction dim (e.g. outer M + inner K
  for mm)" as supported, but every e2e test of that shape is either
  `correctness=False` ("nested tiling + reduction correctness bug") or
  `@pytest.mark.skip` ("inconsistent loop_count across reduction fill/combine
  nodes") in `tests/inductor/test_coarse_tile_e2e.py`. Only *single-level*
  reduction-axis tiling is numerically validated today. A predicate that admits
  known-wrong plans cannot serve as an enumerator's feasibility gate (R1.4).

### The CP-SAT solver

`CpSatLayoutSolver` (`scratchpad/ilp_solver_ortools.py:321`) works in alignment
units, wrapping each buffer in `_LifetimeBufferWithCpVars` (`:149`) or
`_CoreDivisionBufferWithCpVars` (`:244`). The joint wrapper creates:

```python
self.division = m.new_int_var(0, len(b.core_divisions) - 1, f"div_{b.name}")
self.eff_size = m.new_int_var(0, max(per_core), f"eff_size_{b.name}")
self.cores    = m.new_int_var(0, max(cores_used), f"occ_{b.name}")
m.add_element(self.division, per_core,   self.eff_size)
m.add_element(self.division, cores_used, self.cores)
```

Constraints: residency gated on pairwise division compatibility with every
consumer — `constrain_residency` (`:282`) loops the consumers, reads their pairs
from the precomputed `cd_parent_matches` via `match_pairs()` (`:279`), and
delegates each edge to the generic `_gate_divisions` helper (`:132`) with
`in_buffer` as the enforce literal; in-place reuse as shared-offset relaxation
(`_add_inplace_relaxation`, `:520`); and, called from that relaxation, global 2D
no-overlap over optional rectangles
`[start_time, end_time) × [offset, offset + eff_size)` with presence literal
`in_buffer` (`_add_no_overlap_2d`, `:568`).

Objective (`_run`, `:446`) is two-phase lexicographic: minimize
`sum(spill_cost)`, lock it with a rounded inequality
(`model.add(sum(hbm_terms) <= round(solver.ObjectiveValue()))`, `:479`), then
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
@dataclass(frozen=True)
class TileOption:
    """One candidate coarse tiling of a single op, in that op's own frame."""
    dims: tuple[tuple[int, int], ...]     # (host_dim, split_count), outermost-first
    dedup_key: Hashable                   # R2.4 dedup only — not a compatibility key

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

Two things about `TileOption` are load-bearing and easy to get wrong.

**`host_dim` is op-local; a level id is not.** `host_dim` indexes
`op_out_coords` (`pass_utils.py:363`) — the same frame
`SpanOverflowTileLevel.selected_host_dim` already uses, and the frame
`_candidate_host_dims`, `_split_candidates_for_host_dim`,
`can_conform_pointwise_tile` and `_dims_to_hints` all speak. It deliberately does
**not** carry a level or hint id. `coarse_tile`'s `levels` are keyed by
`hint_id`, which is a *group-scoped* identity: each member resolves a hint_id to
its own dimension through `op.dim_hints` → `loop_var` →
`_loop_var_to_ranges_pos` (`coarse_tile.py:729`). Under §3 a group does not exist
until the solve finishes, so a hint id cannot be assigned to a per-op option
enumerated before it. `span_overflow_groups` already sequences this correctly —
op-local signatures during grouping, then `next_hint_id` allocated only when a
group closes (`coarse_tile_span_overflow.py:303-310`, `:543-544`), then
`_dims_to_hints` (`:152`) per member, each op resolving its own `loop_var` from
its own output coordinates. The solver mints ids the same way, post-solve (R4.5).

**Compatibility is a relation, not a key.** `dedup_key` serves R2.4 and nothing
else. Two ops in one group routinely tile *different* host dims — the producer
its V output dim, the consumer the corresponding N dim — so equality of any
per-op key is the wrong test; `span_overflow_groups` matches split counts and
then verifies loop-variable correspondence through the dep. And
`can_conform_pointwise_tile` (`span_overflow_hint_analysis.py:1421`) is
asymmetric and non-equality: it asks whether one op can *adopt* another's
`split_by_host_dim`, checking divisibility, stick boundaries, and *sufficiency*
(the adopted split must fully cover the adopting op's own span pressure). §2's
`(tile_src, tile_dst, cut)` triple table holds a pairwise predicate natively,
so this costs nothing to express — see R4.7.

`CoreDivisionBuffer.core_divisions` becomes `configs: list[PartitionConfig]`,
`chosen_division` becomes `chosen_config`, and `cd_parent_matches` becomes
`config_matches`. Today's behaviour is exactly the `tile=None` slice of the new
space, so parity is directly checkable.

The first two renames are mechanical; **`config_matches` is not** (R2.6).
`_cd_parent_matches` (`allocator.py:1973`) does not compare coeff-keyed
signatures — it compares *physical per-core views* built from the producer's
`write_dep.index` and the consumer's `read_dep.index` through
`_prepare_per_core_view` / `_per_core_view_on_buf`, deliberately, because (per
its own docstring) that is "correct across reductions/reshapes, where a
coeff-keyed signature would conflate axes." Those indices and device layouts are
precisely what `_divide_ranges` and `_resize_device_layout` rewrite. So a
match entry is a property of the *config* pair, not the division pair, and it
cannot be read off the IR the solver sees.

Note also what the cross product is not: a filter. Divisions come from
`enumerate_work_division_candidates`, whose per-dim factors are
`divisors(basis)` over the op's *iteration space* — which tiling has already
divided. Tiling both **loosens** the span guard (`get_per_core_span` is
evaluated on the tiled space, so smaller tiles admit divisions that were
infeasible before — the R2.3 direction) and **tightens** divisibility
(`divisors(M/T)` is smaller than `divisors(M)`, and `adjust_it_space_for_sticks`
shifts the stick basis too). Each tiling option therefore has its own division
candidate set, enumerated against its own iteration space.

### 2. Model variables

The per-buffer vars live on a new wrapper, `_TilingBufferWithCpVars`, extending
`_CoreDivisionBufferWithCpVars` — the same subclass-and-override step that class
made over `_LifetimeBufferWithCpVars`. This is deliberate, not incidental: the
base wrapper exists so "one object flows through the solve instead of a buffer
list shadowed by a parallel `name -> {var}` dict" (`ilp_solver_ortools.py:150`),
and the tiling vars are exactly the kind of per-buffer state that would otherwise
accrete into such a dict. New vars are created in `__post_init__` after
`super().__post_init__()`, and the constraints tying them together are added by
the hooks, as today.

The candidate space is **two-level**, mirroring how §1 enumerates it — tiling
first, then that tiling's own division set:

- `tile[b] ∈ [0, |T_b|)` — which tile option. Slot 0 is reserved for the **unity**
  option (no dimension split), which every op has.
- `div[b] ∈ [0, |D_b(tile)|)` — which core division, *within* the chosen tiling.
- `config[b]` — the flat pair index, tied by a single
  `AddAllowedAssignments([tile[b], div[b], config[b]], <valid triples>)`. The
  division sets are ragged (a tiling both loosens the span guard and tightens
  divisibility, §1), so an allowed-assignment table is what expresses that
  exactly; a rectangular `tile × D_max` encoding would admit pairs that do not
  exist.
- `eff_size[b]`, `cores[b]`, `tile_count[b]` — each via
  `AddElement(config, <table>, var)`, generalizing the two existing
  `add_element` calls.
- `in_buffer[b]`, `offset[b]`, `top[b]`, `merge_vars[b][parent]` — unchanged.
- `cut_parents[b]`/`cut_children[b]`, `escapes[b]`, `boundary_op[b]`,
  `full_size[b]`, `boundary_view[b]` — new; §2 defines them.

Keeping tiling as its own level is what makes `tiled[b]` free:

```text
tiled[b]  ⟺  tile[b] != 0
```

No 0/1 table and no `AddElement` — being tiled is *not selecting the unity
option*, read straight off the variable. A flat `config[b]` would have had to
recover the same fact through a lookup, having just discarded it.

It also shrinks the cut tables. Whether two ops can share a loop nest is a
property of their **tilings**, not their divisions — `can_conform_pointwise_tile`
takes `(op, split_by_host_dim, sencores)`, where `sencores` is the machine's core
count, not either op's chosen division (R4.7). So the per-edge table is keyed on
`(tile_src, tile_dst, cut)`, `|T_src| × |T_dst|` rows rather than
`|T_src|·|D_src| × |T_dst|·|D_dst|`. Division compatibility stays where it
already lives, on the slicing gate (`config_matches`), which is conditional under
`in_buffer` rather than unconditional like cut.

`cut[i]` itself stays at graph level. It is indexed over *program-order
adjacency*, not dataflow, so the two ops it joins need not be related to any one
buffer — and its feasible values come from the `(tile_src, tile_dst, cut)`
triple for that op pair. A multi-output op compounds this: one edge, several
buffers. It therefore has no per-buffer home.

What each wrapper holds is a **neighbour view**, split by direction:
`cut_parents[b]` and `cut_children[b]`, built once in `__post_init__`. Edge `i`
joins ops `i` and `i+1`; for each edge `b` spans, the wrapper mints one bool
and indexes it twice — under the edge's upstream op (its *parent*, `i`) in
`cut_parents[b]`, and under its downstream op (its *child*, `i+1`) in
`cut_children[b]`. The split is what makes each claim's direction explicit: an
op name alone does not identify an edge end — an op interior to `b`'s span is
the child of the edge entering it and the parent of the edge leaving it — and
R4.7's admitting predicate is directional, so which end a claim refers to must
be carried by the dict identity, not by convention. Both dicts mirror
`merge_vars` (neighbour name → var) and `cd_parent_matches` (neighbour name →
table), so the boundary machinery indexes the way the rest of the wrapper
already does.

Each wrapper mints its **own** bools, exactly as `merge_vars` does — the wrapper
stays self-contained and depends on nothing outside itself. Where two buffers
span the same edge, their bools are tied by an **equality**, so the duplicates
are duplicates in name only.

Building the dicts costs nothing. The wrapper is already a `LifetimeBoundBuffer`,
whose `uses` is the sorted list of op indices at which the buffer is accessed,
with `start_time`/`end_time` derived from it (`plan_solver.py:56`) and already
serving as the time axis of `_add_no_overlap_2d`. The edges spanned by `b` are
`[start_time, end_time - 1)` — producer up to last use, excluding the edge
after the last use, which no consumer crosses (`end_time` is `uses[-1] + 1`).

**This structure depends on the op order being fixed** (R4.4). Because the
solver never reorders, program-order adjacency is static and fully known when the
wrappers are constructed, so the topology can be baked in up front. Were
reordering a solver decision, adjacency would itself be variable and the claim
dicts could not be built up front at all.

Buffers spanning a shared edge are tied by one equality per claim, posted by an
`_add_cut_equalities(model, tensors)` sweep in `_run`, alongside the existing
`_add_inplace_relaxation` / `_add_core_division` steps. No new type: the
per-edge tables and pins ride on the buffer the same way `core_divisions` and
`cd_parent_matches` already do, each wrapper installs its own
`AddAllowedAssignments` and pins from them, and the sweep only reconciles
claims. CP-SAT presolve substitutes equality-linked bools away, so the solved
model is the size it would have been with one shared var.

Tying by equality rather than by sharing one variable is what makes the
invariant **checkable**: the sweep sees every claim on every edge, so "all
claimants agree" is an assertion it can make. With claims direction-indexed,
orientation is assertable too: every claim knows whether its key op is the
edge's parent or its child, so the sweep can require all claimants of an edge
to agree on which op is which — the same producer-then-consumer orientation
R4.7's predicate is evaluated in and the `(tile_src, tile_dst, cut)` table is
built in. A claim keyed the wrong way round surfaces as a reconciliation
failure instead of silently tying the wrong pair. A design where wrappers alias a
single var has nothing to assert — the object is either the right one or it
silently is not.

Reading the plan back works the same way. R4.5 needs `groups` in `coarse_tile`'s
`(ops, levels)` shape, which means turning solved cuts into contiguous runs; the
equalities guarantee every claimant of an edge reports the same value, so
`_extract` can walk the wrappers in op order and cut where any claim is 1.

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

`cut[i]` is *determined*, not merely constrained: a precomputed table of
`(tile_src, tile_dst, cut)` triples installed with `AddAllowedAssignments`
admits `cut == 0` only on tiling-compatible config pairs, so an incompatible
pair forces a cut. This is the same shape the per-edge helper `_gate_divisions`
(`:132`) already uses for divisions, widened by one column.

`cut[i]` is indexed over *program-order adjacency* in `graph.operations`, because
that is what a loop nest is. It is distinct from `config_matches` (§1, today's
`cd_parent_matches`), which stays a plain pair set rather than a triple table
because `constrain_residency` applies it *conditionally*, under `in_buffer`,
whereas cut holds unconditionally.

Stickification/relayout optimization — choosing configs to *avoid* a restickify,
which would need a second per-edge `relayout[e]` variable over *dataflow*
producer→consumer edges — is **out of scope** for this RFC (R6, R9).

Crucially, `cut[i]` is indexed over **all** adjacent pairs in `graph.operations`,
not just tileable ones, and is **pinned to 1** at any boundary where either side
cannot be tiled — or was tiled on a reduction axis by the hint pass (R4.6). That
is what makes §3's contiguity guarantee structural rather than aspirational: an
untileable op can never end up inside a cut-free run.

The reduction-axis case is pinned for a sharper reason than untileability. The
rule governing such a group quantifies over the whole run
(`_plan_is_loop_invariant_at_reduction_levels`), and pairwise compatibility
provably does not compose into it, so no widening of the triple table would make
it expressible. R4.6 carries the counterexample; R1.8 keeps reduction-axis
options out of the enumerated space so the situation arises only from hints.

#### A cut adds a buffer, it does not evict its tiled producer

A cut does not evict its tiled producer. Both branches of `_propagate_tiled_op` end
with `output_tiled_dims = []`, and the second says so outright
(`coarse_tile.py:1367`): *"The tiled op's own buffer is always loop-internal
scratch here: it is fully drained by the copy op inserted above before the next
iteration overwrites it."* So `b` keeps its tile-sized layout, stays out of
`mutated_buffers` (`full_buf` is the mutation target, not `b`), is not
tiled-advancing, and is not read-advancing — the copy op's read of it is fixed.
`b` is LX-eligible either way. What a cut changes is **how many tile-sized
eligible buffers exist**:

| Solver's choice | Eligible allocations | Which |
|---|---|---|
| no cut | **1** | `b`, read in-group |
| a cut | **2** | `b` (still scratch) **+** the read copy in the consuming group |
| `in_buffer[b] == 0` | **0** | — |

The second buffer is the point. `b` is never rewritten into `full_buf`;
`_allocate_full_buffer` mints a *separate* HBM buffer, `_insert_copy_op` drains
`b` into it, and `_insert_read_copy_ops` allocates a fresh tile-sized buffer in
the consuming group. Only that last one is new to the packing model.

In the applied IR the two no longer meet: outside consumers are patched to
read `full_buf`, so `b`'s applied lifetime ends at its last in-group reader —
the write-side copy op inserted immediately after the tiled op, or a later
in-group consumer — while the read copy lives in the consuming group. The
**model** cannot claim that ordering: it runs pre-mutation (§5), so `b`'s
interval is derived from pre-mutation `uses`, whose last entry is still the
outside consumer — on the model's time axis the two rectangles overlap at the
consumer's tick. Phase 1 keeps that overlap as deliberate conservatism: `b`
retains its full pre-mutation extent whatever the cut variables say, the read
copy is an additional optional rectangle, and no stacking is assumed. The
error direction is safe — a cut's LX footprint is over-, never under-stated —
and it biases the solver toward fewer cuts, never toward a wrong address; the
placement re-solve prices the real, applied intervals. Expressing the trade
exactly (a cut-conditional interval end, e.g. a complementary
optional-rectangle pair sharing one offset var) is deferred until this
pessimism is shown to matter.

So the model needs no residency constraint tying `in_buffer[b]` to `cut[i]`. What
it does need is for the second rectangle's **existence** to be conditional, since
it exists only when the cut does:

- **`tiled[b]`** — bool, `tile[b] != 0` (§ *Model variables*): the chosen tiling
  is not the unity option. Gates what materializes on each side of a cut — a
  tiled producer yields a `full_buf` and a copy op, an untiled one yields neither
  (`_apply_plan`'s propagation loop visits only ops *inside* a group).
- **`cut_parents[b]` / `cut_children[b]`** — the direction-indexed claim dicts
  over the edges `b` spans (producer to last use): one minted bool per spanned
  edge, keyed by the edge's upstream (parent) op in the first and by its
  downstream (child) op in the second. Buffers sharing an edge are tied by
  equality, not by aliasing one var.
- **`escapes[b]`** — bool. Does any consumer land outside `b`'s own run? Cut-free
  runs are contiguous, so that is exactly "some spanned edge is cut" — an OR
  over the spanned-edge bools (either dict enumerates them exactly once), with
  no per-consumer bookkeeping.
- **`boundary_op[b]`** — bool. Does a `full_buf` get allocated for `b`? An
  existence flag, not a residency gate: the buffer it names is real IR that
  `coarse_tile` inserts, which shifts every downstream liveness tick (R7.1).

```text
escapes[b]      ⟺  OR(cut_children[b].values())  ∨  b is a graph output
boundary_op[b]  ⟺  tiled[b] ∧ escapes[b]
```

The graph-output disjunct is not redundant with the existing
`"graph output (no clone)"` gate. Under `clone_at_graph_boundaries()` a graph
output *may* reside (`allocator.py:334-342`), but `_find_outside_consumers`
treats a graph output as escaping, so a **tiled** graph output still materializes
a `full_buf` even with no consumer outside its run.

#### Read copies are conditional rectangles

`_insert_read_copy_ops` allocates one tile-sized buffer per full-extent input a
tiled op reads, deduped by buffer name. These are ordinary allocations with their
own tile-local strides — LX-eligible, and exactly the buffers tiling exists to
make resident. They enter `_add_no_overlap_2d` as real rectangles.

But they **do not exist until the solver decides they do**, and in two different
ways, per `_full_buffer_read_deps`:

- reading a cross-group producer — exists only if a cut separates them;
  conditional on `escapes[producer]`;
- reading a graph input, constant, or untiled producer — exists whenever the
  consumer is tiled; conditional on `tiled[consumer]`.

Neither is unconditional, so neither rectangle's presence is `in_buffer` alone.
Treating them as always-present over-reserves LX in the cut-free case; omitting
them under-reserves, with a known direction — the model would **undervalue
tiling**, since the omitted buffers are precisely the ones tiling makes pinnable
(R7.1). Both rectangles are therefore optional, with the literal above as the
presence condition and `in_buffer` gating residency on top of it.

The second bullet carries an eviction side too. When the full-extent input is
an untiled `ComputedBuffer` producer and the consumer's dep advances along a
tiled dim, the realized read copy evicts that producer (*Background*, row 3).
The model must say so: the read copy's existence literal enforces
`in_buffer[producer] == 0` on such edges. Advancement is a property of the
*(edge, consumer tile option)* pair, not of the edge alone — a dep touching
host dim `N` is invariant under an option that tiles only `M` and advancing
under one that tiles `N`, and mixed cases are common (broadcast inputs above
all). The implication is therefore keyed per pair: precomputed at
table-construction time by evaluating `_tiled_dims_for_dep` under each
option's per-level extents, and enforced only under the consumer tile options
whose tiled dims intersect the dep's free symbols. Options that leave the dep
invariant omit it. This is a narrower rule than the general eviction
constraint R4.8 forbids — it fires only on row-3 edges, under only the tile
options that realize them, from the mechanism the allocator will actually
apply.

Who creates what is split cleanly between solve and apply. The wrappers and
tables for these predicted rectangles are built by model construction from the
enumerated configs — pure prediction, no IR. The buffers themselves are
created only by the apply step (§5), which reads the solver's declared per-op
config and cut assignment and runs `coarse_tile` as today. Predicted
rectangles take **interstitial time coordinates**: the tick axis is scaled by
a small constant so inserted-op positions land between the integer ticks of
real ops — the write copy just after its producer, the read copy just before
its consumer — and a predicted insertion never renumbers a real buffer's
lifetime. R7.5's index-shifting concern thereby applies to realized IR only,
where `calculate_liveness` recomputes ticks before the placement re-solve;
R7.2's fidelity check compares predicted against realized lifetimes after
normalizing both to rank order — as equality for buffers no cut touches, and
as containment (predicted ⊇ realized) for a cut producer, whose model
rectangle deliberately keeps the pre-mutation extent while its applied
lifetime ends at its last in-group reader.

#### Boundary shape and core division

`boundary_op[b]` says a full buffer appears; its shape and slicing are
deterministic functions of the producer's chosen config, since
`_allocate_full_buffer` derives the full buffer's device layout by scaling the
per-tile one up with `_resize_device_layout`. Both are table entries, bound
exactly as `eff_size[b]` and `cores[b]` are:

- **`full_size[b]`** — `AddElement(config[b], <full-extent footprint>, ...)`.
  This is *not* `eff_size[b] × tile_count[b]`: the full buffer is stickified at
  the full host extent, so per-tile padding to stick boundaries does not survive
  the scale-up. At `eps = 64`, a stick dim of 320 split 4 ways is `4 × ceil(80/64)
  = 512` elements per-tile but `ceil(320/64) = 320` as one buffer.
- **`boundary_view[b]`** — the full buffer's per-core view, from the same table
  mechanism, so agreement against the consumer's read is expressible. Without it
  the model cannot distinguish two orthogonal producer/consumer divisions that
  happen to share a slice count from a genuine match — the conflation that has
  already produced one wrong-output bug in the co-optimizing path (R2.6).

**Pricing is out of scope for this RFC.** What a cut *costs* — the HBM traffic
through `full_buf`, the loop overhead, how the cut's two rectangles (sequential
only in applied IR, §2) trade against the one they replace — is specified by a
separate cost-model design. This section fixes
only what exists and what is eligible; the objective consumes `boundary_op[b]`,
`full_size[b]` and `boundary_view[b]` but does not define them.

### 3. Tiling groups fall out of the cut variables

**A tiling group is a maximal cut-free run of operations.** This is the central
simplification, and it buys three things at once:

1. `_validate_contiguous` is satisfied *structurally*. Because `cut[i]` ranges
   over adjacent pairs of `graph.operations` and is pinned to 1 wherever either
   side is untileable (§2), a maximal cut-free run is by construction a
   contiguous slice of `graph.operations` — exactly what `_validate_contiguous`
   (`:785`) checks against the `op_to_position` map `coarse_tile` builds.
   Contiguity never has to be expressed as a constraint. An untileable op
   between two ops that would otherwise merge simply forces a cut; hoisting such
   interlopers out of the way stays `reorder_unhinted_interlopers`'s job (R4.4),
   not the solver's.
2. Grouping is driven purely by the objective, as required: the solver merges
   two ops into a group exactly when agreeing on a tiling scores better than
   paying for the boundary copies a cut would materialize.
3. A cut is never priced *directly*. It is priced through the consequences it
   materializes, and this RFC's job is to make those consequences **visible** to
   the objective rather than to price them: `boundary_op[b]` says a `full_buf`
   exists, `full_size[b]` and `boundary_view[b]` say at what shape and slicing,
   and the read copies enter the packing model as optional rectangles under the
   same literals (§2). There is deliberately no `n_groups` penalty term — the
   performance profile is read off the real outcomes (tile shape, LX pinning
   status), not off a proxy for them.

   What those consequences are *worth* — HBM traffic through `full_buf`, loop
   overhead, and how a cut's two tile-sized rectangles (sequential in applied
   IR, conservatively concurrent in the model — §2) trade against the one they
   replace — is specified by a **separate cost-model design** and is
   out of scope here. Note only that the consequences are producer-dependent, so
   any price must be too: a cut between two tiled configs materializes a
   full-size HBM buffer plus a mutation copy op, while a cut whose producer is
   *untiled* materializes only a read copy in the consumer — though it can
   still cost that producer its LX candidacy (*Background*, row 3). A uniform
   per-cut constant would overprice every untiled → tiled edge, which is most
   forced cuts.

`validate_coarse_tile_groups`'s invariant (a hint scope must not split across
groups) becomes a constraint: `cut[i] == 0` is forced for every `i` interior to
a hint scope.

### 4. Injected sympy objective

A new module `torch_spyre/_inductor/scratchpad/cost_expr.py` provides:

- **A symbol namespace** the solver binds to model variables. Per-buffer:
  `size`, `read_count`, `in_lx`, `spilled`, `cores`, `tile_count`,
  `is_intermediate`. Aggregator: `SumOverBuffers`. Globals:
  `total_hbm_bytes`, `peak_lx_bytes`, `idle_cores`.
  `peak_lx_bytes` is defined as the packing high-water mark — one
  `AddMaxEquality` over the `top[b]` vars `_add_no_overlap_2d` already creates
  (`:568`) — *not* a time-indexed occupancy sum, which would cost a constraint
  per timestep and is why the naive reading of "peak" is rejected.
  `tile_count` and `in_lx` are the primitives from which a performance profile
  is derived; there is no group-count symbol.

  `SumOverEdges` is **reserved, not provided**. With relayout deferred (R6) no
  edge-indexed term survives, so shipping the aggregator with nothing to
  aggregate would be dead API. The name is held for R6.3's
  `relayout_bytes = SumOverEdges(relayout[e] * bytes[e])`, and `relayout_bytes`
  is likewise absent for now.

  Anything beyond this list must arrive as a per-config scalar, precomputed into
  the §2 table and bound by one more `AddElement`. The tile *shape* is
  deliberately not a symbol (R3.8).
- **A lowering** `lower(expr, bindings) -> cp_model.LinearExpr` over an
  explicitly bounded sympy subset (R3.3). Anything outside that subset raises
  `CostExpressionError` naming the offending node. Silently approximating an
  objective is worse than a compile error.
- **A default objective** built from today's terms:
  `SumOverBuffers(spill_cost) - SumOverBuffers(cores)`.

The objective is a **single expression minimized in one phase** — the model
computes one total cost and minimizes it (`Minimize(expr)`), with no
lexicographic sequence and no per-phase locking. `CostSpec` therefore wraps a
single sympy expression; a bare expression is the normal form. This is a
deliberate move away from today's two-phase lexicographic solve: the hard
guarantee that parallelism can never buy a spill is no longer structural but a
matter of relative weight — the default weights the spill term to dominate the
core term so the practical outcome tracks today's spill-first intent (R3.2,
R3.5).

**The predictor is load-bearing for what this objective *means*, not only for how
accurate it is.** §3 prices a cut partly through residency the run's interior
loses, and no symbol here expresses that directly — there is no "this buffer
would have been scratch under a different cut assignment" term. The pricing works
because the *predicted buffer set itself* varies with the cut assignment: a
different assignment yields a different set of buffers to sum `spill_cost` over.
So R7.1's predictor is not merely an accuracy input to the objective, it is part
of the objective's definition. A predictor that omits the buffers tiling creates
(R7.5) does not make the objective slightly wrong — it makes it price a different
question.

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

`_maybe_coarse_tile_span_overflow` is subsumed **on the success path**:
span-forced tiling becomes a feasibility constraint on configs rather than a
separate pass. The pass itself is retained — skipped when the solve succeeds,
run verbatim when it raises (the R8.3 failure path) — so a graph whose
feasibility requires tiling still compiles when the joint solve fails.

The final placement solve remains **authoritative for addresses**. The joint
model decides on a *predicted* buffer set; addresses are then computed over the
real post-tiling buffers. A misprediction therefore degrades to a spill, never
to a wrong address.

### 6. Tractability

- **Whole-graph model.** The solve is a single CP-SAT instance over the entire
  graph — it is **not** decomposed at matmul or any other op boundary, and makes
  no op-specific segmentation assumption. Matmul operands come from HBM, so
  matmuls are untileable; §2's untileable-pinning already forces cuts on their
  edges as a consequence of that, not as a special case.

  Those forced cuts are also **cheaper than they look**. A cut whose untileable
  op is the producer is a row 3 or row 4 of the cut-cost table (*Background*):
  it materializes a read copy at most, never a full-size HBM buffer — though a
  row-3 edge still costs the tile copy and, when the read advances, the untiled
  producer's LX candidacy. A tiled chain feeding an untileable consumer is
  row 2 and does pay `full_buf` plus the mutation copy. The common reading —
  that *every* cut costs a full-size round-trip — would still make a
  whole-graph model look far more expensive than it is, and is the main reason
  segmentation at matmul boundaries is unnecessary rather than merely unwanted.
- **No per-op config cap.** Model size is controlled by external pruning of the
  enumerated config set rather than a fixed ceiling (see R2.4 and *Open
  questions*).
- **Enumeration cost, not just model size.** Two per-op costs now scale with the
  tiling-option count rather than being paid once:
  `enumerate_work_division_candidates` runs per option (R2.2), and
  `_views_for_divs`'s sympy-heavy prep is no longer candidate-invariant, so its
  cache key gains the tile (R2.6) and it is built once per `(op, dep, buf, tile)`
  rather than once per `(op, dep, buf)`. The prep was introduced precisely to keep
  view cost proportional to ops rather than candidates; tiling reinstates a factor
  of the option count. Solve time is already reported against the gate-off
  baseline (testing item 10) — enumeration time must be reported alongside it,
  since the two have different mitigations.
- **Warm-start** via `AddHint` with the current heuristic's plan. Combined with
  the R8.3 failure path — `SolveError` caught at the new pass slot, then the
  existing tiling method with the greedy solver — this yields an anytime
  property and a never-worse-than-today floor.

## Requirements

### R1 — Tiling enumeration

- **R1.1** Add `enumerate_tile_options(op, *, max_dims, max_splits_per_dim,
  max_options) -> list[TileOption]`, returning **all** feasible options within
  bounds, tiling **output ranges only** (R1.8). This is the behavioural change
  from `_search_min_cost_tile_plan`
  (`:1269`), which returns the *first* combo in `_combo_cost` order. Both of its
  failure modes are preserved: it *raises* `Unsupported` when no combo passes
  (`:1395`, `:1399`), and returns `None` only when there are no candidate host
  dims at all (`:1303`).
- **R1.2** `_combo_cost` is retained solely as the ranking used to truncate to
  `max_options`, and as the deterministic tie-break.
- **R1.3** The untiled option `TileOption(dims=())` is always present, so
  feasibility is never worse than today.
- **R1.4** Validity predicates must be **reused, not reimplemented**. From
  `wsr/span_overflow_hint_analysis.py`: `_within_stick_host_dim` (`:240`),
  `_post_tile_stick_alignment_error` (`:263`), `_candidate_host_dims` (`:911`),
  `_cap_split_candidates` (`:979`), `_input_stick_alignment_error` (`:1042`),
  `_split_candidates_for_host_dim` (`:1100`), `_iter_split_combos` (`:1195`),
  `_combined_tile_stick_alignment_error` (`:1215`),
  `_host_dim_has_legal_nontrivial_split` (`:935`, the R1.9 candidate source), and
  `can_conform_pointwise_tile` (`:1421`, the R4.7 adoption predicate). From
  `pass_utils.py`: `coeff_through_floor` (`:848`, sub-stick guard) and
  `op_out_coords` (`:363`, the frame `host_dim` indexes).

  `_validate_reduction_tiling` (`coarse_tile.py:1233`) and
  `_seed_buffer_for_carry` (`:575`) are deliberately **not** in this list. They
  guard reduction-axis tiling, which R1.8 excludes from the enumerated space, and
  `_validate_reduction_tiling` over-approves in any case (*Background*), so it
  could not serve as a feasibility gate even if reduction options were
  enumerated. Both continue to run inside `coarse_tile` on the hint path,
  unchanged.
- **R1.5** Derived quantities come from the existing zero-mutation planner —
  `_planned_tile_extents_per_level` (`:309`) for the extents themselves, and
  `_tiled_dims_for_dep` (`:421`) to filter them per dep. No new extent arithmetic
  is written.
- **R1.6** Every returned option must be applicable *and* numerically correct.
  Applicability alone is insufficient: a test that applies each option via
  `_apply_plan` and asserts no `Unsupported` is raised would pass on the known
  wrong-numerics nested reduction shapes (*Background*), because their failure
  mode is a silent wrong answer, not an exception. The enumerator test therefore
  applies each option **and** compares against CPU, in the manner of
  `run_coarse_tile_test(..., correctness=True)`.
- **R1.7** Existing caps are the defaults, and stay where they are defined today
  in `wsr/span_overflow_hint_analysis.py` rather than migrating to `config.py`:
  `_MAX_TILE_DIMS = 3`, `_MAX_TILE_COMBOS = 512`, `_MAX_SPLITS_PER_DIM = 16`
  (`:143-145`), `_MAX_AUTO_TILE_SPLIT_COUNT = 64` (`:149`).
- **R1.8** **No reduction-axis options.** `enumerate_tile_options` never emits a
  `TileOption` that divides a `reduction_ranges` entry; every level tiles an
  output range. This matches what the automatic planner already does
  (`SpanOverflowTileLevel.is_reduction` is hardcoded `False`) and is the
  enumeration-side half of the R9 non-goal. Hint-driven reduction-axis tiling is
  unaffected — it still applies pre-stickification through
  `_maybe_coarse_tile_hints` — but those ops enter the model pinned (R4.6, R5.6).
- **R1.9** **Candidate dims are not span-pressure-only.** `_candidate_host_dims`
  (`:911`) takes `list[SpanOverflowCandidate]`, so it surfaces only dims already
  under span pressure — which is correct for the span-overflow planner and wrong
  here. An op with no span overflow yields no candidate dims, hitting R1.1's
  "returns `None` when there are no candidate host dims at all" path. An
  enumerator built strictly on R1.4's list would therefore return **only the
  untiled option for exactly the ops where tiling-to-buy-LX-residency matters**,
  silently nullifying the second defect in *Motivation*.

  Candidate dims are instead the union of the span-pressure dims and every host
  dim passing `_host_dim_has_legal_nontrivial_split` (`:935`) — an existing
  helper, already built on `_split_candidates_for_host_dim`. Ordering stays
  pressure-first (`_candidate_host_dims`'s own ordering, then the remainder), so
  when `_combo_cost` truncation to `max_options` binds it discards the
  speculative options before the pressure-relieving ones.

### R2 — Config construction and joint feasibility

- **R2.1** `PartitionConfig` pairs a `CoreDivision` with a `TileOption | None`
  and precomputes `per_core_bytes`, `cores_used`, `tile_count`.
- **R2.2** Divisions continue to come from `enumerate_work_division_candidates`
  (`work_division.py:753`) unchanged, including all five of its guards: at most
  one reduction dim split (`:812`), no coordinate-masked dim split (`:819`),
  TOPK left unsplit (`:776`), the core budget `prod(splits) <= max_cores`
  (`:810`), and a per-core span within `MAX_SPAN_BYTES` on every tensor dep
  (`:814-818`). It is called **once per tiling option**, against that option's
  divided iteration space, because its per-dim factors are `divisors(basis)` over
  `iteration_space_from_op` and its span guard is evaluated on the same space
  (§1). Each tiling option therefore carries its own division candidate set;
  neither set is a subset of the untiled one.
- **R2.3** **Span feasibility is evaluated on the pair, not per subsystem.** A
  config is feasible iff its per-core, per-tile span is within `MAX_SPAN_BYTES`
  (`work_division.py:73`, `65535 * 4096` ≈ 256 MiB). This generalizes the
  per-core-only check R2.2 already performs at `:814-818`, and discharges the
  `core_split_estimate = 1` TODO.

  Stating the over-tiling defect in config terms: tiles are **sequential** loop
  iterations and cores are **parallel**, but both draw down the same divisibility
  budget on a dimension. `T=4` tiles × `C=32` cores over `M=512` and `T=1` × `C=32`
  cut the span by different factors but only the second spends the whole budget on
  parallelism. Today the two decisions each spend that budget as if alone, which
  is why `MAX_SPAN_BYTES` ends up satisfied twice over. A joint feasibility check
  is what lets the objective spend it once.
- **R2.4** Configs are deduped by signature; there is **no fixed per-op cap** —
  model size is controlled by external pruning of the enumerated set (*Open
  questions*). The pair (work-division seed, untiled) is always retained, so the
  model's feasible set always contains today's answer.
- **R2.5** An op with no feasible config raises `Unsupported` at the same
  pipeline point it does today.
- **R2.6** **`config_matches` needs tiling-aware per-core views.** This is the
  one part of the `CoreDivision` → `PartitionConfig` migration that is not a
  rename, and it is on the critical path: `config_matches` gates residency
  through `constrain_residency`, so it cannot be deferred alongside the relayout
  work that shares its machinery (R6.3).

  `_views_for_divs` (`allocator.py:2079`) caches the sympy-heavy prep under
  `(op name, dep, buf_name)` on the explicit assumption that it is
  candidate-invariant — true when a candidate is only a core division, false
  once a candidate also carries a tiling. Three consequences:

  - The prep cache key gains the tile: `(op name, dep, buf_name, tile)`.
    Divisions of one op under different tilings must not share a prep.
  - `_prepare_per_core_view` (`pass_utils.py:1467`) and `_per_core_view_on_buf`
    (`:1696`) must accept a *predicted* post-tiling frame — divided ranges and
    the resized device layout — rather than reading the op's current layout.
    The solve runs before `coarse_tile` applies (§5), so at match-construction
    time no tiled IR exists to read.
  - That predicted frame is the same artefact R7.1 produces. R2.6 and R7.1 share
    one predictor; a divergence between them is a wrong-residency bug, not a
    mispredicted size, so it does **not** enjoy R7.4's degrade-to-spill safety.

  The existing conservatism is retained: a candidate whose slicing is
  unrepresentable is excluded from matching and the producer falls back to HBM.
  A tiling whose predicted frame cannot be built is excluded the same way — never
  pin on a slicing that cannot be verified.

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
  `log_lx_usage` by keyword (`allocator.py:184`, `:1348`). The four concrete
  overrides move in lockstep — `ilp_solver_ortools.py:348`, `greedy_solver.py:134`,
  `firstfit_bestfit_solver.py:186`, `simulated_annealing.py:122`.

- **R3.2** The objective is a **single total expression minimized in one phase**
  (`Minimize(expr)`). There is no lexicographic sequence and no per-phase
  locking: today's two-phase lexicographic solve (`ilp_solver_ortools.py:479`) is
  **replaced, not generalized**. The hard guarantee that parallelism can never
  buy a spill becomes a weighting choice (R3.5) — a term whose scale must
  dominate another is expressed by its coefficient, subject to the
  `COST_SCALE`/overflow rules in R3.4.
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
- **R3.5** `objective=None` selects the default single-phase objective built from
  today's terms (`SumOverBuffers(spill_cost) - SumOverBuffers(cores)`), with the
  spill term weighted to dominate. Because the solve is single-phase rather than
  two-phase lexicographic, **exact bit-identity with today's plans is not
  required**; the guarantee is spill-parity (no plan spills a buffer today's
  objective would have kept resident) with no core-count regression at equal
  spill.
- **R3.6** The four placement-only solvers (`greedy`, `firstfit`, `bestfit`,
  `simulated_annealing`; registry at `allocator.py:2108-2113`) accept the
  parameter for ABC conformance, ignore a non-`None` objective, and log a warning
  once. The ABC docstring states this explicitly — the contract must not imply
  support these solvers lack. Note `LAYOUT_SOLVER` has a fifth value, `cpsat`,
  which is handled ahead of that registry (`allocator.py:2159`) and is the one
  solver for which `objective` is honoured.
- **R3.7** `SPYRE_COST_EXPR` parses a string against the exported namespace, so
  objectives can be explored without a code change.
- **R3.8** **Symbol binding is bounded.** Every symbol resolves to either an
  existing model variable or a single `AddElement` lookup over a per-config table
  computed at enumeration time. No symbol may add constraints scaling with
  anything but the buffer count and the adjacent-pair count (the edge count
  rejoins this list when R6.3 lands `relayout[e]`) — which is why `peak_lx_bytes` is the
  packing high-water mark over the existing `top[b]` vars rather than a
  time-indexed occupancy sum, and why the tile *shape* is not a symbol. A term
  needing shape sensitivity precomputes a per-config scalar instead. Adding a
  symbol that violates this is a design error, not a performance trade-off.

### R4 — Tiling groups

- **R4.1** Groups are **not** pre-computed. `cut[i]` booleans over adjacent pairs
  of `graph.operations` define them, fixed by `AddAllowedAssignments` over the
  per-edge `(tile_src, tile_dst, cut)` triple table, which admits `cut == 0`
  only on tiling-compatible config pairs.
- **R4.2** `cut[i]` is pinned to 1 at every boundary where either side is
  untileable. Contiguity is therefore structural and `_validate_contiguous`
  (`coarse_tile.py:785`) passes by construction — a maximal cut-free run of a
  list whose untileable positions are all cut is a contiguous slice of that
  list. A test asserts this directly rather than relying on the argument.
- **R4.3** `cut[i] == 0` is forced for every `i` interior to a hint scope,
  preserving `validate_coarse_tile_groups`'s invariant.
- **R4.4** `reorder_unhinted_interlopers` continues to run as a pre-step. The
  solver does not reorder operations. **Graph reordering is out of scope.**

  Execution order is the topological order of `graph.operations`, established at
  lowering and guaranteed by `GraphLowering` (`passes.py:404`). It is *a*
  topological order, not the only valid one, so an unhinted op sitting between
  two ops the solver would like to fuse is an artifact of that linearization
  rather than a semantic constraint — which is why `reorder_unhinted_interlopers`
  can relocate such ops on a dataflow legality check alone.

  The solver nonetheless takes the order as given, and every interloper the
  hint-driven pre-step did not move becomes a **forced cut**. That is a
  plan-quality ceiling, not a correctness problem, and a bounded one: by the cut
  table most forced cuts are the untiled → tiled row, which allocates no
  `full_buf` — though when the consumer's read advances it still costs the
  untiled producer its LX candidacy (*Background*, row 3). The expensive
  tiled → tiled row requires an interloper between two ops the solver actively
  wanted to fuse.

  Lifting this is harder than re-running the existing pass, which is why it is
  not phase 2 either. Reordering sits at pipeline position 5, deliberately
  **before** stickification; the joint solve runs late — post-stickification
  (§5), after layouts are committed and after `optimize_restickify_locations`
  has chosen restickify sites against the current order — with the
  placement-only re-solve last at `_maybe_scratchpad_planning`. A solve → relocate →
  re-solve loop would move ops past decisions already made on the assumption they
  would not move. Hoisting the tiling decision earlier instead conflicts with
  `_maybe_coarse_tile_span_overflow` being post-stickification precisely because
  it needs `device_layout` for span arithmetic.

  The fixed order is also what makes the cut claims static dicts (§2):
  program-order adjacency is static, so each buffer's neighbour topology — and
  each spanned edge's parent/child orientation — is fully known when its
  wrapper is constructed.
- **R4.5** The solver emits **two** artefacts, mirroring `span_overflow_groups`
  exactly, because `coarse_tile` alone is not enough:

  - `groups` in the documented `(ops, levels)` shape, passed with the existing
    `group_idx_offset` parameter so emitted `loop_group_id`s do not collide with
    those the hint pass already stamped; and
  - `dim_hint_assignments` — `(op, list[DimHint])` pairs built by
    `_dims_to_hints` (`coarse_tile_span_overflow.py:152`) from each op's
    `TileOption.dims` and the hint ids minted for its run.

  `op.dim_hints` is an **input** to `plan_coarse_tile_groups`, not an output of
  it: the hint lookups that build `hint_id_to_ranges_pos` read it. The
  span-overflow path assigns them explicitly before calling `coarse_tile`
  (`passes.py:357-365`, "a pure planning step: it decides each op's dim_hints but
  does not set them"). The apply step does the same, in the same order, and
  derives `group_idx_offset` from existing `loop_group_id[0]` values the same way.
  No new application path is introduced.
- **R4.6** `cut[i]` is pinned to 1 on **both** boundaries of any op the hint pass
  tiled on a reduction axis, exactly as for an untileable op (R4.2). Such an op
  is therefore always a singleton group and never shares a loop nest with a
  solver-chosen neighbour.

  This is not merely conservative — it is what keeps the pairwise cut table
  sound. The invariant governing a reduction-tiled group is
  `_plan_is_loop_invariant_at_reduction_levels` (`coarse_tile.py:559`): at every
  level where *some* member tiles a reduction dim, *every* other member must be
  loop-invariant at that level. That quantifies over the whole group, and
  adjacent-pair compatibility does not compose into it. Counterexample: `A` tiles
  a reduction dim at level `L`, `B` is loop-invariant at `L`, `C` tiles an output
  dim at `L`. Pair `(A,B)` is legal and pair `(B,C)` is legal — `B`'s invariance
  says nothing about `C` — yet the run `{A,B,C}` violates the invariant. Pairwise
  tables can express "these two agree"; they cannot express a predicate
  quantified over a run whose membership they are simultaneously deciding.
  Lifting R4.6 needs run-identity in the model (per-run, per-level literals), not
  a wider triple table.
- **R4.7** "Tiling-compatible" in R4.1 is a **pairwise predicate evaluated at
  table-construction time**, not equality of a per-op key. For an ordered adjacent
  pair, `cut == 0` is admitted iff:

  1. the consumer can adopt the producer's split —
     `can_conform_pointwise_tile(op, split_by_host_dim, config.sencores)`
     (`span_overflow_hint_analysis.py:1421`), which checks divisibility, stick
     boundaries, and sufficiency; **and**
  2. the loop variables correspond through the dep — the symbol tiling the
     consumer's dim must appear in the producer's tiled coordinate as seen
     through the read, the check `_reduction_shares_group_tiled_dim`
     (`coarse_tile_span_overflow.py:68`) performs. Matching split counts alone is
     necessary but not sufficient: two unrelated dims can split into the same
     count.

  Both are reused, not reimplemented (R1.4). Because
  `can_conform_pointwise_tile` is **directional**, the predicate is evaluated in
  program order — producer then consumer — matching the direction
  `span_overflow_groups` already conforms in. Any pair for which correspondence
  cannot be established fails closed to `cut == 1`, preserving the existing
  conservatism: an unverifiable pair is never fused into a possibly-desynchronized
  loop.
- **R4.8** What a cut materializes is a **model variable**, not a post-hoc
  property. Per buffer the model carries `cut_parents[b]`/`cut_children[b]`
  (the direction-indexed dicts over the cut vars on the edges `b` spans, §2),
  `tiled[b]`, `escapes[b]`, and
  `boundary_op[b] ⟺ tiled[b] ∧ escapes[b]` (§2). The solve precedes IR mutation
  (R7.1), so `full_buf` and the copy op do not exist yet and their own LX
  rejections are not available to the model; what it can see is the producer's
  own output and the buffers a cut would add.

  A cut does **not** evict its *tiled* producer. `_propagate_tiled_op` sets
  `output_tiled_dims = []` on both branches (`coarse_tile.py:1367`), so `b` stays
  loop-internal tile-sized scratch and remains LX-eligible whether or not it
  escapes. There is no `boundary_op[b] ⟹ in_buffer[b] == 0` constraint, and any
  model that adds one is wrong. The one eviction the model does carry is the
  row-3 rule (§2): a read copy whose advancing read crosses an untiled → tiled
  edge enforces `in_buffer[producer] == 0`, because the realized copy stamps
  that producer `"tiled (advancing)"` (*Background*).

  `tiled[b]` is `tile[b] != 0` — the candidate space is two-level (tiling, then
  that tiling's divisions), so being tiled is read off the variable rather than
  looked up. It is required in the conjunction, not optional: an untiled → tiled
  cut materializes no `full_buf` at all (R4.2's forced cuts are all of this
  form), so `escapes[b]` alone would claim a boundary buffer that never gets
  allocated.

  These vars live on a `_TilingBufferWithCpVars` wrapper extending
  `_CoreDivisionBufferWithCpVars`. They are **not** kept in a parallel
  `name -> {var}` dict — avoiding exactly that is why the wrapper hierarchy
  exists (`ilp_solver_ortools.py:150`).
- **R4.9** The read-side tile copies are **optional rectangles**, not
  unconditional ones. `_insert_read_copy_ops` creates a tile-sized LX-eligible
  buffer per full-extent input of a tiled op, and whether it exists is a solver
  decision: conditional on `escapes[producer]` for a cross-group read, on
  `tiled[consumer]` for a graph input, constant, or untiled producer. Presence in
  `_add_no_overlap_2d` is that literal; `in_buffer` gates residency on top of it.
  Modelling them as always-present over-reserves LX on cut-free runs; omitting
  them undervalues tiling, since they are precisely the buffers tiling makes
  pinnable.
- **R4.10** The boundary buffer's **shape and core division** are model variables
  too — `full_size[b]` and `boundary_view[b]`, bound by `AddElement` over
  per-config tables like `eff_size[b]`/`cores[b]`. `full_size[b]` is not
  `eff_size[b] × tile_count[b]`; the full buffer is stickified once at the full
  host extent, so per-tile stick padding does not survive the scale-up.
  `boundary_view[b]` exists so producer/consumer agreement at the boundary is
  checked on the physical per-core view (R2.6), not on a slice count two
  orthogonal divisions can share. Both are inputs the objective consumes; what
  they are **worth** is the separate cost-model design's concern, not this one's.

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
- **R5.6** A hint that tiles a reduction axis still applies, and
  `enable_reduction_tiling` (`config.py:82`) keeps its current default and
  meaning. The affected op enters the model as a pinned single-config buffer
  (R5.1) with both its `cut[i]` boundaries pinned to 1 (R4.6). Setting
  `SPYRE_INDUCTOR_ENABLE_REDUCTION_TILING=0` makes such a hint raise
  `Unsupported`, unchanged by this RFC.

### R6 — Stickification / relayout: out of scope

Choosing configs to *avoid* a restickify — a "stickification optimization" that
models each producer→consumer edge's relayout cost and lets the objective trade
it off — is **out of scope for this RFC** (R9). The `relayout[e]` variable, its
per-edge triple table, and the `relayout_bytes` objective term are all deferred.

- **R6.1** The compiler keeps inserting restickifies wherever configs force one,
  exactly as today (`insert_restickify`, `passes.py:438`). The solver neither
  models nor minimizes that cost.
- **R6.2** *Implication — accepted pessimism.* Because relayout cost is invisible
  to the objective, the solver may pick a (tiling, division) config whose
  physical per-core view disagrees with a neighbour's and thereby force a
  restickify a relayout-aware model would have avoided. Those bytes are real but
  unpriced, so plans can be **pessimistic on relayout-driven HBM traffic**. This
  never makes a plan *infeasible* — a restickify can always be inserted — it only
  means the model cannot prefer the cheaper-to-stickify config. In the worst case
  the joint solve is no better than today on relayout, and possibly worse, since
  tiling introduces new per-core views that today's un-tiled graph never had.
- **R6.3** Lifting this is follow-on work, but only the *pricing* half is
  deferred. The two halves must not be bundled:

  - **Lands in this RFC (R2.6):** tiling-aware physical per-core views —
    `_prepare_per_core_view` (`pass_utils.py:1467`) and `_per_core_view_on_buf`
    (`:1696`) evaluated against a predicted post-tiling frame. `config_matches`
    depends on this to gate residency, so it is not optional and cannot wait for
    the relayout work.
  - **Deferred:** `relayout[e]` as a *determined*, cost-only edge variable,
    precomputed once per edge at enumeration time from those same views, plus a
    `relayout_bytes = SumOverEdges(relayout[e] * bytes[e])` term in the objective
    namespace, inside R3.8's linear-binding rule.

  An earlier draft filed the view extension under the deferred half. That would
  have left the non-deferred `config_matches` path depending on a mechanism this
  RFC never builds; the follow-on work is a variable and a cost term over views
  that already exist by then.

### R7 — Prediction fidelity and application

- **R7.1** A pure predictor maps a candidate config set to the predicted buffer
  set (sizes, lifetimes, boundary copies) with **no IR mutation**. R1.8 bounds
  what it has to model: output-range tiling materializes only the boundary copies
  of `_allocate_full_buffer`/`_insert_copy_op`/`_insert_read_copy_ops`. The
  accumulator, identity fill, and combine op that reduction-axis tiling would add
  are never predicted — hints apply pre-stickification (`passes.py:430`), so by
  the time the solve runs those buffers are already real IR the model simply
  sees.

  The predictor's second output is the **post-tiling frame** — divided ranges
  plus the resized device layout — that R2.6 evaluates per-core views against.
  One predictor serves both; they must not drift apart.
- **R7.2** `SPYRE_VERIFY_TILE_PREDICTION=1` applies the plan and asserts
  predicted per-buffer sizes and lifetimes match the realized ones, **and** that
  each predicted per-core view equals the one recomputed from the post-tiling IR
  (R2.6). The view check is the more important of the two: a mispredicted size
  degrades to a spill under R7.4, whereas a mispredicted view means residency was
  gated on a slicing agreement that does not hold, which is a wrong-data bug. This
  is the highest-risk area in the design and gets its own test suite.
- **R7.3** Application order is: decide → `coarse_tile` → commit divisions →
  placement-only re-solve. Addresses always come from the final solve over real
  buffers.
- **R7.4** The placement re-solve is warm-started from the joint solve's
  residency intent. A buffer that no longer fits degrades to a spill with a
  distinct per-buffer `residency_reason` (`plan_solver.py:68`) — which surfaces
  through the solver-level `spill_reasons` map (`plan_solver.py:219`) and the
  allocator's `reject_reasons` mirror (`allocator.py:137`) — so the mispredict is
  visible rather than silent.
- **R7.5** The predictor models **inserted operations and their positions**, not
  just the buffers those operations allocate. `coarse_tile` splices the full
  buffer in before the group's first op, the write copy after the tiled op, and
  read copies before their consumer (*Background*). Liveness is index-based
  (`calculate_liveness` over `graph.operations`), so in realized IR each
  insertion shifts lifetime ticks for everything downstream — a systematic
  offset, not noise. The model sidesteps the renumbering by placing predicted
  insertions at §2's interstitial coordinates on a scaled tick axis, so every
  real buffer's lifetime stays stable in the model; the realized offsets appear
  only after apply, are recomputed wholesale by `calculate_liveness`, and are
  compared against the prediction under rank-order normalization (R7.2).

  It must equally model the LX candidates tiling **creates**: the interior
  per-tile scratch (whose `output_tiled_dims` becomes `[]`, making it eligible)
  and the read-side tile copies (ordinary tile-sized allocations, also eligible).
  These do not exist until `coarse_tile` runs, so omitting them leaves the final
  placement re-solve holding buffers the joint objective never scored. The bias
  has a known direction — the model would **undervalue tiling**, since the
  buffers it fails to see are exactly the ones tiling exists to make pinnable —
  so this is not a wash that averages out across a graph.

### R8 — Robustness, gating, determinism

- **R8.1** New gate `UNIFIED_TILING` / `config.unified_tiling`, **default off**.
  The bare `UPPER_SNAKE` form matches the LX-planning family this gate composes
  with (`LX_PLANNING`, `CO_OPTIMIZING_LX_PLANNING`, `LAYOUT_SOLVER`;
  `config.py:22-25`, `:111`), while the diagnostic flags below keep the newer
  `SPYRE_`-prefixed style of `SPYRE_INDUCTOR_*` — the split is deliberate, not
  accidental. Requires `LAYOUT_SOLVER=cpsat` and `CO_OPTIMIZING_LX_PLANNING=1`,
  the latter itself default-off today (`config.py:23-25`); warn and no-op
  otherwise.
- **R8.2** Warm-start the model via `AddHint` with the current heuristic's plan,
  so hitting the time limit yields today's answer rather than a worse one.
- **R8.3** Time limit → `SolveError` (`plan_solver.py:27`), caught by a **new
  handler at the `unified_partition_solve` slot** — the existing try/except at
  `allocator.py:2211` wraps only `_maybe_scratchpad_planning` (pass 455) and
  never sees this pass. On failure the joint plan is discarded whole and the
  pipeline reverts to the existing tiling method with the greedy solver:
  `_maybe_coarse_tile_span_overflow` runs exactly as today (retained, §5), the
  heuristic division passes at 451-452 proceed unchanged, and placement at
  pass 455 drops straight to placement-only greedy — `allocator.py:2211`'s
  fallback path, entered directly rather than after a second `SolveError`,
  since a joint solve that just timed out makes another CP-SAT attempt a poor
  bet. The graph is unmutated when the solve raises — the solve precedes
  `coarse_tile` — so the fallback starts from clean IR. This is what makes the
  floor hold on the failure path: the graphs span-forced tiling exists to
  rescue still compile.
- **R8.4** Determinism: keep `num_search_workers = 1` under
  `torch.are_deterministic_algorithms_enabled()` and `random_seed = 0`. Tiling
  adds symmetry, so add symmetry-breaking over equal-cost configs and document
  the tie-break, so plans are reproducible across runs.
- **R8.5** `ortools` remains the optional extra named `cpsat`
  (`pyproject.toml:36-38`, `ortools>=9.0`); the import stays guarded
  (`ilp_solver_ortools.py:86-93`) and `_make_cpsat_solver` (`allocator.py:2116`)
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
- **No stickification / relayout optimization.** The solver does not model or
  minimize restickify cost (R6); configs are chosen blind to relayout, which can
  be pessimistic on relayout-driven HBM traffic. Deferred to follow-on work.
- **No reduction-axis tiling.** The solver never chooses to tile a reduction
  range (R1.8), and hint-driven reduction-axis tiling is pinned out of the
  grouping decision (R4.6, R5.6). Three reasons, in descending order of how hard
  they are to work around: the group invariant is not pairwise so the cut model
  cannot express it (R4.6); only *single-level* reduction-axis tiling is
  numerically validated today, while `_validate_reduction_tiling` also admits the
  known-wrong nested shapes (*Background*); and it is not a pure working-set
  reduction, so pricing it needs the accumulator and fill buffers in the cost
  model rather than the tiled op's footprint alone. The capability itself is
  untouched — `enable_reduction_tiling` keeps its default and hints keep working.
  Follow-on work is described under *Open questions*.

## Files

**New**

- `torch_spyre/_inductor/scratchpad/cost_expr.py` — symbol namespace,
  `CostSpec`, sympy→CP-SAT lowering, `CostExpressionError`.
- `torch_spyre/_inductor/wsr/enumerate_tilings.py` — `enumerate_tile_options`,
  built on the R1.4 predicates; output ranges only (R1.8).

**Modified**

- `scratchpad/plan_solver.py` — `TileOption` (op-local `dims`, §1),
  `PartitionConfig`; `CoreDivision` retained as a config field; the R3.1
  signatures.
- `scratchpad/ilp_solver_ortools.py` — new `_TilingBufferWithCpVars` subclass of
  `_CoreDivisionBufferWithCpVars` (`:244`) carrying the two-level `tile`/`div`
  pair in place of `division`, plus the boundary vars (R4.8–R4.10); per-buffer
  direction-indexed cut-claim dicts (`cut_parents`/`cut_children`) reconciled
  by an `_add_cut_equalities` sweep in `_run`; read
  copies as optional rectangles in `_add_no_overlap_2d` (`:568`); relayout
  deferred (R6); single-phase objective driven by `CostSpec`; `_extract` writes
  `chosen_config` and reconstructs `groups` from the solved cuts.
- `scratchpad/allocator.py` — `_enumerate_core_divisions` (`:1558`) becomes config
  enumeration; `_cd_parent_matches` (`:1973`) becomes `_config_matches`, on
  tiling-aware views (R2.6, **not** a rename); `_views_for_divs` (`:2079`) takes
  the predicted frame and its `prep_cache` key gains the tile;
  `_commit_divisions` (`:1605`) also emits `groups` **and**
  `dim_hint_assignments` for `coarse_tile` (R4.5).
- `pass_utils.py` — `_prepare_per_core_view` (`:1467`) and `_per_core_view_on_buf`
  (`:1696`) accept a predicted post-tiling frame instead of reading the op's
  current ranges and device layout (R2.6).
- `passes.py` — insert `unified_partition_solve` (with its R8.3 `SolveError`
  handler) and the apply step; skip `_maybe_coarse_tile_span_overflow` when
  the solve succeeds — retained verbatim as the R8.3 fallback tiler, not
  deleted.
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
   coverage lives in `JointDivisionSolverTests:776` and
   `TestCpSatPlacementOnly:1121`), `test_scratchpad_use.py`, `test_coarse_tiling.py`,
   `test_coarse_tile_e2e.py`, `test_span_overflow_hint_analysis.py` all pass
   unchanged. Note `test_coarse_tiling.py` has no CI config yaml under
   `tests/configs/torch_spyre_tests/inductor/`, unlike its siblings, so it must be
   run explicitly rather than assumed covered.
2. **Parity, gate on with tiling disabled.** Spill outcomes must match today's
   CP-SAT output and core counts must not regress at equal spill. Exact
   bit-identity is **not** required, because the objective is now single-phase
   (R3.2, R3.5); this is the regression guard for the `CoreDivision` →
   `PartitionConfig` migration.
3. **Cost lowering.** Unit tests for the R3.3 accept/reject table and R3.4
   scaling (single-phase `Minimize` of one total expression; no per-phase
   locking), with a `CostExpressionError` case per rejected construct. Plus R3.8:
   every namespace symbol adds at most one `AddElement` (or the single
   `AddMaxEquality` for `peak_lx_bytes`), and model size grows linearly in buffer
   count and adjacent-pair count as the graph scales. Also assert `SumOverEdges`
   and `relayout_bytes` are **absent** from the exported namespace (§4) — a
   reserved name that silently resolves would let an objective reference a term
   the model never constrains.
4. **Cut tables and structural contiguity.** The `cut[i]` triple table is total —
   every `(tile_src, tile_dst)` pair appears exactly once — and `cut[i]` is
   pinned to 1 at every untileable boundary and on **both** boundaries of every
   hint-driven reduction-axis-tiled op (R4.6), so such an op is always a
   singleton group. Then the property that
   §3 rests on: for any solution, every maximal cut-free run is a contiguous
   slice of `graph.operations` (R4.2), asserted directly rather than argued.

   For R4.7, assert the admitting predicate is evaluated **directionally**: a
   pair whose consumer can adopt the producer's split but not the reverse admits
   `cut == 0`, and a pair where loop-variable correspondence cannot be
   established fails closed to `cut == 1`. Assert the claim indexing carries the
   same orientation: for every wrapper, each spanned edge's `cut_parents` entry
   and `cut_children` entry resolve to the same bool after the equality sweep,
   and the edge's triple table takes `tile_src` from the edge's parent op and
   `tile_dst` from its child. For R4.5, assert the emitted
   `(groups, dim_hint_assignments)` pair round-trips — applying the hints then
   calling `coarse_tile` reproduces exactly the grouping the solver chose, with
   no `loop_group_id` collision against hint-pass groups.
5. **Enumerator completeness and scope.** Brute-force reference on small shapes;
   the enumerator's set must equal the reference's. No returned option divides a
   `reduction_ranges` entry (R1.8) — asserted over the reduction ops in
   `test_coarse_tile_e2e.py`'s Group 4 and Group 5 shapes, which are the ones
   that would otherwise produce reduction-axis candidates. Every option applies
   **and** matches CPU numerically (R1.6), not merely applies without
   `Unsupported`.

   The R1.9 guard needs its own case, because it is the one that fails silently:
   for an op under **no** span pressure but with a legally splittable host dim,
   assert the enumerator returns more than the untiled option. Built on
   `_candidate_host_dims` alone this returns a single option and the solver simply
   never tiles that op — no error, no warning, and the LX-residency motivation
   quietly does nothing.
6. **Prediction fidelity.** Run the coarse-tiling and scratchpad suites under
   `SPYRE_VERIFY_TILE_PREDICTION=1`.
7. **Boundary buffer LX status.** For a graph tiled into two groups, assert the
   `full_buf` that `_allocate_full_buffer` produces has a non-`None`
   `residency_reason` (`"mutation target"` or `"tiled (advancing)"`), that the
   copy op's own output is likewise rejected, and — the positive half — that the
   interior per-tile scratch and the read-side tile copies *are* LX candidates.
   This is the table in *Background* asserted rather than argued. Assert too that
   the producer **keeps** a `None` `residency_reason` across a cut: both branches
   of `_propagate_tiled_op` set `output_tiled_dims = []`, so a cut must not evict
   it, and a regression here would resurrect the eviction constraint R4.8 rules
   out. Also assert the predicted lifetimes agree with the realized ones after
   `coarse_tile` has inserted its ops, under the rank-order normalization of
   §2's interstitial coordinates (R7.5) — equality for buffers no cut touches,
   containment (predicted ⊇ realized) for cut producers, whose model rectangles
   deliberately keep the pre-mutation extent (§2) — since insertion renumbers
   every downstream tick in the realized IR.

   Then the **model side**, which is the half that can go wrong silently
   (R4.8–R4.10). Assert the 1 / 2 / 0 rule directly: a cut-free run yields one
   LX-eligible tile-sized buffer for `b`, a cut yields two (`b` plus the
   consuming group's read copy), and `in_buffer[b] == 0` yields none. Assert the
   read copies are **optional** rectangles (R4.9) — that a cut-free solution
   reserves no space for a read copy that will not be created, and that a
   solution cutting an untiled producer still creates one in the consumer.
   Assert `full_size[b]` equals the realized `full_buf` footprint and
   `boundary_view[b]` the realized per-core view, both after `coarse_tile` runs;
   drift there feeds the cost model wrong numbers while the solve still reports
   optimal.

   Then the untiled→tiled case, which is the cut-cost table's most surprising
   row and the one most forced cuts land on: assert **no** `full_buf` is
   allocated and no `MutationLayoutSHOULDREMOVE` op appears. For the producer,
   assert both halves of the row-3 rule (§2): with an advancing consumer read it
   is stamped `"tiled (advancing)"` — evicted, and the model's
   `in_buffer[producer] == 0` implication agrees — while with a read invariant
   along every tiled dim it keeps a `None` `residency_reason` and stays a
   candidate. Left unasserted, this row is the one a future change to the
   propagation loop would silently break.
8. **Per-core view prediction (R2.6).** For every config of every op in the
   coarse-tiling suites, assert the predicted per-core view equals the one
   recomputed by `_prepare_per_core_view` after `coarse_tile` has actually run.
   Separately, assert `_views_for_divs`'s `prep_cache` never returns a prep built
   under a different tiling — the stale-prep failure is silent, and it produces a
   `config_matches` entry claiming two configs slice a buffer identically when
   they do not. Pair with a negative test: two configs of one op differing *only*
   in tiling must not share a cache entry.
9. **Over-tiling fix.** A case where span overflow forces tiling *and* work
   division splits the same dim: assert the joint model picks a strictly smaller
   tile count than the `core_split_estimate = 1` path.
10. **End-to-end performance.** `mlp-linear-kn.t` and `mha_4h` at `SENCORES=32`,
    the two benchmarks already tracked in `scratchpad_planning.md`, measured
    against the baselines recorded there: `mlp-linear-kn.t` at ~79%
    process-engine utilization after pointwise seeding, ~17% below its
    pre-seeding fused kernel time (`:519`); `mha_4h` converging on `B/4·M/8`
    with the scores matrix pinned (`:508`), but with the reduction option
    pushing search into tens of seconds (`:566`). Report PE utilization, fused
    kernel time, *and* solve time against the gate-off baseline — the last
    matters because tiling enlarges the model.
11. **Determinism.** Identical plans across two runs under
    `torch.use_deterministic_algorithms(True)`.
12. **Fallback.** With the gate on and the solver forced to fail (an epsilon
    time limit or an injected `SolveError` — a zero limit is skipped by the
    `if self._time_limit_seconds` guard, `ilp_solver_ortools.py:457`), a graph
    that requires span-forced tiling compiles through the
    retained span-overflow path and matches today's plan; assert the failed
    solve left no trace in the IR (the solve precedes `coarse_tile`, R8.3).

## Alternatives considered

**A separate CP-SAT tiling stage ahead of layout planning.** Cleaner to land and
test, but it reproduces the current defect in a new place: a tiling chosen
without seeing LX occupancy or the core division still has to guess. Rejected in
favour of one joint model.

**Keeping the hardcoded objective and adding tiling terms to it.**
Requires editing the solver for every cost experiment, and the interesting
question — how to trade HBM traffic against parallelism against loop overhead —
is exactly the one that needs iteration. Rejected in favour of injection. (Today's
objective is two-phase lexicographic; §4 replaces it with a single-phase weighted
one independently of the injection question — see R3.2.)

**Pre-computing tiling groups from producer/consumer connectivity, then having
CP-SAT pick one tiling per group.** A smaller model, but the grouping heuristic
becomes a second place where a wrong guess is unrecoverable, and grouping is
precisely what the objective should decide. Rejected in favour of cut variables.

**Modelling tiling and division as independent variables** rather than a
precomputed config cross product. Keeps the model smaller in variable count, but
reintroduces the products and divisibility conditions as nonlinear constraints.
The config encoding absorbs them into `AddElement` table lookups at the cost of
enumeration, which is bounded by the caps in R1.7.

## Resolved design decisions

These four were raised as open questions and have been resolved for phase 1:

- **Segmentation granularity — resolved: whole-graph.** The solve is a single
  CP-SAT instance over the entire graph, not decomposed at matmul or any other op
  boundary (§6). No op-specific break is assumed; matmul cuts fall out of
  untileability, not a segmentation rule.
- **How cuts should be priced — resolved: loops are free.** A cut is priced only
  through the consequences it materializes; no `n_groups` or per-cut term (§3). A
  cut that materializes no boundary copy costs nothing. This is a working
  assumption and can be revised with a loop-overhead term if a later cost model
  justifies it.
- **Config cap per op — resolved: no cap.** The model does not cap configs per op;
  model size is controlled by external pruning of the enumerated set (§6, R2.4).
- **Default objective — resolved: keep today's terms, single-phase.** The default
  stays today's spill and core terms, now combined into one single-phase weighted
  objective rather than the two-phase lexicographic solve (§4, R3.2, R3.5).

## Open questions

- **Objective tuning.** The single-phase default reproduces today's terms with
  the spill term weighted to dominate. What weighting, and what additional terms
  (tile count / loop overhead, `peak_lx_bytes`), should the default carry once the
  mechanism is trusted?
- **Stickification.** Relayout cost is unmodelled (R6), which can be pessimistic
  on relayout-driven HBM traffic. When is the follow-on `relayout[e]` work worth
  landing, and does tiling make that pessimism large enough to reprioritize it?
- **Reduction-axis tiling.** Excluded for phase 1 (R9). Bringing it in scope has
  three prerequisites, and the first two are independent of this RFC: fix the
  nested output+reduction wrong-numerics so `_validate_reduction_tiling`'s stated
  contract matches reality; extend the per-config cost tables to include the
  accumulator and fill buffers, so the solver does not read reduction tiling as
  free LX relief. Only the third is a modelling question here — expressing the
  group invariant needs per-run, per-level literals rather than pairwise cut
  tables (R4.6), which is a real increase in model size. Is that worth paying for,
  or is reduction-axis tiling better left permanently hint-only?
