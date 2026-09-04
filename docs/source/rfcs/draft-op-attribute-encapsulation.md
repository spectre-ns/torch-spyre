# Encapsulating op, graph and layout attributes

Written against `enable-default-cooptimization` at `5819bfbf`. Every count below was
measured this session by walking the AST or reading the file; line numbers were
verified at `faafe4e1`, whose diff against `5819bfbf` touches one test file and none
of the sources cited here.

The goal: the scratchpad planner stops interrogating objects with `getattr` /
`hasattr` / `isinstance` and instead asks types that answer for themselves — Spyre
ops, the Spyre graph, and the Spyre layout. Reflection disappears as a consequence
of the questions acquiring names and owners, not as an end in itself.

## 1. What the planner asks, and why it asks it badly

### 1.1 The census

`torch_spyre/_inductor/scratchpad/` contains **159** builtin reflection call sites:
101 `isinstance`, 39 `getattr`, 18 `hasattr`, 1 `setattr`. They are not spread
evenly.

| File | isinstance | getattr | hasattr | setattr | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `allocator.py` | 39 | 14 | 15 | 0 | 68 |
| `ilp_solver_ortools.py` | 25 | 2 | 1 | 0 | 28 |
| `graph_editor.py` | 13 | 5 | 2 | 0 | 20 |
| `utils.py` | 7 | 13 | 0 | 0 | 20 |
| `lx_relayout.py` | 11 | 1 | 0 | 1 | 13 |
| `simulated_annealing.py` | 3 | 0 | 0 | 0 | 3 |
| `coarse_tiling.py` | 1 | 2 | 0 | 0 | 3 |
| `plan_solver.py` | 0 | 2 | 0 | 0 | 2 |
| `contact_profile.py`, `greedy_solver.py` | 2 | 0 | 0 | 0 | 2 |

Six files have none at all — `permutation_layout.py` is 1604 lines with zero.

Classifying the 101 `isinstance` calls by what they narrow: **58** discriminate
upstream Inductor IR classes (`ComputedBuffer` 22, `MemoryDep` 10,
`MutationLayoutSHOULDREMOVE` 7, container unwrapping 7, `Reduction` 4, others),
**26** are sympy / OR-Tools / stdlib narrowing (16 of them inside one CP-SAT
expression printer in `ilp_solver_ortools.py`), and **17** test a Spyre-owned type.

That distribution is the first thing this plan has to be honest about; §2 states the
consequence.

### 1.2 Thirty-one questions, twelve of which are ours

Behind those 159 sites are **31 distinct questions**. Twelve are Spyre questions,
asked 47 times; nineteen are upstream-IR, sympy, OR-Tools or stdlib questions, asked
112 times. The recurring ones:

| Question | Sites | Spellings in use |
| --- | ---: | --- |
| Is this a placeable compute op? | 25 | `isinstance(op, ComputedBuffer)` ×22, `hasattr(op, "data")` ×3 |
| Is this dep indexed (vs a `StarDep`)? | 16 | `isinstance(dep, MemoryDep)` ×10, `hasattr(dep, "index")` ×6 |
| Does this buffer carry a Spyre device layout? | 15 | `isinstance(l, FixedTiledLayout)` ×7, `hasattr(l, "device_layout")` ×7, `getattr(...)` ×1 |
| Is this buffer an in-place mutation alias? | 7 | `isinstance(op.layout, MutationLayoutSHOULDREMOVE)` ×7 |
| Does this op carry a coarse-tile loop nest? | 5 | `getattr(op, "loop_info", None) is not None`, `hasattr`, and both combined |
| What core division was committed for this op? | 5 | `getattr(..., None)` ×4, `hasattr` ×1 |

The count is not the problem. **The multiple spellings are**, because they have
already diverged.

### 1.3 The divergence is live, and it is a wrong-answer class

`device_layout` is assigned in exactly one place (`ir.py:105`) and never deleted, so
`hasattr(layout, "device_layout")` *is* `isinstance(layout, FixedTiledLayout)`. Two
spellings, one meaning — until an upstream wrapper gets involved:

- `allocator.py:1677-1681` unwraps `layout.real_layout()` **before** testing
  `FixedTiledLayout`, so a mutated tiled buffer answers **True**.
- `utils.py:227-229` and `allocator.py:392-396` reject the mutation layout first, so
  the same buffer answers **False**.
- The other ~12 sites never unwrap at all, and answer **False**.

`MutationLayoutSHOULDREMOVE` (upstream `ir.py:4886-4926`) forwards `stride` and
`storage_size()` to `real_layout()` but defines neither `allocation` nor
`device_layout`, and it is upstream code we cannot extend. So the same question,
spelled the same way, has two answers in one package, with no local way to tell
which was intended. That is the defect this plan exists to close; the reflection
count is a symptom.

### 1.4 The same computation, written out several times

**Per-core slicing agreement — three implementations over one geometry kernel.**
"Do this buffer's users agree on how it is sliced across cores?" is answered by
`get_ncores_for_buffers` (`scratchpad/utils.py:537-628`, the fixed-division path),
by `ResidencyEdge` + `_cd_parent_matches` (`allocator.py:1450-1577`, `2560-2596`,
the joint path) and by `_clone_divisions_and_matches` (`allocator.py:2493-2557`, the
input-clone path). All three call the same `_per_core_view_from_prep`
(`pass_utils.py:2941`) and then apply **strictly different guard sets**:
`get_ncores_for_buffers` has a broadcast-read guard the others lack and silently
discards the `representable` flag the others honour; the multi-dimensional-split
matmul rejection exists only in `ResidencyEdge.parent_view`. The comment at
`allocator.py:1478-1479` asserting parity with a matmul guard in
`get_ncores_for_buffers` is false — that file does not mention matmul at all.
Separately, `get_ncores_for_buffers` runs three times per co-optimized solve and
every one of its results is discarded.

**Residency reasons — two ladders.** `_buffer_residency_reason`
(`allocator.py:452-541`, 13 sequential-return branches) has a near-duplicate in
`_input_residency_reason` (`allocator.py:543-587`, 10 branches), sharing seven reason
strings **in a different order**. Branch 2 of the first (`"unsized (no device
layout)"`, `allocator.py:492`) is dead as written: branch 1's
`_op_output_good_for_lx_reuse` already requires `isinstance(op.layout,
FixedTiledLayout)` (`allocator.py:396-399`), and `FixedTiledLayout.__init__` always
sets `device_layout`. On the input path there is no such gate at all, and
`_would_produce_lx_back_gap` runs unguarded at `allocator.py:585`.

**Buffer byte size — one formula, five failure semantics.**
`math.prod(device_size[:-1]) * 128` is written out at `ir.py:490`,
`hbm_pool_planning.py:119`, `scratchpad/utils.py:242`, `allocator.py:878` and
`allocator.py:2374` (plus the C++ original at `csrc/spyre_tensor_impl.cpp:298-303`).
When the layout is not a `FixedTiledLayout` those five sites respectively fall back
silently to the logical numel, assert, return a `-1` sentinel, return `0`, and raise
`AttributeError` unguarded. Some of that spread is real — it encodes which pipeline
phase the caller runs in — and §3.5 says which parts may be merged and which may not.

**LX residency — a string membership test at ~22 sites.** `"lx" in
layout.allocation` is open-coded about 22 times across ten files (seven in
`codegen/compute_ops.py` alone), with an `hbm_pool` twin open-coded 11 more times and
`codegen/ktir.py:841/864` spelling both at once through `INTERNAL_SPACES`. The
`allocation` dict is written in only six places and cleared in two.

**Op identity — four namespaces, one collision.** Pooling is asked as the FX short
name `"avg_pool2d"` (`utils.py:51-57`) and as the SDSC `reduction_type`
`"avgpoolfwd"` (`constants.py:204-207`, `allocator.py:1262`). The string `"conv2d"`
means *depthwise* in the first namespace and *forward convolution* in the second.

### 1.5 Where the metadata actually lives

Op-level Spyre state has no declared type. `_SPYRE_METADATA_ATTRS`
(`loop_info.py:375-391`) names ten attributes and `copy_op_metadata`
(`loop_info.py:394-404`) propagates them by *name* through a
`hasattr`/`getattr`/`setattr` loop. That is only the bulk-copy subset: at least
seventeen Spyre names are monkey-attached in total, plus five more on the
`GraphLowering` itself. Metadata copying is spread across **three** hand-maintained
lists — `_SPYRE_METADATA_ATTRS`, the three-tuple duplicated at
`read_copy_elision.py:170` and `:396`, and the bare `replacement.loop_info = ...` at
`read_copy_elision.py:395` — and attributes in none of them are dropped by every
reconstruction (`_coarse_tile_force_live`, set at `wsr/coarse_tile.py:5183/5263/5489`
and read at `patches.py:148` to defeat scheduler DCE, is the sharpest example).

Two structural facts about that surface shape the whole plan.

**The carriers are heterogeneous.** `loop_info` is asked of `Operation` while
iterating the mixed `graph.operations` list (`passes.py:388-391`,
`allocator.py:1265-1268`). `layouts` is stamped on `TensorBox`
(`propagate_layouts.py:2170`, `:2406`), on `InputBuffer` (`optimize_restickify.py:679`)
and on the `ExternKernel` fallbacks (`propagate_layouts.py:2461-2489`). A
`ComputedBuffer` subtype alone reaches none of those.

**Absence is load-bearing.** Exactly 32 `hasattr` sites over Spyre names are
control-flow predicates, and ten delete sites use removal as a *state
transition* — nine explicit `del` lines plus one `hasattr`-guarded `delattr`
loop over a three-name tuple (`insert_restickify.py:462-464`): `pass_utils.py:1846-1848` sets `op_it_space_splits` and then deletes
`iteration_space_ownership`, and `allocator.py:2226` reads that absence as "skip this
op". Five of those `hasattr` guards are outright blockers if absence stops being
expressible — `wsr/coarse_tile.py:219` would classify every mutated source as
loop-written, `pass_utils.py:1893` would call a helper that *raises* for write-free
ops, `wsr/coarse_tile.py:5669` would invert `_index_consumer`, and
`allocator.py:2226` would commit divisions for ops `_distribute_work` deliberately
skipped. §3.1 is the design that makes this a non-issue rather than a prerequisite.

### 1.6 The planner has no shared context, and rebuilds the same maps

`allocator.py:600-609` says outright that there is no shared context object. The
consequence is an eight-keyword-argument signature at `allocator.py:452-467` and a
set of graph-scoped maps rebuilt two to four times per plan: `_get_buffer_user_deps`
three times on the placement path and four on the co-optimized one,
`get_ncores_for_buffers` two and three, `mem_usage_by_buf` one and three, and
`op_by_name` plus `set(graph.get_output_names())` three times each.

The `cache` parameters on `mem_usage_by_buf` (`utils.py:198`) and
`get_ncores_for_buffers` (`utils.py:538`) are used at exactly one call site each.
`mem_usage_by_buf` is called four times (`allocator.py:947`, `:1977`, `:2248`,
`:2336`) and only the first supplies a cache; that cache is then threaded on to
`get_ncores_for_buffers` at `utils.py:209`, while the two direct calls at
`allocator.py:626` and `:945` supply none. Hoisting a single shared cache is
therefore not a pure speed-up — it makes previously uncached sites cached, which is a
behaviour change on top of the staleness question, and §3.4 treats it as one.

The genuine O(N²) scan is `buffer_not_read_in_full` (`utils.py:254`, full
`for op in graph.operations` at `:296`), called once per buffer from both residency
ladders. The pin ladder's group-set rescan, which an earlier draft flagged, has
already been hoisted out of the per-op loop (`allocator.py:2086-2098`).

### 1.7 Three corrections to the record

`_is_coarse_tiled` (`allocator.py:1266`) is **not** dead. It has exactly one live
call site, `allocator.py:1290`, inside `_drop_reduction_splits_in_coarse_group`
(itself called at `:2134`). It is not a pin in `_division_map`, and
`_determine_in_place_division_invariant` (`:2233-2296`) still contains no coarse-tile
gate. Any statement that it has zero call sites is wrong as of this commit.

The `_division_map` pin ladder is six guards, and its loop-invariant group sets are
already hoisted.

Upstream is *nearly* free of exact-type checks on IR nodes, not entirely:
`torch/_inductor/utils.py:3348` does `type(node) is ir.FallbackKernel` inside
`is_collective`. It is torchrec-specific and only ever answers "not a collective" on
the Spyre path, so it is inert here — but the plan should not rest on an absolute.

## 2. What this plan promises, and what it does not

**It does not promise that reflection goes to zero.** After a perfect refactor,
`scratchpad/` still holds roughly 43 of today's 159 sites: the sympy/OR-Tools
narrowing inside the CP-SAT printer, upstream container unwrapping in
`graph_editor.py`, and the `isinstance` narrowings whose false branch is an
upstream-owned object Spyre never constructs. The empirical proof that better types
do not remove those: `FixedTiledLayout` is exactly the Liskov-substitutable Spyre
subtype of an Inductor class that this refactor wants more of — and it did not remove
a single `isinstance`. It created 62, because callers hold a variable typed `Layout`
or `OutputSpec`.

**The raw count is also the wrong metric**, and a gameable one: `scratchpad/`'s 159
sites are 12.4% of the package's 1285, and the same questions are asked 83-89% of the
time *outside* it (`ComputedBuffer` 104 of 126 sites, `FixedTiledLayout` 55 of 62,
`loop_info` 31 of 36). Moving an `isinstance` one directory up would improve a
scratchpad-local count while making the codebase worse. That is why §4 puts the
predicates in a package-level home and converts `scratchpad/` first, rather than
defining a second spelling inside it.

So this plan is graded on four numbers instead:

| Metric | Today | Target |
| --- | ---: | ---: |
| Spyre questions with no name or owner | 12 | 0 |
| Redundant answer sites for those 12 questions | 35 | 0 |
| Questions with more than one non-equivalent spelling | 4 | 0 |
| Duplicated implementations of one algorithm | 3 slicing-agreement, 2 residency ladders, 5 size formulas | 1 each |

plus one qualitative claim: the two `elif` ladders become registries whose membership
a test can check, so the next guard cannot fall out of one unnoticed.

### 2.1 Static typing will not carry any of this, and does not today

The library is annotated and mypy runs in pre-commit over `torch_spyre` and `tests`
(`.pre-commit-config.yaml:36-42`). But that hook installs only `types-PyYAML`, so its
isolated environment has **no torch** — and with `ignore_missing_imports = true`
(`pyproject.toml:104`), `torch._inductor.ir.ComputedBuffer` resolves to `Any`. A
class deriving from `Any` accepts any attribute. This was tested directly: two probe
files, one assigning `op.loop_info` on a bare `ComputedBuffer` and one declaring a
`ComputedBuffer` subclass and then reading `op.typo_attribute_that_does_not_exist`,
both pass the gating hook — *"Success: no issues found in 196 source files"*.

Three consequences the plan is built around:

1. The `disable_error_code = ["attr-defined"]` carve-out at `pyproject.toml:107-109`
   is **already inert in CI**. Removing it changes nothing observable, and this plan
   does not propose to (the user's call). Saying the refactor "removes the reason for
   the carve-out" is true but buys nothing measurable today.
2. After the refactor, mypy still will not flag a dropped or misspelled Spyre
   attribute. **100% of the safety net is runtime invariants and tests**, and §5 is
   sized accordingly.
3. The escape hatch is not cheap: `.venv/bin/mypy torch_spyre tests` at HEAD reports
   **285 errors in 37 files** (131 `arg-type`, 64 `union-attr`, 35 `assignment`).
   Adding torch to the hook is its own project, and it is out of scope here.

### 2.2 The gate that would grade this refactor does not currently run

Two findings, both verified, that any acceptance criterion has to account for.

**The pre-scheduling pipeline is outside the FX-graph cache key.**
`CustomPreSchedulingPasses` is instantiated at `patches.py:112` and invoked from the
`_spyre_update_scheduler` monkeypatch; it is never assigned into any
`torch._inductor.config` field, and `new_config` (`patches.py:80-97`) registers only
`pre_grad_custom_pass`, `post_grad_custom_pre_pass`, `post_grad_custom_post_pass`,
`_pre_fusion_custom_pass` and `_post_fusion_custom_pass`. `FxGraphHashDetails`
(`codecache.py:1064-1093`) reads exactly those slots plus the two joint ones. So
`CustomPreSchedulingPasses.uuid()` — which `_uuid` (`passes.py:152-159`) computes
faithfully over 16 pass source files — is never consulted. Editing
`scratchpad/allocator.py`, `work_division.py`, `wsr/coarse_tile.py`,
`insert_restickify.py`, `ir.py`, `pass_utils.py` or any new module does **not**
invalidate a warm cache.

And a cache hit is total: the same tiny Spyre compile run twice against one
`TORCHINDUCTOR_CACHE_DIR` gives `fxgraph_cache_miss` with the pre-scheduling pipeline
executing once, then `fxgraph_cache_hit` with it executing **zero** times — work
division, LX planning and coarse tiling do not run at all. `fx_graph_cache` is on by
default and only ~10 test modules disable it. Spyre *config* changes (`SENCORES`,
`LAYOUT_SOLVER`, `CO_OPTIMIZING_LX_PLANNING`) do bust the key; source edits, which
are precisely what this refactor is, do not. A "capture divisions, refactor,
re-capture, diff" gate would therefore report zero changes on stale artifacts — and
the revert would be equally non-restoring. **S0 fixes this before anything else.**

**`use_deterministic_algorithms(True)` does not make a division baseline
reproducible.** It sets `num_search_workers = 1`
(`scratchpad/ilp_solver_ortools.py:816-818`), but the line above sets
`max_time_in_seconds` from `config.cpsat_time_limit_seconds`, default **120**
(`config.py:193-195`), and the three-phase lexicographic solve at `:840-885` accepts
`FEASIBLE` as well as `OPTIMAL` and then locks the next phase from that possibly
truncated incumbent (`:851`, `:881`). A deadline hit in phase 1 installs a *looser*
residency lock, so phases 2 and 3 can commit a genuinely different plan — not merely
a different tie-break. Baselines are reproducible only on graphs that reach OPTIMAL
inside the limit; the baseline harness must pin the limit high and record the solve
status per graph.

## 3. The shape

Four owned types and one ordering concept. Each is independently useful; together
they are what lets the questions have owners.

### 3.1 `SpyreAttr`: a declared attribute that preserves absence

The pivot of the whole design. Every Spyre op attribute becomes a typed data
descriptor whose value lives under a private `_sp_<name>` key of the instance
`__dict__`, and which **raises `AttributeError` when unset**:

```python
class SpyreAttr(Generic[T]):
    __slots__ = ("public", "slot", "default")

    def __set_name__(self, owner, name):
        self.public, self.slot = name, "_sp_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        try:
            return obj.__dict__[self.slot]
        except KeyError:
            if self.default is not _UNSET:
                return self.default
            raise AttributeError(...) from None

    def __set__(self, obj, value):
        obj.__dict__[self.slot] = value

    def __delete__(self, obj):
        del obj.__dict__[self.slot]      # AttributeError if unset
```

That single choice is what makes this refactor tractable. With it,
`hasattr(op, "loop_info")` is still `False` before the attribute is stamped,
`getattr(op, name, default)` still returns `default`, and `del op.name` still works —
so **none of the 32 `hasattr`, ~100 `getattr`-with-default or 10 delete sites has
to change for the type to land**. The three name-list-driven loops keep working too,
for the same reason: `copy_op_metadata`'s `hasattr`/`getattr`/`setattr` walk
(`loop_info.py:402-403`), the two `("layouts", "restick_cost_fn",
"op_it_space_splits")` tuples in `read_copy_elision.py:170/396`, and the
`delattr` loop at `insert_restickify.py:462-464`. The absence-semantics migration catalogued in §1.5
stops being a prerequisite and becomes optional later cleanup, which removes the
largest correctness risk from the critical path.

Two consequences to write into the module docstring:

- A default may be given **only** for a name with no `hasattr` and no `del` site
  anywhere. Today exactly one qualifies (`_coarse_tile_force_live`). Two names have
  *conflicting* absence defaults across call sites and can never become a bare
  attribute read until those are unified — `dim_hints` is fetched with a `[]` default
  at `wsr/coarse_tile_hints.py:54/84` and `wsr/coarse_tile_span_overflow.py:467` but
  with `None` at `pass_utils.py:3278` and `wsr/propagate_named_dims.py:670`;
  `op_it_space_splits` is fetched with `({}, {})` at `pass_utils.py:3229` and `None`
  at three other sites. Unifying those is a separate behaviour change.
- `_ts_cached_read_writes` must be **excluded** from the descriptor set. It is
  addressed through `op.__dict__.get/[]=/pop` at `pass_utils.py:188/191/216` — the
  only public-name `__dict__` writes on an op in the tree. A descriptor there would
  make `invalidate_op_read_writes` silently no-op, and it is called immediately after
  `inner_fn` load-name swaps (`wsr/coarse_tile.py:2100/2206`,
  `scratchpad/graph_editor.py:293`), so the memo would then serve the old body's
  reads and writes to ~30 planner call sites.

### 3.2 `SpyreOpMixin` and the per-carrier subtypes

One mixin declares the ~17 attributes and the predicate methods; per-carrier subtypes
are `class SpyreX(SpyreOpMixin, ir.X): pass`. They add **zero dataclass fields**, so
`__dataclass_fields__` is literally the same object as the base's and `__init__`,
`__eq__` and `__hash__` are the same functions (`__hash__` stays `None` —
`ComputedBuffer` is deliberately unhashable, documented at
`wsr/coarse_tile.py:450-460`).

The carrier set, established from the two exhaustive visitors that dispatch over
`graph.operations` (`propagate_layouts.py:2179-2495` and
`work_division.py:1846-1882`):

| Carrier | Minted by | Spyre metadata it receives |
| --- | --- | --- |
| `ir.ComputedBuffer` | upstream `StorageBox.realize` (`ir.py:10024`) + 12 torch-spyre sites | all ~17 |
| `ir.MultiOutput` | upstream `FallbackKernel.create` | `layouts`, `restick_cost_fn` |
| `ir.FallbackKernel` | upstream `make_fallback` | none today |
| `ir.DeviceCopy` | upstream `to_device` | `layouts`, `restick_cost_fn` |
| the six `ExternKernel` fallbacks in `ir.py` | torch-spyre | `layouts`, `restick_cost_fn` |

The six torch-spyre fallbacks need no promotion at all — they take the mixin at class
definition time. `ConcatKernel`/`NopKernel` are unreachable because torch-spyre owns
the `aten.cat` and `aten.constant_pad_nd` lowerings.

**Liskov rules for the subtype**, each with a reason:

1. **Additive only.** No override of any inherited member. In particular never
   `storage_size()` (upstream's contract is host elements, consumed by
   `is_dense_contiguous_storage_and_layout` → `View.create`), never `__eq__`, never
   `__hash__`.
2. **No `__init__` or `__post_init__` on the mixin.** All six torch-spyre
   `ExternKernel` subclasses call `super().__init__(...)` *positionally* into
   `ExternKernel.__init__`; putting the mixin first in the MRO is safe only while it
   defines no initialiser.
3. **No cached derived properties.** `pass_utils.py:2229` and `:2246-2249` already
   hand-patch cache staleness on IR nodes; a cached property on the subtype would add
   a third instance of the same bug.
4. **Exact-class registry keying**, not an MRO walk. An unknown `ComputedBuffer`
   subclass must be given a subtype *of its own class*, never narrowed to
   `SpyreComputedBuffer`.
5. **Do not restore hashability**, and change the one op-to-op `==` in the tree —
   `assert new_ops[0] == padded_buf` (`pass_utils.py:2670`) — to `is`. The dataclass
   `__eq__` compares `__class__`, so base and subtype never compare equal; both sides
   there are the same object today, and `is` makes a future half-promotion fail loudly
   instead of silently.

### 3.3 Promotion: at birth, at the one funnel, moving values

`GraphLowering.register_operation` (`graph.py:1094-1101`) is the only writer of
`graph.operations` — `self.operations.append(op)` at `:1098` is the sole `append` in
the whole `torch/_inductor` tree. Patching *that method* inside
`enable_spyre_context` means every op is born promoted, including everything minted
by the seven passes that re-enter upstream lowering (`padding.py:727/732`,
`pass_utils.py:2630`, `insert_restickify.py:236/250`, `split_multi_ops.py:479/562`).
That is what dissolves the re-entry problem structurally rather than by sweeping.

Three constraints on that patch:

- **Device-gate it.** `register_operation` is device-agnostic and fires for every op
  of every graph compiled inside `enable_spyre_context`. Mixed CPU/Spyre graphs are
  real — `passes.py:130-148` carries three separate `_*_have_spyre_device` guards for
  exactly that reason. Promote only when `op.get_device().type == DEVICE_NAME`.
- **Promote after the original call**, which asserts `operation_name is None` and
  stamps the name.
- **Move values, never bare-reclass.** An op whose `__dict__` already holds
  `loop_info` under the public name, reclassed into a type where `loop_info` is a
  descriptor, reads back as *unset* while the value sits unreachable in `__dict__`.
  Measured. Two named wrong-code endpoints: `_is_coarse_tiled` goes False, making
  `_drop_reduction_splits_in_coarse_group` a no-op and re-admitting the K-splits its
  own docstring records as ~86% element mismatch on a flash-attention output matmul;
  and `spyre_kernel.py:637/873` stop emitting tile-advance metadata entirely. So
  `promote_operation` pops each public Spyre name out of `__dict__` first and
  re-`setattr`s it after the reclass, and a debug-mode assert checks that no promoted
  op retains a public `__dict__` key colliding with a declared descriptor.

**The funnel is not sufficient on its own, and the plan must not claim it is.** About
twenty torch-spyre sites splice hand-built ops straight into `graph.operations` and
set `operation_name` by hand precisely to bypass `register_operation`
(`insert_restickify.py:300`, `padding.py:129`, `split_multi_ops.py:577`,
`enforce_indirect_access_layout.py:621/633`, `scratchpad/graph_editor.py:241`,
`wsr/coarse_tile.py:3564/4140/4475/5039`, and an in-place replacement at
`pass_utils.py:2234`). Those are covered on day one by converting the 12 construction
sites to the factory — but they get **no safety net** from the funnel, so a
thirteenth site added later would be silently unpromoted. The between-pass hook is
therefore a **validator** (`assert all(is_spyre_op(o) for o in graph.operations)`),
not a second promoter; making it a promoter would mask exactly the drift it exists to
catch.

An unknown carrier class synthesizes a subtype in production and **raises** under
`config.spyre_ir_strict_promotion`, which the test suite sets. A hard raise in
production would turn a torch bump that adds one `Operation` class into a total
compile failure.

### 3.4 `SpyreGraphLowering`, phases, and epochs

The same in-place reclass works for `GraphLowering` (verified), so the graph can
report its own attributes too. It carries five today, in three different idioms:
`restickify_plan` (`insert_restickify.py:607`), `hbm_pool_sizes`
(`hbm_pool_planning.py:222`), `_emitted_layout_targets` (via
`V.graph.__dict__.setdefault`, `scheduler.py:855`), `_spyre_lx_relayout_copies`
(reached through a module constant, `lx_relayout.py:49/104/399`) and
`_spyre_pre_scheduling_complete` (`patches.py:117/128`).

Three of those need individual care rather than a bulk rename:

- `scheduler.py:855`'s `__dict__.setdefault` is correct only while the key is genuinely
  absent, and the dedup it implements is a *correctness* mechanism (a target's device
  layout must be restored exactly once per generated program). Seeding the name to
  `None` makes `setdefault` return `None` and the next `.add()` an `AttributeError`.
- `lx_relayout.py:390-401` reads the registry, asserts it empty, installs it, then
  **mutates the object it just installed**. It breaks under a `None` default
  (`None[...]`) and under a class-level `{}` (process-shared across graphs). It needs
  an explicit `if copies is None: copies = {}` before the `setattr`.
- `insert_restickify.py:773`'s `assert hasattr(graph, "restickify_plan")` is a
  *pass-ordering* guard, not a data check. Any declared field makes it vacuously true
  and degrades its failure into `None.get(...)` at `:790`. It becomes an explicit
  phase check in the same commit that gives the field a default.

**`_update_scheduler` runs more than once per graph** (`graph.py:2568/2587/2607/2640`),
and `patches.py:117`'s flag is the only re-entry guard. If a rename leaves the flag
written under one name and read under another, the entire pre-scheduling pipeline
re-runs on an already-transformed graph — tripping `assert not
materialized_lx_relayouts(graph)` or `"Operation registered twice"`, or double-inserting
restickify buffers. Rename that flag last, alone, with a test that calls
`_update_scheduler` twice.

**Phase.** Six of the predicates this plan exposes are only well-defined after a
particular pass, and today they answer *silently and wrongly* before it:
`_fixed_core_division` returns a one-core division when `iteration_space_ownership` is
absent (`allocator.py:1580-1583`), and `work_division_splits_are_legal` returns
**True — every split legal** when the layout is not yet a `FixedTiledLayout`
(`work_division.py:1012-1016`). An ordered `Phase` enum stamped by the pass runner,
plus `require_phase(...)` assertions in those helpers, converts both into loud
failures. This is the piece that makes "types answer for themselves" safe: a type
alone cannot express that its own answer is not yet meaningful.

**Epochs.** Memoizing the graph-scoped maps of §1.6 is only safe with an invalidation
signal, because the planner mutates the graph mid-solve — `graph_editor.py:240-243`
removes and reinserts operations *inside* `_push_allocation`'s loop, which staleness
every `LifetimeBoundBuffer.uses` list already built. Two counters (`graph_epoch`,
`division_epoch`), bumped at the structural-edit and division-commit choke points,
with a structure fingerprint checked under a config flag. Memoize only
structurally-derived facts; anything ownership-derived keys on `division_epoch`.

Note the caching subtlety from §1.6: hoisting one shared cache makes previously
uncached call sites cached. That is a behaviour change and needs the byte-identical
plan gate, not just a speed measurement.

### 3.5 `FixedTiledLayout`: additive accessors only

The layout is the one type we already own outright, and it is the cheapest, safest
win. Add read-only properties — `device_bytes`, `num_sticks`, `is_lx`,
`is_hbm_pool`, `is_kernel_internal`, `lx_address`, `hbm_pool_address` — plus two
module-level functions over the raw `SpyreTensorLayout` for callers that hold one.

Constraints, each verified at runtime:

- **Never override `__eq__` or `__hash__`.** Today `FixedTiledLayout ==
  FixedLayout` is `True` for the same host geometry, and two `FixedTiledLayout`s
  differing only in `element_arrangement` compare equal *and* hash equal. That is
  wrong-looking, and fixing it is a non-additive change to a member upstream calls
  (`ir.py:6094` asserts layout equality). Pin the current behaviour with a test so a
  future override fails loudly, and file the equality gap rather than fixing it here.
- **Never override `storage_size()`.** Upstream's contract is host elements. This is
  already why `ir.py:485-495` exists as a separate function rather than an override.
- **`allocation` stays a public rebindable dict.** `hbm_pool_planning.py:274-289`
  keys aliasing on `id(layout.allocation)`, and `TensorArg.allocation`
  (`op_spec.py:230`) is the *same dict object* aliased out of the layout
  (`spyre_kernel.py:797`), with `spyre_kernel.py:1310` writing through it back into
  the layout. So `is_lx` is a read-only *view*, not a property with a setter, and the
  five write sites stay as they are.
- **Non-goal:** `NoneLayout` and `MultiOutputLayout` are `OutputSpec` siblings, not
  `Layout` subclasses, and `Buffer.get_layout()` raises for them while `.layout`
  returns them silently. No property anywhere in the `Layout` tree reaches them, so
  the ~12 defensive `hasattr(..., "device_layout")` sites survive this refactor
  unchanged. Claiming otherwise would be the plan's biggest overreach.

The five size formulas collapse onto one function while **all five failure semantics
stay at their call sites**, because they encode pipeline phase rather than layout
state: `hbm_pool_planning.py` can assert because it runs post-fusion;
`scratchpad/utils.py` needs its `-1` sentinel because it legitimately sees plain host
`FixedLayout` buffers.

Finally, the mutation-unwrap question of §1.3 gets two *named* functions instead of
one ambiguous spelling — `own_tiled_layout(op)` (this op's own device layout, `None`
for a mutation alias) and `written_tiled_layout(op)` (the layout this op writes,
unwrapping `real_layout()`). Only the sites that already answer the way the new name
says get converted in that step; any site whose answer would *flip* is a separate,
deliberate change.

### 3.6 Two registries, and the duplications behind them

**Residency.** One rule set serves both ladders, each rule tagged with the subjects
it applies to (op output, graph input, or both), returning `Reason | None`. The
op-path order is preserved exactly; the input path's reason strings shift, and since
nothing in production reads them the change is made deliberately in one commit with
the before/after pair in the message. The dead branch of §1.4 is not deleted — it is
*promoted* to the input path, where the gate it duplicates does not exist and
`_would_produce_lx_back_gap` currently runs unguarded.

The visible contract is narrow and must be pinned: four or five residency literals
appear in tests, all echoed back from a test-supplied value, and the six
allocator-generated strings have **zero** direct test coverage. Zero `_division_map`
pin strings are asserted anywhere — that ladder's strings are debug-log only.

**Division pins.** Six rules, current order, with an exhaustiveness test that walks
the module for `(op: Operation) -> bool` callables and asserts each is registered or
explicitly allowlisted. Two of the six are *graph-level set memberships*
(`ops_in_offset_mutation_component`, `_fused_layout_group_ops`), not op predicates —
so the registry takes a context, and those two cannot become `op.is_*()` methods.
Note also that `_is_coarse_tiled` will need an allowlist entry with its justification,
since it is a live predicate that is deliberately *not* a pin (§1.7).

**The root-cause merges** are the point of this section, not the tables:

- The two closure builders (`ops_in_offset_mutation_component`,
  `_fused_layout_group_ops`) are the same graph walk with different radius and
  alias-following; they become one parameterised walker.
- `_reads_offset_slice` and `_writes_at_constant_offset` are one predicate over a
  side parameter.
- The three per-core-slicing-agreement implementations become one, with each caller
  opting into exactly the guard set it applies today. That merge is *not*
  behaviour-preserving if the guard sets are unified — it is behaviour-preserving
  only if each caller keeps its own set — so the merge lands first and any guard
  change lands after, one commit each, with the LX pin-set delta in the message.

## 4. The steps

Ordered. Each is independently landable and, except where noted, individually
revertable.

| # | Step | Size | Card? | Gate to leave it |
| --- | --- | --- | --- | --- |
| S0 | Put the pre-scheduling pipeline in the FX-graph cache key | S | no | Touching each pass source *and* each new type module changes `uuid()`; a second compile of an edited pipeline misses the cache |
| S0b | Guard the autouse HBM-poison fixture on device presence | S | no | The device-free planner files run green with no card; two meta-tests, one proving a real `DeviceOpenFail` still fails the session |
| S1 | `FixedTiledLayout` accessors; collapse the five size formulas and the ~22 LX open-codings | M | no | Byte-identical plans; the grep for open-coded spellings returns only the five write sites |
| S2 | Land `spyre_ir.py` — `SpyreAttr`, `SpyreOpMixin`, the subtypes, promotion, validator. No call sites | M | no | Unit tests for absence round-trip, idempotency, `__dict__`-move, synthesis fallback; nothing else imports it yet |
| S3 | Give the six torch-spyre `ExternKernel` fallbacks the mixin at class definition | S | yes | `dataclasses.fields` unchanged on all six; planner suite unchanged |
| S4 | Birth-time promotion at `register_operation` + the 12 construction sites + the between-pass validator | L | yes | Validator clean on every test graph in strict mode; compiled-op suite numerically identical vs CPU; **per-op division diff empty on all three corpora** |
| S5 | Move the predicates to a package-level home; convert `scratchpad/` call sites | M | yes | LX pin set and reason strings unchanged; division diff empty |
| S6 | `SpyreGraphLowering` + `Phase` + epochs + the facts memo | L | yes | Phase sequence monotonic; each mutation site bumps the expected counter and only that one; byte-identical plans; `_build_graph_facts` runs once per plan |
| S7 | The two registries, the closure merge, and the slicing-agreement merge | L | yes | Characterization tests from a prior step still pass; exhaustiveness test fails on the tree as it stands and passes after |
| S8 | The LSP defects in our own hierarchies (§4.1) | M | yes | Per sub-step; the `boundary` hoist is the one intentional behaviour change and lands alone |

**S0 is genuinely first.** Every other step's gate is a before/after comparison, and
§2.2 shows those comparisons are currently served from a cache that does not know the
code changed. Until S0 lands, every gate below must additionally set
`force_disable_caches` — and that must be asserted in the harness, not assumed.

**S4 is the one high-risk, effectively one-way step.** Land it alone, with the
measured numerical delta in the commit message so a bisect can attribute a later
regression. Its characteristic failure is an op reaching the planner unpromoted;
because an unset descriptor and an unpromoted op are indistinguishable at every
`hasattr`/`getattr` site, no existing test can detect it. The validator is the only
thing that can, which is why it is part of the same step.

### 4.1 The Liskov defects we can actually fix

"Follow Liskov where possible" is unambiguously achievable inside torch-spyre's own
hierarchies, where we own both sides. Five defects, in order of how much branching
they force on callers:

1. **`boundary` lives only on the subclass.** `ilp_solver_ortools.py:241` sniffs it
   with `getattr(b, "boundary", None)`, and the base's fallback
   (`not b.first_use_is_read`) coincides only for inputs. Hoisting it with an
   `Intermediate` default would *weaken the postcondition* — the base would
   confidently answer "intermediate" for a graph output. The fix is a three-valued
   `boundary` supplied at construction from the graph, plus one shared `spill_cost()`
   replacing the two divergent copies. This is the one intentional behaviour change
   in the set: CP-SAT placement-only plans may change for graphs with LX-resident
   outputs, and each delta must trace to an output buffer losing exactly one `size`
   of credited saving.
2. **`min_footprint` is a value-changing override** — `self.size` on the base, a
   strictly smaller per-core figure on the subclass. Legal, but only because `size`
   means different things in the two classes. Write the unit contract into both
   docstrings and add an invariant assert that a placement-only solver never holds a
   buffer whose division is still free.
3. **`_solve(solver: MemoryPlanSolver, ...)` is overridden with the same base
   parameter type**, forcing `assert isinstance(solver, CoreDivisionLayoutSolver)` at
   two sites. Make `ScratchpadAllocator` generic in its solver type and delete both
   asserts — mypy then proves what the assert checked (locally; see §2.1).
4. **Two capability probes.** `allocator.py:309` `getattr`s
   `supports_paired_buffers` even though the base declares it, and `allocator.py:2694`
   constructs a throwaway solver purely to ask whether the class supports core
   divisions. Both become class-level declarations.
5. **`SaCoOptimizingSolver.plan_layout` raises `NotImplementedError`** for an
   inherited method, and `ExhaustiveSearchSolver` narrows a parameter's accepted
   values. Implement the first by delegation; for the second, either widen the
   implementation or assert the narrowed precondition and document it.

## 5. Validation

Because §2.1 leaves static checking with nothing to say, this section *is* the
guarantee.

**A shared off-device IR builder.** This does not need inventing —
`tests/inductor/test_coarse_tiling.py:1661` (`_make_ftl_op`) and
`tests/inductor/test_hbm_pool_planning.py:120` (`_make_ftl_buffer`) already build a
genuine `ComputedBuffer` over a genuine `FixedTiledLayout` wrapping a real
`SpyreTensorLayout`, inside `V.set_graph_handler`, with the device never
initialising. Promote them to one shared builder that can also produce mutation ops,
extern fallbacks, real `MemoryDep`-carrying read/writes, and a multi-op graph. Every
step above is then testable without a card.

**The conftest tax, and the limit of fixing it.** `tests/inductor/conftest.py:55-75`
is an autouse *session* fixture allocating ~2.8 GB of device tensors; ~1018 planner
tests are device-free but pay 6.2 s and ERROR outright with no card. Guard it on
device *presence*, keep `DeviceOpenFail` fatal (the device is single-tenant, so a
lost race must not silently disable the virgin-zero poison), and never `pytest.skip`
— it is session-scoped, so a skip skips the whole inductor suite. Be honest about
what this does not fix: on a machine where the card exists but is **busy**, the guard
does not fire and the session still errors. The structural fix is making the poison
requested-by-device-tests rather than autouse, which is a larger change.

**Promotion invariants** (the substitute for the type checker): every op in
`graph.operations` is promoted, checked between passes in strict mode; no promoted op
has a public `__dict__` key colliding with a declared descriptor; promotion is
idempotent and does not reset metadata; every re-entrant minting site is covered by
an AST-level test so a thirteenth site fails CI rather than compiling silently.

**The acceptance gate records per-op division.** LX residency and spilled bytes can
be byte-identical while every committed core division changes, and that division is
real per-core kernel slicing — every silent-wrong-output incident in this branch's
history was exactly that (~19% avgpool window split, ~9% coarse-tile group, ~44%
offset-slice read, ~45.5% flash `tile_B_H`). So every baseline records per-op
`chosen_division` and every criterion diffs it, with the solve status per graph
recorded alongside (§2.2), the FX cache disabled, and the regeneration entry point in
the module rather than in a docstring.

**Two traps to avoid, both already present in the tree.**

- *Vacuous coverage tests.* `tests/inductor/test_coarse_tiling.py:7828-7852` is
  named as a `_SPYRE_METADATA_ATTRS` coverage test but asserts only that two names
  *not in the tuple* are not copied — it passes with `copy_op_metadata = lambda src,
  dst: None`, verified. Any exhaustiveness test here must iterate the declared
  descriptors and assert **positive** propagation of each, plus one negative control.
- *Laundered skips.* `tests/conftest.py:97-121` rewrites `skipped` to `wasxfail` for
  any test carrying an xfail mark, and `tests/inductor/` has 55 such marks with only
  7 strict. A device-absence skip on those is reported as XFAIL, indistinguishable in
  the summary from "ran and failed as expected". A gate that reads pass/xfail counts
  cannot tell a cardless run from a real one.

**The fakes sweep is real work, and its failure mode is silent.** There are 23
`MagicMock(spec=ComputedBuffer)` sites and heavy `SimpleNamespace` use (89 in
`test_span_overflow_hint_analysis.py`, 53 in `test_work_division_hint.py`). Each
passes `isinstance(op, ComputedBuffer)` today; if any converted call site narrows to
`isinstance(op, SpyreComputedBuffer)`, those fakes take a *different branch* rather
than crashing — and `_division_map` branches on that isinstance at `allocator.py:2119`
and `:2140`. Keep the narrowing on `ComputedBuffer` unless a step deliberately
strengthens it, and convert the affected fakes in the same commit when it does.

## 6. Collisions

`scratchpad/allocator.py` is simultaneously the target of this refactor and of four
other planned workstreams — coarse-tiling solver integration, core-division hints,
candidate pruning, and restickify instrumentation — which between them want to add an
`elif` to the pin ladder, a seed and a tiling loop to `_enumerate_core_divisions`, and
a table encoding to `_cd_parent_matches`. Left uncoordinated that is three merge
conflicts and three chances to drop a guard silently, which is how the coarse-tile pin
was lost in the first place.

The registry (S7) is what converts contested `elif` insertions into independent table
entries, and its exhaustiveness test is the mechanism that catches the next dead
guard. That argues for S7 landing before the hint and coarse-tiling workstreams touch
the ladder — but S7 is also the largest step here, so the pragmatic split is: S0-S2
immediately (they collide with nothing), S4-S5 next, and S7 negotiated against
whichever of the other workstreams is closest to landing.

Two housekeeping notes. The stale worktrees under `.claude/worktrees/` hold older
copies of these files — line numbers in the sibling compilation plan resolve against
`compiler-next-steps` (`dcd8a184`), not against HEAD, which is why several of its
citations look wrong. And the working tree is not a stable baseline: it changed under
measurement during this session, so the division baseline must be captured from a
pinned SHA in a clean worktree of its own.

## 7. Decisions needed

1. **Does S0 (the cache-key fix) land as its own PR, ahead of everything?**
   Recommendation: yes. It is small, it is independently correct, and without it no
   gate in this plan is trustworthy — including gates for the other four workstreams.
2. **Does the between-pass hook validate or promote?** Recommendation: validate. A
   promoter there would mask drift from the ~20 sites that splice ops in directly,
   which is the exact failure mode the hook exists to catch.
3. **Does the `boundary` hoist (§4.1.1) ship with this refactor or separately?** It
   is the only intentional behaviour change in the LSP set and it can move CP-SAT
   placement-only plans. Recommendation: separately, after S8's mechanical parts.
4. **Do the input-path residency reason strings change?** They must, if the two
   ladders merge into one ordered rule set. Nothing in production reads them and no
   test asserts them, so the cost is only that the characterization tests are updated
   deliberately in the same commit. Confirm that is acceptable.
5. **Is `dim_hints`/`op_it_space_splits` default unification in scope?** Their
   conflicting defaults across call sites (§3.1) are the one thing blocking those two
   from ever becoming plain attribute reads. Recommendation: out of scope here, filed
   as a follow-on behaviour change.
6. **Buffer-side `layouts`** (`InputBuffer`, `ConstantBuffer`, `TensorBox`) — a
   follow-on `SpyreBufferMixin` promoted at `register_buffer`, or left as monkey
   attributes? Recommendation: follow-on. `scratchpad/` reads none of them, so leaving
   them is inert for this plan's goal, and `TensorBox` is not a `Buffer`.

## 8. Risks

- **An unpromoted op is invisible.** An unset descriptor and an unpromoted op answer
  identically at all 32 `hasattr` and ~100 `getattr` sites, and an unpromoted op
  carrying a monkey value still answers *correctly*. Only the validator can see the
  difference, and the existing off-device builders construct base objects — so they
  would keep passing against a completely broken promoter until they are converted.
- **Reclass-without-move is silent metadata loss** with two named wrong-code
  endpoints (§3.3). This is the single defect most worth a dedicated test.
- **`layouts` and `restick_cost_fn` are coupled.**
  `optimize_restickify.py:512/521/575/586/594/622/662/703` gate on `hasattr(op,
  "layouts")` and then dereference `op.restick_cost_fn` unguarded — `hasattr(layouts)`
  is being used as a proxy for "restick_cost_fn is set". They cannot be migrated in
  separate steps.
- **A wholesale metadata copier widens what survives a reconstruction.**
  `_SPYRE_METADATA_ATTRS` is a deliberate subset: `read_copy_elision.py` re-adds three
  names by hand at the two sites that want them, while `replace_computed_buffer_body`
  deliberately does not carry them. Deriving `copy_op_metadata` from the descriptor
  set must preserve that exclusion, with an explicit opt-out list and a test that
  fails when a declared descriptor is neither copied nor listed.
- **Silent wrong output is this area's characteristic failure**, not crashes. Every
  behaviour-changing step lands alone, with numerical comparison against CPU and the
  measured delta in the commit message.
- **Device time is the constraint.** The card is single-tenant and concurrent suites
  fail with a misleading resource-busy error, so these suites must be serialised.
  S0b is what keeps the inner loop off that critical path — for the cardless case,
  not the busy-card case.
