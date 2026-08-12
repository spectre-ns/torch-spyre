# The torch-spyre Compiler: A Stage-by-Stage Developer Deep-Dive

This document traces a PyTorch program through the entire torch-spyre Inductor backend, from `torch.compile` entry to the SuperDSC JSON consumed by IBM's DeepTools (`dxp_standalone`) backend. It names the concrete IR classes at every stage, describes what each pass does (input → transform → output), and consolidates every mechanism for inspecting the IR. Citations are `file:line` relative to the repo root `/home/zhang402/torch-spyre/torch-spyre`.

> **Caveat on upstream line numbers.** torch itself was not importable in the analysis sandbox; upstream PyTorch call-site lines (e.g. `torch/_inductor/graph.py:2260`) were read from a checked-out wheel and may drift across PyTorch versions. The *contracts* — config-key names, the `_update_scheduler → Scheduler` ordering, the five custom-pass hooks — are stable.

> **Post-merge `0afc247` status.** A large merge refactored `passes.py` and the
> work-division/coarse-tiling subsystems. The verified deltas are corrected inline below
> (marked **[post-merge]**) and documented in full in **§10**, which is authoritative where
> it conflicts with the specific line numbers in §1/§4. Headlines: passes now take a
> `GraphLowering`; the pre-scheduling pipeline is **14 steps**; work division is **three
> passes** (`span_reduction`, `cost_model_matmul_division`, `work_distribution` — the old
> `k_fast_division` is gone); coarse tiling stamps a single `loop_info` (`CoarseTileInfo`)
> attribute; the low-level IR is **still SuperDSC** (KTIR is planned, a new `JobPlan`
> runtime layer consumes SuperDSC). See §10.

---

## 1. Pipeline at a glance

```
                          PyTorch program  (examples/softmax.py: torch.softmax(x, dim=0))
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ Dynamo  +  AOTAutograd  +  decomposition                      │   FX graph of ATen ops
        │  ── Spyre decompositions injected by enable_spyre_decompositions│   (decompositions.py:102)
        │     (rewrite ATen → simpler ATen / spyre::* custom ops)        │
        └──────────────────────────────────────────────────────────────┘
                                   │
   compile_fx wrapper (_wrapper, __init__.py:95) detects spyre tensors →
   enters enable_spyre_context (patches.py:38), installs 6 hooks + config patches
                                   │
        ┌──────────────────────────────────────────────────────────────┐
        │ Inductor FX passes (operate on torch.fx.Graph)                │
        │  HOOK 1  pre_grad_custom_pass     CustomPreGradPasses  []      │  ← pre_grad.py:331
        │  HOOK 2  post_grad_custom_pre_pass CustomPrePasses            │  ← post_grad.py:109  (collect_spyre_hints)
        │  HOOK 3  post_grad_custom_post_pass CustomPostPasses          │  ← post_grad.py:180  (recover hints, mm→bmm, ...)
        └──────────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────────────────────────────────────────┐
        │ Inductor LOWERING  (enable_spyre_lowerings, lowering.py:117)  │  ATen FX → LoopLevel IR
        │   each op → Pointwise / Reduction / SpyreReduction, .realize() │  (ComputedBuffer, FixedLayout)
        │   has_large_inner_fn → True  → every op realized (patches.py:104)│
        └──────────────────────────────────────────────────────────────┘
                                   │
   GraphLowering.codegen() → _update_scheduler PATCHED (patches.py:119) →
   HOOK 4 (the heaviest)  CustomPreSchedulingPasses(graph)   ← graph.py:2260, BEFORE Scheduler(self.operations)
        ┌──────────────────────────────────────────────────────────────┐
        │ CustomPreSchedulingPasses.__call__  (passes.py:321-366)       │  takes graph: GraphLowering
        │  14 STEPS IN ORDER  [post-merge 0afc247]:                     │  (mutate graph.operations)
        │   1  deadcode_elimination                                     │
        │   2  propagate_spyre_tensor_layouts                           │
        │   3  optimize_restickify_locations                            │
        │   4  finalize_layouts                                         │
        │   5  insert_restickify                                        │
        │   6  insert_bmm_padding                                       │
        │   7  dedup_and_promote_constants                              │
        │   8  _maybe_chunk_large_tensors    [config.chunk_large_tensors]│
        │   9  propagate_named_dims                                     │
        │  10  assign_dim_hints                                         │
        │  11  _maybe_coarse_tile        [only if hint groups present]  │
        │  12  span_reduction                  ┐ three                  │
        │  13  _distribute_work:                │ work-                  │
        │        cost_model_matmul_division +   │ division               │
        │        work_distribution             ┘ passes                 │
        │  14  _maybe_scratchpad_planning (LX)  [lx_planning]           │
        └──────────────────────────────────────────────────────────────┘
                                   │
   Scheduler builds SchedulerNodes →
   HOOK 5 CustomPreFusionPasses  (propagate_mutation_layouts, build_loop_scheduler_nodes →
                                  CountedLoopSchedulerNodes, BEFORE Inductor fusion)
   fuse_nodes →
   HOOK 6 CustomPostFusionPasses (memory_planning[HBM], spyre_fuse_nodes)
                                   │
        ┌──────────────────────────────────────────────────────────────┐
        │ SuperDSCScheduling.codegen_node (scheduler.py:304)            │  LoopLevel IR → OpSpec
        │   SpyreKernel + SpyreKernelOpsHandler replay node.codegen()   │  (op_spec.py: OpSpec/TensorArg/LoopSpec)
        │   load→TensorAccess, op→PointwiseOp/ReductionOp, store→OpSpec  │
        │   codegen_kernel → emits async_compile.sdsc('name', [specs])  │
        └──────────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────────────────────────────────────────┐
        │ codegen: generate_bundle (bundle.py:49)                       │  OpSpec → SuperDSC JSON
        │   unroll_loop_specs → parse_op_spec (superdsc.py:529)         │  (sdsc_N.json + bundle.mlir)
        │   → generate_sdsc (compute_ops.py:222) → JSON dict            │
        └──────────────────────────────────────────────────────────────┘
                                   │
        subprocess  dxp_standalone --bundle -d <dir>   (async_compile.py:61)
                                   │
                                   ▼
                          DeepTools backend → device binary → SpyreSDSCKernelRunner
```

There are **six** extension hooks. Five are official Inductor extension points wired via `torch._inductor.config.patch` (`patches.py:85-99`); the sixth is a monkeypatch of `GraphLowering._update_scheduler` (`patches.py:119-123`) that fires **HOOK 4** (`CustomPreSchedulingPasses`) — the only hook that runs on lowered Inductor IR *before* `SchedulerNode`s exist.

> **[post-merge]** `passes.py` was refactored into two pipeline base classes —
> `_SpyreGraphPassPipeline` (FX-graph hooks, device-guarded) and `_SpyreNodePassPipeline`
> (scheduler-node hooks) — sharing a `_uuid()` helper, with a `@_runs(...)` tag so
> config-gated wrappers still key the Inductor cache on the *real* passes' source files. Two
> structural moves from the diagram above: `build_loop_scheduler_nodes` now runs in **HOOK 5**
> (`CustomPreFusionPasses`, before Inductor fusion, so `CountedLoopSchedulerNode`s survive),
> and work division (step 13) is the three passes `cost_model_matmul_division` +
> `work_distribution` (the old `k_fast_division` was replaced). Full detail in §10.

---

## 2. The IR at each stage (the crisp answer)

This is the single most important reference. **What is the IR, concretely, at each stage?**

| Stage | IR object(s) | Defining class(es) | Where |
|---|---|---|---|
| **Dynamo/AOT/post-grad** | `torch.fx.Graph` of ATen `call_function` nodes (e.g. `aten.amax`, `aten.sub`, `aten._softmax`), each carrying `meta["val"]` fake tensors on device `"spyre"` | upstream `torch.fx.Node` | post-grad FX |
| **After Spyre decompositions** | same `torch.fx.Graph`, but ATen ops rewritten to simpler ATen or to `spyre::*` custom ops (`spyre::gelu`, `spyre::exx2`, …) | `torch.library.custom_op` targets (`customops.py`) | pre-lowering |
| **LoopLevel IR** | **stock Inductor `Loops` IR** — there is *no* custom Spyre compute dialect. Each op is a `ComputedBuffer` (named `buf3`, …) whose `.data` is a `Pointwise` or `Reduction`/`SpyreReduction`, carrying an `inner_fn(index)` closure that emits scalar `ops.*` calls. Layout starts as `FixedLayout`/`FlexibleLayout`; later becomes `FixedTiledLayout` | `torch._inductor.ir.Pointwise`, `Reduction`; Spyre's `SpyreReduction` (`ir.py:39`), `SpyreConstantFallback` (`ir.py:110`), `SpyreEmptyFallback` (`ir.py:141`), `FixedTiledLayout` (`ir.py:80`) | `lowering.py`, `ir.py` |
| **OpSpec** (backend's own IR) | A flat `list[OpSpec | LoopSpec | UnimplementedOp]`. An **`OpSpec`** is a *single* named device op with explicit iteration space, fully-resolved device-layout tensor args (`TensorArg`), and op-specific `op_info`. The whole RValue expression tree for one buffer is flattened into **exactly one** `OpSpec` | `op_spec.py`: `OpSpec` (`:51`), `TensorArg` (`:26`), `LoopSpec` (`:81`), `UnimplementedOp` (`:77`) | `spyre_kernel.py`, `op_spec.py` |
| **SuperDSC (SDSC)** | A per-op **JSON dict** (one `sdsc_N.json` per OpSpec) keyed `f"{idx}_{opfunc}"`, plus a `bundle.mlir` driver. Intermediate `SDSCSpec`/`SDSCArgs` dataclasses (with `__str__`) precede the JSON | `superdsc.py`: `SDSCSpec` (`:78`), `SDSCArgs` (`:43`); JSON shape in `compute_ops.py:222` | `superdsc.py`, `compute_ops.py`, `bundle.py` |

### 2.1 The `OpSpec` object, field by field (`op_spec.py:51-73`)

```python
@dataclasses.dataclass
class OpSpec:
    op: str                                       # Spyre device op name: "relufwd", "realdiv", "amax",
                                                  #   "identity", "restickify", "batchmatmul", ...
    is_reduction: bool
    iteration_space: dict[Symbol, tuple[Expr, int]]  # loop symbol -> (range/extent, work_division core-split)
    args: Sequence[TensorArg]                     # inputs first, OUTPUT LAST (args[-1])
    op_info: dict[str, Any]                        # op-specific; notably op_info["constants"]
    tiled_symbols: list[Symbol]                    # iteration symbols divided by an enclosing LoopSpec.count
```

`TensorArg` (`op_spec.py:26-48`): `is_input`, `arg_index` (`-1` until `codegen_kernel` assigns the runtime kernel-arg position), `device_dtype` (a `DataFormats` C++ enum, e.g. `SEN169_FP16`), `device_size: list[int]` (the SpyreTensorLayout device shape), `device_coordinates: list[Expr]` (one SymPy expr per device dim over the iteration symbols; the within-stick column dim appears as a pair `floor(c/64)` and `Mod(c, 64)`), `allocation` (dict possibly carrying `pool`/`lx`/`hbm` byte offsets), `stride_map: list[int] | None`, `per_tile_fixed: bool`.

`LoopSpec` (`op_spec.py:81-100`): `count: Expr`, `body: list[OpSpec|LoopSpec|UnimplementedOp]`, per-level `tiled_symbols`.

These are **plain `@dataclass`es** — no custom `__str__`; `repr()` works but the intended human-readable form is the codegen pretty-print (`_codegen_op_spec_list`, `spyre_kernel.py:691`).

> **[post-merge]** The per-op coarse-tiling tag on a `ComputedBuffer` is now a single
> `loop_info: CoarseTileInfo` (`loop_info.py`) carrying `loop_group_id`, `loop_count`,
> `loop_tiled_dims`, and the new `loop_tiled_reduction_dims` (Stage-1 reduction-dim tiling) —
> replacing the earlier separate `loop_group_id`/`loop_count`/`loop_tiled_dims` attributes
> described in §4 Stage 4. See §10.4–§10.5.

### 2.2 The SuperDSC JSON shape (`compute_ops.py:222-552`)

`generate_sdsc` returns `(sdsc_json, base_symbol_values, affine_strides)`. The top-level key is `f"{idx}_{opfunc}"`, mapping to:
- Fold props: `sdscFoldProps_`, `sdscFolds_`, `coreFoldProp_`, `coreletFoldProp_` (affine transforms `{"Affine":{"alpha_":..,"beta_":..}}`).
- Core ownership: `numCoresUsed_`, `coreIdToDsc_`, `coreIdToWkSlice_`, `numWkSlicesPerDim_`, `coreIdToDscSchedule`.
- `dscs_`: one `DesignSpaceConfig` keyed by `opfunc`, containing:
  - `N_`: full iteration extents `{dim+"_": size}`.
  - `coordinateMasking_` / `maskingConstId_`.
  - `dataStageParam_["0"]["ss_"/"el_"]`: per-core sizes (steady-state / epilogue) = `iteration_space[dim] // work_slices[dim]`.
  - `primaryDsInfo_`: per-layout-label `layoutDimOrder_`, `stickDimOrder_`, `stickSize_`.
  - `scheduleTree_`: one `allocate` node per `TensorArg` with `component_` (`lx`/`hbm`), `layoutDimOrder_`, `maxDimSizes_`, `startAddressCoreCorelet_`, optional `backGapCore_`, `coordinates_.coordInfo` (per-dim fold descriptor).
  - `labeledDs_`: per-arg `dsType_` (layout label), `scale_`, `wordLength`, `dataFormat_`, `memOrg_`.
  - `constantInfo_`: encoded constants.
  - `computeOp_`: `{exUnit (pt/sfp), opFuncName, dataFormat_, fidelity_, inputLabeledDs, outputLabeledDs}`.

---

## 3. The sticked/tiled tensor layout (the cross-cutting data structure)

Almost every Spyre-specific transformation revolves around the tiled device layout. A Spyre tensor carries **two layouts simultaneously**, bundled in `FixedTiledLayout` (`ir.py:80-107`), a subclass of Inductor's `FixedLayout`:

- the **host** layout — ordinary `size`/`stride` in logical row-major terms (inherited from `FixedLayout`);
- the **device** layout — `self.device_layout`, a C++ `SpyreTensorLayout` (`csrc/spyre_tensor_impl.h:35`, pybound in `module.cpp:172`), describing the *physical* tiled form. Plus `allocation: dict` and `per_tile_fixed: bool` (`ir.py:96-98`).

`SpyreTensorLayout` fields (`spyre_tensor_impl.h:37-49`):
- **`device_size: list[int]`** — on-device dim sizes, in **decreasing stride order with the stick dimension last**.
- **`stride_map: list[int]`** — maps each device dim to its corresponding **host stride**.
- **`device_dtype: DataFormats`** — `elems_per_stick()` (`:97`) gives the stick width (64 at fp16).

A **stick** is a 128-byte-aligned chunk = `elems_per_stick` elements. The **last entry of `device_size` is the within-stick axis**. Device dims = host dims + 1 (`pass_utils.py:642-644`).

The bridge between host index expressions and device coordinates is the SymPy algebra in **`views.py`**: `compute_coordinates(size, stride, var_ranges, index)` (`views.py:61`) projects a flat `MemoryDep.index` onto per-dim coordinate expressions; `normalize_coordinates`/`Term` (`views.py:203,219`) break them into `num*(var%mod)//den + offset` form; `align_tensors` (`views.py:347`) reconciles multiple operands' iteration spaces and tiles the stick dim ("never fuse last dimension = stick dimension!", `views.py:321`).

> **Important timing:** during *lowering* buffers carry plain `FixedLayout`/`FlexibleLayout` — `device_layout` (the sticks) is **not** visible. It is committed by `finalize_layouts` (`insert_restickify.py:223`) during HOOK 4. To inspect sticks you must look at a post-layout pass.

---

## 4. Stage-by-stage reference

### Stage 0 — Backend registration & compile-context plumbing

**Files:** `__init__.py`, `patches.py`, `passes.py`, `config.py`, `choices.py`, `scheduler.py`, `constants.py`.

This is the backbone. Three responsibilities:

1. **Register `"spyre"` with Dynamo & Inductor.** `_autoload()` (`__init__.py:132-170`, run once per process, triggered from `__init__.py:75-77`) calls `register_backend_for_device(DEVICE_NAME, SuperDSCScheduling, SpyrePythonWrapperCodegen, device_custom_config=config)` (`__init__.py:163-168`) — telling Inductor which `BaseScheduling` and wrapper-codegen class to use, and registering `config.py` as the device-custom config namespace.

2. **Wrap `compile_fx`.** `enable_spyre_compile_fx_wrapper()` (`__init__.py:27-123`) monkeypatches `cfx.compile_fx` once. `_wrapper` (`__init__.py:95-120`) runs `_uses_spyre(gm, example_inputs)` (`__init__.py:60-93`), which checks input fake-tensor devices, the output node's `meta["val"]`, and any node `kwargs["device"]` (catches `aten.zeros(device="spyre")`). If Spyre is involved it enters `enable_spyre_context(...)`.

3. **Install all six hooks + config patches.** `enable_spyre_context` (`patches.py:38-139`) applies `torch._inductor.config.patch(new_config)` (`patches.py:85-99`) wiring the five custom-pass keys plus behavioral flags: `split_reductions=False`, `benchmark_harness=False`, `unroll_reductions_threshold=1`, `permute_fusion=False`, `allow_buffer_reuse=False`. Outside the dict it forces `Loops.has_large_inner_fn = lambda self: True` (`patches.py:104-105`, realize everything), pops the last `joint_graph.pass_patterns` entry to disable softmax mul/div fusion (`patches.py:107-111`), installs `spyre_data_types()` (`patches.py:24-35`) and `V.set_choices_handler(SpyreHeuristics())` (`patches.py:130`), and monkeypatches `_update_scheduler` (`patches.py:119-123`). All patches restored in the `finally` (`patches.py:135-138`).

**Fusion is disabled at three layers**: `SuperDSCScheduling.can_fuse_vertical`/`can_fuse_horizontal` → `False` (`scheduler.py:268-284`); `can_buffer_be_removed_through_fusion` → `False` (`scheduler.py:259-266`); `SpyreHeuristics` (`choices.py:22-60`) `reduction_split_factor=1` and all `can_fuse*` → `False`. Spyre re-implements its own *order-preserving* fusion later (`spyre_fuse_nodes`, §Stage 6).

**Config flags** (`config.py`, installed via `install_config_module`) **[post-merge: several changed]**: `lx_planning` (`LX_PLANNING`, **now default ON `"1"`**), `co_optimizing_lx_planning` (`CO_OPTIMIZING_LX_PLANNING`), `chunk_large_tensors` (`CHUNK_LARGE_TENSORS`), `core_id_k_fast_emission` (`SPYRE_CORE_ID_K_FAST_EMISSION`, default on), `sencores` (`SENCORES`, 1–32, default 32), `global_stick_optimizer` (default on), `allow_all_ops_in_lx_planning` (**new**, `False`), `dxp_lx_frac_avail`, `bundle_symbolic_args` (`BUNDLE_SYMBOLIC_ARGS` — **renamed from `bundle_hbm_symbols`**), `unroll_loops`. **Removed:** `coarse_tiling` / `COARSE_TILING` and `coarse_tiling_groups_fn` (coarse tiling is now gated by hint-group presence, not a flag).

---

### Stage 1 — FX rewriting (decompositions + custom ops)

**Files:** `decompositions.py`, `customops.py`.

Two sub-mechanisms rewrite the ATen FX graph *before* lowering:

- **Decompositions** (`register_spyre_decomposition`, `decompositions.py:60`) — adds each fn to `spyre_decompositions` and (for `aten` ops) registers a `PrivateUse1` dispatch kernel so the same decomposition is reachable in **eager** mode. Examples: `addmm_decomp` (`:366`) → `mat1@mat2 + input`; `spyre_layer_norm` (`:431`) → `spyre::exx2 + layernormscale + layernormnorm`; `spyre_gelu` (`:467`) → `spyre::gelu`; `spyre_topk` (`:455`) → `spyre::topkvalue`/`topkindex`. `enable_spyre_decompositions` (`decompositions.py:102`) is a reentrant CM that injects these into Inductor's decomp table, *removes* `triu`/`tril` (`:40`) and *removes* decomps for ops in `fallbacks.py:fallback_ops` (so they fall back to eager).

- **Custom ops** (`customops.py`) — declares `spyre::*` ops via `torch.library.custom_op` + `register_fake` for shape inference (`softplus:24`, `exx2:62`, `gelu:162`, `clamp:175`, `constant:432`, `empty:193`, `overwrite:251`). The fakes return correctly-shaped empties; the *real* behavior comes from lowering. `overwrite_f` is wired into Inductor's `inplaceable_ops` (`customops.py:318`).

**Important:** `aten._softmax` is **not** in the Spyre decomposition table (`decompositions.py` has no `_softmax` entry), so softmax uses PyTorch's **in-tree** numerically-stable decomposition. See §6.

---

### Stage 2 — Lowering into LoopLevel IR

**Files:** `lowering.py`, `ir.py`, `views.py`, `dtype_ops.py`.

`enable_spyre_lowerings` (`lowering.py:117`) swaps `spyre_lowerings` into Inductor's global dict, unregisters fallback-op lowerings (`:132`), and patches `aten.clamp[_min/_max]`. Each handler (`register_spyre_lowering`, `lowering.py:48`) consumes a post-decomposition FX node and emits Inductor `Loops` IR.

**What a single op looks like.** `lower_gelu` (`lowering.py:555`) turns `spyre::gelu(x)` into a `Pointwise` with `inner_fn = lambda index: ops_wrapper("gelu")(x.make_loader()(index))`, `ranges = x.get_size()`, then `.realize()`s it into a named `ComputedBuffer` (`buf3`). The operator identity ("gelu") survives only as the string passed to `ops_wrapper`.

Op families produced:
- **`Pointwise`** — `gelu`/`softplus`/`clamp`/`layernorm*` (`lowering.py:556,572,591`).
- **`Reduction` / `SpyreReduction`** — `mm`/`bmm` modeled as a reduction over K with `reduction_type = BATCH_MATMUL_OP = "batchmatmul"` (`lowering.py:274,346`, degenerating to a pointwise `mul` at K==1); `mean`/`exx2`/`topk` likewise. `SpyreReduction` (`ir.py:39`) adds an `op_info: Any` side-channel carrying operator constants (`scaling_factor`, `exx2scale`, `useZeroMean`) because reductions bypass the virtualized `ops.*` handler (docstring `ir.py:40-47`).
- **`ExternKernel` fallbacks** — `lower_full`/`lower_constant` → `SpyreConstantFallback` (`ir.py:110`); `lower_empty` → `SpyreEmptyFallback` (`ir.py:141`).
- **`SliceView` + `mutate_to`** — `lower_cat` (`:745`), `lower_constant_pad_nd` (`:768`), `lower_overwrite` (`:640`).

Because `has_large_inner_fn → True` (`patches.py:104`), **every** op is realized into its own `ComputedBuffer`. After lowering the graph is a flat list of named buffers, each holding one Pointwise/Reduction with a host `FixedLayout`. This is the input to HOOK 4.

---

### Stage 3 — HOOK 4: `CustomPreSchedulingPasses`, group A (layout, restickify, padding, constants, hints)

**Files:** `passes.py`, `deadcode_elimination.py`, `propagate_layouts.py`, `optimize_restickify.py`, `insert_restickify.py`, `padding.py`, `dedup_constants.py`, `propagate_named_dims.py`, `propagate_hints.py`, `temp_passes.py`, `pass_utils.py`.

These run on `graph.operations` (a flat, topo-ordered `list[ir.Operation]`), gated on some op being on the `"spyre"` device (`passes.py:240-245`). The order (`passes.py:250-260`) is load-bearing.

1. **`deadcode_elimination`** (`deadcode_elimination.py:70`) — reverse-topo liveness from `V.graph.get_output_names()`; side-effecting/mutation ops always live (`_has_side_effects`, `:54-67`). Removes dead ops in place, adds names to `V.graph.removed_buffers`.

2. **`propagate_spyre_tensor_layouts`** (`propagate_layouts.py:598`) — topo walk; graph inputs read their `device_tensor_layout()` (`:603-624`). For each `ComputedBuffer`, `compute_layouts` (`:531-588`) produces a **list of candidate STLs** stored on `op.layouts` plus a **restick cost function** `op.restick_cost_fn`. Op-type rules: multi-arg pointwise → `AllSameNode` (all share the stick); matmul → `_matmul_layouts` (`:343`) wrapped in `FixedInOutNode` (x sticks on reduction var K, y/output on generated var N, found by sympy set arithmetic, `:272-291`); `exx2`/`layernormnorm` → `FixedInOutNode`; topk → `_topk_layouts`; single-arg → `_single_arg_op_layout` (clone forces row-major → `AnyInNode`).

3. **`optimize_restickify_locations`** (`optimize_restickify.py:505`) — selects exactly one output STL per op to minimize total restickify cost. `EdgeCostMap` (`:44`): per input edge, cost = 0 if stick-compatible, `prod(device_size)` if a feasible restickify, `INF` if infeasible. Two drivers: **greedy** (`greedy_local_min_cost`, default) commits each op's min-cost layout to `op.committed_stl`; **beam** (`beam_global_min_cost`, gated by `config.global_stick_optimizer`, `BEAM_WIDTH=64`).

4. **`finalize_layouts`** (`insert_restickify.py:223`) — wraps each `committed_stl` in a `FixedTiledLayout`, assigns `op.layout`, and for each input edge where the committed input STL differs from what the op requires records a deferred entry in `V.graph.restickify_plan` (`dict[op_name] -> [{arg_name, target_layout}]`). **This is where `device_layout` (sticks) first becomes visible.**

5. **`insert_restickify`** (`insert_restickify.py:203`) — drains the plan. `_create_restickify_node` (`:82`) inserts a `torch.ops.spyre.restickify` FX node, lowers it via `graph_lowering.run_node`, and `insert_restickify_on_node_inputs` (`:146`) patches the consumer's `inner_fn` with a `NameSwapHandler` to read the repacked buffer, reconstructs the consumer `ComputedBuffer`, and splices the restickify op in before the consumer. (A restickify is a device-to-device repack that moves which logical axis lives within the stick; hardware cannot gather/scatter across sticks, `optimize_restickify.py:225-236`.)

6. **`insert_bmm_padding`** (`padding.py:184`) — for each `BATCH_MATMUL_OP` reduction, pads the **y** operand's K dim up to a stick boundary (x is left untouched because hardware masks x's within-stick tail to zero, `padding.py:33-36`). Identifies x vs y *geometrically* via `device_coordinates` (`:245-271`). Emits a 4-op `constant_pad_nd` sequence via `lower_pad_sequence` (`pass_utils.py:509`) and rebuilds the matmul `inner_fn` to read padded y; `reduction_ranges` stays at K (the widening happens at SDSC codegen).

7. **`dedup_and_promote_constants`** (`dedup_constants.py:101`) — groups `SpyreConstantFallback` by `(value, dtype, device)`, redirects duplicate consumers to the canonical via `NameSwapHandler`, drops dups, and **front-loads** surviving constants to the head of `operations` (`:131-138`). Output constants never deduped.

8. **`chunk_large_tensors`** (`chunk_large_tensors.py:426`, gated) — see §Stage 4.

9. **`propagate_named_dims`** (`propagate_named_dims.py:297`) — **no-op unless** the driver called `name_tensor_dims(...)`. Propagates human-readable axis names; stamps `op.named_dims`, `op.loop_var_dims`, `op.reduction_named_dims` (`:214-221`).

10. **`assign_dim_hints`** **[post-merge: now in `propagate_named_dims.py`, not `temp_passes.py`]** — resolves `spyre_hint(...)` scopes into `op.dim_hints: list[DimHint]` (`DimHint` at `propagate_hints.py`: `dim_names`, `split_count`, `loop_var` (None when the op is broadcast w.r.t. that hint), `is_reduction`, `hint_id`). Hints are attached to FX `custom` meta by `spyre_hint` and survive AOT re-tracing via `collect_spyre_hints`/`recover_spyre_hints`. `op.dim_hints` is the contract consumed by coarse tiling. (Work-division hints `spyre_hint(work_div={...})` are kept separate in `op.work_div_loop_info`.)

---

### Stage 4 — HOOK 4 group B (coarse tiling, chunking, work division)

**Files:** `coarse_tile.py`, `chunk_large_tensors.py`, `work_division.py`, `temp_passes.py`, `multi_dim_reduction_pass.py`.

Everything here runs **after** stickification/padding (layouts final, in stick units). The work-division trio runs **after** coarse tiling/chunking so each only sees the *reduced* iteration space.

**`chunk_large_tensors`** (`chunk_large_tensors.py:426`, runs before `span_reduction`) — splits oversized `Pointwise` `ComputedBuffer`s into stick-aligned, memory-safe chunks so work-division need not special-case them. `_needs_chunking` (`:188`): chunk if `per_core_span > MAX_SPAN_BYTES` (256 MB) after best split, or if `total_bytes > MAX_SPAN_BYTES * max_cores`. Chunk 0 is shrunk in place; chunks 1..N-1 become `Scatter` `MutationLayoutSHOULDREMOVE(op)` buffers offset by `chunk_offset` (`_chunk_op`, `:334`). Shares `MAX_SPAN_BYTES` with `work_division.py`.

**`coarse_tile`** **[post-merge: gating changed]** — now run by the `_maybe_coarse_tile` wrapper **only when hint-derived groups exist** (`groups = hints_to_coarse_tile_groups(graph)`; `coarse_tile(graph, groups=groups)` if `groups`), not by a `config.coarse_tiling` flag. `hints_to_coarse_tile_groups` now lives in `coarse_tile.py` (moved out of `temp_passes.py`) and forms groups from each op's `dim_hints`. It is a **stamping + range-rewriting** pass — no loop node is created yet (that happens post-scheduling). `_stamp_group` validates the group is a contiguous slice and stamps a **single `op.loop_info: CoarseTileInfo`** (`loop_info.py`) holding `loop_group_id` (nesting-path tuple), `loop_count`, `loop_tiled_dims`, and `loop_tiled_reduction_dims` — replacing the earlier three separate attributes. `_divide_ranges` divides `data.ranges` (output ranges) by `loop_count`, syncs `op.layout.size`/`stride`, and **rebuilds the `SpyreTensorLayout`** for the smaller per-tile buffer; for tiled **reduction** dims `_divide_reduction_ranges` divides `data.reduction_ranges` instead (Stage 1, §10.5). `insert_tiling_propagation` then handles outside consumers: loop-internal scratch → `per_tile_fixed = True`; used inside+outside (Case 1) → allocate full HBM buffer + insert copy op; used only outside (Case 2) → rewire to `MutationLayoutSHOULDREMOVE`.

**Work division** (`work_division.py`) — three sequential passes plus a heuristic. The unit of work is the iteration space `iteration_space_from_op(op)` (`pass_utils.py:195`). Splits are committed to `op.op_it_space_splits` via `apply_splits` (`:485`) → `splits_by_index_coeff` (`pass_utils.py:241`), which encodes `{Symbol: split}` as a **coeff-keyed** `ItSpaceSplits` pair (output dims keyed by write-index coefficient, reduction dims by read-index coefficient) — stable across the pre-scheduling→codegen boundary and robust to scheduler symbol renaming. Decoded later via `apply_splits_from_index_coeff` (`pass_utils.py:266`).

- **Pass 1 — `span_reduction`** (mandatory) — computes the *minimum* splits so every tensor's per-core span ≤ 256 MB. When no tensor violates the limit it leaves the op untouched. Raises `Unsupported` if >1 *reduction* var must split. topk reductions for k≤4 skipped.
- **Pass 2 — `cost_model_matmul_division`** **[post-merge: replaces `k_fast_division`]** — fires only on `matmul`/`bmm` ops. It enumerates feasible `(b, m, n, k)` splits, prices each with an analytic estimate `cost = (compute + hbm + psum + tie_break) × b^1.4`, and selects the lowest-cost combination. It **declines** (→ Pass 3) when the op is not a matmul/bmm, when Pass 1 already committed a split, when dims are ambiguous, or when its split would use fewer cores than the default Pass 3 split. Returns the claimed ops so Pass 3 skips them. (The K-collaborator ring placement that `core_id_k_fast_emission` used to gate is now a codegen-side permutation.)
- **Pass 3 — `work_distribution`** — distributes remaining cores (+ matmuls the cost model declined): honor committed span splits, fill output dims by decreasing size (`prioritize_dimensions`), then one reduction dim (`multi_dim_iteration_space_split`). `max_cores` from `config.sencores` (1..32). Logs CRITICAL on residual >256 MB. The `_distribute_work` wrapper runs Pass 2 then Pass 3 so every eligible op is finalized by exactly one of them.

  Named work-division hints `spyre_hint(work_div={...})` (resolved into `op.work_div_loop_info`) are committed directly, bypassing Pass 2's cost model and Pass 3's auto-distribution, and are authoritative even over Pass 1 (with a warning). `SPYRE_INDUCTOR_IGNORE_HINTS=1` disables them.

**`multi_dim_reduction_pass.decompose_multi_dim_reductions`** (`multi_dim_reduction_pass.py:165`) — an FX-graph pass that splits multi-dim reductions into chained single-dim reductions. **Not currently wired into any active pipeline** (no caller in `torch_spyre/` or `tests/`); staged/example code.

**IR before vs after group B:** *Before*, each op is a `ComputedBuffer` with full `data.ranges` and a full-size `FixedTiledLayout`; no `loop_*`/`op_it_space_splits`. *After coarse tiling*, tiled ops carry `loop_group_id`/`loop_count`/`loop_tiled_dims` with shrunk ranges/layout. *After work division*, ops carry `op_it_space_splits`.

---

### Stage 5 — HOOK 4 tail: LX scratchpad planning (and the separate HBM planner)

**Files:** `scratchpad/allocator.py`, `scratchpad/utils.py`, `scratchpad/graph_editor.py` **[post-merge: new]**, `scratchpad/firstfit_bestfit_solver.py` **[post-merge: new, replaces the old `plan_solver.py`/`passes.py`]**, `memory_planning.py`. (The scratchpad solver internals were refactored in the merge; the pass-selection detail below — `passes.py:272-276` — now lives in the `_maybe_scratchpad_planning` wrapper.)

There are **two unrelated memory-planning passes**:

| | `scratchpad_planning` (LX) | `memory_planning` (HBM/DDR) |
|---|---|---|
| File | `scratchpad/allocator.py` | `memory_planning.py` |
| Runs in | HOOK 4 (pre-scheduling, on `ir.Operation`s) | HOOK 6 (post-fusion, on `BaseSchedulerNode`s) |
| Pool | per-core 2 MB LX (~1.6 MB usable) | HBM 4 GB intermediates segment (`SEGMENT_SIZE=0x400000000`) |
| Annotation | `layout.allocation["lx"] = address` | `layout.allocation["pool"] = address` |
| Gating | `LX_PLANNING` (**[post-merge] now default ON**) | `SPYRE_INDUCTOR_MEMORY_PLAN` (default on) |

**LX path.** `scratchpad_planning(graph, allocator=None)` (`allocator.py:398`) picks `DefaultAllocator` or, when `CO_OPTIMIZING_LX_PLANNING=1`, `StrategyBCoOptimizingAllocator` (chosen in `passes.py:272-276`). The only durable mutation in the default path is annotating eligible buffers' `FixedTiledLayout.allocation["lx"]` (`_push_allocation`, `:172`). Flow (`plan_allocation`, `:212`): pre-passes (`CloneInputNodesPass`, inert by default) → `_generate_buffers` produces `LifetimeBoundBuffer`s (per-core size, integer liveness `[start,end)`, `in_place_parents`) → `GreedyLayoutSolver.plan_layout` (`plan_solver.py:143`, greedy linear-scan / register-allocation sweep with first-fit `_find_free_block`, **no defrag, no eviction**; address `None` → stays in HBM) → write addresses.

Eligibility (`_filter_ops`, `:91`): dropped if op output not in `OP_OUTPUT_GOOD_FOR_LX_REUSE = ["max","amax","sum","exp","sub"]` (`utils.py:25`), or a `MutationLayoutSHOULDREMOVE`, or `get_ncores_for_buffers` returns `-1` (core-division mismatch, `utils.py:141`), or graph I/O. In-place reuse limited to `OP_GOOD_FOR_LX_INPLACE = ["exp","sub"]`.

**Co-optimizing allocator** (`allocator.py:294-395`) — "Strategy B": jointly optimizes work-division + scratchpad. `_enum_split_options` (`:231`) generates ≤`DEFAULT_VARIANT_CAP=6` alternative splits per op by flipping the single output-dim split onto another divisible dim; `_search` (`:327`) does exhaustive DFS, `_score_layout` (`:380`) = total HBM bytes of un-pinned buffers; winning split committed back to `op.op_it_space_splits`.

**HBM path.** `memory_planning(nodes)` (`memory_planning.py:137`) packs HBM intermediates (not I/O, `FixedTiledLayout`, `"lx" not in allocation` so LX wins) via live-range analysis + greedy first-fit `Allocator` (`:33`), setting `allocation["pool"]` and `V.graph.pool_size`. Note: `V.graph.pool_size = allocator.get_pool_end()` — the **bump-pointer arena extent** (`_pool_end`, `:250`), *not* the tracked `_peak_usage`. `pool_end ≥ peak` because first-fit over variable-size freed blocks can fragment (a freed block too small for the next request forces an extend); `_peak_usage` is used only for the segment-overflow guard (`:70`) and the info log (`:244`). The freeing rule is conservative: a block frees only when its last-read step is strictly `< ` the new buffer's write step (`:215`).

Both annotations feed codegen: a buffer with `lx` or `pool` is removed from kernel args (`spyre_kernel.py:480-498`) and gets a baked-in start address (`superdsc.py:402-407` resolves `pool` > `lx` > `hbm`).

---

### Stage 6 — Scheduler fusion + LoopLevel IR → OpSpec (SpyreKernel)

**Files:** `scheduler.py`, `spyre_kernel.py`, `op_spec.py`, `fusion.py`, `pass_utils.py`, `views.py`.

**Fusion (`spyre_fuse_nodes`, `fusion.py:52`)** runs at HOOK 6, *before* codegen, and is **order-preserving** (never reorders). It greedily groups consecutive `SchedulerNode`s into `FusedSchedulerNode`s (one SDSC bundle each), constrained only by the **non-intermediate tensor budget** `_max_bundle_tensors()` (`:30`); a non-`SchedulerNode` (e.g. `FallbackKernel`) forces a bundle boundary (`:92-99`). Disable with `SPYRE_INDUCTOR_ENABLE_FUSION=0`.

**Driver: `SuperDSCScheduling.codegen_node`** (`scheduler.py:304`). For each node it computes `iteration_space(node)` (`pass_utils.py:184`), splits into `(iter_vars, reduction_vars)`, creates a `SpyreKernel`, and calls Inductor's `node.codegen(index_vars)` (`:326-333`) — replaying the `Loops` inner_fn through the installed ops handler.

**The RValue layer.** `SpyreKernelOpsHandler` (`spyre_kernel.py:298`, a `DefaultHandler`) intercepts every `ops.*` call to build a transient **expression tree of `RValue` nodes** (`spyre_kernel.py:61-97`):
- `load` → `TensorAccess(name, index, layout)` (`:491`).
- any pointwise op → dispatched by `_default` (`:310`) to a `SpyreOpFuncs` method (`:139-296`, the **ATen-op-name → Spyre-device-op-name table**: `truediv→"realdiv"`, `relu→"relufwd"`, `clamp→"clip"`, `exp→"exp"`, `sub→"sub"`, …) returning a `PointwiseOp`.
- `reduction` → `ReductionOp` (`:337`).
- unknown → `UnimplementedOp`.

**`store` → OpSpec (the core transition).** The whole RValue tree for one buffer is flattened into **exactly one** OpSpec at `store`/`store_reduction` (`:509`/`:568`):
- `create_tensor_arg` (`:385`) turns each `TensorAccess` into a `TensorArg`: `concretize_index` (`pass_utils.py:87`) replaces size symbols, `compute_coordinates` (`views.py:61`) projects the flat index onto device coordinates, and only non-scratchpad tensors are registered as kernel args (`:410-414`).
- `create_op_spec` (`:417`) validates dtypes (FP32 only for `SPYRE_FP32_OPS`), folds work-division splits into `it_space_extended = {sym: (range, split)}` via `apply_splits_from_index_coeff`, computes coarse-tile `tiled_symbols` from `loop_info.loop_tiled_dims` **[post-merge: was the separate `loop_tiled_dims` attr]** (plus reduction-tiled symbols appended after the output syms, Stage 1), and constructs the `OpSpec`.
- For a **bare `TensorAccess`** store (pure copy), the op name is chosen *geometrically* (`:548-564`): all-zero input coords → `IDENTITY_OP` (broadcast); stick variable differs → `RESTICKIFY_OP`; else `IDENTITY_OP`.
- `BATCH_MATMUL_OP` is special-cased (two tensor inputs + output, `:597-611`); reduction `op_info` is pulled from `node.data.op_info`.

**Coarse-tile path.** `CountedLoopSchedulerNode` (`scheduler.py:75`) is driven by `_codegen_counted_loop`/`_codegen_loop_body` (`:353-472`), which produce specs into the same kernel and call `wrap_op_specs_in_loop` (`spyre_kernel.py:625`) to wrap them in a `LoopSpec`.

**Finalization: `codegen_kernel`** (`spyre_kernel.py:638`):
1. `simplify_op_spec` (`:773`) runs `align_tensors` (`views.py:347`) → `normalize_coordinates` on each arg, rewriting `iteration_space` and each arg's `device_size`/`device_coordinates` so all tensors share a codegen-ready iteration space. **Mutates the OpSpec in place.**
2. Assigns each `TensorArg.arg_index` and sets `allocation["hbm"]` segment offsets (`:652-658`).
3. `_codegen_op_spec_list` (`:691`) pretty-prints the OpSpec list as **Python source** (sympy exprs as `sympify('…')`, `op_info` via `_serialize_value` `:99`), emitted by `define_kernel` (`scheduler.py:486`) as `async_compile.sdsc('name', [OpSpec(...), ...])`.

**Information added** in this transition: per-dim device coordinate exprs, device dtype/size/stride_map, scratchpad vs HBM allocation + final arg indices, work-division splits, coarse-tile `tiled_symbols`, canonical Spyre op name. **Dropped:** the pointwise expression-tree structure (flattened — one OpSpec per device op), original ATen overload, host strides, Inductor's CSE machinery.

---

### Stage 7 — OpSpec → SuperDSC JSON + host wrapper

**Files:** `codegen/superdsc.py`, `codegen/compute_ops.py`, `codegen/bundle.py`, `codegen/unroll.py`, `wrapper.py`, `execution/async_compile.py`.

`SpyreAsyncCompile.sdsc(name, specs)` (`async_compile.py:46`) creates a temp `output_dir` under `cache_dir()/inductor-spyre/`, calls `generate_bundle`, then `subprocess.run(["dxp_standalone","--bundle","-d",output_dir])` (`:61`), returning a `SpyreSDSCKernelRunner` (or `SpyreUnimplementedRunner` if any `UnimplementedOp` present).

`generate_bundle(name, output_dir, specs)` (`bundle.py:49`):
1. **`unroll_loop_specs`** (`unroll.py:258`, when `unroll_loops=True`, the default — DeepTools lacks `scf.for` support) fully unrolls every `LoopSpec` innermost-first, baking per-iteration addresses into `allocation` (`pool`/`hbm`/`lx`), shrinking `device_size` to the tile size, and clearing `tiled_symbols`.
2. **Pass 1** `_compile_specs` (`:175`): per OpSpec → `compile_op_spec` (`superdsc.py:646`) → `parse_op_spec` (`:529`) → `generate_sdsc` (`compute_ops.py:222`) → writes `sdsc_N.json` (`json.dump indent=2`).
3. **Pass 2** writes `bundle.mlir` (`:123-167`): module-level `#map_N = affine_map<...>` defs, `%sym_N` constants, recursive body via `_emit_specs` (`:274`): `LoopSpec` → `scf.for`, `OpSpec` → `affine.apply` addr + `sdscbundle.sdsc_execute(...) {sdsc_filename="sdsc_N.json", ...}`.

**`parse_op_spec`** (`superdsc.py:529`) — the field-by-field OpSpec → `SDSCSpec` mapping: symbol renaming to canonical labels (`INPUT_DIM_LABELS`/`OUTPUT_DIM_LABELS`/`MATMUL_DIM_LABELS`), `_concretize_for_sdsc` forcing concrete sizes (`:426`, TODO issue#220 for symbolic), work-division → `dim_splits`/`num_cores`/`work_slices`, device dim order + stick dim from `device_coordinates` (`_get_device_dim_order`, `:216`), matmul K-padding (`_extend_matmul_k_to_padded`, `:455`), per-tensor `SDSCArgs` (`_create_sdsc_tensors`, `:319`: per-dim `scales` [1 normal, -1 reduced, -2 stick-reduction], `strides`, `offsets`, `backGap`, layout dedup by `(dim_order, stick_dim_order, stick_size)` → `LAYOUT_LABELS`), padding/masking, opfunc selection (`nonstick` suffix for non-stick reductions), `execution_unit` (`"pt"` matmul / `"sfp"` otherwise), and `core_id → work-slice` mapping (`_get_core_to_slice_mapping`, `:135`; K-fast variant `:156`).

**Address modes** (`compute_ops.py`, controlled by `bundle_symbolic_args` **[post-merge: renamed from `bundle_hbm_symbols`]**, default `False`): `use_symbols=False` bakes concrete HBM byte addresses into `startAddressCoreCorelet_` (`:330`); `use_symbols=True` registers negative symbol IDs + `affine_strides` for runtime `affine.apply` (`:276`).

**Host wrapper** (`wrapper.py`, `SpyrePythonWrapperCodegen`): `make_buffer_allocation` (`:110`) emits `spyre_empty_with_layout(...)` for `FixedTiledLayout` buffers; `generate` (`:75`) injects a shared `_pool` scratch (`allocate_pool`, `:152`) sized from `V.graph.pool_size`; `noop_simplify_loops_impl` (`:164`) disables Inductor's loop-contiguity simplification.

---

## 5. How to inspect & interpret each stage (consolidated)

### 5.1 The one master facility — `logging_utils.py`

All Spyre logging is gated by **`SPYRE_INDUCTOR_LOG=1`** (`logging_utils.py:19,39-49`). Without it, every Spyre logger is pinned at WARNING and all `.info`/`.debug` is silenced. Then:
- `SPYRE_INDUCTOR_LOG_LEVEL=INFO|DEBUG` (`:20`).
- `SPYRE_LOG_FILE=/path` (default stderr) (`:21`).
- Logger names follow `torch_spyre._inductor.<name>`, created by `get_inductor_logger("<name>")` (`:62`).

### 5.2 Per-stage dump table

| Stage | Mechanism | Logger / Env | What it shows | Location |
|---|---|---|---|---|
| Post-grad / lowering FX | `TORCH_LOGS="+inductor"`, `TORCH_COMPILE_DEBUG=1` | upstream | FX graphs, lowering, scheduler decisions; `GraphTransformObserver` before/after for each custom FX pass | `post_grad.py:110`, `pre_grad.py:332` |
| Lowering detail | `lowering` logger (DEBUG) | `torch_spyre._inductor.lowering` | `lower_mm`/`lower_bmm` input/output sizes & layouts | `lowering.py:287,359` |
| LoopLevel IR **before** pre-scheduling | `passes` logger (INFO) | `torch_spyre._inductor.passes` | `_format_operations(operations)`: each op's name, type, `layout`, `allocation`, `op_it_space_splits`, `.data` | `passes.py:247-248` |
| LoopLevel IR **after** pre-scheduling | `passes` logger (INFO) | `torch_spyre._inductor.passes` | same, post all 15 passes | `passes.py:279-280` |
| Per-pass detail | per-module loggers (DEBUG) | `propagate_layouts`, `optimize_restickify`, `insert_restickify`, `padding`, `dedup_constants`, `coarse_tile`, `work_division`, `chunk_large_tensors`, `assign_dim_hints` | restickify plan, beam contents, stamped `loop_*`, cores/splits, chunking decisions; `assign_dim_hints`/`propagate_named_dims` print INFO tables | e.g. `optimize_restickify.py:483`, `coarse_tile.py:612`, `work_division.py:539`, `temp_passes.py:368` |
| LX allocation | inspect `layout.allocation` dict | module logger `...scratchpad.allocator` | presence of key `"lx"`; `LifetimeBoundBuffer.address` (`None`=spilled) | `allocator.py` |
| HBM allocation | `MEMORY_PLANNING` logger (DEBUG/INFO) | `torch_spyre._inductor.MEMORY_PLANNING` | per-buffer `live=[..] size=.. offset=..`, peak/pool summary | `memory_planning.py` |
| Per-load/store device coords | `spyre_kernel` logger (DEBUG) | `torch_spyre._inductor.spyre_kernel` | `kernel_load`/`kernel_store` name + index + device coords | `spyre_kernel.py:501,528,591` |
| Per-op **OpSpec** | `spyre_kernel` logger (DEBUG) | `torch_spyre._inductor.spyre_kernel` | `op_spec: relufwd, is_reduction=…, iteration_space=…, op_info=…` | `spyre_kernel.py:695-720` |
| **OpSpec list as Python** (most useful) | `TORCH_LOGS=output_code` / `torch._inductor.config.debug=1` | upstream | the verbatim `async_compile.sdsc("name", [OpSpec(...), TensorArg(...)])` pretty-print | `spyre_kernel.py:638-665`, `_codegen_op_spec_list:691` |
| SDSC symbol map + **SDSCSpec** | `codegen.superdsc` logger (DEBUG) | `torch_spyre._inductor.codegen.superdsc` | symbol→label map; full `SDSCSpec.__str__` | `superdsc.py:537-540,654` |
| Bundle write | `sdsc_compile` logger (INFO) | `torch_spyre._inductor.sdsc_compile` | "Generating .../bundle.mlir"; unimplemented-op WARNING | `bundle.py:124`, `async_compile.py:51` |
| **On-disk SDSC artifacts** | filesystem | — | `sdsc_N.json` (one per OpSpec, indent=2) + `bundle.mlir` in `cache_dir()/inductor-spyre/<kernel>_<rand>/` | `async_compile.py:35`, `bundle.py:208,123` |

### 5.3 `repr` / `__str__` notes

- `OpSpec`/`TensorArg`/`LoopSpec`/`UnimplementedOp` are plain `@dataclass` (`op_spec.py`) — `repr()` works; intended readable form is `_codegen_op_spec_list` (`spyre_kernel.py:691`). Enumerate nested specs with `_iter_op_specs(kernel.op_specs)` (`:682`) or `find_unimplemented` (`op_spec.py:107`).
- `SDSCSpec`/`SDSCArgs` **do** define `__str__` (`superdsc.py:94,56`).
- `FixedTiledLayout.__str__`/`__repr__` (`ir.py:100`) prints host size/stride + `device_layout`; `SpyreTensorLayout` has pybound `__str__` (`module.cpp:175`) showing `device_size`/`stride_map`.

### 5.4 Programmatic dump (no hardware)

Call `generate_bundle(name, out_dir, specs, use_symbols=…, unroll_loops=…)` directly on a hand-built or captured `[OpSpec]` list — it writes the JSON/MLIR *without* invoking `dxp_standalone` (that subprocess only runs from `SpyreAsyncCompile.sdsc`). Or call `compile_op_spec(0, op_spec, symbols=[])` / `parse_op_spec(op_spec)` to get the JSON dict / `SDSCSpec` in-process.

### 5.5 The one-liner

```
SPYRE_INDUCTOR_LOG=1 SPYRE_INDUCTOR_LOG_LEVEL=DEBUG TORCH_LOGS=output_code python examples/softmax.py
```
yields, in order: LoopLevel IR before/after (`passes` INFO) + per-op OpSpec + SDSCSpec (DEBUG) + the host wrapper containing the OpSpec list. Toggle gated passes with `COARSE_TILING=1`, `LX_PLANNING=1`, `CHUNK_LARGE_TENSORS=1`, `CO_OPTIMIZING_LX_PLANNING=1`, `SPYRE_CORE_ID_K_FAST_EMISSION=0`, `BUNDLE_HBM_SYMBOLS=1`, `UNROLL_LOOPS=0`, `SENCORES=N`.

---

## 6. Worked example: `examples/softmax.py`

`examples/softmax.py` builds `x = torch.rand(512, 1024, dtype=torch.float16)` and runs `torch.softmax(x, dim=0)` three ways: CPU (`:24`), Spyre eager (`:30`, dispatches via PrivateUse1 to `ops/eager.py` — *does not* touch this front-end), and Spyre compiled (`:33-34`). Only the compiled path is traced below. (`dim=0` is the *outer/row* dim of a `512×1024` tensor — a reduction along the **non-stick** axis.)

**1. ATen decomposition.** Softmax is **not** a Spyre decomposition (`decompositions.py` has no `_softmax`), so PyTorch's in-tree numerically-stable decomposition applies:
```
arg0 → amax(dim=0,keepdim) → sub(x,amax) → exp → sum(dim=0,keepdim) → div → output
```
Both reductions are single-dim, so `multi_dim_reduction_pass` is a no-op.

**2. FX passes (HOOK 2/3).** `collect_spyre_hints`/`recover_spyre_hints` run; `convert_constant_with_graph_node`, `mm_to_bmm_pass`, `bmm_unflatten_pass` are no-ops (no matmul). Gated by `_maybe_run_graph_pass` (`passes.py:94`).

**3. Lowering → LoopLevel IR.** `amax`/`sum` → `Reduction` nodes (`reduction_type "max"`/`"sum"`); `sub`/`exp`/`div` → `Pointwise`. Each `.realize()`d into a `ComputedBuffer`. After `propagate_spyre_tensor_layouts` each gets a `FixedTiledLayout`. This is the IR `_format_operations` dumps.

**4. HOOK 4 passes that touch it.** `propagate_spyre_tensor_layouts` assigns candidate STLs (multi-arg pointwise `sub`/`div` → `AllSameNode`); since `dim=0` is the non-stick reduction, `optimize_restickify_locations` may insert restickify ops where the stick axis disagrees between the reduction output and the broadcast consumers; `span_reduction`/`work_distribution` split the iteration space across `SENCORES` cores (config/shape-dependent — confirm from the before/after dumps). `amax`/`sum`/`exp`/`sub` are exactly the ops in `OP_OUTPUT_GOOD_FOR_LX_REUSE`, so with `LX_PLANNING=1` these intermediates are candidates for the scratchpad.

**5. OpSpec.** `SpyreKernel` yields ~4 OpSpecs: a **reduction OpSpec** (`op="amax"`, `is_reduction=True`, `execution_unit→"sfp"`), a pointwise stack for `exp(x - amax)` (`SpyreOpFuncs.sub` → `"sub"`, `exp` → `"exp"`), a second **reduction OpSpec** for `sum`, and a pointwise OpSpec for `exp/sum` (`truediv` → `"realdiv"`). Each carries `iteration_space`, `args` (`TensorArg`s with `device_coordinates`/`allocation`), `op_info`.

**6. SuperDSC.** `parse_op_spec` → `SDSCSpec` per op; `generate_bundle` writes `sdsc_0.json … sdsc_3.json` + `bundle.mlir` into `cache_dir()/inductor-spyre/<kernel>_<rand>/`; `dxp_standalone` compiles them.

Contrast: `examples/mul.py` (one `aten.mul.Tensor`) → one pointwise OpSpec; `examples/mean.py` (one `aten.mean.dim`) → one reduction OpSpec.

### 6.1 Ground-truth capture (`LX_PLANNING=1`, real hardware run)

Captured on the run machine with
`TORCH_LOGS=output_code,ir_post_fusion LX_PLANNING=1 python examples/softmax.py`
(`x = rand(512, 1024, fp16)`, `softmax(dim=0)`, `SENCORES=32`). Saved at
`example_lx.log`. Two corrections to the conceptual trace above and the live numbers:

- **Correction 1 — op count & names.** It is **5** OpSpecs (not ~4), and the `amax`
  reduction emits `op='max'` (not `"amax"`): `max, sub, exp, sum, realdiv`. `div`→`realdiv`.
- **Correction 2 — `ir_post_fusion` is silent.** Only `__output_code` was emitted; the
  `ir_post_fusion` artifact produced **zero** lines for this backend. The LoopLevel /
  scheduler-node IR is therefore **not** observable via `TORCH_LOGS=ir_post_fusion` — this
  is the concrete motivation for adding our own LoopLevel dump pass (§7).

**One fused bundle.** All 5 ops fuse into a single SDSC bundle `sdsc_fused__softmax_0`,
run by one host call `sdsc_fused__softmax_0.run(_pool, arg0_1, buf4)` — softmax is **one
device kernel launch** across 32 cores (confirms order-preserving `spyre_fuse_nodes`).

**Two independent symbol families.** Reduction OpSpecs use `c0,c1`; pointwise OpSpecs use
`d0,d1`. Each OpSpec carries its *own* iteration space (reconciled at SDSC level), so the
symbol names do **not** correspond across ops.

**Iteration space = work division (32 cores).**

| OpSpec | iteration_space | split meaning |
|---|---|---|
| `max` (reduction) | `{c0:(1024,16), c1:(512,2)}` | free dim 1024 → 16 cores; reduction dim 512 → 2 cores (partial reductions). 16×2=32 |
| `sub` (pointwise) | `{d0:(512,32), d1:(1024,1)}` | 32-way split on 512; 1024 unsplit |
| `exp` (pointwise) | `{d0:(512,32), d1:(1024,1)}` | same as sub |
| `sum` (reduction) | `{c0:(1024,16), c1:(512,2)}` | same as max |
| `realdiv` (pointwise) | `{d0:(1024,1), d1:(512,32)}` | 32-way split on 512 |

**Device layout / sticks (every TensorArg).** Full `f16[512,1024]` → `device_size=[16,512,64]`
(the 1024 dim = 16 sticks × 64 elems), `stride_map=[64,1024,1]` (device-dim → host-stride).
Coordinates e.g. `[floor(c0/64), c1, Mod(c0,64)]`: `c0` indexes the 1024 free axis,
decomposed into stick index `floor(c0/64)` + within-stick `Mod(c0,64)`. Reduced
`f16[1,1024]` → `device_size=[1,16,64]`, `stride_map=[64,-1,1]` — the **`-1`** sentinel marks
the reduced/broadcast (512→1) axis.

**The LX story — what `LX_PLANNING=1` actually bought.** Tracing each buffer's home
allocation through the bundle:

| Buffer | producer.out | consumer.in | home |
|---|---|---|---|
| `arg0` (x) | — | max/sub | `hbm:0x4_0000_0000` (input segment, 16 GiB) |
| `amax` | max → `pool:0` | sub reads `pool:0` | HBM intermediates pool |
| `sub`  | sub → **`lx:0`** | exp reads **`lx:0`** | **per-core LX scratchpad** |
| `exp`  | exp → `pool:2048` | sum + realdiv read `pool:2048` | HBM pool (num_users=2) |
| `sum`  | sum → `pool:0` | realdiv reads `pool:0` | HBM pool (**reuses amax's freed slot**) |
| `div`  | realdiv → `hbm:0x8_0000_0000` | — (= `buf4`) | output segment (32 GiB) |

So LX planning placed the **`sub` intermediate (full `[512,1024]`) into per-core SRAM**, and
`exp` consumed it **in-place from LX** — removing one full-size HBM intermediate. This is
exactly the `OP_GOOD_FOR_LX_INPLACE=["exp","sub"]` /
`OP_OUTPUT_GOOD_FOR_LX_REUSE=["max","amax","sum","exp","sub"]` rule firing (`scratchpad/utils.py:25`).
The host `_pool` (HBM) is `(8208,)` — that count is in **sticks**, not bytes
(`pool_size_sticks = ceil(pool_size_bytes/128)`, `wrapper.py:155`), so the arena is
8208 × 128 ≈ **1.0 MiB**, dominated by the 1 MiB `exp` intermediate. **LX usage is
invisible in the host wrapper** — `lx:` addresses are resolved on-device, so the HBM pool
holds only `amax`/`sum` (2 KiB, reused at offset 0) + `exp` (1 MiB, offset 2048); the equally
large `sub` lives in SRAM instead. Without LX, `sub` (live `[1,2]`) and `exp` (live `[2,4]`)
do not overlap-free, so the pool would roughly double.

**Open optimization question (for later):** `exp`'s *output* went back to HBM (`pool:2048`),
not LX — likely because `exp` has `num_users=2` (consumed by both `sum` and `realdiv`) so it
must outlive the in-place LX window, or the greedy first-fit allocator (`plan_solver.py`)
didn't retain it. Keeping `exp` in LX is a candidate optimization to probe once we
instrument the allocator.

### 6.2 Reading a LoopLevel IR dump entry (ground truth)

Captured with `SPYRE_DUMP_IR=1 LX_PLANNING=1` (dump pass in `dump_loop_ir.py`, wired
into `CustomPreSchedulingPasses` at `passes.py`). Saved at `haoyang_logs/loop_ir_dump.log`.
The dump prints the IR twice — BEFORE any pre-scheduling pass and AFTER all 15 — so the
two can be diffed.

**What the diff shows (softmax).** The `inner_fn` (the actual scalar math) is **identical**
before and after; the middle-end only *annotates*. Three things appear in the AFTER record:
1. `FixedLayout` → `FixedTiledLayout` (a `device_layout` / sticks is attached).
2. `op_it_space_splits` (the work-division plan across 32 cores).
3. `allocation={'lx': 0}` on the `sub` op only (LX scratchpad placement).
HBM `pool` allocations are NOT here — they are assigned later by `memory_planning`
(post-fusion), so at LoopLevel you only see `lx`.

**Two naming systems.** The header `opN:` is the *operation* name (`get_operation_name()`);
the `inner_fn` bodies reference *buffer* names `bufN`. Operation `op_i` produces buffer
`buf_i` (1:1, since Spyre realizes every op). So `op1` (sub) reads `buf0` (amax's output)
and writes `buf1`.

**Anatomy of a pointwise entry — `op1` (sub), AFTER:**

```
op1: ComputedBuffer                                       # realized (materialized) buffer
  layout=FixedTiledLayout('spyre:0', torch.float16,
      size=[512,1024], stride=[1024,1],                   # HOST/logical view (row-major)
      device_layout=SpyreTensorLayout(                    # PHYSICAL tiled view (added by passes)
          device_size=[16,512,64], stride_map=[64,1024,1],
          device_dtype=DataFormats.SEN169_FP16))
  allocation={'lx': 0}                                    # memory home (absent => default HBM)
  op_it_space_splits={d0:32, d1:1}                        # work-division: axis d0 over 32 cores
  Pointwise('spyre', torch.float16,                       # op.data: elementwise node
    def inner_fn(index):
        i0, i1 = index                                    # output iterators, OUTERMOST-FIRST
        tmp0 = ops.load(arg0_1, i1 + 1024*i0)             # = arg0_1[i0,i1] (flat row-major index)
        tmp1 = ops.load(buf0, i1)                         # = amax[0,i1]; only i1 => BROADCAST over i0
        tmp2 = tmp0 - tmp1
        return tmp2,
    ranges=[512,1024],                                    # output iteration space
    origin_node=sub)                                      # provenance to the ATen FX node
```

**Loop iterators (`index`).** `index` is the tuple of output loop iterators, listed
**outermost-first** to match `ranges`. For `ranges=[512,1024]`, `index=(i0,i1)`:
`i0` ∈ [0,512) is the **outer** loop (the rows), `i1` ∈ [0,1024) is the **inner** loop (the
contiguous columns). Confirm it from the flat index `i1 + 1024*i0`: the outer iterator
carries the large stride (1024), the inner one is contiguous (stride 1). Which index
variables appear in a load encodes broadcasting — `ops.load(buf0, i1)` uses only `i1`, so
amax's single row is reused for every `i0`. (This is the *logical* iteration space; Spyre
tiles and parallelizes it rather than running literal nested scalar loops.)

**Anatomy of a reduction entry — `op0` (amax), AFTER:**

```
op0: ComputedBuffer
  layout=FixedTiledLayout(... size=[1,1024], device_size=[16,1,64], stride_map=[64,-1,1] ...)
  op_it_space_splits={d0:16, d1:2}
  Reduction('spyre', torch.float16,
    def inner_fn(index, rindex):                          # NOTE second arg: rindex
        _, i1 = index                                     # output is [1,1024]; dim0=1 discarded
        r0_0 = rindex                                     # reduction coordinate (the 512 rows)
        tmp0 = ops.load(arg0_1, i1 + 1024*r0_0)           # = arg0_1[r0_0, i1]
        return tmp0,
    ranges=[1,1024],                                      # OUTPUT axes (surviving dims)
    reduction_ranges=[512],                               # axes being collapsed
    reduction_type=max,                                   # combine op
    origin_node=amax)
```

A reduction has **two kinds of axes**:
- **output axes** (dims that survive) — indexed by `index`, listed in `ranges`;
- **reduction axes** (dims collapsed away) — indexed by `rindex`, listed in `reduction_ranges`.

`reduction_ranges=[512]` means "for each output element, sweep and combine 512 values."
`rindex` (`r0_0`) is the loop variable walking those 512 values; the `inner_fn` only *loads*
`arg0_1[r0_0, i1]`, and `reduction_type=max` supplies the combine. Conceptually:

```
for i1 in range(1024):          # output coordinate
    acc = -inf
    for r0_0 in range(512):     # rindex: the 512 values being combined
        acc = max(acc, x[r0_0, i1])
    out[0, i1] = acc
```

**Work-division split (`op_it_space_splits`) — two ways to use cores for a reduction.**
Think "column-maxes of a 512×1024 matrix across 32 cores":
- **Split the OUTPUT axis (1024 columns) — no merge.** Give each core a slab of columns; it
  computes the *full* max-over-512-rows for its columns alone. Independent, no coordination.
- **Split the REDUCTION axis (512 rows) — needs a merge.** Give one column to 2 cores: A maxes
  rows 0–255, B maxes rows 256–511; each yields a **partial** max, then a final step does
  `max(partialA, partialB)`. Splitting a reduced axis forces this combine ("partial reductions
  then combined").

`{d0:16, d1:2}` = 16 (columns) × 2 (rows) = **32 cores**. Why not split the columns 32 ways?
The 1024 column axis = `16 sticks × 64`, and work-division won't cut *inside* a stick (atomic
unit), so the column split tops out at 16; the leftover factor of 2 goes to the reduction axis.
Contrast pointwise `sub` `{d0:32, d1:1}`: both dims are output dims, so it simply splits the
512-row dim 32 ways (16 rows/core, full sticks intact) — no reduction, no merge. (Passes:
`span_reduction` sets the minimum reduction split needed to fit per-core memory;
`work_distribution` then fills remaining cores across output dims and at most one reduction dim.)

**`SpyreTensorLayout` fields.** `device_size` = physical dims, **stick axis last**
(`[16,512,64]`: the 1024 contiguous host dim becomes 16 sticks × 64 elems/stick; device dims
= host dims + 1). `stride_map` = device-dim → **host stride** (`[64,1024,1]`); the **`-1`**
sentinel (in reduced tensors, `[64,-1,1]`) marks the reduced/broadcast axis. `device_dtype`
fixes `elems_per_stick` (64 at fp16).

**Field glossary.**

| Field | Meaning |
|---|---|
| `opN:` header / `bufN` in body | operation name / the buffer it produces (1:1) |
| `ComputedBuffer` | a realized (materialized) buffer |
| `layout` host `size`/`stride` | logical row-major view PyTorch sees |
| `device_layout` (`SpyreTensorLayout`) | physical tiled form (sticks) — added by layout passes |
| `device_size` | physical dims, stick axis last; host_dims + 1 |
| `stride_map` | device-dim → host-stride; `-1` = reduced/broadcast axis |
| `allocation` | `lx`=scratchpad, `pool`=HBM arena (added later), absent=default HBM |
| `op_it_space_splits` | per-axis core counts; product ≈ cores used — added by work-division |
| `Pointwise` / `Reduction` | the compute node (`op.data`) |
| `inner_fn(index[, rindex])` | symbolic closure emitting scalar compute via `ops.*` |
| `index` / `rindex` | output iterators (outermost-first) / reduction iterator |
| `ops.load(buf, expr)` | scalar load; `expr` = flat index; index vars present ⇒ broadcast pattern |
| `ranges` / `reduction_ranges` | output iteration space / reduced axis sizes |
| `reduction_type` | combine op for a reduction (`max`, `sum`, …) |
| `origin_node` / `origins` | provenance back to the ATen FX node(s) |

---

## 7. Where to add an instrumentation / print pass

Concrete insertion points, in increasing depth, with the function-signature pattern each follows.

**(A) ATen FX stage** — append a callable to `CustomPostPasses.passes` (`passes.py:154-159`). Signature: `def my_dump(graph: torch.fx.Graph) -> None:` iterating `graph.nodes` printing `node.target`, `node.meta["val"]`. Sees the decomposed softmax chain.

**(B) Between individual LoopLevel-IR passes (the cleanest snapshot point)** — interleave extra calls inside `CustomPreSchedulingPasses.__call__` (`passes.py:250-277`). The dump helper already exists:
```python
# pattern — drop between any two passes, e.g. after step 2 and after step 14:
logger.info("AFTER propagate_layouts\n%s", _format_operations(operations))   # passes.py:72-91
```
No new infrastructure needed; only the before/after points (`:247`,`:279`) are wired today. Each call shows `layout`, `allocation`, and `op_it_space_splits` evolving. **This is the recommended place to instrument the LoopLevel IR → OpSpec boundary's *input* side** (run after step 14 `work_distribution` / step 15 `scratchpad_planning`).

**(C) At OpSpec materialization (the LoopLevel IR → OpSpec boundary's *output* side)** — extend `SpyreKernel.codegen_kernel` (`spyre_kernel.py:638`) or iterate the produced specs. Pattern:
```python
for spec in _iter_op_specs(self.op_specs):     # spyre_kernel.py:682
    logger.debug("OPSPEC %r", spec)            # dataclass repr; or render via _codegen_op_spec_list
```
This dumps every `OpSpec` *after* `simplify_op_spec` but *before* SDSC lowering — the isolated OpSpec stage.

**(D) Between OpSpec and SuperDSC** — hook `parse_op_spec`/`compile_op_spec` (`superdsc.py:529,646`). Pattern: log the returned `SDSCSpec` (which already has `__str__`) and/or the JSON dict from `generate_sdsc` before `json.dump`. Equivalently, instrument `generate_bundle`'s Pass 1 (`bundle.py:175`) to print each `sdsc_N.json` dict before write (`:208-211`). For a no-hardware harness, call `generate_bundle(...)` directly on a captured `[OpSpec]` list and read the emitted files.

**Signature pattern for a self-contained LoopLevel dump pass** (mirrors existing passes; takes and returns the op list so it can slot into any pipeline position):
```python
def dump_operations(operations: list[ir.Operation]) -> list[ir.Operation]:
    logger = get_inductor_logger("dump")          # logging_utils.py:62
    logger.info("OPERATIONS\n%s", _format_operations(operations))  # passes.py:72
    return operations
```

---

## 8. Open questions / needs hardware to confirm

1. **Upstream call-site line numbers** (`graph.py:2260`, `post_grad.py:109/180`, `scheduler.py:2329/2333`) were read from an unpinned wheel and may shift; the config-key names and the `_update_scheduler → Scheduler` ordering are stable contracts.
2. **`SpyreReduction.op_info` rationale** — the docstring (`ir.py:40-47`) itself flags uncertainty ("We believe… TODO: validate") about why reductions need `op_info` instead of the `ops.*` handler.
3. **Stick width per dtype** — `elems_per_stick()` is computed in C++ from `DataFormats`; only the fp16=64 value is documented. Needs the build/hardware to confirm others.
4. **Dynamic shapes** — pervasive `concretize_expr`/`_concretize_for_sdsc`/`_concretize_for_cmp` calls (TODO issues #1371/#1372/#1373/#220) exist because `SpyreTensorLayout` and SDSC generation are not yet SymInt-aware; behavior under fully dynamic shapes is unverified, and the intended symbolic `symbolDefinitions_` JSON path is not implemented.
5. **BMM padding correctness** — `insert_bmm_padding` pads only y, relying on hardware masking x's within-stick tail to zero (`padding.py:33-36`); a hardware behavior assumption, not Python-verifiable.
6. **Restickify cost model** — uses `prod(device_size)` (element count) as a proxy for repack latency (`optimize_restickify.py:91`); accuracy needs hardware measurement.
7. **k-fast heuristics** — `_PT_ROWS=8`, the `rows_per_core` windows, and n/k-stick gates are empirically tuned; generalization beyond tested matmul shapes is undocumented. k-fast/span-reduction interaction has explicit TODOs (`work_division.py:580,656`) leaving parallelism on the table.
8. **LX persistence across bundle boundaries** (scratchpad doc §2) — the planner assumes LX survives SuperDSC bundle boundaries, but VF multi-tenancy may wipe LX on a context switch; needs runtime confirmation. `CloneInputNodesPass` is shipped but inert (`"clone"` commented out of `OP_OUTPUT_GOOD_FOR_LX_REUSE`).
9. **`bundle_hbm_symbols`/`scf.for` path** — requires backend `sdscbundle` symbol-table support still under development; only the default `use_symbols=False` / fully-unrolled path is exercised end-to-end.
10. **SuperDSC JSON schema** — defined implicitly by what `generate_sdsc` emits and what proprietary DeepTools consumes; no checked-in schema. Field semantics (`sdscFolds_`, `dataStageParam_.el_`, `maskingConstId_`, fold encodings) and the int32→fp32 identity-op relabel (DeepTools issue #4307) are only verifiable against the backend, which is not in this repo.
11. **`multi_dim_reduction_pass`** — defined but has no caller anywhere; confirm whether it is intended to be wired in or is dead/example code. `assign_dim_hints` lives in `temp_passes.py` (name suggests provisional placement).
12. **softmax specifics** — the exact in-tree `_softmax` op sequence (and whether `half_to_float` introduces a `to_dtype` cast), and whether `dim=0` triggers `insert_restickify`/how span_reduction splits it, are stated from PyTorch internals and should be confirmed by running the `SPYRE_INDUCTOR_LOG` one-liner on hardware/CPU-sim.

---

## 9. External reference (corroboration)

The official "How torch-spyre works" page
(<https://torch-spyre.readthedocs.io/en/latest/getting_started/how_torch_spyre_works.html>)
corroborates this synthesis. Notable confirmations and verbatim phrasings:

- **Two IR levels for Spyre passes** — "The Spyre-specific passes (orange) operate on
  two IR levels. The first set runs on the FX Graph before Inductor lowering. The second
  set runs on the LoopLevel IR itself before codegen." (matches §1 HOOK 1-3 vs HOOK 4).
- **Stick** — "128-byte aligned sticks of 64 fp16 elements (a constant we call
  `BYTES_IN_STICK=128`)"; "On Spyre, the same tensor is physically stored as four tiles
  of 64-element sticks." (matches §3).
- **Layout propagation** — "`propagate_spyre_tensor_layouts()` traverses the scheduler
  graph and converts the standard `FixedLayout` of each tensor into our
  `FixedTiledLayout`." (matches §4 Stage 3, step 2).
- **Work division** — "`span_reduction()` identifies which iteration dimensions can be
  parallelized across cores ... `work_distribution()` then assigns those spans to the 32
  cores." (matches §4 Stage 4).
- **LX scratchpad** — "The 2 MB programmable scratchpad per core ... The compiler decides
  exactly what lives in SRAM at each point." (matches §5 Stage 5).
- **SuperDSC** — "a JSON-based intermediate representation that describes the full
  tile-level compute graph for the 32 cores of Spyre", with components: core fold
  properties, tensor descriptors, schedule tree (HBM/LX allocate nodes), data staging
  (steady-state/epilogue per-core sizes), compute operations (one per op, encoding the
  execution unit). (matches §2.2 / §7).
- **Debugging philosophy** — "We chose JSON as the wire format for SuperDSC because we
  needed to read and diff these artifacts constantly"; inspecting `sdsc_N.json` in a text
  editor "was often the fastest way to diagnose" address-mapping bugs. (matches §5.2/§5.4 —
  dump the on-disk `sdsc_N.json` + `bundle.mlir`).
- **KTIR** — confirmed as the planned MLIR-based replacement for the SuperDSC JSON wire
  format.

---

## 10. Repo refresh (merge `0afc247`) — pipeline changes & claim verification

A large merge landed (`0afc247`, ~179 files, +27k/−4k). This section is authoritative
where it conflicts with §1/§4 specifics. Verified against the merged code and the refreshed
docs (`inductor_frontend.md`, `coarse_tiling_loops.md`, `work_division_planning.md`,
`ktir.md`, `backend.md`).

### 10.1 Two claims, verified

**Claim A — "ops are now fused at tile level, so the LoopLevel IR is just the actual inner
loop": PARTIALLY CORRECT — true ONLY for hint-driven coarse-tiled groups.**

- TRUE for ops inside a `spyre_hint(tiles=/slices=/num_tiles_per_dim={...})` scope.
  `coarse_tile` divides that op's `data.ranges`, `layout.size/stride`, and the
  `device_layout` `device_size` **down to a single tile** (`coarse_tile.py:1058-1148`), and
  codegen wraps the per-tile op sequence in a `LoopSpec(count=K, body=[OpSpec, ...])` via a
  `CountedLoopSchedulerNode` (`scheduler.py`). So for those ops the IR/OpSpec body *is* the
  single-tile inner loop; `loop_count` + `tiled_symbols` + `affine.apply` reconstruct full
  addressing. Intermediates stay in LX across ops within the tile.
- FALSE / not applicable for **un-hinted** ops. Coarse tiling is opt-in: without hints,
  `hints_to_coarse_tile_groups` yields nothing, ops keep their **full** ranges, and codegen
  uses the flat path — no inner loop. **Our plain `examples/softmax.py` has no hints, so it
  is NOT coarse-tiled** — which is exactly why our LoopLevel dump (§6.2) showed full
  `[512,1024]`/`[1,1024]` ranges, not a tile. `OpSpec.tiled_symbols` is non-empty *exactly
  when* the op was codegen'd inside a `CountedLoopSchedulerNode`.
- The loop **wrapper** lives at the scheduler/codegen layers (`CountedLoopSchedulerNode` →
  `LoopSpec`), not in the LoopLevel IR node — the op node only carries the `loop_info` tag
  plus reduced ranges.

**Claim B — "the low-level IR should NOT be SuperDSC": WRONG (as of the merged code).**

- The compiler **still emits SuperDSC**: `generate_bundle` → `sdsc_N.json` + `bundle.mlir`
  → `dxp_standalone` (`async_compile.py:47-65`, `codegen/bundle.py`, `compute_ops.py`).
  Nothing changed at the `OpSpec` → low-level boundary.
- **KTIR** is the *planned* successor (RFC 0682; `ktir.md`, `backend.md` "From SuperDSC to
  KTIR"). The spec is stable and a reference interpreter exists, but "the backend lowering
  path is in development" — **not production**. Only one mention in code (a TODO comment in
  `temp_passes.py`).
- The kernel of truth behind the claim: a **new runtime execution layer** was added —
  **`JobPlan` / `JobPlanStep`** (`csrc/job_plan.*`, `csrc/prepare_kernel.*`, bound in
  `module.cpp` as `prepare_kernel`/`launch_jobplan`). It translates DeepTools' *SpyreCode*
  (itself produced from the SDSC bundle) into an ordered execution plan of H2D / D2H /
  Compute / HostCompute steps. It is **downstream of** SuperDSC (consumes it), opt-in via
  `DUMP_SPYRE_CODE`, and does **not** replace the compiler's low-level IR.

> Net: **SuperDSC = still the low-level compiler IR / wire format.** JobPlan = a new *runtime*
> execution IR *below* the backend. KTIR = future.

### 10.2 Pipeline structure now (six hooks, refactored)

`passes.py` was rewritten around two pipeline base classes — `_SpyreGraphPassPipeline`
(FX-graph passes, device-guarded) and `_SpyreNodePassPipeline` (scheduler-node passes) —
sharing a `_uuid()` helper, with a `@_runs(...)` tag so config-gated wrappers still key the
Inductor cache on the *real* passes' source files. Passes now receive a `GraphLowering`
`graph` (not a `list[Operation]`).

The six hooks (five upstream + the monkey-patched sixth):

| Hook | Runs now |
|---|---|
| `CustomPreGradPasses` | (empty) |
| `CustomPrePasses` | `collect_spyre_hints` |
| `CustomPostPasses` | `recover_spyre_hints`, `convert_constant_with_graph_node`, `mm_to_bmm_pass`, `mark_direct_unit_bmm_pass`, `bmm_unflatten_pass` **+ our `dump_fx_graph`** |
| `CustomPreFusionPasses` | `propagate_mutation_layouts`, `build_loop_scheduler_nodes` (builds `CountedLoopSchedulerNode`s **before** Inductor fusion) |
| `CustomPostFusionPasses` | `memory_planning`, `spyre_fuse_nodes` |
| `CustomPreSchedulingPasses` | the 14-step pipeline below (via `_update_scheduler` monkeypatch) **+ our `dump_loop_ir` BEFORE & AFTER** |

The **14** pre-scheduling steps (`passes.py:321-346`), in order:
`deadcode_elimination` · `propagate_spyre_tensor_layouts` · `optimize_restickify_locations`
· `finalize_layouts` · `insert_restickify` · `insert_bmm_padding` ·
`dedup_and_promote_constants` · `_maybe_chunk_large_tensors` [gated] · `propagate_named_dims`
· `assign_dim_hints` · `_maybe_coarse_tile` [hint-gated] · `span_reduction` ·
`_distribute_work` (`cost_model_matmul_division` + `work_distribution`) ·
`_maybe_scratchpad_planning` [LX-gated].

Changes vs the old §4 model: work division is now **three** passes; `coarse_tile` (step 11)
is new; `assign_dim_hints` moved into `propagate_named_dims.py`; every pass takes `graph`.

### 10.3 Work division is now THREE passes (`work_division_planning.md`)

1. **`span_reduction`** — commits the *minimum* splits to keep per-core span ≤ 256 MB; else
   leaves the op untouched.
2. **`cost_model_matmul_division`** — matmul/bmm only; enumerates `(b,m,n,k)` splits, prices
   each via `cost = (compute + hbm + psum + tie_break) × b^1.4`, picks the cheapest. Declines
   to Pass 3 if not a matmul / Pass 1 already split / dims ambiguous / it would use fewer
   cores than the default.
3. **`work_distribution`** — everything else (+ declined matmuls); fills output dims by
   decreasing size, then ≤ 1 reduction dim.

Named work-division hints `spyre_hint(work_div={...})` resolve per-op and are committed
directly (bypassing auto-distribution and the matmul cost model), authoritative even over
Pass 1 (with a warning). `SPYRE_INDUCTOR_IGNORE_HINTS=1` disables them. Kept **separate** from
coarse-tiling hints `spyre_hint(tiles=...)`.

### 10.4 New op attributes (now shown by `_format_operations` / our loop-IR dump)

- **`loop_info: CoarseTileInfo`** (`loop_info.py`): `loop_group_id` (nesting-path tuple),
  `loop_count` (per-level trip counts), `loop_tiled_dims` (per-level output-range indices),
  `loop_tiled_reduction_dims` (NEW — per-level reduction-range indices, Stage 1).
- **`dim_hints: list[DimHint]`** (`propagate_hints.py`): `dim_names`, `split_count`,
  `loop_var`, `is_reduction`, `hint_id` — the *input* to coarse tiling from `spyre_hint`
  scopes. (passes.py `_format_operations` now also prints `dim_hints`/`loop_info`.)

### 10.5 Stage 1 reduction-dim tiling (#2572)

Coarse tiling can now tile a **non-stick reduction** dim via "fill-init + per-tile combine":
allocate a full-size HBM accumulator, fill it with the reduction identity (`sum`→0/`add`,
`max`→−∞/`maximum`, …), insert an in-loop combine op that merges each tile's partial, mark
the output `per_tile_fixed`, and patch outside consumers. Stick-dim reduction tiling and
mixed output+reduction levels (Stage 2) raise `RuntimeError` for now.

### 10.6 Why the running script likely fails

Static analysis of the merged middle-end **and** our instrumentation/bench is **clean**: all
imports resolve, no conflict markers, pass signatures match the new `GraphLowering` contract,
`dump_fx_graph`/`dump_loop_ir`/`bench`/`profiling` are all compatible, and the #2585
"utilities" refactor touched **only test files**. So the failure is not a Python-level break
we can see here.

Most likely cause: **the C++ extension `_C` needs rebuilding.** The merge changed `csrc/`
heavily — `job_plan.{h,cpp}`, `prepare_kernel.{h,cpp}`, `module.cpp` (+38), `spyre_mem.cpp`,
`spyre_tensor_impl.h` — so a stale installed `_C` will ABI-mismatch or miss symbols. Rebuild
on the run machine (the project's editable/build step) and re-run. If it still fails, the
**traceback is required** (static Python is clean); localize with
`SPYRE_DUMP_IR=1 TORCH_LOGS="+inductor"`. Cosmetic only: `kernel_runner.py:16` carries an
unused `import torch` from the merge resolution (ruff F401, harmless at runtime).

### 10.7 Doc drift to ignore

`getting_started/how_torch_spyre_works.md` and `key_concepts.md` still describe work division
as "two-pass" — **stale**; use the three-pass model in `work_division_planning.md`.
