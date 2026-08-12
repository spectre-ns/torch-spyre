# Spyre Cost Model — Design, Status, and Plan

A self-contained record so work can resume after any context loss. Companion to
[compiler_pipeline_deep_dive.md](compiler_pipeline_deep_dive.md) (how the compiler/IR works)
and the auto-memory `project-goal-compiler-instrumentation.md`.

Repo: `/home/zhang402/torch-spyre/torch-spyre`. Dev machine has **no Spyre runtime/hardware**;
a separate **run machine** (`/home/haoyang/torch-spyre`, same git) executes. Do all
edits/lint here; hand precise commands to the run machine and paste results back.

---

## 0. TL;DR — where we are

- **Goal:** a high-level, *relative* performance model that predicts Spyre kernel device
  latency from the **"LoopLevel IR — AFTER pre-scheduling passes"** graph, to guide
  higher-level compiler optimization. **Not a cycle-accurate simulator.**
- **Measurement is solved.** The GOLDEN measurement is the **torch.profiler "Self SPYRE"
  per-kernel device time** (`sdsc_fused_*`). Our old `SPYRE_PROFILE_SYNC` min measured
  kernel + a non-deterministic, size-scaling Memset/host-setup bucket (tracked separately).
- **Model = TURNAROUND** (`run_profile_sweep.sh` §B–F, encoded in `cost_model.py`):
  **`T = fill + (R+W)/BW_PEAK + α·min(R,W) + c_loop·L`**, R/W = HBM read/write bytes, L = tile
  count. Calibrated: **`fill≈0, BW_PEAK≈150 GB/s, α≈0.0057 ns/byte, c_loop≈860 ns/tile`**. One
  *peak* bandwidth (one-directional) minus a **read/write turnaround penalty on the overlap
  `min(R,W)`** → reproduces the V-shaped effective BW. **LX traffic ~free.** Validated: **~2%
  error on core pointwise + reductions, ~7% overall** (vs ~11% for an additive two-rate).
- **Coarse tiling now modeled (Part 2, §5.4).** Loop-aware bytes (per-arg `loop_factor`: the
  advancing reduced input counts once, the per-tile accumulator counts ×L, LX scratch free) +
  a calibrated per-tile loop overhead **`c_loop≈860 ns/tile`** (the K-sweep slope). So tiling a
  *standalone* reduction is **slower** (~+0.86 µs/tile, no input-traffic saving); LX on/off makes
  no difference for it (partial too small); sum/amax/amin identical. After `c_loop`, tiled
  reductions hit ~6–7% (was −52% at K=16).
- **Verified:** arithmetic-free (gelu/exp == neg on kernel time); HBM BW shared,
  core-independent ≥2 cores; LX ~29× HBM (so ~free); **broadcast operand loaded ONCE at its
  own small device size — NOT zeroed, NOT per-core** (rung-G probe: bcast ≈ bcastcol ≪ add).
- **Bandwidth V-shape (the old ~111 cap EXPLAINED + now MODELED):** effective BW is highest
  one-directional — read-only ~**150**, write-only ~**125–140** — and dips to ~**105** at a
  balanced 1R+1W, *below either pure direction*. The `α·min(R,W)` term captures this dip with
  one constant. The deep mechanism (half-duplex / turnaround / shared bus) still wants
  `aiu-smi` confirmation, but the cost **effect** is now in the model.
- **Stream-count "anomaly" RESOLVED (was Rung 7):** mul/add are **not** anomalous — effective
  BW simply rises with read fraction. The turnaround model fits `mul` (2R1W) at 116 GB/s to
  0.2%, and `add3`/`add4` to ~8% (LX intermediates not *perfectly* free). Do **not** encode a
  per-stream BW table.
- **Known residual biases (B–F):** broadcast pointwise ~**17% faster** than the model (off the
  V-curve; cause OPEN — the most interesting open thread); write-only ~16%; fan-in
  `add3/add4` ~8%; `sumcol` (reducing the outer/partitioned axis) ~19%.
- **Lesson logged** (memory `claim-discipline-perf-modeling`): don't make mechanism claims a
  single data point can't support; name the controlled experiment instead.

---

## 1. Goal

Predict **relative** device latency of a Spyre kernel from its after-pre-scheduling LoopLevel
IR, so we can rank compiler choices (LX placement, work division, fusion, tiling) without
running. Accuracy target: good enough *ordering*, not absolute ns. A simulator would be
useless overkill.

The IR input per op gives: `ranges` (output iter space), `reduction_ranges`,
`op_it_space_splits` (cores per dim), `device_size`/`stride_map` (sticks), `allocation`
(LX or not — only LX is annotated at this stage; everything else is HBM), and the `inner_fn`
(loads → which tensors, broadcast pattern; the arithmetic). See §6.2 of the deep-dive for how
to read an op.

---

## 2. Design principles (agreed with user)

1. **Relative, not simulator.** Few parameters, coarse. Predict which is faster.
2. **Bandwidth + one fixed term, not per-access latency.** A static-dataflow engine streams
   sticks back-to-back; per-access latency is hidden and shows up once as a **fixed
   per-kernel cost** (≈20 µs measured; pipeline fill+drain + device setup + ~7 µs host
   residue). So model traffic ÷ effective bandwidth, plus that fixed term.
3. **Effective bandwidth folds in the messy parts.** Real per-core BW depends on DRAM
   controller buffering, concurrent cores, row-buffer hits — but Spyre access is stick-aligned
   (128 B), streaming, statically scheduled, so those are *constant* for a given shape and
   collapse into one **effective aggregate BW**. For *relative* predictions with the same
   streaming pattern, the absolute BW **cancels** → robust without modeling the controller.
4. **Verification-driven.** Build the model AND a benchmark that checks each assumption; add a
   term only where a benchmark shows the simple model *misranks*. (See §9.)
5. **One knob at a time, simplest first.** Pointwise before reductions; single op before fused.

---

## 3. The model

Per kernel/bundle:

```
T = fill + (R + W) / BW_PEAK + α · min(R, W) + c_loop · L     (LX traffic ~free)
```

where `R` = HBM bytes READ (inputs), `W` = HBM bytes WRITTEN (outputs), and `L` = the
coarse-tiling loop trip count (1 when not tiled, so the last term vanishes for the common
case). `R`/`W` are **loop-aware** when tiled — see the coarse-tiling bullet below.

- `fill` (≈**0**) — the golden **kernel** has *no* fixed term (section-A intercepts ~0). The
  old `fixed≈20µs` was a **separate, non-deterministic, size-scaling Memset/host-setup bucket**
  (the profiler's "Memset (Device)" event), which our old `SPYRE_PROFILE_SYNC` min folded into
  the kernel — it is NOT kernel cost and is tracked separately, not modeled.
- `BW_PEAK` (≈**150 GB/s** == bytes/ns) — the **one-directional peak** HBM bandwidth, reached
  by read-only or write-only traffic. "Aggregate / shared-HBM" assumption (SENCORES rung 5:
  core-independent ≥2 cores).
- `α · min(R, W)` (≈**0.0057 ns/byte**) — the **read/write turnaround penalty.** HBM is a
  shared bus that must switch between read and write; the cost falls on the **overlap**
  `min(R,W)` — 0 for one-directional traffic, maximal at a balanced 1R+1W. This carves the
  measured **V-shaped** effective BW (≈150 read-only → ≈105 balanced → ≈125–140 write-only)
  with **one constant** instead of a second bandwidth. `α` is solved from the balanced neg
  (eff 105): `2/105 = 2/BW_PEAK + α`. (An additive two-rate `R/150 + W/81` fits the common ops
  too but mispredicts pure writes by ~55%, because it taxes writes even with nothing to overlap
  — the turnaround form taxes only *concurrent* R+W, which is physically right.)
- **LX traffic is treated as ~free (no `BW_LX` term).** Rung 4's per-op LX cost (~1 µs) sits
  below the noise; LX is ~29× HBM. `lx_bytes` is computed for inspection but contributes 0.
  (The `add3`/`add4` ~8% under-prediction hints LX isn't *perfectly* free — a future refinement.)
- `R` / `W` / `lx_bytes` — `R` sums input args, `W` the output, each attributed to HBM or LX by
  **allocation propagation** (an op's input memory = its producer's output allocation).
  LX-placed tensors don't count toward HBM. R and W are summed over the **whole bundle** before
  the turnaround term (a fused kernel interleaves all its reads/writes on one bus).
- **Sizes come from the DEVICE layout (sticks), not the torch logical shape.** Each arg's bytes
  use `FixedTiledLayout.device_layout.device_size` (stick-padded: a row of N fp16 rounds up to
  `ceil(N/64)*64`), available post-`finalize_layouts` where the cost dump runs. A reduction's
  reduced input is **naturally full-sized**, and a `[1,N]` broadcast carries its real one-row
  device size.
- **Broadcast/scalar inputs are loaded ONCE — counted, NOT zeroed, NOT per-core.** An input
  whose index references fewer loop vars than the output rank (**including 0 vars** — a scalar
  like the `1.0` in `x+1.0`) is loaded once and reused across the broadcast dim. It is counted
  at its **own small one-row/-col device size** — not scaled to the output, and not dropped to
  zero (the earlier "excluded → free" was wrong). **rung-G verified once, not per core**: at
  cores=32/R=64, `bcast (b[1,C]) ≈ bcastcol (b[R,1]) ≪ add` — a per-core reload would have
  pushed `bcast` up to `add`; it didn't. (A scalar with no resolvable buffer falls back to ~1
  element, not the output size.) Flagged via `n_index_vars < n_out_vars` in the dump.
- **Bulk-load / contiguous assumption.** The model assumes the **default contiguous layout**:
  each core's tile is contiguous and stages in one **bulk DMA read** (not many scattered
  per-stick reads). Spyre's memory requests are limited, so contiguous layout is *required* for
  full bandwidth (`tensors_and_layouts.md`). A strided/scattered layout would move less BW and is
  **NOT modeled** (would need an access-pattern term — see the `sumcol` ~19% miss).
- **Reductions need no special bandwidth.** A reduction is just a kernel with a tiny `W` (the
  small output), so `min(R,W)≈0` → `T ≈ R/BW_PEAK`, the read-only rate, automatically.
  `is_reduction` survives only to add a cross-core ring-combine term (`psum_per_elem_ns`).
- **Coarse tiling — loop-aware traffic + a per-tile loop cost.** A `spyre_hint` splits a
  reduction's reduced axis into `K` tiles; the pass emits `fill + K×(tiled-reduce, combine)`
  inside one unrolled bundle, and stamps `loop_info`/`dim_hints` on each op. Two effects, both
  read from the IR:
  - **Loop-aware bytes (per-arg `loop_factor`).** An arg's bytes scale by its loop factor:
    **1** for an *advancing* arg (the reduced input walks the full tensor once across the `K`
    tiles, so its full `device_size` already covers all of it) or a normal arg; **`L`** for a
    *fixed* arg held at one address across the loop (a per-tile **accumulator** re-read/written
    every iteration). LX scratch is ~free regardless. So a tiled reduction adds only the
    accumulator round-trips `~2·K·D` over the untiled version — it does **not** reduce the input
    read.
  - **Per-tile loop overhead `c_loop·L`.** With `unroll_loops=True` the loop unrolls into `L`
    body copies; each dispatch adds a fixed cost beyond its memory traffic. **CALIBRATED
    `c_loop ≈ 860 ns/tile`** (§5.4). So tiling a *standalone* reduction is **slower** (more
    tiles ⇒ more overhead, no input-traffic saving) — the payoff is fused chains that keep
    intermediates in LX, which is out of this round's scope.

**Per-op vs fused:** model a single op as its own kernel; a fused bundle = sum of its ops'
traffic with **one** fill, and intermediates that stay in LX don't hit HBM. (Our examples run
SDSC-bundle fusion but **NOT** tile fusion — tile fusion needs `spyre_hint` coarse-tile groups,
absent in bare softmax/gelu. Tile fusion is a *separate, later* regime where loop-internal
intermediates become per-tile on-chip scratch.)

**Why two "fusions" — do not conflate:**
- **SDSC bundle fusion** (`spyre_fuse_nodes`): groups ops into one kernel (`sdsc_fused_*`);
  intermediates still live in memory (HBM/LX). ACTIVE in our runs.
- **Tile fusion** (coarse tiling): a `spyre_hint` loop with per-tile scratch. NOW EXERCISED by
  the P2 reduction-tiling runs (ctsum/ctamax/ctamin) and handled by the loop-aware bullet above;
  the LX-residency win for fused *pointwise* chains is still out of scope.

**Calibrated values (fp16, B–F + Part-2 profiler sweeps, encoded in `cost_model.py`):**
`fill≈0`, `BW_PEAK≈150 GB/s`, `α≈0.0057 ns/byte`, `psum_per_elem_ns≈0.14`,
**`c_loop≈860 ns/tile`** (§5.4). **LX traffic ~free** (no `BW_LX` term — rung 4 below noise;
P2 confirmed LX-on/off makes no difference for standalone tiled reductions). Accuracy: **~2% on
core pointwise + reductions, ~7% overall**; tiled reductions ~6–7% after `c_loop` (was −52% at
K=16). Residual biases (broadcast +17%, write-only, fan-in, sumcol/dim0) listed in §0/§11.
See §5 for the data.

---

## 4. Measurement methodology (the instrument)

Static dataflow ⇒ device latency is **deterministic**. Strategy: take the **min over N runs**
(host jitter only *adds* time, so min strips it); confirm via collapsed spread.

**Key facts established:**
- **Host floor ~70 µs/call.** Every compiled-fn call pays ~70 µs host overhead (Dynamo guard,
  wrapper, output alloc, ~7 µs async dispatch). Small-op device compute hides under it →
  end-to-end wall-clock CANNOT measure it.
- **Device sync now exists** (merge `#918`): `torch_spyre._C.synchronize(device=None)` blocks
  on the Flex runtime stream. (Pre-merge the c10 sync hooks were no-op stubs; the merge added
  the `SpyreStream`/stream-pool plumbing that holds a `flex::RuntimeStream*` handle to wait on.
  `elapsedTime`/event-timing is still stubbed — we time the host clock *around* a sync.)
- **`SPYRE_PROFILE_SYNC=1`** makes the per-kernel timer block on the device after each launch →
  the registry's per-kernel **min** is real device latency, for ANY op (small included).
- **Don't use end-to-end `net` or the identity baseline for small ops** — host-wrapper
  differences dominate (gave nonsensical negative nets). Read the per-kernel device min directly.

**GOLDEN measurement = the profiler's kernel device time.** `torch.profiler`
(`ProfilerActivity.PrivateUse1`, needs the kineto-spyre wheel; `examples/profile_ops.py`)
reports a **"Self SPYRE"** column = the TRUE per-kernel (`sdsc_fused_*`) device time. This is
the standard to trust going forward. Cross-check (gelu[512×1024]): kernel **17.3 µs** ≈ our
*traffic* term (18.9 µs) ✓. Our `SPYRE_PROFILE_SYNC` min (~37.9 µs) = kernel + a separate,
**non-deterministic ~20 µs overhead** (the profiler's `Memset (Device)` = host/device setup),
so the old **`fixed ≈ 20 µs` is that OVERHEAD bucket, not kernel cost**. When predicting the
KERNEL time, re-fit `fill_ns` against profiler kernel times across sizes (expect it to drop to a
small device pipeline-fill). The Memset scales (12.5 µs tiny → 28.8 µs at 512×1024) and sits
OUTSIDE our `kernel_timer` bracket — characterize it with a `profile_ops` size sweep.

**Env vars (instrument):**
| var | effect |
|---|---|
| `SPYRE_PROFILE=1` | record per-kernel host launch times (registry, atexit table) |
| `SPYRE_PROFILE_SYNC=1` | sync device after each launch → per-kernel **device** latency |
| `SPYRE_PROFILE_FILE=path` | append profile report to file instead of stderr |
| `SPYRE_DUMP_IR=1` | dump ATen FX + LoopLevel IR (before/after pre-scheduling) |
| `SPYRE_DUMP_COST=1` | dump cost-model features + prediction at compile time |
| `LX_PLANNING=1/0` | LX scratchpad planning (now defaults ON) |
| `SENCORES=N` | number of cores (1–32) |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` | force recompile (else cache hit skips dumps) |

---

## 5. Empirical data gathered (calibration set)

All device-side, deterministic (min-of-N), fp16:

| op | size | elements | device min | notes |
|---|---|---|---|---|
| softmax (bundle) | 512×1024 | 0.5 M | 65–71 µs | 5 ops fused |
| softmax | 512×4096 | 2 M | ~200 µs | |
| softmax | 4096×4096 | 16 M | 1594 µs | 8× elems ⇒ 8× time (linear) |
| add (`a+0`) | 512×1024 | 0.5 M | 32.9 µs | 1-in/1-out memory pass |
| add | 512×4096 | 2 M | ~87 µs | |
| add | 4096×4096 | 16 M | 594 µs | |

**Fits** (`latency = fixed + slope × M-elem`):
- **add ≈ 14.8 µs + 36.2 µs/M-elem** (near-perfect)
- **softmax ≈ 16.3 µs + 98.6 µs/M-elem** (slope ≈ 2.7× add ⇒ ~2.7 DDR passes)
- **fixed ≈ 15 µs** shared by both ⇒ op-independent fixed cost. (3-point fit; the 5-point
  gelu ladder in §5.1 refines this to **~20 µs** — use that.)
- One 0.5 M-elem (2 MB) HBM round-trip ≈ **18 µs** ⇒ **~110 GB/s** effective.

### 5.1 gelu op-ladder calibration (`run_cost_model_plan.sh`, 2026-06-12, fp16)

The pointwise ladder, all DEVICE min via `SPYRE_PROFILE_SYNC`:
- **Rung 1 (gelu size sweep, ROWS=512):** 0.26M→29.5µs, 0.52M→39.2, 1.05M→59.0, 2.1M→93.0,
  4.19M→171.5. Clean linear ⇒ **`fixed≈20.1µs`, `BW_HBM≈111 GB/s`** (slope 35.9µs/Melem;
  2 passes × 2 B). Predicts to ~3%.
- **Rung 2 (arithmetic):** relu 38.4, gelu 38.2, exp 38.4, sigmoid 34.5 (≈equal) ⇒
  **arithmetic FREE, memory-bound. CONFIRMED.** (Also confirms `fixed` is op-independent.)
- **Rung 3 (traffic):** gelu(1-in) 41.0, mul(2-in) 58.1, add 57.7. +1 input ≈ +17µs.
  **BUT** measured mul is ~20% OVER the byte-linear model — UNATTRIBUTED (could be `BW` or a
  per-stream fixed cost; a single size point can't tell). → Rung 3b (below).
- **Rung 4 (LX chain, all-LX):** gelu depth 1/2/4/8/16 → 39.6/34.4/34.7/39.3/43.0 — nearly
  FLAT and **non-monotonic** (16 chained gelus ≈ 1). The per-op LX cost (~1 µs) is *below* the
  run-to-run noise (~5 µs), so `BW_LX` can't be resolved (the ~3200 GB/s "fit" is an artifact).
  Decision: **treat LX as ~free** (drop the term); qualitatively LX is ~29× HBM.
- **Rung 5 (SENCORES):** 1→60.3, 2→40.5, 4→43.5, 8→41.9, 16→40.8, 32→40.5. 1→2 helps then
  FLAT ⇒ **HBM BW SHARED, saturates ~2 cores ⇒ core count NOT a direct model term.**

**Rung 3b (queued)** — controlled multi-input attribution, holding `fixed` op-independent:
a `mul` size sweep (fit its own `fixed`,`BW`), plus EQUAL-total-bytes/different-stream pairs
(`mul[512×1024]` 3-stream == `gelu[512×1536]` 2-stream = 3,145,728 B; and the 6 MB pair
`mul[512×2048]`==`gelu[512×3072]`). The model predicts each pair EQUAL; any gap is the pure
stream-count effect. Gap constant 3MB→6MB ⇒ fixed per-stream cost; gap doubles ⇒ per-byte BW.

**Core division — effect (Rung 5 + reasoning):** for memory-bound pointwise, core *count*
(≥2) has **no direct** effect (BW-bound, flat). It matters **indirectly** via: (1) **LX fit** —
per-core tile = `total/cores`; placement only succeeds if it fits ~1.6 MB, flipping
`lx↔hbm` bytes (the cliff; already in the model through `allocation`); (2) load balance
(uneven splits → max-core); (3) **reductions** — splitting the reduced axis adds a cross-core
combine (a term still owed). Direct access-pattern/locality effects are UNVERIFIED.

**LX experiment** (softmax[512×1024], device min): **LX on = 71.05 µs, off = 91.23 µs ⇒ −20 µs
(22%)**. LX keeps the `sub` intermediate in SRAM, removing its ~18 µs HBM round-trip (matches
the add round-trip cost). Implication: LX planning subtracts a tensor's HBM passes; fusion and
LX are complementary (fusion = one kernel, LX = keeps intermediate off DDR). Predicted cliff:
benefit vanishes once per-core `sub` tile (`total/cores × 2 B`) outgrows ~1.6 MB LX (~8192²).

### 5.2 Broadcast, stream-count, and bandwidth (`run_cost_model_plan.sh`, 2026-06-15/16, fp16)

- **Rung 6 (broadcast) — CACHED, general.** `bcast` (`a+b[1,N]`) 36 µs vs `add` (full) 58;
  `mulbcast` 35 vs `mul` 58. The broadcast operand lands on the **2-pass** (no-broadcast)
  latency ⇒ loaded once, ~free; not add-specific. Scalars too (`x+1.0` ≈ `gelu`). **Encoded.**
- **Rung 7 (stream count) — FALSIFIES a stream-count BW law.** Per-byte rate is NON-monotonic in
  operand count: gelu(2-stream) 116, mul(3) 88, add3(4) 118, add4(5) 124. The n-ary adds fuse to
  one kernel **staging intermediates in LX ⇒ all have just 1 HBM write**; extra reads barely
  cost (reads are cheap). So `{2:111,3:80}` is WRONG. Only plain 2-input `mul`/`add` is
  anomalously ~15-25% slow — **UNEXPLAINED** (known residual).
- **Rung 8 (bandwidth, asymptotes at 16k–64k cols).** read-only (`sum`) ~**176** GB/s, write-only
  (broadcast-fill) ~**146**, balanced 1R+1W (`neg`/`copy`) ~**97**. `neg`≈`copy` confirms the
  scalar is free. **Reads ~saturate the 204.8 LPDDR5 peak; mixing in writes ~halves it** — the
  real reason pointwise caps ~100-110 (not a "half the DRAM peak" mystery).
  *Byte-count sanity check:* at the same tensor size `copy` (1R+1W) moves 2× the bytes of `read`
  (~1R), so the bytes are counted consistently. But the TIME ratio is **not** a clean 2× — it is
  1.3× (small, overhead-compressed) → 3.4× (large), i.e. >2× once size dominates. That excess
  over 2× IS the R+W penalty (`copy` runs at ~half `read`'s BW); a clean 2× would mean no penalty.
- **Rung 9 (copy × SENCORES, large fixed size).** BW rises 1→4 cores (41→112) then plateaus
  ~100; **bigger per-core tiles do NOT help** (1 core is slowest) ⇒ the cap is a **shared-bus
  saturation reached at ~4 cores**, NOT per-core burst/scratchpad size.
- **Rung 10 (write-fraction V-curve) — CONFOUNDED.** Multi-output `w2`/`w3` re-read `x` per
  output (not a shared load), so they aren't clean 1R:NW; do not use. `aiu-smi` is the proper way
  to vary the read/write mix — see `bandwidth_turnaround_experiment.md`.

**`mul`/`add` "anomaly" RESOLVED (turnaround model):** the plain 2-input binary is not
anomalous — it's a 2R+1W kernel (read fraction 0.67), and effective BW simply rises with read
fraction. The turnaround model fits `mul` (2R1W) at 116 GB/s to **0.2%**. The earlier "~15-25%
slow" framing compared it against the wrong baseline (the balanced 1R+1W rate); against its own
read:write mix it is exactly on-model.

### 5.3 GOLDEN re-anchor on profiler kernel time (`run_profile_sweep.sh`, 2026-06-18, section A)

We re-built the model on the **profiler's per-kernel device time** ("Self SPYRE"), discarding
the old SPYRE_PROFILE_SYNC fit (whose ~20 µs "fixed" was non-deterministic overhead). Section A
(neg + gelu size sweep, fp16):

- **Kernel time is linear in I/O, with ~ZERO fixed.** `kernel = fill + bytes/BW`: neg fill −2.4 µs
  (≈0), BW **104** GB/s, R²=0.9997; gelu fill −3.4 µs (≈0), BW **100**, R²=1.000. So **`fill_ns→0`,
  `BW_HBM→102`** (balanced 1R+1W kernel rate). gelu≈neg ⇒ arithmetic-free on the golden time.
- **The Memset/setup overhead is NOT fixed — it scales with I/O.** neg `11.5 µs + 0.0275 ns/elem`,
  gelu `25.9 µs + 0.0244 ns/elem` (fixed part noisy/non-deterministic; the per-elem scaling is
  real, ~60% of the kernel slope). So the old "20 µs" was just the fixed component at small sizes.

**B–F sweep (2026-06-18, `profile_sweep_20260618_210419.log`) → TURNAROUND model.** Effective
BW is set by the read:write mix, not op/size: read-only ~**150**, balanced ~**105**, write-only
~**125–140**, multi-read (`mul`) ~**116**. Fit `T = (R+W)/BW_PEAK + α·min(R,W)` with
**`BW_PEAK=150, α=0.0057`** (α from the balanced neg). Validated: **~2% on core ops, ~7%
overall** (vs ~11% for an additive two-rate, which mispredicts pure writes ~55%). `cost_model.py`
now encodes `fill_ns=0`, `bw_peak_gbps=150`, `rw_turnaround_ns_per_byte=0.0057`. Residual biases:
broadcast **+17%** (off the V-curve, OPEN), write-only ~16%, fan-in `add3/add4` ~8%, `sumcol`
~19%. The reload probe (rung-G) confirmed the broadcast operand is loaded **once, not per core**.

### 5.4 Coarse-tiling reduction (`run_5h_sweep.sh` Part 2, 2026-06-19, fp16)

`ctsum`/`ctamax`/`ctamin` = `spyre_hint(num_tiles_per_dim={"B":K})` over a `[2048, D]` dim-0
reduction, run NORMAL (`LX_PLANNING=0`, partial in HBM) and LX-ON, sweeping `K` and `D`
(`grand_sweep_20260619_151132.log`). Three findings:

- **The coarse-tiling loop is NOT free — `c_loop ≈ 860 ns/tile`.** K-sweep (D=512, K∈{2,4,8,16}):
  `unmodeled = kernel − pred` fits **`1.88 µs + 0.864·K`**. The K-slope is the per-tile loop
  overhead → set `c_loop≈860 ns`; the `1.88 µs` intercept is the dim-0 access penalty (below).
  Data-independent: the D-sweep gives the same slope. With `c_loop=860`, the tiled-reduction
  error drops from **−52% → −7%** at K=16. So tiling a standalone reduction is *slower*, scaling
  ~`+0.86 µs/tile`.
- **LX on/off makes no difference** for standalone tiled reductions (±0.6 µs, no consistent
  sign): the per-tile partial is tiny, so scratchpad residency saves nothing. Confirms LX ~free,
  and that the LX payoff is fused chains (not lone reductions).
- **sum/amax/amin are identical** (22.3/22.8/23.0 µs at K=8) — the combine operator
  (`add`/`maximum`/`minimum`) is free, as assumed.
- **Residual ≈ the dim-0 access penalty.** The untiled dim-0 reduction is already ~15% slow vs
  the read rate (same `sumcol`-style bias) — this is the `1.88 µs` intercept, *not* loop cost,
  and the model's remaining ~7% on tiled reductions.

---

## 6. What's built (files)

**Instrument (committed earlier):**
- `torch_spyre/execution/profiling.py` — `kernel_timer` ctx-mgr (wraps `SpyreSDSCKernelRunner.run`
  in `kernel_runner.py`); `SPYRE_PROFILE` / `SPYRE_PROFILE_SYNC`; `format_report`;
  `set_report_at_exit`; `_device_synchronize` → `_C.synchronize()`.
- `torch_spyre/execution/bench.py` — `measure_device(fn, runs, warmup)` (device-side, default,
  reads registry min, warmup discarded); `measure_latency` (host e2e, kept for user-latency);
  `device_sync`, `LatencyStats`, `net_latency_us`.
- `torch_spyre/_inductor/dump_common.py`, `dump_fx_graph.py`, `dump_loop_ir.py` — IR dumps
  (`SPYRE_DUMP_IR`), wired into `passes.py` (`CustomPostPasses` for FX; `CustomPreSchedulingPasses`
  before/after for LoopLevel).

**Cost model (this round):**
- `torch_spyre/_inductor/cost_model.py` — PURE model: `OpFeatures` (with `read_bytes()` /
  `write_bytes()`), `ArgTraffic`, `CostParams` (**turnaround: `fill_ns=0`, `bw_peak_gbps=150`,
  `rw_turnaround_ns_per_byte=0.0057`, `psum_per_elem_ns=0.14`; LX free; broadcast args counted
  ONCE at their own size**), `predict_ops`, `predict_op`, `explain`. No torch deps ⇒
  path-loadable/testable.
- `torch_spyre/_inductor/dump_cost_model.py` — `extract_features(operations)` over live IR
  (cores from `op_it_space_splits`; per-arg bytes + LX/HBM via allocation propagation). Broadcast
  flag = `n_index_vars < n_out_vars` (**includes 0-var scalars**); a broadcast operand is counted
  once at its own small device size (an unresolved scalar falls back to ~1 elem, not the output).
  Hook wired after the AFTER LoopLevel dump; `SPYRE_DUMP_COST=1` prints both.
- `examples/bench_bandwidth.py` — DRAM bandwidth probe: `BENCH_BW_OP=neg|copy|read|write|w2|w3`,
  `BENCH_BW_SUSTAIN_S=N` (saturate for `aiu-smi` sampling). Computes effective BW vs the 204.8
  peak. Companion: `notes/bandwidth_turnaround_experiment.md`.
- `examples/bench_ops.py` — device-side pointwise ladder. `BENCH_OP=gelu|relu|sigmoid|exp|mul|add`,
  `BENCH_DEPTH=N` (unary chain for LX-BW sweep), `BENCH_LX_ALL=1`
  (`config.allow_all_ops_in_lx_planning=True` → all ops LX-eligible, `allocator.py:97`),
  `BENCH_ROWS/COLS/RUNS/WARMUP`.
- `examples/bench_softmax.py` — device-side softmax bench (prints `LX_PLANNING`).
- `examples/bench_sweep.py` — size sweep (currently host-e2e; TODO align to device-side).

Local `uv` env at `.venv` (Python 3.12, torch 2.11 cpu, ruff, mypy) for lint + standalone tests.
Full `pip install -e .` not possible here (no SDK); on the run machine, rebuild C++ after a
merge with `python setup.py build_ext --inplace` (see `_C` rebuild note — missing symbols like
`ElementArrangement` mean a stale `.so`).

---

## 7. Microarchitecture facts & assumptions

Known / assumed (verify in §9):
- 32 cores (SENCORES). Each core has **2 MB LX scratchpad** (~1.6 MB usable).
- **Stick = 128 B = 64 fp16 elems**; within-stick is the last device dim; work division won't
  split inside a stick.
- Two execution units: **`pt`** (matmul/PE array), **`sfp`** (vector/SIMD — pointwise &
  reductions). Pointwise is memory-bound ⇒ unit throughput not yet modeled.
- Memory hierarchy: HBM (shared) ↔ per-core LX ↔ compute. LX-placed tensor's per-core tile must
  fit ~1.6 MB.
- **DRAM is LPDDR5; aggregate peak = 204.8 GB/s** (`_HBM_BW_GBS` in `work_division.py`; the
  compiler's matmul model uses it with a cohort penalty past 8 cores). Reachable only by
  *unidirectional* streaming — read-only nearly hits it (~176, Rung 8); balanced read+write tops
  out ~half (~97). Naming note: "HBM" in the codebase is legacy; the device DRAM is LPDDR5.
- **Bulk load:** a tile's sticks are stored contiguously, so a whole tile stages in **one bulk
  DMA read** instead of many scattered per-stick reads; memory requests are limited ⇒ contiguous
  stick layout is required for full bandwidth (`tensors_and_layouts.md`).
- Work division (`op_it_space_splits`) distributes the iteration space across cores; cores run
  in parallel ⇒ latency ≈ per-core time (balanced).
- No public microarch spec in-repo ⇒ **infer parameters from micro-benchmarks** (§8/§9).

---

## 8. The op ladder (one knob per rung) + run commands

Run on the run machine. Pair with `SPYRE_DUMP_COST=1` to print predictions.

```bash
# 1) fit fill + BW_HBM (single op, size sweep). slope→BW_HBM, intercept→fill
for n in 512 1024 2048 4096; do BENCH_OP=gelu BENCH_COLS=$n python examples/bench_ops.py; done
# 2) arithmetic-free? relu vs gelu same size (equal ⇒ memory-bound; gelu slower ⇒ add compute term)
BENCH_OP=relu python examples/bench_ops.py ; BENCH_OP=gelu python examples/bench_ops.py
# 3) traffic counting: 1-input vs 2-input (mul ~1.5× gelu: 3 passes vs 2)
BENCH_OP=mul python examples/bench_ops.py
# 4) BW_LX: unary chain depth, all intermediates in LX. slope vs N = 2·|x_tile|/BW_LX
for d in 1 2 4 8; do BENCH_LX_ALL=1 BENCH_DEPTH=$d python examples/bench_ops.py; done
#    (first VERIFY via SPYRE_DUMP_IR=1 that intermediates got `lx` and N ops survived)
# 5) shared vs per-core HBM BW (decides whether `cores` enters the model)
for c in 1 2 4 8 16 32; do SENCORES=$c BENCH_OP=gelu python examples/bench_ops.py; done
# 6) broadcast reuse: bcast (a+b[1,N]) vs add (full); + mulbcast vs mul  [CACHED/free]
# 7) stream count: gelu/mul/add3/add4 size sweep  [stream-count BW law FALSIFIED]
# 8) bandwidth: neg/copy/read/write size sweep  [read 176 / write 146 / 1R+1W 97]
# 9) tile-size: copy x SENCORES at large size   [shared-bus saturates ~4 cores]
# 10) write-fraction: copy/w2/w3                 [CONFOUNDED — w2/w3 re-read x; ignore]
# 11) reductions: sumrow/amax/mean + sumall      [read@~176 + ring combine; INITIAL]
```

The full, current ladder (rungs 1–10) is `examples/run_cost_model_plan.sh`. The `aiu-smi`
mechanism capture is a separate two-terminal run — see `bandwidth_turnaround_experiment.md`.

---

## 9. Verification checklist (earn the right to stay simple)

- [x] kernel time linear in I/O, fill ≈ 0 — section A ✓ (intercepts ~0; old ~20µs was overhead)
- [x] BW shared vs per-core — rung 5 ✓ (SHARED; flat ≥2 cores; cores not a direct term)
- [x] traffic = Σ inputs + output — ✓; the plain 2-in `mul`/`add` is **on-model** under the
  read/write split (2R1W → ~116, not anomalous)
- [x] arithmetic free for pointwise — ✓ (gelu/exp == neg on kernel time)
- [x] **broadcast** reuse — rung 6/G ✓ (**loaded ONCE at own size, NOT per core**; counted, encoded)
- [x] **LX cost** from chain-depth — rung 4 → below noise ⇒ **LX ~free** (term dropped; `add3/4`
  hint it's not *perfectly* free)
- [x] cost-model `extract_features` matches the IR — confirmed (`SPYRE_DUMP_COST` op counts/bytes)
- [x] **stream count → BW?** — rung 7 ✓ **FALSIFIED** (it's the read:write mix, not operand count)
- [x] **read vs write vs balanced BW** — section D ✓ (read ~150, write ~125–140, 1R+1W ~105 on
  KERNEL time) → **TURNAROUND model** `(R+W)/150 + α·min(R,W)`, ~7% overall
- [x] **coarse-tiling reduction (flat-K)** — Part 2 ✓ loop-aware bytes + `c_loop≈860 ns/tile`
  (K-sweep slope); LX on/off no-diff for standalone reductions; sum/amax/amin identical
- [ ] **mechanism of the read+write penalty** (turnaround / half-duplex / shared bus) — EFFECT now
  modeled (α·min(R,W)); the physical cause still wants `aiu-smi`
- [ ] **broadcast pointwise +17% faster than model** (off the V-curve) — cause OPEN
- [~] **output-dim (pointwise) tiling** — `loop_factor` generalized (an op tiling an output dim
  → all args advance; reduction → reduced input; combine → accumulator fixed). Per-arg index
  analysis (broadcast-in-tiled-pointwise) still TODO.
- [ ] **coarse-tiling LX-residency WIN** (fused chain `z=(a+b)*c`, `y` in LX) — `BENCH_OP=chain`
  + sweep Part 3 WIRED, model predicts tiled ≪ untiled (547 vs 864 µs); RUN to confirm + find
  the crossover where the LX saving beats `K·c_loop`

---

## 10. Plan / next steps

1. ~~Rungs 1–G + sweep A–F~~ DONE — **TURNAROUND model** encoded (`fill=0`, `BW_PEAK=150`,
   `α=0.0057`); arithmetic-free, shared-BW, LX-free, broadcast loaded-once (counted, not
   per-core), read/write split (read ~150 / write ~125–140 / balanced ~105) all on KERNEL time.
2. **Broadcast pointwise +17% (THE interesting open thread):** `bcast`/`bcastcol` run faster than
   the V-curve predicts — cause unknown. Design a probe (vary read fraction with/without a cached
   operand) to see whether broadcast reads dodge the turnaround penalty.
3. **`aiu-smi` capture** (copy/read/write/neg) → confirm the physical *mechanism* of the
   turnaround penalty (the cost EFFECT is already modeled). See `bandwidth_turnaround_experiment.md`.
4. **Pin the write peak & fan-in:** the model treats writes at `BW_PEAK` (write-only ~16% off);
   `add3`/`add4` ~8% (LX not perfectly free). Both want a dedicated size sweep before refining.
5. **Reductions — done for the common cases** (read-only rate falls out of the turnaround model;
   `sumrow`/`amax`/`mean`/`sumall` ~2-4%). OPEN: `sumcol`/dim-0 (reducing the outer/partitioned
   axis) ~15–19% slow — an access-pattern penalty, not the ring-combine; needs its own probe.
   Also noted: Spyre does **NOT** fuse pointwise→reduction (the pre-reduction op spills its
   `[R,C]` to HBM — an extra round-trip the extraction already captures via real buffers).
6. **Coarse tiling — the LX-residency WIN (Part-3, model done, run pending).** Flat-K reduction
   tiling is modeled (`c_loop≈860`, §5.4), but for a *standalone* reduction tiling only adds
   overhead. The payoff — and the decision a tiling cost model exists to make — is a fused
   **pointwise** chain (`z=(a+b)*c`) tiled so `y=a+b` stays in LX instead of round-tripping HBM.
   - **[DONE] `loop_factor` generalized to OUTPUT-dim (pointwise) tiling.** `_loop_features` now
     reads `loop_tiled_dims` too: an op that tiles an output dim → all its args advance
     (factor 1); a tiled reduction → only the reduced input advances; a combine/fill (tiles
     neither) → its accumulator stays fixed (factor L). So the chain's `a,b,c,z` count once and
     `y` is LX-free when tiled / HBM-counted when not — the model predicts tiled ≪ untiled
     (e.g. 547 vs 864 µs at K=4, [2048,4096]).
   - **[WIRED] `BENCH_OP=chain` + sweep Part 3.** 4 minimal runs (untiled vs tiled K=8 at two
     sizes) — RUN to validate the model **ranks tile-vs-not** and locate the crossover where the
     LX saving overtakes `K·c_loop`.
   - **[TODO] per-arg advancing** (not just per-op): a broadcast operand in a tiled pointwise op
     may not traverse the tiled dim; the principled rule checks each arg's index free-symbols
     against the tiled loop vars.
   - **[TODO] Nested tiling** (outer output + inner reduction — the two-buffer LX accumulator,
     bmm-like) and **generalize `c_loop`** (D=64 ran high — check for a fixed + size component).
7. Later: matmul (`pt` unit / compute-bound), the LX capacity cliff as a hard constraint, and an
   access-pattern term if non-contiguous layouts matter.

---

## 11. Open questions

RESOLVED: shared-vs-per-core HBM BW (rung 5 → SHARED); LX is **~free** (rung 4; ~29× HBM);
arithmetic-free (gelu/exp == neg); **broadcast operand loaded ONCE, not per core** (rung-G —
counted at its own size, not zeroed); **stream-count law FALSIFIED** (rung 7 — it's the read:write
mix); the **read/write split** (section D: read ~150, write ~125–140, balanced ~105 on kernel
time) → the **TURNAROUND model** `(R+W)/150 + α·min(R,W)`; the **`mul`/`add` "anomaly"** (it's a
2R1W kernel, on-model at ~116); and **coarse-tiling flat-K reductions** (Part 2 — loop-aware
bytes + `c_loop≈860 ns/tile`; LX on/off no-diff for standalone reductions).

Still open:
- **Coarse-tiling LX-residency WIN (THE next step)** — for a *standalone* reduction tiling only
  costs `K·c_loop`; the payoff is a fused **pointwise** chain (`y=a+b; z=y*c`) tiled so `y` stays
  in LX vs round-tripping HBM. Measure tiled-vs-untiled for that chain and confirm the model ranks
  tile-vs-not (incl. the crossover). Then: OUTPUT-dim (pointwise) `loop_factor` via per-arg
  advancing detection; nested (outer-output + inner-reduction) tiling; check `c_loop` is one
  constant (D=64 ran high). See §10.6.
- **Broadcast pointwise +17%** — `bcast`/`bcastcol` run faster than the V-curve. The cached
  operand seems to dodge the turnaround penalty; mechanism unknown. **The most interesting thread.**
- **Mechanism of the read+write penalty** — the cost EFFECT is modeled (`α·min(R,W)`), but *why*
  balanced (~105) runs below read-only (~150) is unconfirmed: DRAM turnaround / half-duplex /
  shared-bus. DECIDER: `aiu-smi` DDR bandwidth + bus-utilization. See
  `bandwidth_turnaround_experiment.md`.
- **Write peak & fan-in** — `BW_PEAK` over-predicts pure-write speed (~16%); `add3/add4` ~8%
  (LX intermediates not perfectly free). Both need a dedicated sweep.
- **`sumcol` ~19%** — reducing the outer/partitioned axis is slower than the read rate; looks
  like an access-pattern penalty, not the ring-combine.
- **Reduction model — calibrated (section F):** reductions need no special bandwidth — a tiny
  `W` makes `min(R,W)≈0`, so the turnaround model gives them the read rate (~150) automatically;
  `sumrow`/`amax`/`mean`/`sumall` land at ~2-4%, and the ring-combine is negligible at these sizes
  (`sumall ≈ sumrow`). Still open: the `sumcol` access-pattern penalty (above), and the
  `reduction_cores` (k) extraction heuristic (`out_elems < cores`) — refine only if it matters.
- LX precise BW unresolvable here (signal < noise) — revisit only with a larger-tile LX sweep.
- **Access-pattern / bulk-load:** does a strided (non-contiguous) layout drop BW vs the modeled
  contiguous case? (Exp A in the bandwidth note; gated on confirming the compiled path honors a
  custom layout.)
- **Re-anchor on the profiler kernel time (golden).** Collect `profile_ops` "Self SPYRE" kernel
  times across sizes/ops and **re-fit `fill_ns`** — the old ~20 µs was the non-deterministic
  host/Memset overhead, not kernel cost; the kernel-time fixed should be small. Also characterize
  the `Memset (Device)` (scales 12.5→28.8 µs; is it per-call work we should surface?).
- `fixed`'s ~7 µs host residue: stable across kernels / would a device-only timer change it?
