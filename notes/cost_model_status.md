# Spyre Cost Model — Status / Handoff (2026-07-07)

Living status + detail doc — **read this first on resume**. The full model write-up is
[cost_model_report.md](cost_model_report.md) (the MAIN doc — the long-form derivation of every
term, with figures and accuracy). This file holds the details that don't belong there:
implementation state, open work, methodology, tooling, next steps. Keep the two in sync — if a
number changes here, update the report.

Goal: a HIGH-LEVEL **relative** cost model over the after-pre-scheduling LoopLevel IR, to
guide optimization (LX placement, coarse-tiling). Bar: correct ranking + ±15-20%. No local HW —
edit locally, hand commands to a run machine, logs pasted back.

## Golden measurement

AIU profiler: `torch.profiler` PrivateUse1, "Self SPYRE" per-kernel device time (harness
`profile_ops.py`). New image leaves the kernel name BLANK → classify **by exclusion** (device
time not Memset/Memcpy). Do NOT use `SPYRE_PROFILE`/`SPYRE_PROFILE_SYNC` (host wall-clocks).

## Model + params (implemented in torch_spyre/_inductor/cost_model.py)

Form: `T = compute + HBM/eff − γ·min(compute, HBM/eff)` (see presentation §1 for term table).

| param | value | status |
|---|---|---|
| `bw_peak_gbps` / `rw_turnaround_ns_per_byte` | 150 / 0.00574 | ✅ pointwise/reduction ±5% |
| `mm_bw_read_gbps` / `mm_bw_write_gbps` | 150 / 150 | ✅ single-rate (see DONE note §Matmul) |
| `mac_peak_per_core_ns` | **1140** (was 1536) | ✅ compute-dominant low-core fit |
| `overlap_gamma` | **0.46** (NEW) | ✅ jointly fit w/ peak, RMS 1.7% |
| `mm_spill_t0 / slope / cap` | **448 / 1.10 / 1.70** (NEW) | ✅ decouple+reread sweeps |
| `bw_restickify / stick_scatter / reduce_outer` | **116 / shape-dep / 113** | cat0 = 252−4·log2R−12.3·log2C (shape sweep, R²0.93) |
| reduction read rate | **min(150, 114+61·e^(−ROWS/3700))** | ✅ reduction-rows sweep (op-independent, 2.6% RMS) |
| `bw_broadcast_gbps` | **118** (2026-07-09) | ✅ copy/bcast/bcastcol/mulbcast; pointwise RMS 5.1→2.0% |
| `write_reread_coef/r_exp/c_exp` | **2.15e-7 / 1.60 / 2.20** (2026-07-10) | ⚠️ EMPIRICAL outer-product term; broadcast 19→7.7% (black-box) |
| `pointwise_arity_derate` | **0.075** (NEW) | ✅ add3/add4 |
| matmul `pt_eff` (`r_full=64`, `exp 0.35`) | matmul only | ✅ unchanged |
| `coarse_underfill` (`r_full 13`, `exp 0.68`, `cap 0.95`) | coarse/softmax only (2026-07-09) | ✅ HW: softmax non-spill RMS 7.2% (spill regime deferred) |
| `psum_per_elem_ns` | **0.14, GATED off matmul** (2026-07-08) | ✅ bug fixed (was +489% on forced `WD_K>1`) |

All matmul + per-op changes are **implemented in cost_model.py + dump_cost_model.py**.
Coarse-tiling terms are NOT reworked yet (open).

### ⚠️ Version hygiene — the recorded `pred_us` is NOT current (2026-07-08)

An adversarial review + direct checks found the dataset **mixes model generations**: `pred_us`
is baked into each log at run time, `sweep_records.csv` spans 14 log-families (June→July), and
26 configs have divergent `pred_us` for identical shapes. The old `db_sweep.log` (the claimed
"validated re-run") **predates the extractor false-positive fix** — `sumall` still shows +37/+40%
and `transpose_outer` +66/+49% in those records — and now also predates the psum gate. **Its raw
file is gone** (records survive only in the CSV). So the earlier "HW-validated ±5% / matmul 8%"
claims are **not backed by current-model records**. Only the measured `kernel_us` is version-
independent and trustworthy.

Parser stamps provenance: `model_sha` (from each sweep's `git:` header), `log_date`, and
`is_current` (= `model_sha == --current-sha`); `--drop-ops chain` retires `chain`.
**RESOLVED 2026-07-09:** a fresh `run_db_sweep.sh` on the current code (sha `078922c`) landed →
**185 `is_current` rows** are now the single clean current-model dataset (filter on `is_current`).
The extractor false-positive fixes are **now confirmed on HW in-record**: `sumall` +2.8/+5.6%
(was +37/+40%), `transpose_outer` −4.6/−14.9% (was +66/+49%), `sumcol` +7.4/−0.7%.

## Findings by category (all HW-validated unless noted)

**Pointwise / reduction / broadcast / transport** — ±5% on anchors, but the adversarial review
(2026-07-08) found several claims thinner than presented. Per-op effective-BW overrides
(extractor `_hbm_pattern` from the IR index/layout): `transpose`→`restickify` 116, `cat0`→
`stick_scatter` (size-dependent, see below), `sumcol`→`reduce_outer` 113; `add3/add4` get the
arity derate. The
false-positive fixes (`reduce_outer` requires a **kept stick dim**, excl. `sumall`;
`stick_scatter` requires a **concat dim**, excl. `transpose_outer`) are **in code but NOT in the
current records** (db_sweep predates them → still +37/+66% there); re-run needed to confirm.
Downgrades: `BW_peak=150`/`α` are a soft decomposition — only the combination is identifiable
from pointwise, and the 2:1 `add`/`mul` ratio prefers `BW_peak≈138–147` (150 imported from
read-only reductions). **`cat0` FIXED (2026-07-10, transport-shape sweep):** effBW is
SHAPE-dependent (falls with row width C, weakly R) — `252 − 4·log2R − 12.3·log2C` clamped
[45,150], R²0.93 over 10 shapes (the earlier io-based exp was WRONG: same bytes give 53–85 GB/s
by aspect). `transpose_outer` shows the SAME C-falloff (−22% at wide C) but carries no IR
pattern tag, so it stays on the default copy model, flagged — tagging it (extractor change)
would let it reuse the cat0 shape model. **Reduction large-ROWS residual FIXED (2026-07-10,
reduction-rows sweep):** the read rate falls op-independently with ROWS (149→115 GB/s over
ROWS 2048→16384, flat across COLS) — `min(150, 114+61·exp(−ROWS/3700))` on standalone
row-reductions (NOT fused softmax: gated on len(ops)==1 so input-dedup isn't broken; NOT sumcol:
reduce_outer). Reduction category 7.9%→**2.6%**. `sumcol reduce_outer=113` (rows never varied)
is still HYPOTHESIS. Arity `0.075` fit from 2 arities; per-op derate is 0.06→0.094
(superlinear); decisive test = add3/add4 with LX on. **`copy` FIXED (2026-07-09):** it is not
1R1W — `x+1.0` lowers to an `add` with a resident broadcast operand, so it is a broadcast op
(`copy`/`bcast`/`bcastcol`/`mulbcast` run ~118 GB/s vs `neg` ~105). New `bw_broadcast_gbps=118`
(applied to ops with a broadcast operand + a full input, via `_is_broadcast_op`; NOT `write`)
→ pointwise RMS 5.1→2.0%, clean broadcast points ±3%. Mechanism empirical. **`write`**
(b[1,C]+c[R,1], both broadcast → outer-product) is slow + super-linear in COLS: operands are
tiny (b=C elems, c stick-inflated R×64), so NOT an operand spill — the cost is in the
outer-product output write. No clean mechanism; modeled by an **empirical** `write_reread`
term (coef·ROWS^1.6·COLS^2.2, 12% RMS, black-box) → broadcast category 19→7.7%.
`cat0`/`cat1` are 2:1 write-heavy, not the "R=W byte-copy" the doc states.

**Matmul (k=1) — ~8% on the calibration envelope, NOT "done".** `compute(peak=1140) + two-rate
HBM + tile-spill`, with compute/HBM `overlap γ=0.46`. Isolation order: HBM (compute-free) →
compute+overlap (low cores) → spill → (K-split now gated off). The 8% holds on pow2/balanced/
mid-size shapes (~half the tested points); **honest full-range k=1 RMS ≈18% (−48…+68%), mean
bias −4%**. **The "fanout penalty" hypothesis was FALSIFIED** (re-read sweep FB/FA: fanout 1→32
at fixed small tile → ~0 effect); the split penalty is per-core **TILE SPILL** and the residual
grows with *large* tiles (so it is NOT misattributed underfill — right sign). Residuals (⚠):
tiny matmuls +54% (fixed-overhead floor), extreme *forced* splits, non-pow2-N (model even
*inverts* the 6144-vs-8192 ranking).

**DONE (2026-07-10): dropped the two-rate HBM, now a SINGLE rate = 150 (user approved).** On the
planner-realistic envelope (k=1, fanout ≤8, non-tiny, pow2-N; n=34) the old two-rate 143/156
scored RMS 7.10%; single `mm_bw_read=mm_bw_write=150` (γ unchanged at 0.46) scores **6.9%** —
equal-or-better AND physically respects the 150 copy peak. The compute-free dominant-operand
rates are ~118–148 (write corners) / ~123–136 (read corners): overlapping, both <150, no distinct
write rate. `BW_w=156` was a fit artifact absorbing overlap. **Shipped**: `mm_bw_read_gbps =
mm_bw_write_gbps = 150.0` in cost_model.py (comments/docstrings updated). Report §8 rewritten as
"adopted". Also captured:
honest matmul regime table — realistic 7.1%, forced K-split −40% (unmodeled combine, planner
avoids), skewed >8 −23%, tiny +2.5 (floor), non-pow2 −16%. γ pinned by the balanced aggregate,
NOT the small-shape GH sweep (GH alone prefers γ=0; floor-contaminated). See report §7–§12.

Adversarial review (2026-07-08) downgraded several matmul terms to HYPOTHESIS pending the re-run
+ decouplers: (a) `BW_w=156 > 150` one-directional peak is physically impossible → the "compute-
free" M1 fit absorbed compute-overlap; re-fit BW_r/BW_w on the FULL model (subtract γ), not raw
bytes/time. *(Now resolved — see the RESOLVED note above.)* (b) `peak=1140 / γ=0.46` sit on a non-identifiable ridge — (1190,0.40)/(1220,0.30)
fit equal or better OOS; γ is pinned by ~one shape → need an HBM-dominant cores-scan to pin γ
alone. (c) overlap `min()` FORM untested (52/79 points cluster at compute≈HBM where all forms
coincide). (d) spill log-curve fit from ~5 pts, cap from ~2, `RB` corrupted by non-pow2 N.

**Coarse-tiling — OPEN (active).** See below.

## Coarse-tiling — softmax now largely isolated (2026-07-08)

Reframe (agreed with user): a coarse-tiled op is ONE **fused kernel** (intermediates in LX),
NOT a sum of per-op kernels. Define `rpc = ROWS/(cores·T)` = per-core rows per tile.

The `softmax_terms` grid (ROWS×T at COLS=2048) + the `coarse_terms` softmax runs (ROWS=16384 at
**both COLS=2048 and 4096**) together isolate the softmax cost, and an adversarial challenge was
run + addressed. Results (units: `us/1k-row/1k-col` ≈ per-byte cost):

+ **SETTLED — the driver is `rpc`, NOT the tile count `L`.** At `rpc=16` the four points span
  `T=4..32` (4× tile count) at ~flat cost → kills any `L`/pipeline-overlap story. Normalized
  cost/row collapses onto `rpc` across 4 ROWS values.
+ **SETTLED — underfill is ROWS-driven, NOT per-core-tile BYTES.** The old confound (COLS fixed →
  rows≡bytes) is broken by the cross-COLS data: at **matched `rpc`, doubling COLS (2× tile bytes)
  leaves per-byte cost unchanged (ratio 0.96–1.02)** across the whole non-spill range. So the
  underfill derate keys on `rpc` (rows), independent of COLS.
+ **cost(`rpc`) is U-shaped** (per-byte): min ~40 at `rpc≈32`; steep underfill rise below
  (`rpc8`≈49, `rpc4`≈78, `rpc2`≈126 — i.e. 1.2×/1.9×/3.1× the floor); MILD rise above
  (`rpc64`≈43, `rpc128`≈46), COLS-independent (so also rows-driven, not LX pressure). The current
  derate `min(1,(rpc/16)**0.35)` is mis-shaped: under-derates `rpc≤4`, ignores the `rpc>32` rise.
+ **HYPOTHESIS (leaning likely) — double-counted `arg0` read.** At the floor softmax runs at
  ~100 GB/s single-read-equiv (≈ the balanced copy rate), matching arg0-read-ONCE on both COLS;
  arg0-read-TWICE implies ~150 GB/s (at/above peak). So the fused kernel likely reads arg0 once
  (2nd read LX-served) and the model's 2× read over-counts the floor ~25%. **Confound (from the
  adversarial agent): not separated from a compute-bound (exp) floor or a BW_peak error** — the
  deciding test is a pure-copy of identical footprint vs the softmax floor.
+ **SETTLED mechanism / under-sampled shape — LX-spill is BYTE-driven, separate from underfill.**
  At `rpc=256`, C4096 (2.1 MB/core tile) SPILLED (per-byte 145, `io_hbm_bytes` itself jumped as
  intermediates went to HBM) while C2048 (1.05 MB/core) did NOT. So spill triggers on per-core
  tile MB (~knee 1–2 MB/core), independent of the rows-driven underfill. Only 1 point past the
  knee → exact threshold + post-knee slope need a finer sweep.
+ **Noise:** VAR (5× within-process) = 0.3%; cross-config agreement at matched rpc ≈ ±2–4%. The
  `rpc>32` "mild rise" (+7…+15%) is real vs that, but **cross-process/thermal variance is still
  unbounded** — bound it before trusting single-digit-% effects.

**IMPLEMENTED + HW-VALIDATED 2026-07-09** (cost_model.py; confirmed in the sha-`078922c` re-run):
+ (a) **Fused-kernel HBM counts each distinct external input ONCE** — `_fused_hbm_bytes(ops)`
  dedups `arg`-named HBM inputs across the bundle (softmax `arg0` read by `amax`+`sub` → once).
  Fixes the ~25% floor over-count. Non-softmax ops unaffected (no `arg` reused across ops).
+ (b) **Re-fit `rpc` underfill, decoupled from matmul** — new `coarse_underfill_eff` +
  `coarse_underfill_{rfull=13,exp=0.68,cap=0.95}`; matmul `pt_eff` untouched. **HW: softmax
  non-spill RMS 7.2%** (n=44, was ~20%), floor (rpc16–32) ±0.6%; residual rpc≤8 (+8–10%) and
  rpc≥64 (−7…−14%, the mild rise the cap omits) — matches the synthetic fit exactly.
+ (c) **NO categorical spill term** — the plan said add one, but the IR check showed the
  extractor **already counts spilled bytes**: when a per-core tile overflows LX (~1–2 MB/core)
  the compiler moves intermediates to HBM and the IR reflects it (LX total collapses,
  `io_hbm_bytes` jumps). A predicted-knee term would double-count. The real residual is that
  spilled traffic runs slower than modeled — the HW re-run now shows the spill regime
  (rpc≥160) at RMS 24% (−18…−40%, 7 pts) vs 7.2% non-spill — a RATE effect, DEFERRED until
  the finer knee sweep gives >1 point to fit the spilled-traffic rate.

Remaining softmax decouplers: finer spill knee at C4096 `rpc∈{160,192,208,224,240,256}` (fit the
spilled-traffic rate); cross-process repeats (bound noise); cross-COLS at a 2nd ROWS. `chain`
DROPPED (per user). `matmul_row_tiling` deferred (needs `pt_eff` keyed on coarse-tile `M/tiles`).

### Decoupler sweeps — WRITTEN + design-review-vetted 2026-07-09 (added to `run_db_sweep.sh`)

Four new sweeps to upgrade the HYPOTHESIS terms; each design was adversarially challenged and
the flaws fixed BEFORE writing (memory: conservative-claims-adversarial-check). Not yet run.
+ `run_pointwise_ratio_sweep.sh` — BW_peak vs α. Vetting reframed it as an explicit **read/write
  asymmetry test**: fit `R/BW_read + W/BW_write + α·min(R,W)` and CHECK BW_read==BW_write (a
  symmetric 2-param fit can never surface the misspecification the 105/138–147/150 tension hints
  at). Adds a streaming `read` probe next to the (circular) reduction anchor; sweeps ROWS for the
  plateau; `write` at small COLS is a flagged low-confidence write anchor.
+ `run_matmul_gamma_sweep.sh` — peak/γ + BW_r/BW_w. Vetting confirmed the compute-dom cores-scan
  recovers peak via a **γ-independent slope** (escapes the ridge); FIXED the γ scan to a
  **spill-free small shape** (M=N=512/768, K=64, per-core tile <448) so spill can't drift into
  the γ slope. BW section is a **rank-2 (R,W) grid** with min(R,W) on both sides (the naive
  fixed-M K-sweep was BROKEN: W constant → BW_w unidentifiable, BW_r/α collinear).
+ `run_nonpow2_n_sweep.sh` — the stick-padding sawtooth is in the **per-core tile N/n**, not full
  N (the naive N∈{2048..8192} step 1024 was BROKEN: all stick-aligned → sawtooth invisible; 8192
  broke the MNK cap). FIXED: forced 4×8×1, N stepped 64 so N/8 sweeps 512→576 across a stick edge.
+ `run_softmax_floor_sweep.sh` — double-count vs exp-compute. Vetting rejected the untiled-copy
  control (tiling-overhead confound); added a NEW matched harness op **`softmax_noexp_row_tiling`**
  (softmax structure, `exp`→`mul`) so `T(softmax) − T(noexp)` at matched [ROWS,COLS,TILES]
  isolates exp by **wall-clock time** (not effBW, which presupposes the byte-count answer).
+ `run_broadcast_sweep.sh` (2026-07-09) — pins the **broadcast effBW** (`bw_broadcast=118`,
  fit on one clean point/op) over COLS at ROWS=2048, AND confirms the **`write` spill**: a
  write ROWS×COLS grid separates C-driven (row operand `b[1,C]` spills → super-linear in C)
  from R-driven (`c[R,1]`). Report §4. `copy` is a broadcast op (increment `x+1.0`).

## Methodology (do NOT repeat past mistakes)

1. Never `measured − model_term` to isolate another term (circular).
2. Isolate each term in a regime where it DOMINATES; subtract only ALREADY-validated terms.
3. **Be conservative on every claim; before pushing a mechanism/parameter, LAUNCH adversarial
   agent(s) to challenge it** (confounds, alternatives, missing controls) and address every
   challenge, or downgrade to "hypothesis + the deciding experiment." (memory: conservative-
   claims-adversarial-check.) This caught real over-claims here (chain underfill, softmax R_eff).
4. Trust measured data, not the in-tree work_division.py model (it's a relative ranker).

## Tooling

+ **Harness** `docs/source/user_guide/examples/profile_ops.py` (BENCH_OP=…; knobs BENCH_ROWS/
  COLS/N, BENCH_TILES, WD_M/N/K, SENCORES, LX_PLANNING).
+ **DB rebuild** `run_db_sweep.sh` — chains all sweeps into ONE `haoyang_logs/db_sweep.log`
  (children write there via `DB_LOG`; per-run `timeout` guard) + auto-parses. New sweeps:
  `run_hbm_ops_sweep.sh`, `run_matmul_compute_sweep.sh`, `run_matmul_psum_sweep.sh`,
  `run_reread_sweep.sh` (RA/RB tile-spill, FB/FA fanout-isolation — falsified fanout),
  `run_decouple_sweep.sh`, `run_split_sweep.sh`, `run_coarse_tiling_sweep.sh`,
  `run_coarse_terms_sweep.sh`, `run_softmax_terms_sweep.sh` (active).
+ **Parser** `notes/parse_sweep_logs.py` → `notes/sweep_records.{json,csv}` (merge by
  `log:lineno`, idempotent). Carries per-op split/model-term breakdown + **provenance**:
  `model_sha` (from `git:` header), `log_date`, `is_current`. Flags: `--drop-ops chain`,
  `--current-sha <sha>` (default: newest parsed log's sha). Also captures `feats` (the
  serialized `OpFeatures`) from each run's `MODEL FEATS` line.
+ **Offline scorer** `notes/eval_model.py` — **recompute accuracy WITHOUT hardware** (the
  measured `kernel_us` is version-independent; only the prediction changes). The harness now
  dumps `MODEL FEATS <json>` (the model's exact input) per run for free, so a new model version
  is scored by `predict_ops(feats)` in pure Python (`cost_model.py` has no torch dep → runs
  locally). `--params k=v,...` re-scores with overridden params instantly; `--verify` checks
  feature fidelity; `--update` writes recomputed `pred_us` back. Rows lacking `feats` (the
  pre-2026-07-09 grand sweep) are reconstructed from the stored `io` block and **self-validated
  against their stored `pred_us`** (mismatches excluded; matmul needs a `feats` re-run — 119/185
  reconstruct today). THE model-iteration loop: edit params/form → `eval_model.py` → new
  accuracy, no Spyre. (`cost_model.op_to_dict`/`op_from_dict`/`ops_to_json` do the (de)serialize.)
+ **Extractor** `dump_cost_model.py`: `_matmul_features` (MACs, M/m, N/n, |A|, |B|, k),
  `_hbm_pattern` (restickify / stick_scatter / reduce_outer from IR index+layout).

## Immediate next steps

1. **Re-run `run_db_sweep.sh` on current code** (psum gate + extractor fixes) → the ONE clean
   current-model dataset. Master runner auto-stamps this sha as `is_current` and drops chain.
   Everything below depends on having current-model `pred_us`.
2. **softmax_terms sweep** (running) → VAR (is the ~19% swing real?) → GR (L vs rows/tile) →
   SP (spill knee). Then adversarial-challenge the conclusion BEFORE modeling.
3. **Decoupler sweeps** the adversarial review proved necessary (fold into the re-run) — for the
   report's "hypothesis → isolation" narrative: pointwise write-only + read-only probes (break
   the `BW_peak`/`α` degeneracy); add3/add4 with LX on (arity mechanism); `cat0` size/aspect +
   `sumcol` reduced-dim (rows) sweeps; matmul HBM-dominant cores-scan (pin γ alone) + BW_r/BW_w
   re-fit on the full model; non-pow2-N handling.
4. If a real, isolable coarse driver: fused-kernel HBM (count reused inputs once) + the
   L-or-rows/tile term + the categorical LX-spill. Re-verify via db_sweep.
5. `matmul_row_tiling` pt_eff keyed on coarse-tile M; recheck.
