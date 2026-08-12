# Report-writing progress tracker (autonomous loop state)

**Goal:** finish `notes/cost_model_report.md` §6–§12 as full external-reader prose
(no internal jargon), each with figure(s) and an end-of-section data table with
per-point error, matching the observation-first style of §1–§3. Every claim must
survive at least one adversarial agent challenge before it goes in.

**Hard constraints (standing user rules):**

- External audience: define every term; no internal words ("grand sweep", script
  names as jargon, "spill" without definition, dev metrics like "RMS 5→2%").
- Be VERY conservative. Launch an adversarial agent to challenge any mechanism/
  parameter claim; address every challenge or downgrade to hypothesis + the
  deciding experiment.
- No unmodeled residual left silently — flag it explicitly if not modeled.
- Do NOT git commit (user manages git).
- No local hardware: cannot run sweeps. Write handoff scripts; write report from
  EXISTING data in `notes/sweep_records.json` (555 records).

**Section order (dependency chain — do NOT skip):**
§6 transport → §7 matmul setup → §8 HBM → §9 split/tile-spill → §10 compute →
§11 overlap → §12 shape residuals. Part IV (coarse tiling) deferred to next report.

## Status log (append newest at bottom)

- 2026-07-10: §1–§5 written & lint-clean. §6–§12 exist as skeleton bullets only.
  Starting: (a) self-trigger cron, (b) reduction ROWS gap sweep script,
  (c) §1–§5 impl-sync verification, (d) §6 full write.
- 2026-07-10: DONE — cron set (every 5 min, session-only); §1–§5 verified synced
  to cost_model.py (0 mismatches; fixed 2.15e-7→2.148e-7); reduction ROWS gap
  sweep written (run_reduction_rows_sweep.sh). §6 Transport WRITTEN + reviewed by
  adversarial agent BEFORE commit: verified transport→`clone` in IR; transpose=116
  flat (settled number, mechanism open); cat1 default (~7% optimistic); cat0 +
  transpose_outer BOTH flagged size-dependent (equal footing — adversarial caught
  the prior inconsistency). fig6_transport.png added. Handoff sweep for the two
  size-dependent copies: run_transport_shape_sweep.sh. Table = 13 pts, RMS 5.6%.
  NEXT: §7–§12 matmul (shared dataset). User: finish ALL 12, no commits.
- 2026-07-10: DONE — §7–§12 matmul WRITTEN with a strong pre-draft adversarial
  challenge; committed ONLY claims that survived. Key honesty corrections baked in:
  (§8) two-rate BW_w=156 is a fit artifact — verified single-rate 150+γ fits the
  realistic regime BETTER (6.87 vs 7.10%); §8 recommends single-rate. (§9) "fanout
  falsified" downgraded (sweeps confounded, no feats) + small-tile over-prediction
  flagged as co-resident residual. (§10) peak clean. (§11) γ NOT pinned by GH sweep
  (GH prefers γ=0, floor-contaminated) — pinned by balanced aggregate, correlated w/
  memory rate. (§12) regime table: realistic 7.1%, K-split −40%, skewed −23%, tiny,
  non-pow2. Figs fig8–fig11 added. External-reader review pass done: added stick/
  systolic/planner/spill/under-fill glosses, moved script name to appendix, added
  §4+§5+§6 error tables. Status doc updated w/ RESOLVED single-rate finding.
  ALL 12 SECTIONS COMPLETE. Lint clean.

## Data-gap handoff (needs run machine; user resting)

- Reduction large-ROWS residual is backed ONLY by `read` + `sumrow` at ROWS=8192.
  Missing: amax/mean/sumcol/sumall at ROWS=8192, and ALL ops at ROWS=4096/16384.
  New script: `run_reduction_rows_sweep.sh` (written, awaiting run).
- Transport `cat0` + `transpose_outer` are size-dependent copies fit on 2–3 square
  shapes only. New script: `run_transport_shape_sweep.sh` (size × aspect, awaiting run).

## STATUS: ALL 12 SECTIONS DRAFTED (2026-07-10)

§1–§12 all have full external-reader prose + figures + end-of-section error tables.
Added §4/§5/§6 tables too (were missing). Lint clean (ruff + PyMarkdown + mypy pass).
Remaining polish for future ticks (NOT re-drafting):
- If reduction/transport gap sweeps get run, fold data in + tighten §5/§6.
- Consider whether to actually flip the model to single-rate 150 (report §8
  recommends it; needs user OK — do NOT change shipped params unprompted).
- cost_model_presentation.md was deleted (redundant with the full report); the report
  is now the single model write-up.
- A second external-reader pass once the user is back for final wording tweaks.
If nothing above is actionable, the loop should report "done" and stop.

## Checklist per section (repeat for each)

1. Pull that op-category's rows from sweep_records.json; confirm what data exists.
2. Draft prose observation→question→hypothesis→experiment→model→validation.
3. Generate/refresh the figure(s) via `notes/plot_report.py`.
4. Adversarial agent challenge on the key claim; fix or downgrade.
5. End-of-section table: every data point, pred vs meas, err%. Use eval_model.py.
6. Lint: `python3 -m pymarkdown scan notes/cost_model_report.md` + ruff on py.
7. Update this file's status log.

- 2026-07-10 (loop tick): Verified §1–§12 complete, no skeleton bullets, all 11
  figures present, PyMarkdown clean. Fixed a dangling "see the appendix" reference
  (§6) by listing the two queued handoff sweeps in the appendix. Nothing else is
  autonomously actionable (remaining items need the run machine, a model-flip OK, or
  the user present). REPORT DONE — stopping the recurring 5-min loop to avoid idle
  token spend. Restart with /loop or a new cron when there is fresh work.

- 2026-07-10 (user OK): FLIPPED matmul model to single-rate 150 (mm_bw_read=
  mm_bw_write=150.0 in cost_model.py; γ unchanged 0.46). Regime-A RMS 7.1→6.9%,
  physical (≤ peak). Propagated: cost_model.py comments/docstrings; report §8
  ("adopted"), §12 regime + representative tables (recomputed preds), fig11
  regenerated; status doc (DONE note + param table 150/150); presentation doc
  (HBM form, constants, isolation step 1, accuracy row 6.9%, representative pts,
  conclusion — also fixed a prior misattribution: the −40% tail is forced K-split,
  NOT non-pow2 N). Non-matmul categories unaffected. All lint clean. Gap-sweep
  logs (reduction ROWS, transport shape) still pending from the run machine.

- 2026-07-10 (user edits, §6 rewrite): §6 trimmed to model-focused prose per user
  (cut long observation, the "copy is a band" bullet, the stick definition, and the
  "why transpose faster" paragraph). Grouped transpose+cat1 as the flat/stable ops.
  BUILT a real size-dependent cat0 model (user ask): stick_scatter is no longer a flat
  60 — now bw = 59 + 50.2·exp(−io_bytes/10.34e6) in cost_model.py (3-pt fit 96/63/59
  GB/s at 3/25/101 MB). cat0 errs now +1.5/+4.3% (was +7/−38 incl. small pt); transport
  RMS 5.6→5.3%. transpose_outer left flagged (2 pts) pending the shape sweep. Synced
  report §6 table+prose, presentation (per-op BW line + transport table), status doc.
  All lint clean. Non-transport categories unaffected.

- 2026-07-10 (new sweeps folded in): parsed reduction_rows + transport_shape logs
  (+63 rows → 618). TWO NEW MODELS: (1) Reduction read rate FALLS with ROWS,
  op-independent: min(150,114+61·exp(−ROWS/3700)) on standalone row-reductions
  (gated len(ops)==1 so fused softmax input-dedup not broken; sumcol=reduce_outer).
  Reduction 7.9%→2.6%. (2) cat0 effBW is SHAPE-dependent (my earlier io-exp model
  was WRONG — same bytes give 53–85 GB/s by aspect): 252−4·log2R−12.3·log2C clamped
  [45,150], R²0.93. transpose_outer has the same C-falloff (−22%) but no IR tag →
  left on default, flagged. §5 rewritten as MODELED; §6 updated (cat0 C-model);
  fig5 + fig8 rebuilt (fig8 now measured-vs-predicted labeled, per user). Also user
  §7/§8 edits: added assumption T=f(compute,memory); §8 concise/model-first (dropped
  old two-rate narrative); §7 trimmed. Synced status + presentation. All lint clean.
  Eval: reduction 2.6, transport 8.5, pointwise 2.2, softmax 10.7 (unregressed), matmul 21.7.

- 2026-07-10 (§9–§11 reorder + fig10 coverage): per user, reordered Part III so the
  compute term is determined BEFORE the tile-spill residual. New order: §8 memory →
  §9 COMPUTE (form MACs/cores/peak + slope fit) → §10 OVERLAP (γ) → §11 SPILL residual
  (now correctly framed as leftover after the full base model). Fixed all cross-refs
  (§7/§8), renamed figs to match (fig9_peak, fig10_overlap, fig11_spill), removed old
  pngs. fig9 (peak) now shows TWO problem sizes (K=2048, K=4096) each swept cores 4→32
  — slope ratio 2:1 matches the MAC ratio, richer coverage. Lint clean.

- 2026-07-10 (figs 9/10 — config marking + coverage + honesty): fig9 (compute) now
  labels every point's m×n split and shows the KEY invariance: at 32 cores the balanced
  splits 4×8/8×4/2×16 collapse to ~385µs (time tracks cores, not the factoring), while
  K-splits (k>1) sit visibly ABOVE the line (labeled m×n×k → §11/§12). fig10 (overlap)
  rebuilt as error% vs MEMORY-FRACTION over ALL 24 balanced configs (square/thin/fat-K,
  cores 4-32, aspect = marker shape): γ=0 over-predicts more as memory grows (→+40%),
  γ=0.46 flattens to ≈0 across shapes — proving γ is not a per-shape fudge. Per user,
  the honest residual is STATED: at memory-fraction>0.5 (thin, memory-dominated) γ=0.46
  scatters ±15-30% → flagged to §8/§12. §9/§10 prose updated to match. Lint clean.

- 2026-07-10 (batch dim + spill-form questions): Investigated end-to-end. FINDINGS:
  (1) Batch: compute (MACs=out_elems·K, out_elems includes B) and memory (device byte
  counts include B) scale with batch AUTOMATICALLY; but the spill term (_matmul_features)
  assumes a 2-D M×N output (picks 2 extreme dims), so batch isn't carried into M/m,N/n or
  operand re-read. Zero batched-matmul data (all arg logical dims = 2). Added report §13
  "Scope: the batch dimension". (2) Spill form: user's insight is right — capacity bounds
  the 2-D tile (area M/m·N/n), not each edge at an independent 448 knee. Data proves it's
  NOT pure area either: at equal area 262144, 1024×256=+136µs vs 512×512=+90µs (shape
  matters). Two decouple sweeps are only 1-D slices → can't fit the 2-D form. Added §11
  3rd flag (separable form is an approximation) + wrote run_matmul_tile_grid_sweep.sh
  (M/m×N/n grid, 19 configs, mmwd, no harness change). Did NOT change the model (need the
  grid data first). All lint clean. Batch validation needs a bmm harness op (follow-up).

- 2026-07-10 (§12 overhaul + full matmul table): Per user: REMOVED §12a non-pow2 N
  (the sweep crashed on exactly the non-stick-aligned N/n cases → hardware doesn't
  allow them; out of scope) and §12b forced-K-split (planner never K-splits) and §13
  batch (no data). REWROTE §12: now §12a = extreme splits, §12b = tiny floor. TRIED to
  model extreme splits (user ask): raising spill cap saturates at −16% mean and can't
  fix it — the real mechanism is an ASYMMETRIC operand re-read (huge N/n breaks it,
  huge M/m is fine), which the symmetric saturating log-spill structurally can't
  express. Did NOT hack a term (conservative); extended run_matmul_tile_grid_sweep.sh
  with a TG2 split sweep (balanced→extreme) to get the data to fit it. Added the FULL
  71-row matmul data table (all points, regime-tagged) after the representative one.
  All lint clean.

- 2026-07-10 (Part IV coarse tiling — started): Removed K-split everywhere (fig9,
  §9/§12 prose, full table → 62 rows). Began Part IV per user framing: the model =
  counting HBM vs LX tensors. Confirmed with data: softmax_row_tiling tiles=1 keeps
  sub/exp intermediates in HBM (7 passes, 9927us) → tiles≥8 moves them to LX (2 passes,
  ~2650us); model tracks it by byte count (floor +2%). softmax = memory-only (no matmul
  op → no compute term = pointwise); matmul_row_tiling = matmul (compute+overlap on the
  fused byte count). Drafted §13 (HBM/LX byte count + softmax tiles table), §14
  (softmax=pointwise / matmul=matmul), §15 (open residuals: the −40% LX-spill boundary
  at tiles=2 where IR tag says LX but hardware spills; + rows/core underfill drift,
  matmul_row ~20% uncalibrated). Lint clean. NEXT: develop the LX-capacity test + the
  underfill; coarse data = softmax 54@11%, matmul_row 9@20%, ctsum @6%.

- 2026-07-10 (Part IV §14/§15 developed + matmul_row analysis): Per user, dropped the
  trivial "softmax=pointwise" §14 (folded to one clause in §13). §14 = LX-SPILL model
  (obs→hyp→exp): error tracks per-core WORKING SET not tile count; collapses across 3
  shapes at LX capacity ≈1 MB/core; offline fix (force over-capacity intermediates to
  HBM) moves the −40% tiles=2 point to −6%. §15 = per-tile UNDERFILL (obs→hyp→exp):
  matmul_row_tiling measured GROWS with tile count at fixed shape (341→440→652 at t2/4/8)
  because each per-tile matmul underfills the systolic array — implied eff ≈0.9/0.5 at
  64/32 rows/core, far steeper than the standalone-matmul underfill (still ~1.0 at 64).
  ROOT CAUSE of the ~20% matmul_row error identified: needs its OWN steeper underfill
  curve (tiled per-tile matmul pays a fill/drain a standalone doesn't). Thin data (1
  shape × 3 tiles) → dedicated tile-count sweep queued. Lint clean.

- 2026-07-10 (Part IV eff model + matmul underfill impl): Per user, added the CALIBRATED
  eff model to report §15: eff=min(0.95,(rpc/13)^0.68), rpc=ROWS/(cores·tiles), RMS 5.9%
  on the LX-fitting softmax regime (mean -1.2, n=45). figs: fig12_coarse_spill (err vs
  working set, spill past ~1MB/core), fig13_coarse_eff (effBW vs rpc + model). IMPLEMENTED
  tiled-matmul underfill: eff=min(1,(rpc/72)^0.85) for tiles_output_dim matmuls only →
  matmul_row 20%→14%, Part III matmul unchanged. Spill NOT yet coded (verified offline:
  force over-cap intermediates to HBM → −40%→−6%); it's the next impl. Overall 12.1%.
  Lint clean.

- 2026-07-10 (broadcast small-COLS): User caught that the fig4 COLS≤1024 "noise" is a
  REAL systematic rise (bcast/bcastcol/mulbcast → ~130/124/132 at COLS=1024; copy flat)
  = the +9..13% §4-table errors. Fixed §4 text (not noise; large-C plateau=118, small-C
  residual flagged, could be a C-dependent rate like cat0). Chose option (2) over hiding
  the 1k points: wrote run_broadcast_smallcols_sweep.sh (COLS 256–4096 + a ROWS control
  at COLS=512) to test if the rise is a real trend below 1024 or a small-tensor artifact.
  Queued in appendix. Lint clean.

- 2026-07-10 (broadcast small-size sweep folded in): User ran run_broadcast_smallcols
  (SC1+SC2; SC3 added after, not yet run). +26 rows → 644. RESOLVED the small-COLS
  question: the >118 lift is a BOUNDED small-tensor speedup (~125-132 GB/s when EITHER
  dim is small, plateaus by COLS=256, → ~118 only when both large), affects ALL FOUR
  ops incl copy — NOT a runaway trend, NOT COLS-specific, so no C-dependent broadcast
  rate warranted; keep 118 (large-size asymptote), flag the ~+10% small-size residual.
  §4 prose + accuracy caption updated (broadcast 77 pts, 7.6%). fig4 rebuilt from
  records (COLS 256-16k, shows the rise+plateau); fig4 neg baseline switched to
  current_only=False (the new-log SHA had shrunk is_current). The 256×16384 −20%
  bcast/mulbcast anomaly still unconfirmed (contradicts small=faster; SC3 re-measure
  queued). Lint clean; eval OVERALL 12.1%.
