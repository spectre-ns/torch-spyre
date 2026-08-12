# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple, high-level analytical cost model for Spyre kernels.

Goal: predict *relative* device latency from the "after-pre-scheduling"
LoopLevel IR to guide higher-level optimization. Deliberately NOT a simulator.

Model (per fused bundle / single-op kernel):

    T   = compute + mem - gamma*min(compute, mem)          mem = HBM / (eff * s_lx)

    HBM = [ (R+W)/BW + alpha*min(R,W) ] * (1 + arity_derate*(n-1)) + spill + write_extra
    s_lx = min(1, (512KB/ws)**0.15)   for a coarse-tiled kernel with ws > 512KB   (else 1)

  where R = HBM bytes READ (inputs), W = HBM bytes WRITTEN (outputs). LX-resident
  traffic is treated as ~free. ``eff`` (<=1) derates the MEMORY term for OUTPUT-dim
  (pointwise) coarse-tiling that shrinks each core's per-tile height; ``s_lx`` (<=1) further
  derates it when the per-core working set overflows LX (spilled traffic runs slower).
  ``compute`` is nonzero only for matmul (see below). A genuine-reduction cross-core ring term,
  and a per-iteration coarse-tiling loop overhead (c_loop*L), once lived here but are dropped
  (the ring is <=~5ns sub-noise; c_loop had no current op to validate it). For a normal untiled
  non-matmul kernel eff = 1, s_lx = 1 and compute = 0, so this reduces to the bandwidth model.

- ``fill`` ~= 0: the golden kernel has no fixed term (section-A intercept ~0; the old
  ~20us "fixed" was a separate non-deterministic Memset/host-setup bucket, not kernel).
- ``BW_PEAK`` ~150 GB/s (== bytes/ns) is the PEAK HBM bandwidth, reached when traffic is
  one-directional (read-only or write-only). HBM is a shared bus that must turn around
  between reading and writing, so a kernel doing BOTH pays a penalty on the overlap.
- ``alpha * min(R, W)`` is that read/write turnaround penalty. min(R,W) is the overlap:
  0 for pure read or pure write (no switching), maximal at a balanced 1R+1W. This gives
  the measured V-shaped effective BW -- ~150 read-only, dipping to ~105 at balanced,
  back up for write-only -- with one extra constant instead of a second bandwidth.
  Verified on the B-F profiler sweep: ~2% error on core pointwise + reductions, ~7%
  overall (turnaround) vs ~11% for an additive two-rate. Using a single aggregate
  BW_PEAK is the "shared HBM" assumption (rung-5: core-independent for >=2 cores).
- memory traffic counts each tensor-arg's bytes once, attributed to HBM or LX by
  its allocation. LX-placed tensors don't touch HBM, and their LX traffic is treated
  as ~free (the measured per-pass LX cost is below run-to-run noise). Broadcast inputs
  are loaded ONCE and reused across the broadcast dim, so they are counted at their own
  (one-row/-col) DEVICE size -- NOT scaled up to the output size (the rung-6 runs proved
  a core does not re-read the operand per output element), but NOT dropped to zero
  either (it is still a real one-time load). That one load is tiny vs the output, so the
  bcast/mulbcast runs still land ~on the 2-pass latency. They are flagged (``broadcast``
  in :class:`ArgTraffic`) for visibility. The "once, not per core" count is VERIFIED on
  device (rung-G reload probe, cores=32, R=64): bcast (b[1,C]) ~= bcastcol (b[R,1]) ~=
  30-33us, both far below the full 3-pass add (52us). A per-core reload would have added
  ~cores*C and pushed bcast up toward add; it did not -- so the operand costs a single
  load regardless of how the work splits across cores.

Byte counts use each arg's DEVICE layout (stick-padded ``device_size``), not the torch
logical shape -- so a reduction's reduced input is naturally full-sized and stick
rounding is captured. REDUCTIONS are a tiny WRITE + a full READ, so they run at a read
rate. That rate is NOT flat: it falls with ROWS (op-independent -- read/sum/amax/mean/
sumall collapse to one curve), ``reduction_read_bw = min(150, 114+61*exp(-ROWS/3700))``,
applied to a STANDALONE row-reduction (single op; a fused softmax stays on bw_peak so its
input dedup is not broken; ``sumcol`` uses reduce_outer). (A cross-core ring-combine term
for a split reduced axis was dropped as sub-noise -- provably <=~5ns on us kernels.)

MATMUL (reduction_type batchmatmul) adds a compute term that OVERLAPS the HBM term:
``T = compute + HBM - gamma*min(compute, HBM)``, with
``compute = MACs / cores / (mac_peak * pt_eff)`` (MACs = M*N*K, mac_peak=1140 SUSTAINED,
gamma=0.46). The matmul HBM uses a SINGLE rate = the copy peak (150; the old two-rate
143/156 is retired, 156>150 was an unphysical artifact) plus an operand RE-READ
"tile spill": ``(|A|+|B|)*f(area)`` at that rate, f a saturating log of the per-core
output-tile area (M/m)*(N/n) past the on-chip capacity ~64K elems). Fanout is NOT separately
identified (the falsification sweeps were confounded). K is
always kept whole (WD_K=1) so the K-split psum ring term is 0. Fit on the db_sweep +
decouple/re-read sweeps: ~8% RMS across cores 4->32, MNK 2e9..3.4e10. Isolation order and
per-term derivation: notes/cost_model_report.md.

COARSE-TILING (fused kernel, e.g. ``softmax_row_tiling``): a coarse-tiled op is ONE fused
kernel with intermediates kept in LX -- NOT a sum of per-op kernels. Two things follow.
(1) HBM = distinct EXTERNAL inputs ONCE + outputs once (``_fused_hbm_bytes``): softmax
reads ``arg0`` in both ``amax`` and ``sub``, but the fused kernel loads it from HBM once
and serves the 2nd read on-chip -- the naive per-op sum double-counts it (~+25% at the
floor; at the floor softmax runs at ~100 GB/s = the balanced-copy rate, i.e. 1 read +
1 write, confirming the once-count). (2) UNDERFILL: tiling an output dim cuts each core's
per-tile height (rpc = ROWS/(cores*tiles)); a short tile underfills the streaming pipeline.
``coarse_underfill_eff = min(cap, (rpc/r_full)**exp)`` (cap~0.95, r_full~13, exp~0.68),
SEPARATE from the matmul ``pt_eff``. Re-fit on the softmax rpc sweep; the cross-COLS control
(COLS 2048 vs 4096 at matched rpc -> per-byte cost equal to +/-4%) proved the derate keys on
ROWS (rpc), NOT tile bytes, and that it is the tile-SIZE effect, not the tile COUNT L (four
T=4..32 points at rpc=16 cost the same). Efficiency plateaus ~0.95 at rpc 16-32 and cliffs
below (rpc4~0.45, rpc2~0.28). KNOWN residuals: a mild efficiency decline above rpc~32
(rpc128~0.82, unmodeled, rows-driven); and when the per-core WORKING SET overflows the
practically available LX (~512 KB/core) the intermediates SPILL to HBM. The extractor already
counts those spilled bytes as HBM (they show up read+written), so it is NOT a byte miss -- but
that spilled traffic runs SLOWER than the modeled rate, so the effective bandwidth is DERATED
(``_lx_spill_bw_derate``: BW *= (cap/ws)**0.15 for ws>cap). Softmax-calibrated (11.0%->5.7% RMS),
gated to non-matmul coarse tiling. See report S14.

ACCESS-PATTERN effective-BW overrides (db_sweep; ``OpFeatures.hbm_pattern``, set by the
extractor from the LoopLevel IR index/layout -- these fold turnaround into one measured
rate): "restickify" transpose (stick swapped -> ~116, FASTER), "stick_scatter" cat0 on a
partition dim (fine sub-stick interleave -> SHAPE-dependent, falls with row width C:
clamp(intercept - rows_coef*log2(R) - cols_coef*log2(C), floor, bw_peak)), "reduce_outer"
sumcol (cross-row reduce -> ~113). Multi-pass pointwise chains (add3/add4: intermediates round-trip HBM) get a
``pointwise_arity_derate`` (~7.5%/extra op). All land within ~4% after these.
BROADCAST-operand ops (copy/bcast/bcastcol/mulbcast: a full input + a small broadcast
operand) run ~118 GB/s -- ``bw_broadcast_gbps`` (mechanism open). ``write`` (b[1,C]+c[R,1],
BOTH operands broadcast -> outer-product) is slow + super-linear; modeled by an EMPIRICAL
extra-traffic term (``write_reread_*``: coef*ROWS^r*COLS^c, ~12% RMS, black-box).
KNOWN residual (not yet modeled): transpose_outer at large size ~-14% (size-dependent);
reductions ~-15% at large ROWS (stick-inflated scattered output write).

CALIBRATION NOTE: the golden per-op measurement is the torch.profiler "Self SPYRE"
(sdsc_fused) KERNEL device time -- NOT our old SPYRE_PROFILE_SYNC min (which folded in a
non-deterministic Memset/host-setup bucket, the source of the obsolete ~20us fill).

Parameters live in :class:`CostParams`, calibrated from device measurements
(``examples/run_cost_model_plan.sh``).
"""

import builtins
import dataclasses
import math

import sympy


@dataclasses.dataclass
class ArgTraffic:
    """Traffic for one tensor argument of an op."""

    name: str
    role: str  # "input" | "output"
    is_lx: bool
    elems: int  # device element count = prod(dims) (its own one-load size)
    broadcast: bool = False  # loaded once & reused across the broadcast dim
    # DEVICE (stick) shape, e.g. [4, 512, 64]
    dims: list = dataclasses.field(default_factory=list)
    # LOGICAL torch shape, e.g. [512, 1024] -- shown next to dims so the stickification
    # (a row of N rounds up to ceil(N/64)*64 sticks) is visible per tensor.
    logical: list = dataclasses.field(default_factory=list)
    # Coarse-tiling loop multiplier on this arg's bytes. 1 for a normal arg or an
    # ADVANCING tiled arg (it walks the full tensor once across the loop, so its full
    # device_size already covers all tiles). L (= loop trip count) for a FIXED arg held
    # at one address across the loop (a per-tile accumulator re-read/written each
    # iteration). LX-resident args are ~free regardless (excluded from read/write).
    loop_factor: int = 1

    @property
    def mem(self) -> str:
        if isinstance(self.is_lx, bool):
            return "lx" if self.is_lx else "hbm"
        raise ValueError("Symbolic is_lx not supported for this operation")


@dataclasses.dataclass
class OpFeatures:
    """Cost-relevant features of one LoopLevel-IR op."""

    name: str  # origin op name (e.g. "gelu", "mul", "sub")
    is_reduction: bool
    out_elems: int
    cores: int
    dtype_bytes: int
    args: list  # list[ArgTraffic]
    reduction_cores: int = 1  # cores splitting the REDUCED axis (1 = none → no combine)
    loop_trip: int = 1  # coarse-tiling loop trip count for this op (prod of loop_count)
    # OUTPUT-dim (pointwise) coarse-tiling: True when this op tiles an output dim, so
    # each core's per-tile height shrinks and the underfill derate applies. False for
    # reduction-dim tiling and for untiled ops.
    tiles_output_dim: bool = False
    # Per-core per-tile pass-row height (output-tiled ops only): the streamed tile's
    # "rows" / cores. Drives ``eff_underfill``; 0.0 = unknown / not applicable -> no
    # derate.
    tile_rows_per_core: float = 0.0
    # MATMUL (reduction_type batchmatmul): adds an ADDITIVE compute term. matmul_macs =
    # M*N*K (total multiply-accumulates); matmul_rows_per_core = M/m (per-core M tile,
    # drives pt_eff). K-split k is carried in ``reduction_cores`` (-> the combine/PSUM
    # term). All zero/False for non-matmul ops.
    is_matmul: bool = False
    matmul_macs: int = 0
    matmul_rows_per_core: float = 0.0  # M/m (per-core A tile height -> A re-read)
    matmul_cols_per_core: float = 0.0  # N/n (per-core B tile width  -> B re-read)
    matmul_a_bytes: int = 0  # |A| = M*K device bytes (re-read scales with M/m)
    matmul_b_bytes: int = 0  # |B| = K*N device bytes (re-read scales with N/n)
    # Access-pattern HBM effective-BW override (from the LoopLevel IR index/layout):
    # "restickify" (transpose: write-stick var read with coeff!=1), "stick_scatter"
    # (cat on a partition dim -> a device dim <64 just inside the stick), "reduce_outer"
    # (cross-row reduction: reduced var read with coeff!=1). "" -> default 150+turnaround.
    hbm_pattern: str = ""

    def read_bytes(self) -> int:
        """HBM bytes READ (input args). Each HBM arg is counted at its own device size,
        scaled by ``loop_factor`` (L for a per-tile accumulator re-read every iteration,
        1 for an advancing tiled arg or a normal arg). A broadcast operand carries its
        real (one-row/-col) ``elems`` -- loaded once, NOT scaled to the output.
        """
        return (
            sum(
                a.elems * a.loop_factor * (1 - a.is_lx)
                for a in self.args
                if a.role == "input"
            )
            * self.dtype_bytes
        )

    def write_bytes(self) -> int:
        """HBM bytes WRITTEN (output args), scaled by ``loop_factor``."""
        return (
            sum(
                a.elems * a.loop_factor * (1 - a.is_lx)
                for a in self.args
                if a.role == "output"
            )
            * self.dtype_bytes
        )

    def hbm_bytes(self) -> int:
        """Total HBM traffic = read + write (kept for the dump / LAST_IO totals)."""
        return self.read_bytes() + self.write_bytes()

    def lx_bytes(self) -> int:
        return sum(a.elems * a.is_lx for a in self.args) * self.dtype_bytes


def op_to_dict(op: "OpFeatures") -> dict:
    """Serialize one OpFeatures (incl. its ArgTraffic list) to a plain JSON-able dict.

    This is the model's INPUT feature vector. Dumped next to the measured kernel time so
    a NEW model version can be scored OFFLINE (predict_ops on the stored features) without
    re-running on hardware -- the measurement is version-independent, only the prediction
    changes. See notes/eval_model.py.
    """
    return dataclasses.asdict(op)


def op_from_dict(d: dict) -> "OpFeatures":
    """Rebuild an OpFeatures from :func:`op_to_dict` output. Robust to schema drift:
    unknown keys are ignored and missing ones fall back to the dataclass defaults, so an
    old dataset still loads against a newer OpFeatures/ArgTraffic definition."""
    afields = {f.name for f in dataclasses.fields(ArgTraffic)}
    args = [
        ArgTraffic(**{k: v for k, v in a.items() if k in afields})
        for a in d.get("args", [])
    ]
    ofields = {f.name for f in dataclasses.fields(OpFeatures)}
    kw = {k: v for k, v in d.items() if k in ofields and k != "args"}
    return OpFeatures(args=args, **kw)


def _jsonable(o):
    """Fallback encoder: coerce non-native leaves (sympy ``Integer``/``Float`` from the
    inductor size expressions, numpy scalars, …) to plain int/float so ``json`` accepts
    them; anything else falls back to ``str``."""
    try:
        f = float(o)
    except (TypeError, ValueError):
        return str(o)
    return int(f) if f.is_integer() else f


def ops_to_json(ops: list) -> str:
    """Serialize a fused bundle (list of OpFeatures) to a single JSON string. Sizes coming
    off the IR are often sympy ``Integer``, so a ``default`` coercer is required."""
    import json

    return json.dumps(
        [op_to_dict(o) for o in ops], separators=(",", ":"), default=_jsonable
    )


def ops_from_json(s: str) -> list:
    """Deserialize a bundle serialized by :func:`ops_to_json`."""
    import json

    return [op_from_dict(d) for d in json.loads(s)]


@dataclasses.dataclass
class CostParams:
    """Fittable parameters for ``T = fill + (R+W)/BW_PEAK + alpha*min(R,W)``.

    Predicts the GOLDEN per-kernel device time (torch.profiler "Self SPYRE"). Fitted on
    the B-F profiler sweep (examples/run_profile_sweep.sh, fp16):
    - fill ~0       -- no fixed kernel term (section A intercept ~0).
    - BW_PEAK ~150 GB/s (== bytes/ns) -- the one-directional peak; read-only reductions
      and read probes land at ~145-155.
    - alpha ~0.00574 ns/byte -- the read/write turnaround penalty, calibrated so a
      balanced 1R+1W neg (R=W) lands at its measured ~105 GB/s effective:
      2/(2/BW_PEAK + alpha) = 105. min(R,W) is the read/write overlap (0 for
      one-directional traffic, maximal at balanced) -> reproduces the V-shaped
      effective BW. ~2% error on core ops, ~7% overall (see module docstring biases).
    LX traffic ~FREE (rung-4 below noise). Verified: arithmetic-free (gelu/exp == neg);
    broadcast operand loaded ONCE (rung-G, not per core); HBM BW shared / core-
    independent >=2 cores (rung 5).
    """

    fill_ns: float = 0.0  # golden kernel has ~no fixed term (section A: intercept ~0)
    bw_peak_gbps: float = 150.0  # one-directional peak HBM BW (read-only / write-only)
    # Read/write turnaround penalty (ns per overlapping byte). HBM is a shared bus that
    # must switch between read and write; the cost falls on the overlap min(R,W). Solved
    # from balanced neg (eff 105): alpha = 2/105 - 2/BW_PEAK.
    rw_turnaround_ns_per_byte: float = 0.00574
    # Genuine-reduction cross-core ring combine: (k-1) hops each touching every output
    # element. Fires ONLY for real reductions (NOT matmul -- gated in predict_ops), and
    # only when out_elems<cores, so it is bounded by ~cores*psum (<=~4.5ns) -- effectively
    # inert (kept for structure). The matmul K-split PSUM ring is deliberately NOT modeled:
    # the planner keeps K whole (WD_K=1), and forcing WD_K>1 made this term explode (+489%).
    # (A per-iteration coarse-tiling loop overhead c_loop*L was removed: it was calibrated on
    # the dropped chain/ctsum reduction-dim sweeps and no current op exercises it.)
    # Pipeline-fill (underfill) derate for OUTPUT-dim (pointwise) coarse-tiling:
    # eff = min(1, (rows_per_core / (pass_rows * target_passes)) ** exponent). Same FORM
    # as the matmul pt_eff (work_division.py); the 8-row pass is the shared hardware
    # constant, target_passes differs by op structure. PROVISIONAL -- guessed from the
    # chain K-sweep (flat to ~16 rows/core, cliff at 8; data hints exponent ~0.4). To be
    # calibrated by the untiled-small-ROWS underfill-confirm runs.
    underfill_pass_rows: float = 8.0  # PT / stream pass granularity (matmul _PT_ROWS)
    underfill_target_passes_pointwise: float = 2.0  # pointwise full-fill ~2 pass (=16)
    # Falloff exponent. CALIBRATED 0.35 from the Section-B chain sweep (rc 16->2): eff
    # rc8=0.72 rc4=0.63 rc2=0.50 (sqrt over-derated rc2 to 0.35). rc4/rc2 imply ~0.335;
    # rc8 is slightly concave (~9% residual). rc1 is NOT underfill -- there the planner
    # splits COLUMNS not rows, so row_split=1 -> full row tile (handled by the extractor
    # keying tile_rows_per_core on the row-dim split, not total cores).
    underfill_exponent: float = 0.35
    # COARSE-TILING (fused pointwise / softmax) underfill -- SEPARATE from the matmul
    # pt_eff above. Re-fit 2026-07-08 on the softmax rpc sweep (rpc = per-core rows per
    # tile = ROWS/(cores*T)); the cross-COLS control (COLS 2048 vs 4096 at matched rpc,
    # per-byte cost equal to +/-4%) proved the derate keys on ROWS (rpc), NOT tile bytes.
    # eff = min(cap, (rpc/r_full)**exp): plateau ~0.95 at rpc~16-32, steep underfill cliff
    # below (rpc4~0.45, rpc2~0.28). A MILD rise above rpc~32 (rpc128 eff~0.82) is a known,
    # unmodeled residual (rows-driven, mechanism TBD -- within the +/-15-20% bar).
    coarse_underfill_rfull: float = 13.0
    coarse_underfill_exp: float = 0.68
    coarse_underfill_cap: float = 0.95
    # LX-SPILL bandwidth derate. When a coarse-tiled kernel's per-core working set (its live
    # intermediate tiles, ~2 of them) overflows the practically available LX (~512 KB/core),
    # those intermediates spill to HBM. The extractor ALREADY counts the spilled bytes as HBM
    # (they show up read+written), so this is NOT a byte miss -- but that spilled traffic runs
    # SLOWER than the modeled rate, so the effective bandwidth is derated:
    #   BW *= min(1, (lx_spill_cap / ws_per_core) ** lx_spill_exp),  ws > cap only.
    # Softmax-calibrated (11.0% -> 5.7% RMS); gated to non-matmul coarse tiling.
    lx_spill_cap_bytes: float = 524288.0  # ~512 KB/core practically available LX
    lx_spill_exp: float = 0.15
    # MATMUL compute term. T_matmul = compute + HBM - gamma*min(compute, HBM), where
    # compute = MACs/cores/(mac_peak*pt_eff). mac_peak=1140 (sustained) fit on the
    # compute-DOMINANT low-core runs (cores 4-8, compute 80-90% of the kernel; the old
    # 1536 datasheet was ~33% optimistic). A single peak over-predicts cores=32 -- the
    # fix is overlap_gamma (compute/HBM pipeline). Fit jointly: peak=1140, gamma=0.46,
    # RMS 1.7% across cores 4->32. pt_eff reuses the underfill derate (~1 for M/m>=64).
    mac_peak_per_core_ns: float = 1140.0  # MAC/ns/core (sustained; compute-isolate fit)
    underfill_target_passes_matmul: float = 8.0  # matmul full-fill ~8 passes (=64 rows)
    # (A coarse-tiled matmul's per-tile array underfill is NOT modeled -- data too thin;
    # tiled matmuls take pt_eff=1. See predict_ops and report Part IV.)
    overlap_gamma: float = 0.46  # compute/HBM overlap: min(compute,HBM) partly hidden
    # Matmul operand RE-READ (tile spill): the per-core OUTPUT-accumulator tile has area
    # (M/m)*(N/n); once it exceeds the on-chip capacity (~64K fp16 elems/core) it no longer
    # stays resident, so the operands are re-streamed from HBM. The re-read magnitude is the
    # operand bytes; the fraction grows with how far the tile overflows:
    #   reread = (|A| + |B|) * f(area),  f(area) = min(cap, slope*log2(area/area0)).
    # Fit on the decouple + re-read sweeps (fanout proven NOT a term; K-split never used, so
    # the old psum ring term = 0).
    mm_spill_area0: float = (
        65536.0  # per-core output-tile area (elems) below which no spill
    )
    mm_spill_slope: float = 0.45
    mm_spill_cap: float = 1.50
    # Matmul HBM: a SINGLE effective rate = the pointwise copy peak (150). The earlier
    # two-rate fit (143 read / 156 write) is retired: 156 > 150 is unphysical (a write
    # cannot beat the copy peak) and was a compute-free-fit artifact absorbing the overlap
    # term. On the planner-realistic envelope a single 150 + turnaround + overlap scores
    # better than the old two-rate (RMS ~5.8% with the area-spill term) -- equal-or-better
    # AND physical. Read/write are not separately identifiable from these data. (See §8.)
    mm_bw_read_gbps: float = 150.0
    mm_bw_write_gbps: float = 150.0

    # Access-pattern effective HBM BW (GB/s) for non-matmul ops whose stick layout is
    # reorganized -- these fold turnaround into the single rate (measured io/kernel on
    # the db_sweep). Keyed by OpFeatures.hbm_pattern; default ops keep bw_peak+turnaround.
    bw_restickify_gbps: float = (
        116.0  # transpose: stick swapped, LESS turnaround (faster)
    )
    # cat on the partition (64-elem block) dim: fine sub-stick interleave. Its effective
    # BW is SHAPE-dependent -- it falls mainly with the row width C (more sticks per row to
    # scatter), weakly with R. Fit on the transport-shape sweep (R2=0.93 over 10 shapes):
    # effBW = intercept - rows_coef*log2(R) - cols_coef*log2(C), clamped to [floor, peak].
    bw_stick_scatter_intercept: float = 252.0
    bw_stick_scatter_rows_coef: float = 4.0
    bw_stick_scatter_cols_coef: float = 12.3
    bw_stick_scatter_floor_gbps: float = 45.0  # large-C saturation clamp
    bw_reduce_outer_gbps: float = 113.0  # cross-row (dim0) reduction (sumcol)
    # Row-reduction READ rate falls with ROWS (the read pipeline degrades as each core
    # streams more rows), op-independent, saturating. Fit on the reduction-rows sweep
    # (read/sumrow/amax/mean/sumall collapse to one curve): effBW = floor + amp*exp(-ROWS/
    # scale), clamped to peak. sumcol (reduce_outer) is exempt -- different access pattern.
    red_read_bw_floor_gbps: float = 114.0
    red_read_bw_amp_gbps: float = 61.0
    red_read_bw_scale_rows: float = 3700.0
    # Ops that stream a FULL input plus a small BROADCAST operand (loaded once) -- copy
    # (x+const), bcast, bcastcol, mulbcast -- run FASTER than a plain 1R:1W op (~118 vs
    # ~105 GB/s; mechanism open). NOT `write` (both operands broadcast, no full input).
    bw_broadcast_gbps: float = 118.0
    # `write` (b[1,C] + c[R,1]: BOTH operands broadcast -> an outer-product write) is slow
    # and SUPER-LINEAR: the operands are re-read in the outer-product and the cost grows
    # steeply with COLS (and, more weakly, ROWS). No clean mechanism yet; the extra HBM
    # traffic is fit EMPIRICALLY on the write sweep -- extra_bytes = coef * ROWS^r * COLS^c,
    # charged at bw_peak. ~12% RMS over the sweep (worst ~-30% at mid sizes). Black-box,
    # to be replaced when the mechanism is understood (see report §4).
    write_reread_coef: float = 2.148e-7
    write_reread_r_exp: float = 1.60
    write_reread_c_exp: float = 2.20
    # Multi-pass pointwise chains (add3/add4: intermediates round-trip HBM) run slower per
    # byte -- ~7.5% per extra op beyond the first. Fit on the arity sweep (add/add3/add4).
    pointwise_arity_derate: float = 0.075


def underfill_eff(
    rows_per_core: float,
    params: CostParams | None = None,
    target_passes: float | None = None,
) -> float:
    """Pipeline-fill efficiency (<=1) for a per-core tile of ``rows_per_core`` rows.

    The streaming / PT pipeline processes a core's tile in passes of
    ``underfill_pass_rows`` (8) rows; a tile shorter than ``pass_rows * target_passes``
    cannot amortise pipeline fill/drain, so effective throughput derates as
    ``(rows / r_full) ** exponent``, capped at 1. ``target_passes`` defaults to the
    pointwise value (coarse-tiling); pass ``underfill_target_passes_matmul`` for the
    matmul compute term (same FORM, deeper pipeline). ``rows_per_core <= 0`` (unknown)
    -> 1.0 (no derate).
    """
    p = params or CostParams()
    if rows_per_core <= 0:
        return 1.0
    tp = p.underfill_target_passes_pointwise if target_passes is None else target_passes
    r_full = p.underfill_pass_rows * tp
    if r_full <= 0:
        return 1.0
    return min(1.0, (rows_per_core / r_full) ** p.underfill_exponent)


def coarse_underfill_eff(rpc: float, params: CostParams | None = None) -> float:
    """Pipeline-fill efficiency for a COARSE-tiled (fused pointwise / softmax) kernel whose
    per-core tile is ``rpc`` rows tall (rpc = ROWS/(cores*tiles)). DISTINCT from the matmul
    pt_eff (``underfill_eff``): re-fit on the softmax rpc sweep, where efficiency plateaus
    at ``coarse_underfill_cap`` (~0.95) around rpc 16-32 and derates as
    ``(rpc/r_full)**exp`` below (rpc4~0.45, rpc2~0.28). The cross-COLS control proved this
    keys on ROWS (rpc), not tile bytes. ``rpc<=0`` (unknown/untiled) -> 1.0 (no derate).
    (A mild efficiency decline above rpc~32 is a known, unmodeled residual.)"""
    p = params or CostParams()
    if rpc <= 0:
        return 1.0
    return min(
        p.coarse_underfill_cap,
        (rpc / p.coarse_underfill_rfull) ** p.coarse_underfill_exp,
    )


def _lx_spill_working_set(ops: list) -> float:
    """Per-core LX working set (bytes) of a coarse-tiled bundle: ~2 live intermediate tiles,
    each ``tile_rows_per_core * cols`` elements. 0.0 if nothing is output-tiled."""
    ws = 0.0
    for o in ops:
        if o.tiles_output_dim and o.tile_rows_per_core > 0:
            cols = max((a.logical[-1] for a in o.args if a.logical), default=0)
            ws = max(ws, 2.0 * o.tile_rows_per_core * cols * o.dtype_bytes)
    return ws


def _lx_spill_bw_derate(ops: list, params: CostParams | None = None) -> float:
    """Bandwidth derate when a coarse-tiled kernel's per-core working set overflows LX. The
    spilled intermediates are already counted as HBM bytes, but that traffic runs slower, so
    ``BW *= (lx_spill_cap / ws)**lx_spill_exp`` for ``ws > cap``. Gated to non-matmul coarse
    tiling (softmax-calibrated); 1.0 when it does not apply."""
    p = params or CostParams()
    if any(getattr(o, "is_matmul", False) for o in ops):
        return 1.0
    ws = _lx_spill_working_set(ops)
    if ws <= p.lx_spill_cap_bytes:
        return 1.0
    return (p.lx_spill_cap_bytes / ws) ** p.lx_spill_exp


def mm_spill_frac(tile_area: float, params: CostParams | None = None) -> float:
    """Operand RE-READ fraction for a matmul: once the per-core output-accumulator tile
    of area ``(M/m)*(N/n)`` exceeds ``mm_spill_area0`` (the on-chip capacity) the operands
    no longer stay resident and are re-streamed from HBM. Saturating log growth
    ``min(cap, slope*log2(area/area0))``; 0 at/below area0."""
    p = params or CostParams()
    if tile_area <= 0:
        return 0.0
    return min(
        p.mm_spill_cap,
        p.mm_spill_slope * math.log2(max(1.0, tile_area / p.mm_spill_area0)),
    )


def max(*args, **kwargs):
    """``max``, but symbolic-aware: dispatches to ``sympy.Max`` when an arg is
    a sympy expression (whose truth-valued comparisons the builtin can't
    resolve), otherwise defers to the builtin -- including its ``key``/
    ``default`` kwargs and single-iterable form, neither of which ``sympy.Max``
    supports."""
    if any(isinstance(a, sympy.Basic) for a in args):
        return sympy.Max(*args)
    return builtins.max(*args, **kwargs)


def min(*args, **kwargs):
    """``min`` counterpart of :func:`max`; see its docstring."""
    if any(isinstance(a, sympy.Basic) for a in args):
        return sympy.Min(*args)
    return builtins.min(*args, **kwargs)


def _fused_hbm_bytes(ops: list) -> tuple:
    """(read, write) HBM bytes for a FUSED bundle, counting each distinct EXTERNAL graph
    input (name starts ``arg``) ONCE even if several fused ops read it -- a fused kernel
    loads it from HBM once and serves the re-reads on-chip/LX (softmax reads ``arg0`` in
    both ``amax`` and ``sub``; the naive per-op sum double-counts it, ~+25% at the floor).
    Internal-buffer traffic is taken as the IR reports it: LX buffers are ~free (excluded),
    and a buffer that SPILLED to HBM and is re-read stays counted (the spill is exactly why
    it can't be reused on-chip). Outputs summed as-is (distinct per op)."""
    r = w = 0
    ext_in: dict = {}  # external input name -> its one-load HBM bytes (dedup across ops)
    for o in ops:
        for a in o.args:
            b = a.elems * a.loop_factor * o.dtype_bytes * (1 - a.is_lx)
            if a.role == "input" and a.name.startswith("arg"):
                if a.name in ext_in:
                    ext_in[a.name] = max(ext_in[a.name], b)
                else:
                    ext_in[a.name] = b
            elif a.role == "input":
                r += b
            else:
                w += b
    r += sum(ext_in.values())
    return r, w


def _is_broadcast_op(o) -> bool:
    """True for an op that streams a FULL HBM input AND a small BROADCAST operand (loaded
    once): copy (x+const), bcast, bcastcol, mulbcast. These run at ``bw_broadcast_gbps``,
    faster than a plain 1R:1W op. Excludes matmul/reduction and ``write`` (both operands
    broadcast -> no full input, and a different, super-linear regime)."""
    if getattr(o, "is_matmul", False) or o.is_reduction:
        return False
    ins = [a for a in o.args if a.role == "input" and a.mem == "hbm"]
    return any(a.broadcast for a in ins) and any(not a.broadcast for a in ins)


def _is_outer_broadcast(o) -> bool:
    """True for a `write`-like op where EVERY HBM input is a broadcast operand (no full
    streamed input) -- an outer-product write ``b[1,C] + c[R,1]``. Its cost is slow and
    super-linear (empirical ``write_reread_*`` term)."""
    if getattr(o, "is_matmul", False) or o.is_reduction:
        return False
    ins = [a for a in o.args if a.role == "input" and a.mem == "hbm"]
    return bool(ins) and all(a.broadcast * (1 - a.is_lx) for a in ins)


def _outer_broadcast_extra_bytes(o, p) -> float:
    """Empirical extra HBM bytes for an outer-product write (see CostParams). 0 if the
    output's logical [R, C] shape is unavailable."""
    out = next(
        (
            a
            for a in o.args
            if a.role == "output" and a.mem == "hbm" and len(a.logical) >= 2
        ),
        None,
    )
    if out is None:
        return 0.0
    rows, cols = out.logical[-2], out.logical[-1]
    return p.write_reread_coef * rows**p.write_reread_r_exp * cols**p.write_reread_c_exp


def _logical_rc(o):
    """(rows, cols) from the op's output logical [.., R, C], or None."""
    out = next(
        (
            a
            for a in o.args
            if a.role == "output" and a.mem == "hbm" and len(a.logical) >= 2
        ),
        None,
    )
    return (out.logical[-2], out.logical[-1]) if out else None


def stick_scatter_bw(o, p):
    """cat-on-block-dim (cat0) effective BW: falls with row width C, weakly with R."""
    rc = _logical_rc(o)
    if rc is None:
        return p.bw_stick_scatter_floor_gbps
    rows, cols = rc
    bw = (
        p.bw_stick_scatter_intercept
        - p.bw_stick_scatter_rows_coef * math.log2(max(2, rows))
        - p.bw_stick_scatter_cols_coef * math.log2(max(2, cols))
    )
    return min(p.bw_peak_gbps, max(p.bw_stick_scatter_floor_gbps, bw))


def _reduction_rows(o):
    """ROWS of a reduction's input (governs its read rate), from the largest HBM input."""
    ins = [
        a
        for a in o.args
        if a.role == "input" and a.mem == "hbm" and len(a.logical) >= 2
    ]
    return max(ins, key=lambda a: a.elems).logical[-2] if ins else 0


def reduction_read_bw(rows, p):
    """Row-reduction read rate: peak at small ROWS, falling+saturating as ROWS grows."""
    return min(
        p.bw_peak_gbps,
        p.red_read_bw_floor_gbps
        + p.red_read_bw_amp_gbps * math.exp(-rows / p.red_read_bw_scale_rows),
    )


def predict_ops(ops: list, params: CostParams | None = None) -> float:
    """Predicted device latency (ns) for a bundle of ops (one fused kernel).

    ``T = fill + [(R+W)/BW_PEAK + alpha*min(R,W)] / eff_underfill`` where R/W are the
    bundle's HBM read/write bytes (LX ~free), already
    loop-scaled per arg (see ArgTraffic.loop_factor). R/W come from ``_fused_hbm_bytes``:
    a fused kernel loads each distinct EXTERNAL input from HBM ONCE (re-reads served
    on-chip), so a shared input is not double-counted; reads and writes are summed over
    the bundle before the turnaround term (shared bus). ``eff_underfill`` derates the
    bandwidth term when OUTPUT-dim (coarse) tiling shortens each core's per-tile height
    (``coarse_underfill_eff``, keyed on per-core rows per tile).
    Matmul ops add an ADDITIVE compute term (MACs/cores/(mac_peak*pt_eff)).
    """
    p = params or CostParams()
    r, w = _fused_hbm_bytes(ops)
    # HBM. Matmul uses a SINGLE effective rate (mm_bw_read==mm_bw_write==150, the copy
    # peak) plus the read/write turnaround -- same form as pointwise. The old two-rate
    # read<write model is retired (156>150 was unphysical, a compute-free-fit artifact).
    # Pointwise/reduction/transport keep the single-BW turnaround model.
    _pat_bw = {
        "restickify": p.bw_restickify_gbps,
        "reduce_outer": p.bw_reduce_outer_gbps,
    }

    def _eff_bw(o):  # per-op effective-BW override, or None -> default turnaround
        pat = getattr(o, "hbm_pattern", "")
        # TODO: uncomment
        # if pat == "stick_scatter":  # cat0: shape-dependent rate (falls with C)
        #    return stick_scatter_bw(o, p)
        if pat in _pat_bw:  # restickify (transpose), reduce_outer (sumcol)
            return _pat_bw[pat]
        # TODO: uncomment
        # if _is_broadcast_op(o):
        #    return p.bw_broadcast_gbps
        return None

    if any(getattr(o, "is_matmul", False) for o in ops):
        # Operand re-read: when the per-core output tile of area (M/m)*(N/n) overflows the
        # on-chip capacity, both operands (|A|+|B|) are re-streamed by the same fraction.
        # Read-rate bytes. (Fanout was proven NOT a term by the re-read sweep.)
        spill = sum(
            (o.matmul_a_bytes + o.matmul_b_bytes)
            * mm_spill_frac(o.matmul_rows_per_core * o.matmul_cols_per_core, p)
            for o in ops
            if getattr(o, "is_matmul", False)
        )
        mem = (
            r / p.mm_bw_read_gbps
            + w / p.mm_bw_write_gbps
            + spill / p.mm_bw_read_gbps
            + p.rw_turnaround_ns_per_byte * min(r, w)
        )
    elif any(_eff_bw(o) is not None for o in ops):
        # Per-op effective BW (access-pattern transports OR a broadcast operand); these
        # fold turnaround into the rate. Ops without an override keep the default
        # single-BW + turnaround.
        mem = 0.0
        for o in ops:
            ro, wo = o.read_bytes(), o.write_bytes()
            bw = _eff_bw(o)
            if bw:
                mem += (ro + wo) / bw
            else:
                mem += (ro + wo) / p.bw_peak_gbps + p.rw_turnaround_ns_per_byte * min(
                    ro, wo
                )
    elif (
        len(ops) == 1
        and ops[0].is_reduction
        and not getattr(ops[0], "is_matmul", False)
        and not ops[0].tiles_output_dim
    ):
        # A STANDALONE row-reduction (sum/amax/mean/read over the last axis, or sumall)
        # reads at a rate that FALLS with ROWS. The rate is fit as (R+W)/time, so it
        # already includes the read/write turnaround -- do NOT add it again. sumcol takes
        # the reduce_outer path above; a FUSED coarse kernel (len>1, e.g. softmax) stays on
        # bw_peak below so its input dedup is not broken.
        # TODO: skip the cost model in the solver when there's only one op
        mem = (r + w) / reduction_read_bw(_reduction_rows(ops[0]), p)
    else:
        mem = (r + w) / p.bw_peak_gbps + p.rw_turnaround_ns_per_byte * min(r, w)
        # Multi-pass pointwise chain (add3/add4): intermediates round-trip HBM, so the
        # bundle runs slower per byte than a single op. ~7.5% per extra HBM op.
        n_pw = sum(
            1
            for o in ops
            if not o.is_reduction
            and not getattr(o, "is_matmul", False)
            and not o.tiles_output_dim
        )
        if n_pw > 1:
            mem *= 1.0 + p.pointwise_arity_derate * (n_pw - 1)
    # `write` outer-product re-read: empirical extra HBM traffic, super-linear in the
    # output shape (both operands broadcast, no full input). Charged at bw_peak.
    # TODO: fix this
    # mem += (
    #    sum(_outer_broadcast_extra_bytes(o, p) for o in ops if _is_outer_broadcast(o))
    #    / p.bw_peak_gbps
    # )
    # OUTPUT-dim (pointwise) coarse-tiling underfill: a short per-core tile underfills
    # the streaming pipeline, derating the bandwidth term. The smallest tile in the
    # bundle governs (worst underfill). 1.0 (no derate) when nothing is output-tiled.
    eff = 1.0
    for o in ops:
        if o.loop_trip > 1 and o.tiles_output_dim and o.tile_rows_per_core > 0:
            eff = min(eff, coarse_underfill_eff(o.tile_rows_per_core, p))
    # LX-SPILL bandwidth derate: a coarse-tiled kernel whose per-core working set (~2 live
    # intermediate tiles) overflows LX spills to HBM, and that spilled traffic runs slower
    # than the modeled rate. Bytes are already counted as HBM; here we derate the BW.
    spill_derate = _lx_spill_bw_derate(ops, p)
    mem_t = p.fill_ns + mem / eff / spill_derate
    # MATMUL compute = MACs/cores derated by pt_eff (PT-array fill).
    compute = 0.0
    for o in ops:
        if (
            o.is_matmul
            and o.matmul_macs > 0
            and (not isinstance(o.cores, int) or o.cores > 0)
        ):
            # A coarse-tiled matmul appears to underfill the array MORE per tile than a
            # standalone one, but the current data (thin, non-current, partly U-shaped) is
            # too weak to fit -- so it is NOT modeled: tiled matmuls take pt_eff=1 (flagged;
            # a clean tile-count sweep is queued). Standalone matmuls use the array-fill derate.
            if o.tiles_output_dim:
                pt_eff = 1.0
            else:
                pt_eff = underfill_eff(
                    o.matmul_rows_per_core, p, p.underfill_target_passes_matmul
                )
            compute += o.matmul_macs / o.cores / (p.mac_peak_per_core_ns * pt_eff)
    # compute/HBM OVERLAP: memory transfers pipeline with the systolic compute, so the
    # smaller of the two is partly hidden (gamma=0.46). For a non-matmul bundle compute=0
    # -> min(0, mem_t)=0 -> t = mem_t (unchanged).
    if compute == 0.0:
        t = mem_t
    else:
        t = compute + mem_t - p.overlap_gamma * min(compute, mem_t)
    # (A genuine-reduction cross-core ring-combine term once lived here; it is provably
    # bounded by ~cores * a tiny per-elem cost <= ~5 ns -- below run-to-run noise --
    # so it is dropped as inert. K is never split for matmul, so there is no matmul analogue.
    # A per-iteration coarse-tiling LOOP overhead (c_loop*L) also once lived here, calibrated
    # on the now-dropped chain/ctsum reduction-dim sweeps; no current op exercises it, so it
    # is removed rather than carried unvalidated.)
    return t


def predict_op(op: OpFeatures, params: CostParams | None = None) -> float:
    """Predicted device latency (ns) for a single op (as its own kernel)."""
    return predict_ops([op], params)


def explain(ops: list, params: CostParams | None = None) -> str:
    """Human-readable breakdown of the prediction for a bundle of ops."""
    p = params or CostParams()
    lines = []
    for o in ops:
        r, w, lx = o.read_bytes(), o.write_bytes(), o.lx_bytes()
        loop = f" loop_trip={o.loop_trip}" if o.loop_trip > 1 else ""
        pat = f" [{o.hbm_pattern}]" if getattr(o, "hbm_pattern", "") else ""
        lines.append(f"  {o.name:<12} read={r}B write={w}B lx={lx}B{loop}{pat}")
        for a in o.args:
            bc = " broadcast (loaded once)" if a.broadcast else ""
            lf = f" xL={a.loop_factor}" if a.loop_factor > 1 else ""
            counted = a.elems * a.loop_factor * o.dtype_bytes * (1 - a.is_lx)
            dev = a.dims if a.dims else [a.elems]
            log = f"torch {a.logical} -> " if a.logical else ""
            # One line per DEVICE-LAYOUT tensor: name, role, logical->device dims,
            # residency, byte calc, the HBM bytes the model counts, and the loop factor.
            lines.append(
                f"      {a.role:<6} {a.name:<22} {log}device {dev} in {a.is_lx}"
                f"  | {a.elems} elems x {o.dtype_bytes}B = {a.elems * o.dtype_bytes} B"
                f" (hbm counted: {counted} B){lf}{bc}"
            )
    # Prediction with the rough calculation spelled out, so SPYRE_DUMP_COST shows the
    # same step-by-step breakdown (base + turnaround, then the underfill derate for
    # pointwise tiling).
    R, W = _fused_hbm_bytes(ops)  # external input counted once (fused kernel)
    is_mm = any(getattr(o, "is_matmul", False) for o in ops)
    base = (
        (R / p.mm_bw_read_gbps + W / p.mm_bw_write_gbps)
        if is_mm
        else (R + W) / p.bw_peak_gbps
    )
    turn = p.rw_turnaround_ns_per_byte * min(R, W)
    # Underfill derate (output-dim tiling): smallest per-core tile governs.
    eff, eff_rows = 1.0, 0.0
    for o in ops:
        if o.loop_trip > 1 and o.tiles_output_dim and o.tile_rows_per_core > 0:
            e = coarse_underfill_eff(o.tile_rows_per_core, p)
            if e < eff:
                eff, eff_rows = e, o.tile_rows_per_core
    # Matmul compute (additive): sum the per-op compute term for any matmul ops.
    mm_us, mm_lines = 0.0, []
    for o in ops:
        if o.is_matmul and o.matmul_macs > 0 and o.cores > 0:
            pe = underfill_eff(
                o.matmul_rows_per_core, p, p.underfill_target_passes_matmul
            )
            c_ns = o.matmul_macs / o.cores / (p.mac_peak_per_core_ns * pe)
            mm_us += c_ns / 1000
            mm_lines.append(
                f"     compute = MACs/cores/(mac_peak*pt_eff) = {o.matmul_macs}/"
                f"{o.cores}/({p.mac_peak_per_core_ns:.0f}*{pe:.3f}) = {c_ns / 1000:.2f}"
                f" us  (M/m={o.matmul_rows_per_core:.0f}, pt_eff={pe:.3f})"
            )
    t = predict_ops(ops, p)
    parts = "(R+W)/BW_PEAK + a*min(R,W)"
    if eff < 1.0:
        parts = f"[{parts}] / eff_underfill"
    if mm_us > 0:
        parts = f"compute + {parts}"
    lines.append(f"  -- prediction (turnaround): T = {parts} --")
    lines.append(f"     R={R}B (read)   W={W}B (write)")
    lines.extend(mm_lines)
    if is_mm:
        blab = f"R/{p.mm_bw_read_gbps:.0f} + W/{p.mm_bw_write_gbps:.0f}"
    else:
        blab = f"(R+W)/{p.bw_peak_gbps:.0f}"
    lines.append(f"     base = {blab} = {base / 1000:.2f} us")
    lines.append(
        f"     turn = a*min(R,W) = {p.rw_turnaround_ns_per_byte}*{min(R, W)} "
        f"= {turn / 1000:.2f} us"
    )
    if eff < 1.0:
        lines.append(
            f"     eff_underfill = min({p.coarse_underfill_cap},"
            f"({eff_rows:.1f}/{p.coarse_underfill_rfull:.0f})"
            f"**{p.coarse_underfill_exp}) = {eff:.3f}  "
            f"-> (base+turn)/eff = {(base + turn) / eff / 1000:.2f} us"
        )
    lines.append(f"     => T_model = {t / 1000:.2f} us")
    return "\n".join(lines)
