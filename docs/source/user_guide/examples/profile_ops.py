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

"""Cross-check our SPYRE_PROFILE_SYNC min-of-N against the PyTorch profiler.

The torch.profiler ``PrivateUse1`` activity (needs the kineto-spyre wheel -- see
docs/source/user_guide/profiling/pytorch_profiler.md) reports a **"Self SPYRE"**
column = the TRUE per-kernel device time. Two uses:

1. **Validate our timer.** Our SPYRE_PROFILE_SYNC min brackets the launch+sync, so it
   = device time + ~7 us host residue. Run the SAME op/size here and in bench_ops.py:
   ``our_min  ~=  Self_SPYRE(sdsc_fused_*)  +  ~7us`` confirms the measurement.
2. **Decompose the ~20 us fixed term.** The profiler surfaces a separate
   ``Memset (Device)`` event (~12.5 us on a tiny add). If it stays ~constant across
   sizes, a big chunk of our "fixed" is a real DEVICE memset, not host overhead.

NOTE: this is a TIME + memory-allocation profiler -- it has NO DRAM bandwidth / read-
vs-write / bus-utilization counters, so it CANNOT explain why read+write halves
bandwidth (that needs aiu-smi). It only gives cleaner device time.

Knobs: BENCH_OP, BENCH_ROWS, BENCH_COLS, BENCH_WARMUP. BENCH_OP is any of the
bench_ops/bench_bandwidth ops: neg copy gelu relu sigmoid exp | mul add | add3 add4 |
read sumrow sumall amax mean | bcast mulbcast | write.

Examples:
    # one op (prints the table + a parseable SUMMARY line with kernel_us / memset_us)
    BENCH_OP=neg BENCH_COLS=1024 \
        python docs/source/user_guide/examples/profile_ops.py
    # full golden re-sweep (rebuild the model from kernel time):
    bash docs/source/user_guide/examples/run_profile_sweep.sh
"""

import os

# Enable the cost-model dump so we read ITS device-layout I/O size (the same byte
# accounting the model uses), and force compile so the dump fires.
os.environ.setdefault("SPYRE_DUMP_COST", "1")
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from torch_spyre._inductor import cost_model, dump_cost_model  # noqa: E402

DEVICE = torch.device("spyre")
OP = os.environ.get("BENCH_OP", "gelu")
ROWS = int(os.environ.get("BENCH_ROWS", "512"))
COLS = int(os.environ.get("BENCH_COLS", "16384"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
SENCORES = os.environ.get("SENCORES", "")  # core count (read only to tag the SUMMARY)
TILES = int(os.environ.get("BENCH_TILES", "0"))  # coarse-tile dim0 into K (>=2 on)
LX = os.environ.get("LX_PLANNING", "1")  # scratchpad planning on(1)/off(0); SUMMARY tag
NCOLS = int(os.environ.get("BENCH_N", str(COLS)))  # matmul N dim (M=ROWS, K=COLS, N)
WD_M = os.environ.get("WD_M")  # forced matmul work-div split (spyre_hint work_div):
WD_N = os.environ.get("WD_N")  # cores = WD_M*WD_N*WD_K. Unset dim -> stays 1 (the hint
WD_K = os.environ.get("WD_K")  # is FINAL, not floor-filled). Used by the `mmwd` op.

torch.manual_seed(0xAFFE)


def _rand(*shape):
    return torch.rand(*shape, dtype=torch.float16).to(DEVICE)


def _sum_all(*ts):
    acc = ts[0]
    for t in ts[1:]:
        acc = acc + t
    return acc


# Same ops as bench_ops/bench_bandwidth so the GOLDEN kernel times map 1:1.
_UNARY = {  # 1 read + 1 write (gelu/relu/sigmoid/exp also probe arithmetic-free)
    "neg": lambda x: -x,  # cleanest balanced 1R+1W (no constant)
    "copy": lambda x: x + 1.0,  # scalar 1.0 is a cached broadcast -> still 1R+1W
    "gelu": F.gelu,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "exp": torch.exp,
}
_BINARY = {"mul": lambda a, b: a * b, "add": lambda a, b: a + b}  # 2R + 1W
_NARY = {"add3": 3, "add4": 4}  # n inputs summed (intermediates staged in LX)
_REDUCE = {  # read-dominated; sumall reduces to a scalar -> ring combine
    "read": lambda x: x.sum(dim=-1),
    "sumrow": lambda x: x.sum(dim=-1),  # reduce COLS -> [ROWS] (within-stick axis)
    "sumcol": lambda x: x.sum(dim=0),  # reduce ROWS -> [COLS] (the other axis)
    "sumall": lambda x: x.sum(),  # -> scalar: reduced axis split across all cores
    "amax": lambda x: x.amax(dim=-1),
    "mean": lambda x: x.mean(dim=-1),
}
_BCAST = {"bcast": lambda a, b: a + b, "mulbcast": lambda a, b: a * b}  # b = [1, COLS]
# Data-movement ("transport") ops -- a RESTICKIFY / copy that moves the SAME bytes as
# neg/copy but reorganizes the stick layout (scattered access). Access patterns differ:
#   transpose : [R,C]->[C,R], the stick dim MOVES (heavy intra-stick reshuffle)
#   cat0/cat1 : concat 2 copies along rows (non-stick) / cols (stick dim)
# Compare vs `neg` (contiguous 1R+1W) via effective BW to read the access-pattern cost.
# (`transpose_outer` -- a 3D stick-PRESERVING outer-dim swap -- is a separate branch.)
_TRANSPORT = {
    "transpose": lambda x: x.transpose(0, 1).contiguous(),  # [R,C]->[C,R]: stick C->R
    "cat0": lambda x: torch.cat([x, x], dim=0),  # append rows (non-stick dim)
    "cat1": lambda x: torch.cat([x, x], dim=1),  # append cols (stick dim -> interleave)
}
# Coarse-tiled dim0 reductions (mirror coarse_tile/run_{sum,amax,amin}_dim0_tiled.py):
# reduce a [B=ROWS, D=COLS] tensor over B, tiling B into BENCH_TILES chunks. With
# BENCH_TILES>=2 a spyre_hint wraps it in a K-iteration loop (fill + K x reduce/combine)
# BENCH_TILES<=1 is the plain untiled baseline. Run each under LX_PLANNING=0/1.
_CT_REDUCE = {"ctsum": "sum", "ctamax": "amax", "ctamin": "amin"}


# Per-call setup hook: the coarse-tile path sets this to re-declare its named dims
# before every traced invocation (the example declares once + compiles once; this
# profiling harness traces repeatedly). None for all non-tiled ops.
_PREPARE = None


def _ct_workload(rtype: str):
    """Coarse-tiled dim0 reduction, mirroring coarse_tile/run_*_dim0_tiled.py + utils.py
    ``_compile_and_run``: a CPU input (sum scaled 0.1 as the example does), eager
    ``declare_tensor_dim``, ``name_tensor_dims`` + ``spyre_hint`` inside fn, an eager
    reference call of fn, then a dynamo/Fx cache reset right before compile.
    """
    import torch_spyre._inductor.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    b, d = ROWS, COLS
    scale = 0.1 if rtype == "sum" else 1.0  # example scales sum to avoid fp16 growth
    x_cpu = torch.randn(b, d, dtype=torch.float16) * scale

    def _declare():
        pnd.declare_tensor_dim("B", b)
        pnd.declare_tensor_dim("D", d)

    _declare()  # eager, as the example does in main() before compiling

    def reduce_fn(x):
        return getattr(x, rtype)(dim=0)

    if TILES >= 2:

        def fn(x):
            pnd.name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"B": TILES}):
                return reduce_fn(x)

        # Re-declare EAGERLY (not inside fn -> no trace perturbation) before each traced
        # call so propagate_named_dims always resolves the named-dim sizes.
        _PREPARE = _declare
    else:
        fn = reduce_fn

    fn(x_cpu)  # eager reference call (mirror compare_with_cpu's cpu_result = fn(...))
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (x_cpu.to(DEVICE),)


def _chain_workload():
    """Fused pointwise chain ``z = (a + b) * c`` over [A=ROWS, B=COLS] -- the Part-3
    LX-residency probe (mirrors the scratchpad example in coarse_tiling_loops.md).
    BENCH_TILES>=2 tiles the A (row) dim so the intermediate ``y = a + b`` stays in LX
    instead of round-tripping HBM; ``allow_all_ops_in_lx_planning`` makes y LX-eligible.
    Untiled (BENCH_TILES<=1) is the baseline where y is a full HBM buffer. Toggle
    LX_PLANNING to compare. The IO dump shows y in lx (free, tiled) vs hbm (counted)."""
    import torch_spyre._inductor.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import config, spyre_hint

    global _PREPARE
    a_n, b_n = ROWS, COLS
    config.allow_all_ops_in_lx_planning = True  # let the intermediate y be LX-placed
    xa = torch.randn(a_n, b_n, dtype=torch.float16)
    xb = torch.randn(a_n, b_n, dtype=torch.float16)
    xc = torch.randn(a_n, b_n, dtype=torch.float16)

    def _declare():
        pnd.declare_tensor_dim("A", a_n)
        pnd.declare_tensor_dim("B", b_n)

    _declare()

    def _name(a, b, c):
        pnd.name_tensor_dims(a, ["A", "B"])
        pnd.name_tensor_dims(b, ["A", "B"])
        pnd.name_tensor_dims(c, ["A", "B"])

    if TILES >= 2:

        def fn(a, b, c):
            _name(a, b, c)
            with spyre_hint(num_tiles_per_dim={"A": TILES}):
                return (a + b) * c

        _PREPARE = _declare
    else:

        def fn(a, b, c):
            return (a + b) * c

    fn(xa, xb, xc)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (xa.to(DEVICE), xb.to(DEVICE), xc.to(DEVICE))


def _softmax_row_tiling():
    """Softmax over dim=-1, row-tiled over NROW (mirrors coarse_tile/run_softmax_row_
    tiling.py + the new_coarse_tiling IR): the 5 softmax ops (max, sub, exp, sum, div)
    fuse into ONE NROW-tiled loop with the intermediates LX-resident. BENCH_TILES>=2
    tiles NROW into that many chunks; else untiled. Measured via the harness AIU
    profiler (torch.profiler "Self SPYRE"), NOT the SPYRE_PROFILE host-launch path."""
    import torch_spyre._inductor.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    nrow, ncol = ROWS, COLS
    x_cpu = torch.randn(nrow, ncol, dtype=torch.float16)

    def _declare():
        pnd.declare_tensor_dim("NROW", nrow)
        pnd.declare_tensor_dim("NCOL", ncol)

    _declare()

    if TILES >= 2:

        def fn(x):
            pnd.name_tensor_dims(x, ["NROW", "NCOL"])
            with spyre_hint(num_tiles_per_dim={"NROW": TILES}):
                return torch.softmax(x, dim=-1)

        _PREPARE = _declare
    else:

        def fn(x):
            return torch.softmax(x, dim=-1)

    fn(x_cpu)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (x_cpu.to(DEVICE),)


def _mm_workload():
    """Matmul ``a @ b`` with a FORCED work-division split (spyre_hint work_div), so the
    (m, n, k) core split is controlled instead of planner-chosen -- for term isolation
    (compute / hbm / psum). M=ROWS, K=COLS, N=BENCH_N; WD_M/WD_N/WD_K give the per-dim
    split (cores = product, FINAL). Mirrors tests/inductor/test_work_division_hint.py:
    eager declare_tensor_dim, name the inputs' dims, hint inside fn; the coarse-tile
    _PREPARE hook re-declares before each traced call.
    """
    import torch_spyre._inductor.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    m, k, n = ROWS, COLS, NCOLS
    wd = {
        nm: int(os.environ[ev])
        for nm, ev in (("M", "WD_M"), ("N", "WD_N"), ("K", "WD_K"))
        if os.environ.get(ev)
    }
    xa = torch.rand(m, k, dtype=torch.float16)  # CPU; timing only, values irrelevant
    yb = torch.rand(k, n, dtype=torch.float16)

    def _declare():
        pnd.declare_tensor_dim("M", m)
        pnd.declare_tensor_dim("K", k)
        pnd.declare_tensor_dim("N", n)

    _declare()

    def fn(a, b):
        pnd.name_tensor_dims(a, ["M", "K"])
        pnd.name_tensor_dims(b, ["K", "N"])
        with spyre_hint(work_div=wd):
            return a @ b

    _PREPARE = _declare
    fn(xa, yb)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (xa.to(DEVICE), yb.to(DEVICE))


def _softmax_noexp_row_tiling():
    """MATCHED CONTROL for the softmax double-count/exp test: identical NROW-tiled fused
    structure as ``_softmax_row_tiling`` -- 2 reductions (amax, sum) + 3 pointwise (sub,
    MUL, div) -- but with the transcendental ``exp`` REPLACED by a cheap ``mul``. Same
    tiling, same fusion depth, same LX-resident intermediates; the ONLY difference is exp.
    So (softmax_time - softmax_noexp_time) at matched [ROWS,COLS,TILES] isolates the exp
    cost -- the design-review-mandated control that an untiled copy cannot provide."""
    import torch_spyre._inductor.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    nrow, ncol = ROWS, COLS
    x_cpu = torch.randn(nrow, ncol, dtype=torch.float16)

    def _declare():
        pnd.declare_tensor_dim("NROW", nrow)
        pnd.declare_tensor_dim("NCOL", ncol)

    _declare()

    def _sm_noexp(x):  # amax, sub, mul (<- exp), sum, div : softmax minus the exp
        y = x - x.amax(dim=-1, keepdim=True)
        z = y * 0.5  # cheap pointwise stand-in for exp (same pipeline stage, no exp)
        return z / z.sum(dim=-1, keepdim=True)

    if TILES >= 2:

        def fn(x):
            pnd.name_tensor_dims(x, ["NROW", "NCOL"])
            with spyre_hint(num_tiles_per_dim={"NROW": TILES}):
                return _sm_noexp(x)

        _PREPARE = _declare
    else:

        def fn(x):
            return _sm_noexp(x)

    fn(x_cpu)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (x_cpu.to(DEVICE),)


def _matmul_row_tiling():
    """Matmul ``a @ b`` COARSE-TILED over the M (row) dim via
    spyre_hint(num_tiles_per_dim={"M": TILES}) -- mirrors
    coarse_tile/run_matmul_row_tiling.py. DISTINCT from `mmwd`, which forces a
    work-division CORE split: this creates a sequential M-tile LOOP. M=ROWS, K=COLS,
    N=BENCH_N; BENCH_TILES>=2 tiles M, else untiled. Measured via the AIU profiler."""
    import torch_spyre._inductor.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    m, k, n = ROWS, COLS, NCOLS
    xa = torch.rand(m, k, dtype=torch.float16)  # CPU; timing only, values irrelevant
    yb = torch.rand(k, n, dtype=torch.float16)

    def _declare():
        pnd.declare_tensor_dim("M", m)
        pnd.declare_tensor_dim("K", k)
        pnd.declare_tensor_dim("N", n)

    _declare()

    if TILES >= 2:

        def fn(a, b):
            pnd.name_tensor_dims(a, ["M", "K"])
            pnd.name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"M": TILES}):
                return a @ b

        _PREPARE = _declare
    else:

        def fn(a, b):
            return a @ b

    fn(xa, yb)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (xa.to(DEVICE), yb.to(DEVICE))


def make_workload():
    if OP in _UNARY:
        return torch.compile(_UNARY[OP]), (_rand(ROWS, COLS),)
    if OP in _BINARY:
        return torch.compile(_BINARY[OP]), (_rand(ROWS, COLS), _rand(ROWS, COLS))
    if OP in _NARY:
        xs = tuple(_rand(ROWS, COLS) for _ in range(_NARY[OP]))
        return torch.compile(_sum_all), xs
    if OP in _REDUCE:
        return torch.compile(_REDUCE[OP]), (_rand(ROWS, COLS),)
    if OP in _BCAST:  # row-vector broadcast: a[R,C] + b[1,C] (b cached across rows)
        return torch.compile(_BCAST[OP]), (_rand(ROWS, COLS), _rand(1, COLS))
    if OP in _TRANSPORT:  # data movement (restickify): same bytes as a copy, scattered
        return torch.compile(_TRANSPORT[OP]), (_rand(ROWS, COLS),)
    if OP == "transpose_outer":  # 3D [R,8,C]: swap OUTER dims, stick (last dim C) kept
        tp = lambda x: x.transpose(0, 1).contiguous()  # noqa: E731
        return torch.compile(tp), (_rand(ROWS, 8, COLS),)
    if OP == "bcastcol":  # col-vector broadcast: a[R,C] + b[R,1] (b cached across cols)
        return torch.compile(lambda a, b: a + b), (_rand(ROWS, COLS), _rand(ROWS, 1))
    if OP == "write":  # write-only: both inputs broadcast -> cached
        return torch.compile(lambda b, c: b + c), (_rand(1, COLS), _rand(ROWS, 1))
    if OP in _CT_REDUCE:  # coarse-tiled dim0 reduction (BENCH_TILES, LX_PLANNING)
        return _ct_workload(_CT_REDUCE[OP])
    if OP == "chain":  # fused pointwise chain z=(a+b)*c, tiled -> y in LX (BENCH_TILES)
        return _chain_workload()
    if OP == "softmax_row_tiling":  # softmax(dim=-1) NROW-tiled -> 5 ops fuse in LX
        return _softmax_row_tiling()
    if OP == "softmax_noexp_row_tiling":  # matched control: softmax structure, exp->mul
        return _softmax_noexp_row_tiling()
    if OP == "matmul_row_tiling":  # a@b coarse-tiled over M (num_tiles, not core split)
        return _matmul_row_tiling()
    if OP == "mm":  # matmul [M=ROWS, K=COLS] @ [K=COLS, N=BENCH_N]: a Reduction over K
        # -> work-division Pass 2 (cost_model_matmul_division) picks the (m,n,k) split.
        mm = lambda a, b: a @ b  # noqa: E731
        return torch.compile(mm), (_rand(ROWS, COLS), _rand(COLS, NCOLS))
    if OP == "mmwd":  # matmul with a FORCED (m,n,k) split via WD_M/WD_N/WD_K
        return _mm_workload()
    known = (
        list(_UNARY)
        + list(_BINARY)
        + list(_NARY)
        + list(_REDUCE)
        + list(_BCAST)
        + list(_TRANSPORT)
        + [
            "bcastcol",
            "write",
            "chain",
            "softmax_row_tiling",
            "softmax_noexp_row_tiling",
            "matmul_row_tiling",
            "mm",
            "mmwd",
            "transpose_outer",
        ]
        + list(_CT_REDUCE)
    )
    raise SystemExit(f"unknown BENCH_OP={OP!r} (use {known})")


def _print_io(io: dict) -> None:
    """Per-tensor device-layout I/O the COST MODEL counts: dims, residency, bytes.

    Lines are prefixed ``IO `` so the sweep (run_profile_sweep.sh) can grep them
    alongside the SUMMARY line instead of dropping the breakdown.
    """
    print("IO -- device-layout I/O (cost model, stick-padded) --")
    for o in io.get("ops", []):
        red = " [reduction]" if o.get("is_reduction") else ""
        print(f"IO   op {o['name']}{red}")
        for a in o["args"]:
            bc = " broadcast (loaded once)" if a["broadcast"] else ""
            lf = a.get("loop_factor", 1)
            xl = f" xL={lf}" if lf > 1 else ""
            log = f"torch {a['logical']} -> " if a.get("logical") else ""
            print(
                f"IO     {a['role']:<6} {a.get('name', '?'):<22} "
                f"{log}device {a['dims']} in {a['mem'].upper()} = "
                f"{a['elems']} elems x 2B = {a['bytes']} B"
                f"  (hbm counted: {a['hbm_counted']} B){xl}{bc}"
            )
    print(
        f"IO   => HBM I/O total = {io.get('hbm_bytes', 0)} B  "
        f"(lx {io.get('lx_bytes', 0)} B, ~free)"
    )


def _print_model(feats: list) -> float:
    """Print the cost model's ESTIMATED kernel time + rough calc, after the I/O dump.

    Lines are prefixed ``MODEL `` so the sweep greps them. Returns the predicted us so
    the SUMMARY can carry it next to the measured kernel_us.
    """
    if not feats:
        print("MODEL (no features extracted)")
        return 0.0
    p = cost_model.CostParams()
    r, w = cost_model._fused_hbm_bytes(
        feats
    )  # external input counted once (fused kernel)
    lp = max((o.loop_trip for o in feats), default=1)
    is_mm = any(getattr(o, "is_matmul", False) for o in feats)
    base = (
        (r / p.mm_bw_read_gbps + w / p.mm_bw_write_gbps)
        if is_mm
        else (r + w) / p.bw_peak_gbps
    )
    turn = p.rw_turnaround_ns_per_byte * min(r, w)
    # Underfill derate (output-dim tiling): smallest per-core tile governs.
    eff, eff_rows = 1.0, 0.0
    for o in feats:
        if o.loop_trip > 1 and o.tiles_output_dim and o.tile_rows_per_core > 0:
            e = cost_model.coarse_underfill_eff(o.tile_rows_per_core, p)
            if e < eff:
                eff, eff_rows = e, o.tile_rows_per_core
    red_trip = max((o.loop_trip for o in feats if not o.tiles_output_dim), default=1)
    pw_trip = max((o.loop_trip for o in feats if o.tiles_output_dim), default=1)
    loop_ns = (p.c_loop_ns * red_trip if red_trip > 1 else 0.0) + (
        p.c_loop_pointwise_ns * pw_trip if pw_trip > 1 else 0.0
    )
    # Matmul compute term (additive).
    mm_us, mm_lines = 0.0, []
    for o in feats:
        if getattr(o, "is_matmul", False) and o.matmul_macs > 0 and o.cores > 0:
            pe = cost_model.underfill_eff(
                o.matmul_rows_per_core, p, p.underfill_target_passes_matmul
            )
            c_ns = o.matmul_macs / o.cores / (p.mac_peak_per_core_ns * pe)
            mm_us += c_ns / 1000
            mm_lines.append(
                f"MODEL   compute = MACs/cores/(mac_peak*pt_eff) = {o.matmul_macs}/"
                f"{o.cores}/({p.mac_peak_per_core_ns:.0f}*{pe:.3f}) = {c_ns / 1000:.2f}"
                f" us  (M/m={o.matmul_rows_per_core:.0f}, pt_eff={pe:.3f})"
            )
    t = cost_model.predict_ops(feats, p)
    parts = "(R+W)/BW_PEAK + a*min(R,W)"
    if eff < 1.0:
        parts = f"[{parts}] / eff_underfill"
    if mm_us > 0:
        parts = f"compute + {parts}"
    if loop_ns > 0:
        parts += " + c_loop*L"
    print(f"MODEL -- estimate (turnaround): T = {parts} --")
    print(f"MODEL   R={r} B (read)   W={w} B (write)   loop_trip L={lp}")
    for ln in mm_lines:
        print(ln)
    blab = (
        f"R/{p.mm_bw_read_gbps:.0f}+W/{p.mm_bw_write_gbps:.0f}"
        if is_mm
        else f"(R+W)/{p.bw_peak_gbps:.0f}"
    )
    print(f"MODEL   base = {blab} = {base / 1000:.2f} us")
    print(
        f"MODEL   turn = a*min(R,W) = {p.rw_turnaround_ns_per_byte}*{min(r, w)} "
        f"= {turn / 1000:.2f} us"
    )
    if eff < 1.0:
        print(
            f"MODEL   eff_underfill = min({p.coarse_underfill_cap},"
            f"({eff_rows:.1f}/{p.coarse_underfill_rfull:.0f})"
            f"**{p.coarse_underfill_exp}) = {eff:.3f} "
            f"-> (base+turn)/eff = {(base + turn) / eff / 1000:.2f} us"
        )
    if loop_ns > 0:
        lab = f"c_loop*{red_trip}" if red_trip > 1 else f"c_loop_pw*{pw_trip}"
        print(f"MODEL   loop = {lab} = {loop_ns / 1000:.2f} us")
    print(f"MODEL   => T_model = {t / 1000:.2f} us")
    # Machine-readable feature vector (the model's INPUT) so a NEW model version can be
    # scored OFFLINE against the stored measured time -- no hardware re-run. Prefixed
    # `MODEL ` so the sweeps' `^MODEL ` grep already captures it; parse_sweep_logs.py
    # pulls it into the record's `feats`. See notes/eval_model.py. Best-effort: a
    # serialization hiccup must NEVER fail the run (the kernel_us is what matters).
    try:
        print(f"MODEL FEATS {cost_model.ops_to_json(feats)}")
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"MODEL FEATS_SKIPPED {type(exc).__name__}: {str(exc)[:120]}")
    return t / 1000


def _run():
    compiled, args = make_workload()
    for _ in range(WARMUP):  # compile (-> cost-model dump fires) + warm the kernel
        if _PREPARE is not None:  # coarse-tile: re-declare named dims before each trace
            _PREPARE()
        compiled(*args).cpu()
    io = dict(dump_cost_model.LAST_IO)  # device-layout I/O the model computed
    feats = list(dump_cost_model.LAST_FEATS)  # raw OpFeatures for predict_ops()
    io_hbm_bytes = io.get("hbm_bytes", 0)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        if _PREPARE is not None:  # coarse-tile: re-declare before the profiled trace
            _PREPARE()
        compiled(*args).cpu()

    print(f"== {OP}[{ROWS}x{COLS}] -- profiler kernel time vs cost-model I/O ==")
    ka = prof.key_averages()
    print(ka.table(sort_by="cuda_time_total", row_limit=20).replace("CUDA", "AIU"))
    _print_io(io)
    pred_us = _print_model(feats)  # cost-model estimate + rough calc (after I/O dump)

    # Parseable one-liner for a size sweep -> re-fit fill + BW on the GOLDEN kernel
    # time, and track the (non-deterministic) Memset overhead separately. The fused
    # compute kernel is the SPYRE device time that is neither a Memset nor a Memcpy
    # (DtoH/HtoD copy). We classify by EXCLUSION, not by name: older runtimes named it
    # sdsc_fused_*/inductor-spyre-*, but the new image leaves the kernel event name
    # BLANK -- so a name match silently reports 0. Memcpy copies go to `other`.
    kernel = memset = other = 0.0
    for ev in ka:
        us = getattr(ev, "self_device_time_total", 0) or getattr(
            ev, "self_cuda_time_total", 0
        )
        if not us or us <= 0:
            continue
        key = ev.key or ""
        if "Memset" in key:
            memset += us
        elif "Memcpy" in key:
            other += us
        else:
            kernel += us  # fused compute kernel (sdsc_fused / inductor-spyre / BLANK)
    # Effective BW from the GOLDEN kernel time and the model's device-layout I/O.
    bw = io_hbm_bytes / (kernel * 1000) if kernel > 0 else 0.0
    err = (pred_us - kernel) / kernel * 100.0 if kernel > 0 else 0.0
    # ACTUAL cores from the model (split product), not the SENCORES budget tag -- a
    # forced (mmwd) split uses cores = m*n*k, which may be < SENCORES.
    mcores = max((getattr(o, "cores", 0) for o in feats), default=0)
    cores_tag = mcores if mcores > 0 else (SENCORES or "-")
    print(
        f"SUMMARY op={OP} rows={ROWS} cols={COLS} cores={cores_tag} "
        f"tiles={TILES} lx={LX} io_hbm_bytes={io_hbm_bytes} "
        f"kernel_us={kernel:.3f} pred_us={pred_us:.3f} err_pct={err:+.1f} "
        f"bw_gbps={bw:.1f} memset_us={memset:.3f} "
        f"other_dev_us={other:.3f} total_dev_us={kernel + memset + other:.3f}"
    )


def main():
    # Self-report failures: print the full traceback (stderr) AND a parseable FAILED
    # SUMMARY (stdout) carrying the reason, so a sweep that greps SUMMARY still records
    # WHY a run died instead of a bare "FAILED".
    try:
        _run()
    except Exception as exc:  # noqa: BLE001 - diagnostic wrapper
        import traceback

        traceback.print_exc()
        print(
            f"SUMMARY op={OP} rows={ROWS} cols={COLS} tiles={TILES} lx={LX} "
            f"FAILED reason={type(exc).__name__}: {str(exc)[:140]}"
        )


if __name__ == "__main__":
    main()
