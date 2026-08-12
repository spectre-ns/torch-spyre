# Spyre cost-model tooling

Three additive features we built on top of the torch-spyre compiler. Each is a
no-op unless its env var is set, so normal runs are unaffected. The example
commands use the profiler harness
[docs/source/user_guide/examples/profile_ops.py](../docs/source/user_guide/examples/profile_ops.py)
(knobs: `BENCH_OP`, `BENCH_ROWS`, `BENCH_COLS`, `BENCH_TILES`, `SENCORES`,
`LX_PLANNING`).

> **Note:** the dumps and cost-model prediction only fire on a fresh compile, so
> clear the TorchInductor cache before every run — otherwise a cached graph is
> reused and nothing is dumped. The reliable way is to delete the on-disk cache
> (`rm -rf /tmp/torchinductor_*`).

## 1. Dump FX graph and loop-level IR

`SPYRE_DUMP_IR=1` prints the post-grad ATen FX graph and the after-pre-scheduling
LoopLevel IR at compile time (to stderr, or to `SPYRE_DUMP_IR_FILE`). The
LoopLevel dump also shows coarse-tiling metadata — `dim_hints`, `loop_info`
(`loop_count` / `loop_tiled_reduction_dims`), `op_it_space_splits` — when a
`spyre_hint` is present.

- Example FX-graph output: [haoyang_logs/fx_graph_dump.log](../haoyang_logs/fx_graph_dump.log)
- Example LoopLevel-IR output: [haoyang_logs/loop_ir_dump.log](../haoyang_logs/loop_ir_dump.log)
- Coarse-tiled reduction IR (fill + K×(reduce,combine)):
  [haoyang_logs/coarse_tile_sum.log](../haoyang_logs/coarse_tile_sum.log)

```bash
SPYRE_DUMP_IR=1 BENCH_OP=gelu BENCH_ROWS=512 BENCH_COLS=1024 \
    python docs/source/user_guide/examples/profile_ops.py
```

## 2. Per-kernel device time via the PyTorch profiler

The golden measurement is the **`torch.profiler` "Self SPYRE" per-kernel device
time** of each `sdsc_fused_*` kernel (needs the kineto-spyre wheel: [docs/source/user_guide/profiling/pytorch_profiler.md](../docs/source/user_guide/profiling/pytorch_profiler.md)).
The separate **"Memset (Device)"** event is non-deterministic, size-scaling
host/setup overhead — reported, but NOT kernel time.

Our `profile_ops.py`
emits a parseable `SUMMARY … kernel_us=… pred_us=… bw_gbps=…` line per run;
[profile_test.py](../profile_test.py) is a minimal standalone demo that prints the
`sdsc_fused` kernel time next to the cost-model prediction.

- Example profiled output (kernel time + prediction per op):
  [haoyang_logs/grand_sweep_20260619_151132.log](../haoyang_logs/grand_sweep_20260619_151132.log)

```bash
# one op: prints the profiler table + a SUMMARY with kernel_us / bw_gbps
BENCH_OP=gelu BENCH_ROWS=2048 BENCH_COLS=4096 \
    python docs/source/user_guide/examples/profile_ops.py
# minimal standalone profiler demo (kernel time vs cost model):
python profile_test.py
```

## 3. Cost model

`SPYRE_DUMP_COST=1` extracts cost features from the after-pre-scheduling LoopLevel
IR and prints, at compile time, the **per-tensor device-layout I/O**
(dims · residency · byte calc · `hbm counted` / `xL` loop factor) and the
**step-by-step prediction**. The full model, per fused kernel (each term is 0 / 1
when it does not apply; `R`, `W` = HBM bytes read / written):

```
T   = compute + mem − γ·min(compute, mem)          mem = HBM / (eff · s_lx)

  HBM     = [ (R + W)/BW + α·min(R, W) ] · (n-ary derate) + spill + write_extra
  compute = MACs / cores / (peak · pt_eff)
  s_lx    = min(1, (512KB / ws)^0.15)   for a coarse-tiled kernel with ws > 512KB   (else 1)
```

| term | form | what it is |
|---|---|---|
| `(R+W)/BW` | `BW≈150 GB/s`; `BW_red(ROWS)` for row-reductions; per-op `BW_eff` for transport / broadcast | memory bandwidth (LX-resident and broadcast operands counted once / ~free) |
| `α·min(R,W)` | `α≈0.0057 ns/B` | read↔write bus **turnaround** (0 for one-directional traffic) |
| n-ary derate | `× (1 + 0.075·(n_ops−1))` | multi-pass pointwise chain (`add3`/`add4`) |
| `spill` | `(A+B)·min(1.5, max(0, 0.45·log₂(area/65536)))`, `area=(M/m)·(N/n)` | matmul operand **re-read** when the per-core output tile overflows on-chip capacity |
| `write_extra` | `2.148e-7·ROWS^1.6·COLS^2.2` | `write` outer-product (empirical) |
| `compute` | `MACs/cores/(peak·pt_eff)`, `peak≈1140 MAC/ns/core` | **matmul** only (else 0) |
| `pt_eff` | `min(1, (rows/64)^0.35)` | systolic-array fill (per-core rows) |
| `eff` | `min(0.95, (h/13)^0.68)`, `h = ROWS/(cores·tiles)` | coarse-tiling streaming-pipeline fill (memory-bound) |
| `s_lx` | `min(1, (512KB/ws)^0.15)` for `ws > 512KB`, `ws = 2·(rows/core)·COLS·2B` | coarse-tiled kernel whose per-core working set overflows LX (spilled traffic runs slower) |
| `γ·min(compute,HBM)` | `γ ≈ 0.46` | compute/HBM **overlap** (0 when `compute=0`) |

The **full derivation of every term and its accuracy** are written up in
**[cost_model_report.md](cost_model_report.md)**.

```bash
SPYRE_DUMP_COST=1 BENCH_OP=gelu BENCH_ROWS=2048 BENCH_COLS=4096 \
    python docs/source/user_guide/examples/profile_ops.py
```
