# DRAM read/write bandwidth: why does mixed R+W cap at ~98 GB/s?

## The observation (established)

Latency-inferred asymptotic bandwidth, fp16, 32 cores (Rung 8, slope of the two
largest sizes):

| workload | what it does | asymptote | % of 204.8 peak |
|---|---|---|---|
| `read` | `x.sum(dim=-1)` — read-only | ~172 GB/s | 84% |
| `write` | `b[1,N]+c[N,1]` — write-only (broadcast inputs cached) | ~146 GB/s | 71% |
| `copy` | `x+1.0` — balanced 1 read + 1 write | **~98 GB/s** | 48% |

Both *unidirectional* streams run near peak. Doing both at once collapses to ~98 —
**below even the one-way read rate.** So the binding constraint is running reads and
writes **together**, not the write rate (writes alone are fast). Every elementwise op
reads inputs and writes an output, so all pointwise ops pay this; that is why the
practical ceiling is ~100-110 while the compiler's matmul model assumes the full
204.8 (`_HBM_BW_GBS`, reachable only by unidirectional streaming).

## The mechanism is NOT yet established

The fact above is solid; the *cause* is not. Candidates:

1. **DRAM read/write bus turnaround** — the bidirectional data bus wastes cycles
   switching read↔write (tWTR/tRTW). Frequent switching ≈ halves effective BW.
2. **Burst-size / scratchpad pressure** — `copy` stages BOTH an input and an output
   tile per core; `read`/`write` stage only one. Half the footprint per direction →
   smaller DMA bursts → fewer bytes per "limited" memory request
   (see `tensors_and_layouts.md`: contiguous layout required for full BW).
3. **Half-duplex link** — reads and writes share one physical path.

Confounds in the table: the three are *different kernels* (`read` is a reduction with
combine overhead, so 172 is a lower bound on pure-read BW; access patterns differ), so
"copy < serialized(read,write)=158" is suggestive, not rigorous.

## Experiments to discriminate

### A. Latency side (automated, in `run_cost_model_plan.sh`)

- **Rung 9 — tile-footprint / burst-size control.** `copy[512x16384]`, sweep
  `SENCORES` (1→32). Fewer cores → bigger per-core tile → bigger bursts. **BW rises
  with tile size ⇒ burst/scratchpad-limited (#2); flat ⇒ intrinsic to mixing (#1/#3).**
- **Rung 10 — write-fraction V-curve.** Same single read of `x`, add writes:
  `copy`(0.50) → `w2`(0.67) → `w3`(0.75); with `read`(0.0) and `write`(1.0) from
  Rung 8. **A V-shaped dip with the minimum near balanced 1:1 is the turnaround
  signature (#1); a flat/monotonic curve disfavors it.** Valid only if `w2`/`w3` fuse
  to one kernel (1 read + N writes) — check the kernel count + `SPYRE_DUMP_COST`.

Latency alone cannot fully isolate turnaround (we don't control the memory schedule,
and ops lower to structurally different kernels), so these *corroborate* but do not
settle it. The decisive measurement is B.

### B. Decisive: actual DDR bandwidth + bus utilization via `aiu-smi`

`aiu-smi` reports the real DDR controller throughput (and, depending on build,
separate read/write counters + utilization) independent of our latency model. The
tell:

- **`copy` shows ~98 GB/s total with the bus < 100% utilized (idle gaps)** ⇒ the link
  cannot stay busy ⇒ **turnaround / half-duplex (#1/#3)**. Quantify: turnaround tax ≈
  `1 − utilization`, i.e. the achievable-if-no-idle BW is `98 / utilization`.
- **`copy` shows ~98 GB/s at ~100% utilization** ⇒ a real *throughput* limit for mixed
  traffic in the controller (not idle from switching).
- Cross-check: `read` should show ~all-read traffic (~172, ~0 write), `write` ~all-write
  (~146). In `copy`, read-BW + write-BW summing to ~98 (≈49+49) while the bus can do
  172 one-way is the bidirectional penalty made explicit.

#### Setup (run machine, PF mode — see `user_guide/profiling/device_monitoring.md`)

```bash
# one-time: install the monitor (PF mode only)
uv pip install <ibm_aiu_monitor wheel from device_monitoring.md>
uv pip install psutil
```

#### Two-terminal recipe (repeat for op = copy, read, write)

```bash
# Terminal 1 — workload (sustained window so aiu-smi can sample):
export DTCOMPILER_KEEP_EXPORT=true
export SENLIB_DEVEL_CONFIG_FILE=<venv>/etc/senlib_config_aiusmi.json
BENCH_BW_OP=copy BENCH_COLS=65536 BENCH_BW_SUSTAIN_S=30 python examples/bench_bandwidth.py
#   -> prints "SUSTAINED I/O START ... END" around a ~30s saturated window,
#      plus a host-side BW cross-check.

# Terminal 2 — monitor (start when you see "SUSTAINED I/O START"):
export DEEPRT_EXPORT_DIR=<workload-directory>
aiu-smi            # or: aiu-smi dmon   (read the DDR bandwidth + utilization channels)
```

Record, per op: DDR read GB/s, DDR write GB/s, total GB/s, and bus utilization %.
That table settles which mechanism (and how big the turnaround tax is).

## Recording results

Put findings in `cost_model_design.md` only after B: state the *fact* (balanced R+W ≈
half the one-way rate; pointwise caps ~100 while 204.8 needs unidirectional streaming)
and the *mechanism* with the number from `aiu-smi` — not an assumed cause.
