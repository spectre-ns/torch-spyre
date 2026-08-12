# How we measure per-op device time

Three methods, in order of fidelity. The **GOLDEN** one is the **PyTorch profiler**'s
per-kernel device time (`sdsc_fused_*` "Self SPYRE"), which *isolates* the kernel from a
separate, non-deterministic "Memset (Device)" host/setup overhead. The two earlier ones
are a custom per-launch device-sync (`SPYRE_PROFILE_SYNC`, which bundled kernel + that
overhead together) and an end-to-end `.cpu()` drain. (Companion to `cost_model_design.md`.)

## At a glance

| | **Golden** — `torch.profiler` | Custom sync — `measure_device` | Old — `measure_latency` |
|---|---|---|---|
| what is timed | per-kernel **device** time, Memset separated | one kernel + ~7µs host residue | whole compiled call (host+device+D2H) |
| sync point | profiler (kineto-spyre) | `_C.synchronize()` per launch | `.cpu()` (D2H copy) per sample |
| granularity | per `sdsc_fused_*` kernel | per SDSC bundle (= per op) | end-to-end |
| enabled by | `ProfilerActivity.PrivateUse1` | `SPYRE_PROFILE_SYNC=1` | always (pre-merge) |

All report the **min over many samples**: Spyre is static-dataflow ⇒ device latency is
deterministic; host/OS jitter only *adds* time, so the minimum is the true latency. The
custom `SPYRE_PROFILE_SYNC` min measured *kernel + the Memset bucket together* (its
~20µs "fixed" term was that overhead, NOT kernel cost) — which is why we re-anchored the
cost model on the profiler's clean per-kernel time.

## Golden method — PyTorch profiler (`sdsc_fused` kernel time)

`torch.profiler` with `ProfilerActivity.PrivateUse1` (kineto-spyre wheel; see
[../docs/source/user_guide/profiling/pytorch_profiler.md](../docs/source/user_guide/profiling/pytorch_profiler.md))
reports a **"Self SPYRE"** device time per event. The per-kernel `sdsc_fused_*` rows are
the true kernel latency; the separate **"Memset (Device)"** row is the host/setup
overhead and is NOT kernel cost. `examples/profile_ops.py` and `profile_test.py` sum the
`sdsc_fused_*` Self-SPYRE times into `kernel_us` and report Memset separately:

```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
    compiled(*args).cpu()
for ev in prof.key_averages():
    if "sdsc_fused" in ev.key: kernel += ev.self_device_time_total  # golden kernel time
    elif "Memset" in ev.key:   memset += ev.self_device_time_total  # host/setup overhead
```

## Earlier custom method — per-kernel device sync (`SPYRE_PROFILE_SYNC`)

> Superseded by the profiler for cost-model calibration: this min bundles the kernel with
> the non-deterministic Memset/host-setup overhead. Still handy without the kineto-spyre
> wheel, and for the at-exit per-kernel report.

A `perf_counter_ns()` timer brackets the device launch inside `SpyreSDSCKernelRunner.run`
([kernel_runner.py:44](../torch_spyre/execution/kernel_runner.py#L44)); with
`SPYRE_PROFILE_SYNC=1`, `kernel_timer`
([profiling.py:96](../torch_spyre/execution/profiling.py#L96)) calls `_C.synchronize()` in its
`finally`, *before* stopping the clock:

```python
with kernel_timer(self.kernel_name):       # kernel_runner.py
    launch_kernel(self.code_dir, args)     # async launch
# kernel_timer:  start = perf_counter_ns(); yield (launch runs);
#                if sync: _device_synchronize(); elapsed = perf_counter_ns() - start
```

- **Why the sync:** the launch (`executeProgramAsync`,
  [spyre_stream.cpp:217](../torch_spyre/csrc/spyre_stream.cpp#L217)) is asynchronous and
  returns before the device finishes. The sync (`handle->synchronize()`,
  [spyre_stream.cpp:128](../torch_spyre/csrc/spyre_stream.cpp#L128)) makes the bracketed region
  include the device execution. Without it, the timer sees only the ~7 µs host dispatch.
- **Bracketed region:** `[host dispatch ~7µs] + [device kernel] + [host sync return]` →
  device time + a small (~7 µs) op-independent host residue.
- **Executions per measurement = 1.** Each launch self-syncs, so one execution is already one
  clean device-latency sample — no amortization needed.
- **Driver** `measure_device` ([bench.py:173](../torch_spyre/execution/bench.py#L173)):
  `warmup` launches → `profiling.reset()` → `runs` (=100) launches → report the per-kernel
  **`min_ns`**.
- **Granularity:** one `run()` = one SDSC bundle. Single-op example → that op; a fused
  multi-op kernel (e.g. softmax = 5 ops in `sdsc_fused__softmax_0`) → one number for the whole
  bundle.

## Old method — `.cpu()` end-to-end sync

Before the merge added `_C.synchronize()`, there was **no Python-callable device wait** (the
c10 stream-sync hooks were no-op stubs and `elapsedTime` returned `0`). The only way to force
completion was to **copy a dependent output to host with `.cpu()`** — the runtime serializes
that device→host (D2H) transfer, so it can't return until the data is ready.

`measure_latency` ([bench.py:116](../torch_spyre/execution/bench.py#L116)) timed the **whole
compiled call** and ended each sample with `.cpu()` (`_default_sync`,
[bench.py:38](../torch_spyre/execution/bench.py#L38)). Because one `.cpu()` (full drain +
~1 MB D2H ≈ 1 ms) dwarfs a µs-scale kernel, it fired **`inner` (=400) launches per sample**,
synced **once**, and divided by `inner`.

- **Executions per measurement = `inner` (=400)** — to amortize the costly `.cpu()`.
- **Limitations:** times end-to-end, not the kernel; the ~70 µs host-per-call floor hid
  small-op device compute (small ops gave no usable signal); drains only because the *output*
  is copied, so no per-kernel number for an internal kernel.

`.cpu()` remains `measure_latency`'s default — it's the right tool for true **user-visible**
latency (which legitimately includes the D2H transfer). `bench.device_sync`
([bench.py:52](../torch_spyre/execution/bench.py#L52)) is the no-copy `_C.synchronize()`
alternative.

## Env vars

| var | effect |
|---|---|
| `SPYRE_PROFILE=1` | enable the per-kernel timer + the at-exit report |
| `SPYRE_PROFILE_SYNC=1` | sync the device after each launch → per-kernel **device** latency |
| `SPYRE_PROFILE_FILE=path` | write the report to a file instead of stderr |

Without `SPYRE_PROFILE_SYNC`, the same timer reports host **dispatch** latency (~7 µs) only.
