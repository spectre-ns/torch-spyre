# Torch-Spyre Profiling Infrastructure Analysis

**Date:** 2026-06-11  
**Analyzed Files:**
- `examples/bench_softmax.py`
- `torch_spyre/execution/profiling.py`
- `torch_spyre/execution/bench.py`
- `torch_spyre/execution/kernel_runner.py`
- `torch_spyre/profiler/__init__.py`

---

## Overview

Torch-spyre has **two separate profiling/benchmarking systems** that serve different purposes:

1. **Custom Kernel Profiling** (`SPYRE_PROFILE=1`) - Host-side dispatch timing
2. **End-to-End Benchmarking** (`bench.py`) - Wall-clock latency measurement

---

## System 1: Custom Kernel Profiling (`SPYRE_PROFILE=1`)

### Purpose
Per-kernel host-side dispatch timing with automatic reporting at process exit.

### Location
- Implementation: [`torch_spyre/execution/profiling.py`](torch_spyre/execution/profiling.py)
- Integration point: [`torch_spyre/execution/kernel_runner.py:46`](torch_spyre/execution/kernel_runner.py:46)

### How It Works

```python
# In SpyreSDSCKernelRunner.run()
with kernel_timer(self.kernel_name):
    if self.jobplan:
        launch_jobplan(self.jobplan, args)
    else:
        launch_kernel(self.code_dir, args)
```

**Flow:**
1. `kernel_timer(name)` context manager wraps each kernel launch
2. Records: `count`, `total_ns`, `min_ns`, `max_ns` per kernel name
3. At process exit (`atexit.register(report)`), prints table to stderr or `SPYRE_PROFILE_FILE`

### Environment Variables
- `SPYRE_PROFILE=1` - Enable profiling (default: off)
- `SPYRE_PROFILE_FILE=path` - Output file (default: stderr)

### Output Format
```
==== SPYRE_PROFILE: per-kernel host launch timing (min strips host jitter) ====
kernel                                     count     min_us     mean_us      max_us
sdsc_fused__softmax_0                        100      72.450      75.123      89.234
```

### Key Characteristics
✅ **Zero overhead when disabled** - Single env check, no-op context manager  
✅ **Automatic reporting** - No code changes needed  
✅ **Per-kernel granularity** - Tracks each compiled kernel separately  
⚠️ **Host-side only** - Measures dispatch latency, not device execution time

---

## System 2: End-to-End Benchmarking (`bench.py`)

### Purpose
Accurate wall-clock measurement of complete workloads with device synchronization.

### Location
- Implementation: [`torch_spyre/execution/bench.py`](torch_spyre/execution/bench.py)
- Example usage: [`examples/bench_softmax.py`](examples/bench_softmax.py)

### How It Works

```python
from torch_spyre.execution.bench import measure_latency

stats = measure_latency(
    lambda: compiled_sm(x),
    runs=100,
    warmup=3,
    inner=400,  # Amortize sync overhead
    label="softmax[4096x4096]"
)
print(stats)  # min=X.XXXus median=X.XXXus max=X.XXXus spread=X.X%
```

**Flow:**
1. Warmup iterations (absorb compilation cost)
2. For each run:
   - Execute `inner` launches back-to-back
   - Force sync with `.cpu()` (or custom sync function)
   - Record total time, divide by `inner`
3. Return `LatencyStats` with min/median/max/spread

### Key Features

#### Inner Loop Amortization
```python
inner=400  # Launch 400 times, sync once
# Amortizes the ~1ms .cpu() sync overhead
# Reveals per-launch device cost
```

**Why needed:** Device launch is async (`executeProgramAsync`), but `.cpu()` sync costs ~1ms. Single launches are sync-dominated. Multiple launches + one sync amortizes this floor.

#### Determinism Checking
```python
spread = (max - min) / min
# Should be <5% for deterministic device
# >5% indicates host jitter or non-deterministic behavior
```

#### Baseline Subtraction
```python
softmax_stats = measure_latency(lambda: compiled_sm(x), ...)
baseline_stats = measure_latency(lambda: compiled_id(x), ...)  # identity op
net_compute = net_latency_us(softmax_stats, baseline_stats)
# Removes fixed launch + transfer overhead
```

### Environment Variables (Example Usage)
```bash
BENCH_RUNS=100        # Number of samples (default: 100)
BENCH_WARMUP=3        # Warmup iterations (default: 3)
BENCH_INNER=400       # Launches per sample (default: 400)
BENCH_ROWS=4096       # Problem size
BENCH_COLS=4096
```

### Output Format
```
softmax[4096x4096]: min=245.123us median=248.456us max=252.789us  spread=3.1%  [deterministic]  (n=100 inner=400)
identity[4096x4096]: min=72.450us median=73.123us max=75.234us  spread=3.8%  [deterministic]  (n=100 inner=400)
net softmax compute (min - baseline_min): 172.673 us
```

---

## System Comparison

| Feature | `SPYRE_PROFILE=1` | `bench.measure_latency` |
|---------|-------------------|-------------------------|
| **Activation** | Environment variable | Explicit API call |
| **Granularity** | Per-kernel (automatic) | Per-workload (manual) |
| **Synchronization** | None (async dispatch) | Forced (`.cpu()`) |
| **Measures** | Host dispatch latency | End-to-end wall-clock |
| **Overhead** | Zero when disabled | Always present |
| **Output** | Automatic at exit | Returned object |
| **Use Case** | Quick kernel inventory | Accurate benchmarking |
| **Device Time** | ❌ No (unless sync happens) | ✅ Yes (forced sync) |

---

## Critical Caveat: Asynchronous Execution

### The Problem

From [`profiling.py:22-29`](torch_spyre/execution/profiling.py:22-29):

> **CAVEAT:** device execution is asynchronous (`executeProgramAsync`) and the
> stream-sync hooks are no-ops, so this measures host-side *dispatch* latency
> unless a synchronization (e.g. copying a dependent output to host) forces
> device completion.

**What this means:**
- `SPYRE_PROFILE=1` times **only** the host-side kernel launch call
- Device execution happens asynchronously in the background
- Reported times are **NOT** device execution times unless:
  1. A sync (like `.cpu()`) happens to occur, OR
  2. The runtime is effectively synchronous (time scales with problem size)

### When Each System Gives True Device Time

#### `SPYRE_PROFILE=1` ✅ True device time IF:
- The workload naturally syncs (e.g., output is immediately copied to host)
- The runtime is synchronous (time scales with problem size = device is blocking)
- **Probe:** If per-kernel time scales with problem size, it's device-inclusive

#### `bench.measure_latency` ✅ Always true device time:
- Forces sync with `.cpu()` at the end of each timed region
- Guaranteed to include device execution
- **Recommended for accurate benchmarking**

---

## Integration with PyTorch Profiler

### Current State: Minimal Integration

**Only one integration point found:**
```python
# torch_spyre/execution/async_compile.py:62
with torch.profiler.record_function(f"dxp_standalone:{kernel_name}"):
    subprocess.run(["dxp_standalone", "--bundle", "-d", output_dir], check=True)
```

This records **compilation time** (the `dxp_standalone` subprocess), not kernel execution.

### Missing: Runtime Kernel Profiling

The `torch_spyre/profiler/__init__.py` module exists but is **not implemented**:

```python
def is_available() -> bool:
    # more to be implemented later
    return False

__all__: list[str] = []
```

**Implication:** Torch-spyre kernels do **NOT** appear in PyTorch profiler traces (e.g., `torch.profiler.profile()` with `ProfilerActivity.CPU`/`CUDA`).

---

## Identified Issues

### Issue 1: `SPYRE_PROFILE=1` Misleading Name

**Problem:** The name suggests it profiles device execution, but it only measures host dispatch.

**Evidence:**
- Documentation explicitly warns about async execution
- No synchronization in `kernel_timer` context manager
- Only wraps `launch_kernel`/`launch_jobplan` calls

**Impact:** Users may misinterpret the numbers as device execution time.

**Recommendation:**
- Rename to `SPYRE_DISPATCH_PROFILE` or `SPYRE_HOST_PROFILE`
- Update documentation to emphasize "host dispatch timing"
- Add a note in the output table: `[host dispatch only; see bench.py for device-inclusive timing]`

### Issue 2: No PyTorch Profiler Integration

**Problem:** Spyre kernels are invisible to `torch.profiler.profile()`.

**Evidence:**
- `profiler.is_available()` returns `False`
- No `record_function` calls around kernel execution
- Only compilation is profiled, not runtime

**Impact:**
- Cannot use standard PyTorch profiling tools
- Cannot see Spyre kernels in Chrome traces
- Cannot compare Spyre vs CUDA performance in unified view

**Recommendation:**
Implement proper PyTorch profiler integration:

```python
# In SpyreSDSCKernelRunner.run()
with torch.profiler.record_function(f"spyre_kernel:{self.kernel_name}"):
    with kernel_timer(self.kernel_name):  # Keep custom profiling
        if self.jobplan:
            launch_jobplan(self.jobplan, args)
        else:
            launch_kernel(self.code_dir, args)
```

This would:
- ✅ Make kernels visible in PyTorch profiler traces
- ✅ Allow Chrome trace visualization
- ✅ Enable comparison with CUDA/CPU ops
- ✅ Keep existing `SPYRE_PROFILE=1` functionality

### Issue 3: No Device-Side Profiling

**Problem:** No way to get per-op device execution time from hardware counters.

**Current workarounds:**
1. Use `bench.measure_latency` with single-kernel programs
2. Disable fusion (`SPYRE_INDUCTOR_ENABLE_FUSION=0`)
3. Subtract baseline

**Limitation:** Cannot profile multi-kernel bundles or get per-op breakdown within a bundle.

**Recommendation:**
- Investigate DeepTools profiling APIs
- Add device-side instrumentation if available
- Document the single-kernel + baseline methodology as best practice

### Issue 4: `inner` Parameter Not Well Documented

**Problem:** The `inner` parameter in `measure_latency` is critical but subtle.

**Why it matters:**
- Single launches are dominated by ~1ms `.cpu()` sync overhead
- `inner=400` amortizes this to reveal per-launch device cost
- Wrong value → misleading results

**Current state:** Documented in docstring but not in example comments.

**Recommendation:**
- Add prominent comment in `bench_softmax.py` explaining `inner`
- Provide guidance on choosing the value (e.g., "increase until spread <5%")
- Consider auto-tuning: run with increasing `inner` until spread stabilizes

---

## Usage Recommendations

### For Quick Kernel Inventory
```bash
SPYRE_PROFILE=1 python my_script.py
# See which kernels launched and how many times
# Useful for: debugging fusion, checking kernel count
```

### For Accurate Benchmarking
```python
from torch_spyre.execution.bench import measure_latency, net_latency_us

# Single-kernel program (or disable fusion)
compiled_fn = torch.compile(my_op)
baseline_fn = torch.compile(lambda x: x + 0.0)

stats = measure_latency(
    lambda: compiled_fn(x),
    runs=100,
    warmup=5,
    inner=400,  # Tune until spread <5%
    label="my_op"
)
baseline = measure_latency(
    lambda: baseline_fn(x),
    runs=100,
    warmup=5,
    inner=400,
    label="baseline"
)

print(stats)
print(f"Net compute: {net_latency_us(stats, baseline):.3f} us")
```

### For Multi-Kernel Profiling
**Current limitation:** No good solution.

**Workaround:**
1. Profile end-to-end with `measure_latency`
2. Use `SPYRE_PROFILE=1` to see kernel breakdown (host-side only)
3. Manually correlate if times scale with problem size

---

## Comparison with CUDA Profiling

| Feature | CUDA (`nsys`, `nvprof`) | Spyre (current) |
|---------|-------------------------|-----------------|
| Device execution time | ✅ Hardware counters | ❌ No device profiling |
| Per-kernel breakdown | ✅ Automatic | ⚠️ Host dispatch only |
| PyTorch profiler integration | ✅ Full support | ❌ Not implemented |
| Chrome trace visualization | ✅ Yes | ❌ No |
| Async execution handling | ✅ Automatic sync | ⚠️ Manual `.cpu()` |
| Multi-kernel bundles | ✅ Per-kernel | ❌ Bundle-level only |

---

## Proposed Improvements

### Short Term (Low Effort)
1. ✅ Rename `SPYRE_PROFILE` → `SPYRE_DISPATCH_PROFILE`
2. ✅ Add `record_function` calls around kernel launches
3. ✅ Document `inner` parameter prominently in examples
4. ✅ Add warning in profile output about host-side timing

### Medium Term (Moderate Effort)
1. Implement `torch_spyre.profiler` module
2. Add device-side profiling if DeepTools supports it
3. Auto-tune `inner` parameter in `measure_latency`
4. Create unified profiling guide

### Long Term (High Effort)
1. Full PyTorch profiler integration with custom backend
2. Chrome trace export
3. Per-op device timing within bundles
4. Integration with PyTorch Profiler TensorBoard plugin

---

## Summary

Torch-spyre has **functional but limited** profiling:

✅ **What works:**
- Host-side dispatch timing (`SPYRE_PROFILE=1`)
- End-to-end benchmarking with forced sync (`bench.py`)
- Determinism checking via spread metric

⚠️ **What's misleading:**
- `SPYRE_PROFILE=1` name suggests device profiling but only measures host dispatch
- No clear documentation of async execution caveat

❌ **What's missing:**
- PyTorch profiler integration (kernels invisible in traces)
- Device-side execution profiling
- Per-op breakdown within multi-kernel bundles
- Chrome trace visualization

**Recommendation:** The current system is adequate for basic benchmarking but needs:
1. Better naming/documentation to avoid confusion
2. PyTorch profiler integration for ecosystem compatibility
3. Device-side profiling for accurate per-op timing

---

**Analysis Completed:** 2026-06-11  
**Ready for discussion of identified issues**