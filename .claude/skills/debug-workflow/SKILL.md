---
name: debug-workflow
description: "End-to-end methodology for debugging torch-spyre: reproduce, set up a worktree env, isolate the failure mode, instrument the failure site, spike-and-revert a fix, and decide whether to file an issue. Use when triaging test failures, a bug, or unexpected behavior on the Spyre backend. For pipeline-stage error lookup, use debug-compilation."
---

# Debugging torch-spyre

This is the **methodology** skill: how to drive a debugging session in this
repo from a failing test or bug report to a verified root cause. It captures
hard-won, non-obvious facts about the build, the hardware, and the test
harness that are easy to get wrong.

**Related skills:**

- `debug-compilation` — maps a `torch.compile` error to the pipeline stage
  that produced it (8 stages, file-by-file) plus an error → cause → fix table
  in its `common-errors.md`. Reach for it once you know the failure is in
  compilation.
- `project-overview` — repo layout, compilation pipeline, key abstractions.
- `write-spyre-op-test` — the compiled-path test framework.

---

## The loop

1. **Reproduce** with an exact command and the right knobs (`SENCORES`,
   `LX_PLANNING`).
2. **Isolate the failure mode** — almost every Spyre failure falls into one of
   a few canonical buckets (below). Bucketing first tells you whether this is a
   Python-layout bug, a codegen bug, or a hardware-compiler limit.
3. **Instrument the failure site** — print the ground-truth layouts/deps at the
   point that raised, don't reason from the source alone.
4. **Spike a fix, then check the numbers** — a fix that compiles is not a fix.
   Always verify against CPU. Revert spikes; keep the tree clean.
5. **Decide the outcome** — real fix, or file/correct a GitHub issue with the
   triage format.

---

## 1. Reproduce

### Knobs that change behavior

| Knob | Why it matters when debugging |
|---|---|
| `SENCORES=1` | Single core. Bypasses multi-core work division — **isolates** layout/codegen bugs from core-division bugs. Many bugs only repro at `SENCORES=32`; many others *only* repro at 1. Try both. |
| `SENCORES=32` | Default. Exercises `core_division.py`. |
| `LX_PLANNING=1` | Enables LX scratchpad planning (`scratchpad.py`). A large class of failures is LX-planning-only — confirm whether the bug survives with it off. |
| `TORCH_SPYRE_DEBUG=1` | C++ debug logging **and an `-O0` rebuild** (slow build, needed for C++-level bugs/segfaults). |

Reproduce a single failing test minimally first:

```bash
python3 -m pytest tests/inductor/test_inductor_ops.py -k <test_name> -v -p no:cacheprovider
```

`-p no:cacheprovider` is standard here — the test harness and device state
make pytest's cache misleading across runs.

### The hardware is a single exclusive device

There is **one** Spyre VFIO device (`/dev/vfio/41`, `AIU_WORLD_SIZE=1`). The
runtime holds it for the whole process lifetime. Consequences:

- **No naive parallelism.** Two processes (or `pytest -n` / xdist) that both
  open the device get `Device or resource busy`, and the second silently
  produces empty output. Run test groups serially, or serialize device access
  with an flock around `_C.start_runtime` (see `run_lx_parallel.py` /
  `spyre_device_lock.py` at repo root for the working pattern).
- **Long runs go in the background** with a log file, and you poll it — a full
  suite is ~25-55 min:

  ```bash
  TEST_LX_PLANNING_RUN_SKIPS=1 TEST_LX_PLANNING_FULL=1 \
    python3 -m pytest tests/inductor/test_inductor_ops_lx_planning.py -vvv \
    -p no:cacheprovider > /tmp/run.log 2>&1 &
  ```

---

## 2. Worktree env setup (REQUIRED before tests run in a worktree)

This repo is **pip-installed from the primary working directory**. In a fresh
worktree, `import torch_spyre` resolves to the worktree's source (cwd shadows
the install), but the worktree has **no built `_C.so` / `_hooks.so`** and **no
`torch_spyre.egg-info`** — and the egg-info is what provides torch's autoload
entry point. Without it the `"spyre"` device never registers and every test
errors at import/device creation.

If the **native C++ sources are identical** between the worktree and the
installed tree (i.e. you only changed Python), symlink the prebuilt artifacts
from the installed tree instead of rebuilding:

```bash
# from inside the worktree
cd <worktree>/torch_spyre
ln -sf /home/dahubley/torch-spyre/torch_spyre/_C.so     _C.so
ln -sf /home/dahubley/torch-spyre/torch_spyre/_hooks.so _hooks.so
cd .. && ln -sfn /home/dahubley/torch-spyre/torch_spyre.egg-info torch_spyre.egg-info
```

`_C.so` / `_hooks.so` are gitignored; `egg-info` is untracked. Verify:

```bash
python3 -c "import torch, torch_spyre; print(torch.device('spyre'))"
```

If you **changed C++**, you must actually rebuild (`pip install -e .`) — the
symlink shortcut will run stale native code and mislead you.

---

## 3. Isolate the failure mode

Most compiled-path failures fall into these buckets. Bucket *before* you dig —
it tells you which layer owns the bug.

| Bucket | Signature | Owner / meaning |
|---|---|---|
| **A — numerical** | `assert_close` mismatch; compiles & runs fine | fp16 precision, or genuinely wrong codegen. Distinguish by magnitude: tiny → tolerance; structured (zeros, only first span populated) → wrong codegen. |
| **B — stick-expr guard** | `Unsupported: Unexpected stick expression Mod(dX, N)` at `pass_utils.py` `_check_stick_expr_supported` (N ≠ 64) | A view/reshape merged or split the **stick (innermost) dim** so traversal lands at a partial span inside a 64-padded stick. The guard is **correct** — it prevents silently-wrong codegen. Don't "fix" it by relaxing the guard. |
| **C — `dxp_standalone` SIGABRT** | backend compiler crashes (`SIGABRT`), often `fused_*` kernel | A genuine **hardware-compiler limit** (e.g. partial-stick in-kernel reads). Not a Python bug. |
| **D — missing op / lowering** | `AttributeError` on `SpyreOpFuncs`, `Unsupported("...")`, missing lowering | Use `add-spyre-operation`. |
| **E — layout propagation** | `"does not have FixedTiledLayout"`, `dim_map` errors | Stickify pass couldn't tile a node. See `debug-compilation` Stage 4. |

The key discriminator between B and C: B is a **propagation-time guard** that
fails fast and cleanly; C is **runtime/backend** and crashes. Relaxing a B
guard frequently converts the case into a C crash or a silently-wrong A — which
is exactly the evidence that the underlying support is genuinely missing.

---

## 4. Instrument the failure site

Reason from **ground truth**, not from the source. At the line that raised,
print the actual layouts and dependency indices. For layout/stick bugs, capture
for each relevant arg:

- the committed `SpyreTensorLayout`: `device_size`, `stride_map`,
  `elems_per_stick`
- the consumer's read: `dep.index` and its ranges (`{d0:.., d1:..}`)
- the host read coordinates and **which dim is the stick**, and whether that
  stick coordinate is full (`Mod(var, 64)`) or **partial** (`Mod(var, N<64)`)

A worked example of exactly this — flatten merging a size-4 stick dim — lives
in the project memory (`flatten-ground-truth-stls`); use it as a template for
what to capture and how to write it up.

Inductor / Spyre logging:

```bash
export SPYRE_INDUCTOR_LOG=1                 # enable torch_spyre._inductor loggers
export SPYRE_INDUCTOR_LOG_LEVEL=DEBUG       # ERROR|WARNING|INFO|DEBUG
export SPYRE_LOG_FILE=/tmp/spyre_inductor.log   # default: stderr
export TORCH_LOGS="+inductor"               # PyTorch Inductor internals
export TORCH_COMPILE_DEBUG=1                # dumps graphs/JSON to torch_compile_debug/
```

For C++ / runtime / segfault bugs, `TORCH_SPYRE_DEBUG=1` (also forces `-O0`)
and `DT_DEEPRT_VERBOSE=-1` to cut runtime noise.

---

## 5. Spike a fix — then check the numbers

The single most important discipline in this repo:

> **A change that compiles is not a fix. Verify against CPU.**

Relaxing a guard or widening a constraint often makes a case *compile* while
producing **wrong numbers** (e.g. only the first partial span populated, the
rest zeroed). That is a regression dressed as a fix. Always run the test and
compare to CPU (`compare_with_cpu`), and look at the *structure* of any
mismatch — structured zeros mean wrong codegen, not tolerance.

When a spike proves a direction is unviable (e.g. the materialize-via-restickify
route for partial-stick reads still SIGABRTs because the inserted intermediate
copy is itself an in-kernel SDSC op), that **negative result is the finding** —
record it and revert. **Leave the tree clean**; spikes are throwaway.

---

## 6. Decide the outcome

- **Real fix** — implement, add/adjust a test, run pre-commit
  (`pre-commit run --all-files`), sign off (`git commit -s`).
- **Unviable in Python / needs deeper work** — when the root cause is a
  hardware-compiler or runtime limit (bucket C), say so explicitly and propose
  the heavier directions rather than forcing a layout hack.
- **File or correct a GitHub issue** — when triaging a batch of failures,
  summarize them. Mirror the established issue format (see issue #2062): title
  `... Failure Summary - YYYY-MM-DD`; `Totals: X failed, Y passed, Z xfailed`;
  a "Summary by Category" table (#|Category|Count|%|Symptom); per-category
  tables of `Test (base name) | Class`; then Patterns and Suggested Remedies,
  numbered by priority. **Backtick exact error messages and `file:line`.** If
  an existing issue's stated root cause doesn't match what you actually
  observed, correct it — don't inherit a wrong framing.

---

## Quick reference

```bash
# minimal repro of one test
python3 -m pytest tests/inductor/test_inductor_ops.py -k <name> -v -p no:cacheprovider

# isolate from multi-core
SENCORES=1 python3 -m pytest ... -k <name> -v -p no:cacheprovider

# full debug logging
SPYRE_INDUCTOR_LOG=1 SPYRE_INDUCTOR_LOG_LEVEL=DEBUG TORCH_LOGS="+inductor" \
  TORCH_COMPILE_DEBUG=1 python3 -m pytest ... -k <name> -v -p no:cacheprovider

# C++/runtime/segfault (forces -O0 rebuild)
TORCH_SPYRE_DEBUG=1 DT_DEEPRT_VERBOSE=-1 python3 -m pytest ... -k <name> -v
```
