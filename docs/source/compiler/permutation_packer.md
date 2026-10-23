# The permutation layout packer

`NativePermutationLayoutSolver` is the placement engine the simulated-annealing
co-optimizer (`SaCoOptimizingSolver`, `config.layout_solver = "simulated_annealing"`) composes.
Given a set of buffers, each with a size and a half-open lifetime `[start, end)`, and an
allocation order, it decides where each buffer sits in a fixed-capacity scratchpad, and keeps the
use-weighted total size of the buffers that fit. The search over orders and divisions that drives it
is described in [Joint core-division + LX placement](sa_co_optimization.md).

The code:

- **`torch_spyre/csrc/perm_layout_native.cpp`** — the packer. A *permutation* is an allocation
  order; `NativePermutationLayoutSolver` places each buffer on top of the earlier-placed buffers it
  overlaps in time (with in-place reuse) and re-places under `swap`/`rotate`/`resize`/`set_eligible`.
  Its from-scratch counterpart, `ReferencePermutationBasedLayoutSolver`, lives in
  `tests/inductor/reference_perm_layout.py` and is a test oracle only.
- **`torch_spyre/_inductor/scratchpad/plan_solver.py`** — the shared `LifetimeBoundBuffer` data
  type and the `MemoryPlanSolver` ABC.
- **`torch_spyre/_inductor/scratchpad/first_fit.py`** — `FirstFitLayoutSolver`, which seeds the
  co-optimizer's initial permutation.

**Native packer.** The packer is C++ only. A Python implementation was the original one and
served as the canonical spec until the C++ port took over as the default; it was measured at
roughly 14× slower end-to-end on mid-sized problems at representative capacity (see
[Native packer performance](native_packer_performance.md)) and has since been removed. Because the
`_C` extension is required for torch-spyre to function at all, a missing native packer means a
stale or incomplete build and raises rather than falling back.

**Validation philosophy:** every operation is checked against the from-scratch reference oracle in
`tests/inductor/reference_perm_layout.py` — randomized *differential* tests, a gated *stress* suite
(`TORCH_SPYRE_STRESS_SCRATCHPAD=1`, tens of thousands of seeds), and in places *exhaustive*
enumeration of all small configurations. This is what makes the subtle in-place edge cases
trustworthy. The oracle is deliberately the naive placer: it is slow, but its correctness is
meant to be evident from reading it.

**Key invariants that recur:** lifetimes are half-open; per column, address order equals permutation
order (weakly — ties only for in-place reuse); at most two buffers share one address at one tick
(in-place legality caps it); and an in-place pair overlaps at exactly one "transition" tick.

## How it works

### The placement engine

`NativePermutationLayoutSolver` is the placement engine. Given an allocation order (a permutation of
buffer indices), it places each buffer at `align_up(max top of the earlier-placed buffers it overlaps
in time)`, with an in-place child allowed to reuse a parent's slot. A buffer that would cross the
capacity line is *evicted* — its address is `None`, and eviction is upward-closed, so anything that
would rest on it is evicted too. `quality()` — the use-weighted total size of buffers that fit under
capacity — is maintained as a running sum and read in O(1).

Placement is a pure function of (permutation, sizes, eligibility, lifetimes), so every mutating
operation simply re-places every buffer in permutation order. What makes that affordable is the
static data computed once in the constructor and shared by reference with every `copy()`: the
per-buffer time-overlap sets (so a buffer's candidates are a filter over its overlaps rather than a
scan of all earlier buffers), the sparse in-place-partner sets (so placement never probes an
unrelated candidate for an in-place relationship), and the lifetime-interval decomposition.

Two things ride on that interval decomposition. A buffer with no in-place partner reads its floor
straight off a per-interval running maximum instead of gathering candidates at all. And the placement
loop carries a **saturation early-stop**: an interval is *done* once it already holds an evicted
buffer or every buffer alive on it has been placed, and once all intervals are done every remaining
buffer must rest (transitively) on an evicted one, so the tail is bulk-evicted. The early-stop is
result-identical to running the loop out; it only changes the work.

The mutators short-circuit where the re-place provably cannot change anything: a `swap` of two
buffers that do not overlap in time, or where either is ineligible (routed to HBM and so outside the
stacking order), moves no address and returns a zero delta after updating the order alone. `rotate`
does the same for an ineligible buffer.

None of this changes the asymptotics: both the packer and the naive placer are quadratic in the
worst case, and the shortcuts buy constants — which is the whole of the win, and enough of it that
porting the genuinely sub-quadratic algorithm the original Python packer used was judged not worth
the roughly 2.5× more C++ it would take (see
[Native packer performance](native_packer_performance.md)).

`ReferencePermutationBasedLayoutSolver` in `tests/inductor/reference_perm_layout.py` is the
obviously-correct counterpart: it scans all earlier-placed overlapping buffers for each placement and
takes none of the shortcuts above. It ships nowhere; it is the differential-test oracle and a
readable reference spec of the placement semantics.

### `rotate`

`rotate(i, j)` takes one permutation entry out and reinserts it at another position. The permutation
and its inverse are edited in one pass and the layout is re-placed, so the cost is independent of
`|i − j|` — which is what the co-optimizer's long reinsertion moves need.
