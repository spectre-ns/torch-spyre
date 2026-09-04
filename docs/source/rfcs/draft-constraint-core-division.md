# A constraint model for core division, without a candidate menu

Written against `enable-default-cooptimization` at `cb5cb166`. Line numbers are
from that commit. Every measurement below was taken this session on the in-tree
capture corpora (`tests/inductor/cooptimization_captures{,_large,_regen}.json`),
no hardware required.

The question this answers: **can the CP-SAT joint solve stop enumerating core
divisions, deciding them from constraints on per-axis split factors instead —
and how much `AddElement` is actually unavoidable?**

Short answer, in three parts:

1. `AddElement` is unavoidable **nowhere**. Every table in the model is a unary
   function of one small-domain variable, and every such function becomes a
   *linear expression* once that variable is one-hot encoded. That is strictly
   better than a table, because the objective's LP relaxation can see through it.
2. `AddMultiplicationEquality` is needed in exactly **one** place — the per-core
   footprint `eff_size x partition = size` — and even there it is avoidable.
   The other bilinear relation (the prefix-product chain that produces core
   counts and core-mapping strides) is better discharged by *half-reified linear*
   equalities under the one-hot literals, because a small explicit domain
   propagates and an `int_prod` does not.
3. De-enumeration is worth doing, but **not for the reason it looks like**.
   Menu cardinality is a mild cost driver; the *encoding* of the menu is a large
   one. Section 4 is a measured, objective-identical 1.6x that keeps the menu.
   Section 5 removes the menu, and its payoff is precompute, guard
   expressiveness and headroom — not the wall-clock the encoding fix already got.

## 1. Where enumeration lives today

Four distinct places, all downstream of one decision: that a core division is an
**opaque index into a materialized list**.

**(a) The menu itself.** `enumerate_work_division_candidates`
(`work_division.py:981-1000`) is a cross product over `factor_domain(v)` for each
divisible axis, filtered by `WorkDivisionContext.is_legal`
(`work_division.py:861-874`). `CoOptimizingAllocator._division_map`
(`allocator.py:2046-2126`) calls it per op and stores the result on
`CoreDivisionBuffer.core_divisions`.

**(b) The index variable and its tables.**
`_CoreDivisionBufferWithCpVars.__post_init__` (`ilp_solver_ortools.py:279-334`)
creates `division` over `[0, len(core_divisions)-1]` and ties four families of
derived quantity to it with `add_element`: one per split axis (line 321), plus
`eff_size`, `cores` and `core_cost` (lines 330-332). The cost-expression path
adds two more per axis — `log2_` via `add_element` and `inv_` via
`AddDivisionEquality` (`_print_Symbol`, lines 549-577).

**(c) The pairwise compatibility table.** `_cd_parent_matches`
(`allocator.py:2522-2558`) materializes, per producer/consumer edge, every
`(parent_idx, consumer_idx)` pair whose `PerCoreView`s agree
(`ResidencyEdge.match_pairs`, `allocator.py:1509-1527`). `_gate_divisions`
(`ilp_solver_ortools.py:149-162`) then spends **one `BoolVar` and two enforced
linear constraints per pair**.

**(d) The pin ladder.** Every guard in `_division_map` (windowed pool,
`keep_by_index` group, fp8 matmul group, offset mutation, indirect access, offset
slice read, CPU/host buffer) is expressed by *handing the solver a one-element
menu*. A restriction on the decision is encoded as surgery on the enumeration.

### What that costs, measured

Model size after `_add_inplace_relaxation` + `_add_core_division`, before any
solve (`cooptimization_captures_regen.json`):

| graph | buffers | menu | match pairs | CP-SAT vars | constraints |
| --- | ---: | ---: | ---: | ---: | ---: |
| softmax | 6 | 108 | 102 | 216 | 387 |
| sdpa | 9 | 180 | 158 | 312 | 552 |
| flash_attention | 44 | 1077 | 674 | 1550 | 2769 |
| block_x4 | 52 | 1068 | 605 | 1433 | 2447 |
| flash_big | 80 | 2299 | 1614 | 3530 | 6463 |

On `flash_big`, **2684 of 3530 variables are anonymous pair literals** — the
`NewBoolVar("")` that `_gate_divisions` creates. Roughly three quarters of the
model is the compatibility table.

Time is dominated by the first lexicographic phase, not by model build
(single worker, `flash_big`): build 18 ms, residency 1413 ms, parallelism 585 ms,
balance 170 ms. The Python-side menu and pair-table construction is *not* visible
in the captures at all, because the captures are dumps taken after it ran.

## 2. The one structural fact that makes de-enumeration possible

`ResidencyEdge.compatible` (`allocator.py:1494-1507`) is

```text
compatible(p, c)  <=>  parent_view(p) is not None
                  and  parent_view(p) == consumer_view(c)
                  and  p.cores_used == c.cores_used
```

`parent_view` depends only on `p`; `consumer_view` only on `c`. So the relation
is, by construction, **an equality of two independently computed keys** — never
an arbitrary bipartite graph. The pair table is the outer product of that
equality, materialized.

Verified against every captured edge:

```text
edges with pairs: 607   rectangular (= key-equality relations): 607
distinct slicing classes per edge: min=1  max=48  mean=10.9
```

All 607, no exceptions. That licenses everything below: the gate can be written
as "these two derived quantities are equal", and the only design question is what
representation of the key propagates best.

## 3. What the key actually decomposes into

`PerCoreView` (`pass_utils.py:2742-2764`) is three fields, and each decomposes
into per-device-dimension scalars:

| field | decomposition |
| --- | --- |
| `work_slice_dims` | `F_d` = split factor landing on device dim `d` (1 if unsplit) |
| `core_to_slot` | `Mod(floor(core_id / sigma_d), F_d)` — so `sigma_d`, the core-mapping stride, plus `F_d` |
| `num_cores` | product of all per-symbol splits = `cores_used` |

Therefore

```text
view_p == view_c   <=>   forall d:  F_p[d] == F_c[d]  and  sigma_p[d] == sigma_c[d]
                    and  cores_p == cores_c
```

Two supporting facts, both checked in the source rather than assumed:

**`sigma` is a prefix product.** `core_to_slice_mapping`
(`core_mapping.py:28-71`) walks `dim_order` accumulating `stride *= split`, so the
stride of a symbol is the product of the splits of the symbols ordered before it.

**The matmul reorder is unconditional-safe.** `contiguous_dim` moves the last
symbol first *only if* `splits[contiguous_dim] > 1` — but a symbol with split 1
contributes a factor of 1 to every prefix product and is pruned from
`core_to_slot` entirely (it is absent from `sym_to_device_dim`). So moving it
unconditionally changes nothing, and the model needs no case split. This matters:
it is the only place the mapping looked value-dependent.

**The symbol-to-device-dim placement is candidate-invariant.** Everything
`_per_core_view_from_prep` (`pass_utils.py:2941-3154`) uses to place a split —
`device_stride_to_dim`, `stick_host_stride`, `num_stick_dim`, `dep_coeff` — comes
from `_ViewPrep`, which is computed once per `(op, dep, buffer)`. Only five things
in that function depend on the split *values*, and each is a small predicate:

1. the multi-stick rescue (`split * k == num_stick`) — a single admissible value;
2. `device_size[dev_dim] % split != 0` — a domain restriction;
3. two split symbols landing on one device dim — an at-most-one;
4. the interleaving check (a split symbol whose axis carries an unsplit symbol of
   larger coefficient) — a pairwise implication;
5. the stickified-extent overflow check — candidate-invariant, a static kill.

## 4. Stage one: fix the encoding, keep the menu

This is separable, lands on its own, and is where the measured win is.

Replace the `division` index and its `add_element` tables with a **one-hot over
the menu**, and channel every derived quantity linearly:

```python
z = [m.new_bool_var(f"z_{name}_{i}") for i in range(n)]
m.add_exactly_one(z)
m.add(self.eff_size  == sum(per_core[i]   * z[i] for i in range(n)))
m.add(self.cores     == sum(cores_used[i] * z[i] for i in range(n)))
m.add(self.core_cost == sum(core_cost[i]  * z[i] for i in range(n)))
```

and replace `_gate_divisions`' per-pair literals with a gate over the recovered
slicing classes (Section 2 proves the classes exist). Two variants were measured,
both provably equivalent to the pair table:

* **B, support clauses** — for each parent candidate `i`,
  `lit AND z_p[i] => OR_{j in supp(i)} z_c[j]`; unmatched parent candidates
  forbidden under `lit`.
* **D, class-indicator equality** — for each slicing class `g`,
  `sum_{i in g} z_p[i] == sum_{j in g} z_c[j]` under `lit`.

Measured on both large corpora, `num_search_workers = 8` (the count
`draft-compilation-next-steps.md` Section 1.6 recommends; the win is
*larger* at 8 than at the current `os.cpu_count()` = 192, so it is not an
artifact of oversubscription):

Median of 3 runs per encoding:

| graph | A landed | B support | D class-eq |
| --- | ---: | ---: | ---: |
| large/flash_big | 1176 ms | 664 ms | 617 ms |
| regen/flash_big | 1969 ms | 1011 ms | 1021 ms |
| regen/flash_attention | 677 ms | 331 ms | 342 ms |
| regen/block_x4 | 419 ms | 239 ms | 260 ms |
| regen/sdpa | 100 ms | 63 ms | 67 ms |
| **corpus total (15 graphs)** | **5780 ms** | **3528 ms** | **3577 ms** |

**1.64x at the same optimum.** On all 15 graphs, all three lexicographic
objective values — spill traffic, summed core count, summed balance cost — are
identical, and so is the resident set, buffer for buffer. This is not the
quality/time trade that candidate truncation is.

What does move is the *argmin*: on 4 of 15 graphs (`regen/flash_attention`,
`regen/flash_big`, `regen/mlp`, `regen/swiglu`) between 1 and 7 buffers commit a
different division at identical balance cost — a genuine tie broken differently
because the one-hot changes the search order. Addresses move too, for the reason
`draft-compilation-next-steps.md` Section 1.6 already records: the offset
dimension carries zero objective weight and is the model's remaining
structural symmetry. So the acceptance gate for this change is **objective
equality plus resident-set equality**, not a byte-identical plan. A gate demanding
byte-identical divisions would fail here, and would be measuring the symmetry, not
the change.

Two negative results worth recording, because both are the obvious thing to try:

**Maximal compactness is the weakest of the three, and the least stable.**
Collapsing each edge to a *single* linear equality over class-id-weighted
one-hots — `sum_i k_p(i) z_p[i] == sum_j k_c(j) z_c[j]`, one constraint per edge,
no auxiliary booleans at all — is still objective-identical and still a win, but
only **1.22x**, and it is the one variant that *regresses*: slower than the landed
encoding on 5 of 15 graphs, including 0.77-0.82x on all three large-corpus
`block_x*` cases. Under the current `num_search_workers = os.cpu_count()` default
(192 on this host) it inverts outright, to 0.74x overall and 0.44x on
`regen/flash_big`. B and D regress nowhere above the ~20 ms noise floor.

A single equality with arbitrary integer coefficients propagates almost nothing;
the Boolean structure is what the solver was using. So the rule is to replace a
table with **Boolean structure**, not with integer arithmetic. Constraint count is
not the objective — and a measurement of this taken at the default worker count
would have reported the opposite sign, which is its own warning.

**Per-edge `AddAllowedAssignments` is not the answer either.** That was tried
previously (`draft-compilation-next-steps.md` Section 7) and measured 1.2-1.35x
on some graphs and 0.67x on `block_x4`. It keeps the relation as a table; B and D
dissolve it into clauses over the one-hot, which is why they are uniformly
better.

Prefer **B**. It and D are within noise of each other on the corpus, B is simpler
(pure clauses, no arithmetic), and D's advantage shows only under menu inflation —
which Section 5 removes anyway.

## 5. Stage two: the structural model

### 5.1 Variables

Per operation `o`, per divisible axis `v` in `it_space_adjusted`, one-hot over the
axis's legal factors:

```python
dom = ctx.factor_domain(v)                 # divisors, narrowed by allowed_splits
                                           # and span floors -- already computed
z[o, v] = {f: m.new_bool_var(...) for f in dom}
m.add_exactly_one(z[o, v].values())
s[o, v] = sum(f * z[o, v][f] for f in dom)  # a LinearExpr, not a new var
```

`factor_domain` is already memoized on `WorkDivisionContext`. Every unary function
of the axis is now a linear expression, with **no table and no auxiliary
variable**:

| quantity | today | structural |
| --- | --- | --- |
| split factor | `add_element(division, raw, cp_var)` | `sum(f * z[f])` |
| per-core extent `N // f` | not modelled (folded into the menu) | `sum((N // f) * z[f])` |
| `f**2` (balance) | `add_element(division, core_cost, var)` | `sum(f*f * z[f])` |
| `log2(f)` (cost expr) | `add_element(division, values, var)` | `sum(round(S*log2 f) * z[f])` |
| `1/f` (cost expr) | `AddDivisionEquality(var, S, split)` | `sum((S // f) * z[f])` |
| "is split" | implied by the index | `1 - z[1]` |

That is every `add_element` in the file, gone, and both `AddDivisionEquality`
uses with them.

### 5.2 The bilinear part, and how to spend it

Only one relation is genuinely bilinear: the running product of split factors,
which yields both `cores_used` (the full product) and each symbol's core-mapping
stride `sigma` (a prefix of it). Order the symbols as `core_to_slice_mapping`
does — output axes first so the prefix after them is `output_partition`, then
reduction axes so the final prefix is `cores_used` — and build a chain:

```python
P[0] = 1
for i, v in enumerate(order):
    P[i+1] = m.new_int_var_from_domain(
        Domain.from_values(achievable_partial_products(order[:i+1], max_cores)),
        f"P_{o}_{i+1}")
    for f in dom_v:
        m.add(P[i+1] == f * P[i]).only_enforce_if(z[o, v][f])
sigma[v] = P[index_of(v)]        # stride, free
output_partition = P[n_output]   # free
cores_used       = P[-1]         # free, and P[-1] <= max_cores by its domain
```

`achievable_partial_products` is a set intersection over already-known small
domains; on the corpus it never exceeds a few dozen values.

This is the recommendation on `AddMultiplicationEquality`. It *works* here —
ortools 9.15 even honours enforcement literals on `int_prod`, which was checked —
but a chain of half-reified linear equalities under the one-hot literals is
better, for the same reason the class-id equality lost in Section 4: an explicit
small domain propagates and an `int_prod` is opaque. Cost is `O(sum |dom_v|)`
linear constraints per op, all of which the presolve folds.

Keep `AddMultiplicationEquality` for the two cases where no small domain exists:

* `_print_Mul` (`ilp_solver_ortools.py:529-547`) multiplying two genuinely wide
  cost-model variables — unchanged, and correct as written;
* the exact ceiling division for `eff_size`, *if* the partition domain is not
  small enough to channel (see below).

### 5.3 Per-core footprint

`size` is a compile-time constant, so `eff_size = ceil(size / output_partition)`
is a unary function of a variable whose domain has at most a dozen values. Channel
it and there is no multiplication at all:

```python
y = {p: m.new_bool_var(...) for p in partition_domain}
m.add_exactly_one(y.values())
m.add(output_partition == sum(p * y[p] for p in partition_domain))
m.add(eff_size == sum(ceil_div(size, p) * y[p] for p in partition_domain))
```

If coarse tiling later folds into the solve, `p` becomes `p * tile_count` and the
achievable set may grow. The general fallback is exact and needs two products:

```python
m.add_multiplication_equality(prod, [eff_size, PT])
m.add(prod >= size)
m.add(prod <= size + PT - 1)
```

### 5.4 Legality as constraints

`WorkDivisionContext.is_legal`'s six clauses map one-to-one:

| clause | structural form |
| --- | --- |
| `_factors_in_domain` | the one-hot's domain. Zero constraints. |
| `_in_split_domains` (blocked, allowed) | drop literals from the one-hot. Zero constraints. |
| `meets_span_floors` | drop literals below the floor. Zero constraints. |
| `_within_core_budget` | the last prefix domain. Zero constraints. |
| `_one_reduction_split_at_most` | `sum_{v in reduction} (1 - z[v][1]) <= 1`. One constraint. |
| `_spans_within_cap` | one linear constraint per tensor dep — see below. |

The span cap is the one clause that needs care. `get_per_core_span`
(`work_division.py:468-509`) is linear once the per-axis extents are variables:
device coordinates are sums of independent single-variable terms, so

```text
per_core_size(d) = 1 + sum_v coeff(v, d) * (q[v] - 1)      with q[v] = N_v // s_v
```

and `q[v]` is already a linear expression over the one-hot. But the function
returns at the **first** device dim whose `per_core_size > 1`, and which dim that
is depends on the split. Posting the cap for every dim would reject splits that
are legal today. Reproduce the early return exactly:

```python
deg[d]  <=>  per_core_size(d) == 1          # one bool per device dim
m.add(per_core_size(d) * stride_elems * itemsize <= MAX_SPAN_BYTES) \
 .only_enforce_if([deg[0], ..., deg[d-1], ~deg[d]])
```

`O(dims)` bools per dep. This is the single place where a careless structural
translation would silently change behaviour, so it deserves a targeted
differential test against `get_per_core_span` over the enumerated menu.

### 5.5 The residency gate

Per `(op, dep, buffer)` the allocator emits a **placement record** derived from
the existing `_ViewPrep` — the candidate-invariant half of
`_per_core_view_from_prep`, refactored out rather than reimplemented:

```python
@dataclass(frozen=True)
class DepPlacement:
    place:      dict[Symbol, tuple[int, int]]   # sym -> (device dim, stick multiplier k)
    rescue:     dict[Symbol, int]               # sym -> the one split the rescue admits
    unplaceable: frozenset[Symbol]              # no device dim at any split
    interleaved: tuple[tuple[Symbol, Symbol], ...]   # (split sym, outer sym) pairs
    divisible:  dict[Symbol, frozenset[int]]    # factors dividing device_size[dim]
    static_kill: bool                           # prep is None / topk / extent overflow
```

Then, per dep, per device dim `d` with symbol group `G(d)`:

```python
# at most one split symbol per device dim (else the view is unrepresentable)
m.add(sum(1 - z[v][1] for v in G(d)) <= 1).only_enforce_if(lit)

# F_d: linear, because at most one term is nonzero under the constraint above
F[d] == 1 + sum(v in G(d)) sum(f != 1) (k_v * f - 1) * z[v][f]

# sigma_d: pick up the prefix var of whichever symbol is split
m.add(Sig[d] == sigma[v]).only_enforce_if([lit, nontrivial[v]])   # per v in G(d)
m.add(Sig[d] == 1).only_enforce_if([lit, no_split_in_group[d]])
```

and the edge itself, for producer `p` writing buffer `b` and consumer `c` reading
it, all under `in_buffer[b]`:

```python
forall d:  F_write[p][d] == F_read[c][b][d]
forall d:  Sig_write[p][d] == Sig_read[c][b][d]
cores[p] == cores[c]
sum_{v in reduction(p)} (1 - z[p][v][1]) == 0        # no partial-reduction write
sum_d (1 - no_split_in_group[d]) <= 1                # matmul producer, one dim only
```

plus the per-dep representability restrictions: `z[v][f] == 0` for `f` not in
`divisible[v]`, `nontrivial[v] == 0` for `v in unplaceable` (except its `rescue`
value), and `nontrivial[v'] >= nontrivial[v]` for each interleaved pair.

Size: `O(#device_dims + #symbols)` per dep, **independent of menu cardinality**.
And a buffer with five consumers builds one write record and five read records,
against today's five `(P + C)` view evaluations plus five `P x C` comparisons.

### 5.5.1 Operations of different arity

Two ops joined by an edge rarely share an iteration space: a matmul producer has
a K axis its output buffer does not carry, a pointwise consumer may be split on
an axis belonging to a different operand entirely, and a stickified host dim
becomes two device dims so the buffer's rank matches neither side. Nothing in
Section 5.5 needs to special-case that, because the comparison frame is the
**buffer**, never an op.

`device_size` and `stride_map` come from
`V.graph.get_buffer(buf_name).layout.device_layout` (`pass_utils.py:2883-2894`),
so the index set `d` is fixed by the buffer both sides touch. Each op's symbols
reach it by projection: `dep_coeff` is that op's per-symbol host stride *on this
dep* (`pass_utils.py:2917`), and `host_stride == 0` drops the symbol from the
geometry outright (`pass_utils.py:2998`) — the canonical case being a K-split on
the output dep. Re-keying by device-dim index is what
`_per_core_view_from_prep` already does, and for exactly this reason: "two ops
with the same per-core slicing on this buffer compare equal even if they name
their iter axes differently" (`pass_utils.py:3141-3143`).

The three quantities therefore divide as follows:

| quantity | projected into the buffer frame? |
| --- | --- |
| `F_d` | **yes** — `G(d)` is per `(op, dep)` and may be empty on one side; `F_d = 1` then |
| `sigma_d` | **no** — prefix product over the op's *full* axis order, including axes that never touch this buffer |
| `cores_used` | **no** — product over every axis |

`sigma` staying unprojected is the point, not an oversight. A producer whose K
axis is ordered before M carries `sigma = K_split` where a consumer without that
axis carries `sigma = 1`; the two slice the buffer identically but assign the
slices to different physical cores, and `core_to_slot` exists to reject that
(`pass_utils.py:3126-3127`: "matching split factors alone is insufficient").
`cores_used` equality likewise stops a producer on 4 cores matching a consumer on
8 whose extra cores hold no copy.

So arity affects the *variable* count — one one-hot and one prefix var per axis,
so a wide op simply has a longer chain — and never the *constraint* count per
edge, which is the buffer's device rank. The only place it reaches the constraint
form is selecting which prefix var `sigma_d` reads: with `|G(d)| == 1`, the common
case, it is a plain equality with no literal; only reshaped or collapsed axes,
where several symbols land on one device dim, need the half-reified pick.

Two degenerate cases fall out rather than needing handling. A consumer split only
on axes absent from this buffer gets `F_d = 1` for every `d` and matches only a
producer that also left the buffer unsliced *at the same core count* — today's
behaviour exactly, since `work_slice_dims` is empty while `num_cores` still
counts every axis. And a frame-changing broadcast clone, where the arity mismatch
is genuinely unrepresentable, never reaches the comparison: `build_residency_edge`
excludes it statically via `_is_frame_changing_clone`, so it is an edge drop, not
a constraint.

### 5.6 Objective and read-back

The objective is unchanged in meaning and strictly better in form. Residency
(`sum spill_cost * (1 - in_buffer)`) does not touch the division at all.
Parallelism becomes `sum_o cores[o]` where `cores` is the chain's last prefix var.
Balance becomes `sum_o sum_v sum_f f*f*z[o][v][f]` — a pure linear expression the
LP relaxation can see, where today it is an `add_element` result.

Read-back drops the index entirely: `chosen_splits = {v: value(s[o][v])}`, and
`_commit_divisions` (`allocator.py:2167-2200`) keeps its `_split_option_is_legal`
assertion as a post-solve check.

### 5.7 Migration

`SaCoOptimizingSolver` and `ExhaustiveSearchSolver` genuinely need the menu, so
`core_divisions` stays. Add a `division_space` field carrying the axis domains and
placement records; CP-SAT consumes it, the other engines consume the list, and one
equivalence test keeps them honest:

```text
for every captured op:  set(space.enumerate()) == set(buffer.core_divisions)
for every captured edge: space.compatible(p, c) == ((p, c) in cd_parent_matches)
```

That test is the whole safety argument for the change, and it runs on the existing
corpora with no hardware.

## 6. What this actually buys, honestly

It is not primarily wall-clock. Solve time is **sublinear in menu size**, measured
on `regen/flash_big` at 8 workers by truncating and by replicating the menu (the
replication simulates a coarse-tiling cross product where tiling must agree across
a matched edge):

| menu | A landed | D class-eq |
| ---: | ---: | ---: |
| 311 (N=4) | 680 ms | 674 ms |
| 619 (N=8) | 1028 ms | 599 ms |
| 1177 (N=16) | 1905 ms | 1015 ms |
| 2299 (full) | 1974 ms | 1018 ms |
| 4598 (x2) | 3484 ms | 1608 ms |
| 6897 (x3) | 4702 ms | 2312 ms |

A 22x menu costs A 6.9x and D 3.4x. So at `sencores = 32`, where
`prod(splits) <= 32` caps the menu at a few dozen per op, **the menu is not the
wall**. Anyone proposing de-enumeration on speed grounds alone is proposing it for
the wrong reason — and Section 4 gets most of that speed while keeping the menu.

What Section 5 does buy:

* **The Python precompute disappears.** `_cd_parent_matches` evaluates `P + C`
  per-core views per edge and compares `P x C` pairs. None of that appears in the
  numbers above, because the captures are dumps taken after it ran. It is real
  compile-time cost and the structural model deletes it.
* **The pin ladder becomes constraints.** Every guard in `_division_map` is
  currently "hand the solver a one-element menu". Structurally each is a domain
  restriction (`s[v] == fixed`), which cannot be silently lost by a later menu
  transformation. A large share of the co-optimization wrong-code bugs on this
  branch were menu-surgery bugs.
* **Headroom.** The menu grows as the *product* of axis domains; the structural
  model grows as their *sum*. That gap is invisible at 32 cores and decisive if
  `sencores` rises or coarse tiling folds its options into the same solve.
* **Legality moves into the model.** The span cap and core budget stop being
  enumerator filters, which is where
  `draft-compilation-next-steps.md` Section 7 located the real hole.

## 7. Hazards

**Every constraint must be enforced under `in_buffer`.** The gate's current
failure mode is benign: no compatible pairs forces `in_buffer == 0`. In the
structural model the same situation is an unsatisfiable conjunction — which
correctly forces `in_buffer == 0` *only if every constraint in the conjunction
carries the enforcement literal*. One unconditional constraint turns a
non-resident buffer into an infeasible model. This is the highest-risk detail in
the change and is worth an assertion at build time that no representability
constraint was posted without a literal.

**Fail closed on representability.** Today an unrepresentable view simply makes a
candidate absent from the pair list. Structurally it must be an explicit
constraint, and the two known unhandled shapes catalogued at
`pass_utils.py:3094-3110` — collapsed-axis information loss and partial multi-stick
coverage — currently reach the `unrepresentable` fall-through. If the structural
encoding does not reproduce that fall-through it will pin a buffer whose per-core
slicing it cannot express, which is silent wrong code. Mitigation, and it is
cheap: keep `_per_core_view_on_buf` as a **post-solve verifier** and assert, for
every resident buffer, that the producer's write view equals every consumer's read
view under the divisions actually chosen. `O(edges)`, no hardware, and it makes
the whole change safe to land.

**The span-cap early return.** Section 5.4. Differential-test it.

**Coarse tiling is not in the solve on this branch.** `enumerate_tile_options`
(`wsr/enumerate_tilings.py:216`) has no production caller here; tiling is applied
by pre-scheduling passes before the solver runs. The `x2`/`x3` rows above are a
simulation of what folding it in would cost, not a measurement of it.

## 8. Plan

**Step 1 — encoding, no semantic change.** Replace the `division` index and its
`add_element` tables with a menu one-hot; replace `_gate_divisions` with the
support-clause gate (variant B). Acceptance: equal spill / core / balance
objectives and an equal resident set on all three corpora, plus the corpus-total
time — *not* byte-identical divisions or addresses, for the reason in Section 4.
Measured at 1.64x with the worker-count fix in place. Independent of everything
below.

**Step 2 — the placement record.** Refactor the candidate-invariant half of
`_per_core_view_from_prep` into `DepPlacement`, and re-express
`_per_core_view_from_prep` in terms of it. Pure refactor, guarded by the existing
view tests. No solver change.

**Step 3 — `DivisionSpace`.** Add the axis-domain carrier and the equivalence test
of Section 5.7 (`space.enumerate() == core_divisions`,
`space.compatible == cd_parent_matches`) over all three corpora. Still no solver
change; this is the step that proves the structural description is faithful before
anything depends on it.

**Step 4 — the structural solver.** Behind a config flag, build the model of
Section 5. Land the post-solve view verifier of Section 7 *first*, and run it on
both paths. Acceptance: equal objectives and resident set on all three corpora
(same gate as Step 1), verifier clean, and the differential span test green.

**Step 5 — retire the menu on the CP-SAT path** once Step 4 has run clean,
keeping `core_divisions` for the SA and exhaustive engines.

Steps 1 and 2 are worth doing regardless of whether 4 and 5 ever land.

## 9. Reproducing the measurements

The scripts are not in the tree; they monkey-patch
`_CoreDivisionBufferWithCpVars` and re-run `plan_layout_and_core_divisions` over
`load_captures(...)`, comparing solve time and objective. Each is about 80 lines.
Anything measured on `cooptimization_captures.json` and `_large.json` alone is
untrustworthy — `_regen.json` has ~2.8x larger menus and is where the previously
proposed truncation budget regressed. Measure on all three, and set
`num_search_workers` explicitly: at `os.cpu_count()` on a 192-core host the
run-to-run spread swamps everything reported here.
