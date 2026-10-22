# Copyright 2026 The Torch-Spyre Authors.
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

"""How a :class:`SpyreOptimizationGraph` becomes a CP model.

An :class:`EncodingStrategy` encodes the graph one node, edge or relayout
layout at a time, in the order ``SpyreOptimizationGraph._finalize`` walks
them. It owns everything about *how* decisions are searched: how each node's
work split is enumerated (a pinned value, a table of candidates), the
relations over each edge's two ends, the candidate relayout copies, and every
CP variable (:class:`NodeVars`). Nodes and edges stay pure data.

:class:`TableStrategy` is the default: each open work split is a ``Table``
over a candidate menu and every relation a table over menu indices.
:class:`EnumerationFactory` builds a strategy from configuration alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TYPE_CHECKING

import sympy

from torch_spyre._inductor import config
from torch_spyre._inductor.scratchpad.buffer_graph import (
    DependencyEdge,
    ExecutionEdge,
    RelayoutLayout,
    ResourceNode,
    SpyreOptimizationGraph,
    WorkSplit,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    BufferType,
    CoreDivision,
    CoreDivisionBuffer,
    LifetimeBoundBuffer,
    RelayoutCopyBuffer,
    ceil_div,
    division_symbol,
)

if TYPE_CHECKING:
    from torch_spyre._inductor.scratchpad.allocator import CoOptimizingAllocator
    from torch_spyre._inductor.scratchpad.lx_relayout import RelayoutCandidate
    from torch_spyre._inductor.scratchpad.table_candidates import TableCandidates

Var = Any  # a backend variable or literal
Model = Any  # cp_model.CpModel until 3a, a Backend after
Item = ResourceNode | RelayoutLayout
AxisSetting = Literal["table", "lazy", "pruned"]
RelayoutSetting = Literal["data", "pruned", "lazy"]


def _gate_divisions(model, compatible, src_div, dst_div, enforce_lit) -> None:
    """Enforce, when ``enforce_lit`` is true, that ``(src_div, dst_div)`` is
    one of the ``compatible`` (i, j) pairs. With no compatible pairs the
    relation is unsatisfiable, so ``enforce_lit`` is forced false."""
    if not compatible:
        model.Add(enforce_lit == 0)
        return
    # A table, instead of a Boolean and two implications per compatible pair.
    model.AddAllowedAssignments([src_div, dst_div], compatible).OnlyEnforceIf(
        enforce_lit
    )


def _sym_core_divs_of(name: str, divisions: Sequence[CoreDivision]) -> dict:
    """``CoreDivisionBuffer.sym_core_divs`` for a buffer named ``name``."""
    keys = dict.fromkeys(k for cd in divisions for k in cd.splits)
    return {
        key: sympy.Symbol(f"split_{name}_{key}", integer=True, positive=True)
        for key in keys
    }


@dataclass(frozen=True)
class PrinterView:
    """What ``cpsat_expr_printer`` reads about a ``Table`` node: its
    candidates and the cost model's symbols."""

    core_divisions: tuple[CoreDivision, ...]
    sym_core_divs: Mapping[Any, sympy.Symbol]
    sym_division: sympy.Symbol


@dataclass(eq=False)
class NodeVars:
    """The variables of one node or relayout layout. ``division``, ``buffer``
    and the ``*_raw`` columns (the per-candidate values the cost lowering
    tabulates over) exist only for a ``Table``."""

    model: Model
    in_buffer: Var
    offset: Var
    eff_size: Var | int
    merges: dict[str, Var] = field(default_factory=dict)
    splits: dict[Any, Var | int] = field(default_factory=dict)
    tile_counts: dict[Any, Var | int] = field(default_factory=dict)
    cores: Optional[Var | int] = None
    core_cost: Optional[Var | int] = None
    division: Optional[Var] = None
    buffer: Optional[PrinterView] = None
    splits_raw: dict[Any, list[int]] = field(default_factory=dict)
    tile_counts_raw: dict[Any, list[int]] = field(default_factory=dict)
    cores_used: list[int] = field(default_factory=list)
    per_core: list[int] = field(default_factory=list)
    name: str = ""
    fixed_index: Optional[int] = None  # the index, when nothing is left to choose
    _is: dict[int, Var] = field(default_factory=dict)
    _consts: Optional[tuple[Var, Var]] = None

    def division_is(self, i: int) -> Var:
        """Table only: the literal ``division == i``, made on first use."""
        if self._consts is not None:
            true, false = self._consts
            return true if i == self.fixed_index else false
        if i not in self._is:
            lit = self.model.BoolVar(f"div_{self.name}_is_{i}")
            self.model.Add(self.division == i).OnlyEnforceIf(lit)
            self.model.Add(self.division != i).OnlyEnforceIf(lit.Not())
            self._is[i] = lit
        return self._is[i]


@dataclass(frozen=True)
class Provenance:
    """Which call posted a range of the model (debugging side table)."""

    source: str
    element: str
    variables: range
    constraints: range

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "element": self.element,
            "variables": [self.variables.start, self.variables.stop],
            "constraints": [self.constraints.start, self.constraints.stop],
        }


# ---- enumerations of the work-split axis, internal to the strategies -------


@dataclass(frozen=True)
class _Fixed:
    """Known before the solve: ``index`` into the menu for a pinned value,
    ``None`` for COMMITTED, which posts nothing."""

    value: WorkSplit
    index: Optional[int] = None


@dataclass(frozen=True)
class _Table:
    """Explicit candidates: an index var plus element ties."""

    candidates: tuple[CoreDivision, ...]


@dataclass(eq=False)
class EncodingStrategy:
    """Encodes a graph into a CP model, one element at a time. Created from
    configuration alone; ``prepare(graph)`` resets the per-solve state, and
    the ``encode_*`` methods add to it."""

    division: AxisSetting = "table"
    tiling: AxisSetting = "table"
    relayout: RelayoutSetting = "data"
    # per-solve state
    layouts: tuple[RelayoutLayout, ...] = ()
    vars: dict[Item, NodeVars] = field(default_factory=dict)
    served: dict[DependencyEdge, list[tuple[RelayoutLayout, Var]]] = field(
        default_factory=dict
    )
    provenance: list[Provenance] = field(default_factory=list)
    cut_terms: list[Var] = field(default_factory=list)
    tile_terms: list[Var] = field(default_factory=list)

    # ---- whole-graph hooks ----

    def prepare(self, graph: SpyreOptimizationGraph) -> None:
        """Reset the per-solve state, decide how each node's axes are
        enumerated and generate the candidates of table-based axes."""
        raise NotImplementedError

    def encode_no_overlap(self, graph: SpyreOptimizationGraph, model: Model) -> None:
        """The 2-D no-overlap over every node's and layout's rectangle."""
        raise NotImplementedError

    def tiling_segments(self, graph: SpyreOptimizationGraph) -> dict[ResourceNode, int]:
        """The coarse-tiling segment of each tileable op on the chain."""
        raise NotImplementedError

    # ---- per element, called by _finalize in emission order ----

    def encode_node(self, node: ResourceNode, model: Model) -> None:
        raise NotImplementedError

    def encode_layout(self, layout: RelayoutLayout, model: Model) -> None:
        raise NotImplementedError

    def encode_merge(self, edge: DependencyEdge, model: Model) -> None:
        raise NotImplementedError

    def limit_merges(self, node: ResourceNode, model: Model, incoming: bool) -> None:
        raise NotImplementedError

    def encode_capacity(self, item: Item, model: Model) -> None:
        raise NotImplementedError

    def encode_op_constraints(self, item: Item, model: Model) -> None:
        raise NotImplementedError

    def encode_read(self, edge: DependencyEdge, model: Model) -> None:
        raise NotImplementedError

    def encode_relayout(self, layout: RelayoutLayout, model: Model) -> None:
        raise NotImplementedError

    def encode_tiling_vars(self, item: Item, model: Model) -> None:
        raise NotImplementedError

    def encode_chain_head(self, node: ResourceNode, model: Model) -> None:
        raise NotImplementedError

    def encode_nest(self, edge: ExecutionEdge, model: Model) -> None:
        raise NotImplementedError

    def encode_cuts(
        self, item: Item, segments: Mapping[ResourceNode, int], model: Model
    ) -> None:
        raise NotImplementedError

    def encode_tile_counts(self, item: Item, model: Model) -> None:
        raise NotImplementedError

    def encode_clone(self, node: ResourceNode, model: Model) -> None:
        raise NotImplementedError

    # ---- objective inputs and extraction ----

    def chooses_split(self, item: Item) -> bool:
        """Whether ``item``'s work split has vars (it is not COMMITTED)."""
        raise NotImplementedError

    def objective_terms(self, item: Item) -> tuple[Optional[Var], Optional[Var]]:
        raise NotImplementedError

    def cost_bindings(self) -> tuple[dict, dict]:
        raise NotImplementedError

    def chosen_split(self, item: Item, solver) -> WorkSplit:
        raise NotImplementedError

    def footprint(self, item: Item, solver) -> int:
        raise NotImplementedError

    def record_fired(self, edge: DependencyEdge, solver) -> None:
        raise NotImplementedError


class TableStrategy(EncodingStrategy):
    """The default strategy: every open work split is a ``Table`` over its
    candidate menu, a pinned one ``Fixed``, and the view, relayout and nest
    relations are tables over menu indices.

    Until candidate generation moves here, the candidates are the allocator's
    (:meth:`from_buffers`)."""

    def __init__(
        self,
        menus: Mapping[str, Sequence[CoreDivision]] | None = None,
        matches: Mapping[tuple[str, str], Sequence[tuple[int, int]]] | None = None,
        relayout_candidates: Mapping[tuple[str, str], Sequence[RelayoutCandidate]]
        | None = None,
        copies: Sequence[RelayoutCopyBuffer] = (),
        copy_reasons: Mapping[str, str] | None = None,
        **settings,
    ) -> None:
        super().__init__(**settings)
        self.menus = {k: tuple(v) for k, v in (menus or {}).items()}
        self.matches = {k: list(v) for k, v in (matches or {}).items()}
        self.relayout_candidates = {
            k: list(v) for k, v in (relayout_candidates or {}).items()
        }
        self.copies = list(copies)
        self.copy_reasons = dict(copy_reasons or {})
        self.fired: dict[DependencyEdge, RelayoutCandidate] = {}
        self.solved: dict[Item, int] = {}

    @classmethod
    def from_buffers(
        cls,
        buffers: Sequence[LifetimeBoundBuffer],
        reasons: Mapping[str, str] | None = None,
        **settings,
    ) -> TableStrategy:
        """The allocator's final candidates, read once from its buffers: the
        menus, the view tables per edge, the relayout candidates on groups
        with a copy in this solve, and the copies in copy order."""
        reasons = reasons or {}
        names = {b.name for b in buffers}
        copies = [b for b in buffers if isinstance(b, RelayoutCopyBuffer)]
        groups = {c.group_key for c in copies}
        menus, matches, relayouts = {}, {}, {}
        for b in buffers:
            if isinstance(b, RelayoutCopyBuffer) or not isinstance(
                b, CoreDivisionBuffer
            ):
                continue
            if b.core_divisions:
                menus[b.name] = b.core_divisions
            for parent in dict.fromkeys([*b.parents, *b.in_place_parents]):
                if parent not in names:
                    continue
                matches[(parent, b.name)] = b.cd_parent_matches.get(parent, [])
                cands = [
                    c
                    for c in b.cd_parent_relayouts.get(parent, ())
                    if c.group_key in groups
                ]
                if cands:
                    relayouts[(parent, b.name)] = cands
        return cls(
            menus=menus,
            matches=matches,
            relayout_candidates=relayouts,
            copies=copies,
            copy_reasons={c.name: reasons[c.name] for c in copies if c.name in reasons},
            **settings,
        )

    # ---- whole-graph hooks ----

    def prepare(self, graph: SpyreOptimizationGraph) -> None:
        self.vars = {}
        self.served = {}
        self.provenance = []
        self.cut_terms = []
        self.tile_terms = []
        self.fired = {}
        self.solved = {}
        self._capacity = graph.capacity
        self._serves: dict[RelayoutLayout, list[Var]] = {}
        self._candidates_of: dict[RelayoutLayout, tuple[RelayoutCandidate, ...]] = {}
        self._priced: dict[RelayoutLayout, list[int]] = {}
        self._incoming: dict[Item, list[Var]] = {}
        self._enum: dict[Item, Any] = {}
        layouts = []
        for copy in self.copies:
            layout = RelayoutLayout(
                producer=graph.nodes.get(copy.relayout_parent),
                name=copy.name,
                group=copy.group,
                size=ceil_div(copy.size, graph.alignment),
                read_count=copy.read_count,
                division=copy.core_divisions[0],
                start_time=copy.start_time,
                end_time=copy.end_time,
                residency_reason=self.copy_reasons.get(copy.name),
            )
            layouts.append(layout)
            self._candidates_of[layout] = copy.candidates
            self._priced[layout] = sorted(copy.cost_by_source_division)
            self._enum[layout] = _Table((layout.division,))
        self.layouts = tuple(layouts)
        self._layout_by_group = {
            (c.relayout_parent, c.group): lay for c, lay in zip(self.copies, layouts)
        }
        for node in graph.get_nodes():
            self._enum[node] = self._enumeration(node)
        # Coarse-tiling nests, interned over every item with a work split; the
        # untiled nest is 0, so loop_id != 0 means tiled.
        self._nest_ids: dict[tuple[int, ...], int] = {(): 0}
        for node in graph.get_nodes():
            if self.chooses_split(node):
                for cd in self._candidates(node):
                    self._nest_ids.setdefault(
                        cd.tiling.level_counts, len(self._nest_ids)
                    )
        for layout in self.layouts:
            self._nest_ids.setdefault(
                layout.division.tiling.level_counts, len(self._nest_ids)
            )
        self._tiling_on = len(self._nest_ids) > 1
        self._loop_id: dict[Item, Var] = {}
        self._tiled: dict[Item, Var] = {}
        self._group: dict[Item, Var] = {}
        self._join_prev: dict[ResourceNode, Optional[ResourceNode]] = {}
        self._chain_len = 0

    def _enumeration(self, node: ResourceNode) -> _Fixed | _Table:
        if node.committed:
            return _Fixed(WorkSplit.COMMITTED)
        menu = self.menus.get(node.name)
        assert menu, f"{node.name} has an open work split and no candidates"
        if node.work_split is not None:
            return _Fixed(node.work_split, self.menu_index(node.name, node.work_split))
        return _Table(menu)

    def menu_index(self, name: str, split: WorkSplit) -> int:
        """The first candidate of ``name``'s menu equal to ``split``."""
        for i, cd in enumerate(self.menus[name]):
            if WorkSplit.of(cd) == split:
                return i
        raise ValueError(f"{name}: work split {split} is not a candidate")

    def _candidates(self, item: Item) -> tuple[CoreDivision, ...]:
        if isinstance(item, RelayoutLayout):
            return (item.division,)
        return self.menus.get(item.name, ())

    def chooses_split(self, item: Item) -> bool:
        enum = self._enum.get(item)
        return not (isinstance(enum, _Fixed) and enum.index is None)

    def open_choices(self, item: Item) -> int:
        """How many values the item's work split may take in the solve."""
        enum = self._enum.get(item)
        return len(enum.candidates) if isinstance(enum, _Table) else 1

    # ---- 1. vars ----

    def encode_node(self, node: ResourceNode, model: Model) -> None:
        v = self._base_vars(node, model)
        for e in node.inputs:
            if e.in_place:
                v.merges[e.src.name] = model.BoolVar(f"merge_{e.src.name}_{node.name}")
        self.vars[node] = v
        if self.chooses_split(node):
            self._split_vars(node, v, model)
        # A caller-pinned decision fixes its var; a pinned work split is
        # already a Fixed enumeration.
        if node.resident is not None:
            model.Add(v.in_buffer == int(node.resident))
        if node.offset is not None:
            model.Add(v.offset == node.offset)

    def encode_layout(self, layout: RelayoutLayout, model: Model) -> None:
        v = self._base_vars(layout, model)
        self.vars[layout] = v
        self._split_vars(layout, v, model)

    def _base_vars(self, item: Item, model: Model) -> NodeVars:
        return NodeVars(
            model=model,
            in_buffer=model.BoolVar(f"in_buffer_{item.name}"),
            offset=model.IntVar(0, max(0, self._capacity - 1), f"off_{item.name}"),
            eff_size=item.size,
            name=item.name,
        )

    def _syms(self, item: Item) -> tuple[Mapping, Mapping]:
        if isinstance(item, RelayoutLayout):
            return _sym_core_divs_of(item.name, (item.division,)), {}
        return item.sym_core_divs, item.sym_tile_counts

    def _split_vars(self, item: Item, v: NodeVars, m: Model) -> None:
        """The candidate index and the per-candidate quantities tied to it:
        per-core footprint (the output partition and coarse tiling both
        shrink it), cores used, the balance cost (summed squared split
        factors), and the split and tile-count vars of the cost model."""
        menu = self._candidates(item)
        sym_core_divs, sym_tile_counts = self._syms(item)
        v.per_core = [
            ceil_div(item.size, cd.output_partition * cd.tiling.output_tile_count)
            for cd in menu
        ]
        cores_used = [cd.cores_used for cd in menu]
        core_cost = [sum(s**2 for s in cd.splits.values()) for cd in menu]
        v.cores_used = cores_used
        v.buffer = PrinterView(menu, sym_core_divs, division_symbol(item.name))
        enum = self._enum[item]
        fixed = (
            enum.index if isinstance(enum, _Fixed) else (0 if len(menu) == 1 else None)
        )
        if fixed is not None:
            # Nothing to choose: constants instead of element lookups on a
            # fixed index, which keeps presolve linear in the copy count.
            v.fixed_index = fixed
            v.division = m.Constant(fixed)
            v.eff_size = v.per_core[fixed]
            v.core_cost = core_cost[fixed]
            v.cores = cores_used[fixed]
            only = menu[fixed]
            v.splits = {key: only.splits.get(key, 1) for key in sym_core_divs}
            v.splits_raw = {key: [val] for key, val in v.splits.items()}
            v.tile_counts = {key: only.tiling.count_for(key) for key in sym_tile_counts}
            v.tile_counts_raw = {key: [val] for key, val in v.tile_counts.items()}
            v._consts = (m.Constant(1), m.Constant(0))
            return
        name = item.name
        v.division = m.IntVar(0, len(menu) - 1, f"div_{name}")
        v.eff_size = m.IntVar(0, max(v.per_core), f"eff_size_{name}")
        v.core_cost = m.IntVar(0, max(core_cost), f"core_cost_{name}")
        v.cores = m.IntVar(min(cores_used), max(cores_used), f"occ_{name}")
        for key, symbol in sym_core_divs.items():
            assert isinstance(symbol, sympy.Symbol)
            raw = [cd.splits.get(key, 1) for cd in menu]
            var = m.IntVar(1, config.sencores, symbol.name)
            m.AddElement(v.division, raw, var)
            v.splits[key] = var
            v.splits_raw[key] = raw
        for key, symbol in sym_tile_counts.items():
            raw = [cd.tiling.count_for(key) for cd in menu]
            var = m.IntVar(min(raw), max(raw), symbol.name)
            m.AddElement(v.division, raw, var)
            v.tile_counts[key] = var
            v.tile_counts_raw[key] = raw
        m.AddElement(v.division, v.per_core, v.eff_size)
        m.AddElement(v.division, cores_used, v.cores)
        m.AddElement(v.division, core_cost, v.core_cost)

    # ---- 2. merges ----

    def encode_merge(self, edge: DependencyEdge, model: Model) -> None:
        """An active merge pins parent and child to one base, both resident;
        a child that chooses a work split must also reuse the parent's exact
        per-core storage: equal footprints under divisions that slice it
        alike."""
        src, dst = self.vars[edge.src], self.vars[edge.dst]
        lit = dst.merges[edge.src.name]
        model.Add(src.offset == dst.offset).OnlyEnforceIf(lit)
        model.AddImplication(lit, src.in_buffer)
        model.AddImplication(lit, dst.in_buffer)
        if self.chooses_split(edge.dst):
            model.Add(dst.eff_size == src.eff_size).OnlyEnforceIf(lit)
            _gate_divisions(model, self._pairs(edge), src.division, dst.division, lit)
        self._incoming.setdefault(edge.dst, []).append(lit)

    def limit_merges(self, node: ResourceNode, model: Model, incoming: bool) -> None:
        """A storage slot is handed off linearly: at most one merge in, and at
        most one out."""
        if incoming:
            lits = list(self.vars[node].merges.values())
        else:
            lits = [
                self.vars[e.dst].merges[node.name] for e in node.outputs if e.in_place
            ]
        if len(lits) > 1:
            model.AddAtMostOne(lits)

    def _pairs(self, edge: DependencyEdge) -> list[tuple[int, int]]:
        return self.matches.get((edge.src.name, edge.dst.name), [])

    # ---- 3. placement ----

    def encode_capacity(self, item: Item, model: Model) -> None:
        v = self.vars[item]
        model.Add(v.offset + v.eff_size <= self._capacity).OnlyEnforceIf(v.in_buffer)

    def encode_no_overlap(self, graph: SpyreOptimizationGraph, model: Model) -> None:
        """Every item's optional rectangle ``[start, end) x [offset, offset +
        eff_size)``. An active incoming merge shortens the child (never the
        parent, whose bytes above a smaller child are still read on that tick)
        by the one handoff tick, so the two abut and may share an offset."""
        x_intervals, y_intervals = [], []
        items: list[Item] = [*graph.get_nodes(), *self.layouts]
        for item in items:
            v = self.vars[item]
            ins = self._incoming.get(item, [])
            if ins:
                start = model.IntVar(
                    item.start_time, item.end_time, f"start_{item.name}"
                )
                model.Add(start == item.start_time + sum(ins))
                x_start: Any = start
                x_size: Any = item.end_time - start
            else:
                x_start = item.start_time
                x_size = item.end_time - item.start_time
            x_intervals.append(
                model.NewOptionalIntervalVar(
                    x_start, x_size, item.end_time, v.in_buffer, f"x_{item.name}"
                )
            )
            # An interval end must be affine, so the top of a division-dependent
            # footprint gets its own var; its range covers buffers larger
            # than LX itself.
            y_end = model.IntVar(
                0, max(0, self._capacity - 1) + item.size, f"top_{item.name}"
            )
            model.Add(y_end == v.offset + v.eff_size)
            y_intervals.append(
                model.NewOptionalIntervalVar(
                    v.offset, v.eff_size, y_end, v.in_buffer, f"y_{item.name}"
                )
            )
        model.AddNoOverlap2D(x_intervals, y_intervals)

    # ---- 4. op constraints ----

    def encode_op_constraints(self, item: Item, model: Model) -> None:
        if item.residency_reason is not None:
            model.Add(self.vars[item].in_buffer == 0)

    # ---- 5. reads ----

    def encode_read(self, edge: DependencyEdge, model: Model) -> None:
        """A resident producer's division must slice it as the consumer's
        does, or the consumer reads it through a resident relayout copy
        ("match or served"). Gated only when both ends choose a work split,
        and once per read the consumer lists."""
        if not (self.chooses_split(edge.src) and self.chooses_split(edge.dst)):
            return
        for _ in range(edge.reads):
            self._encode_one_read(edge, model)

    def _encode_one_read(self, edge: DependencyEdge, model: Model) -> None:
        src, dst = self.vars[edge.src], self.vars[edge.dst]
        by_layout: dict[RelayoutLayout, list] = {}
        for c in self.relayout_candidates.get((edge.src.name, edge.dst.name), ()):
            layout = self._layout_by_group.get(c.group_key)
            if layout is not None:
                by_layout.setdefault(layout, []).append(c)
        served: dict[RelayoutLayout, Var] = {}
        for layout, cands in sorted(by_layout.items(), key=lambda i: i[0].group):
            lit = model.BoolVar(
                f"served_{edge.src.name}__{edge.dst.name}__g{layout.group}"
            )
            _gate_divisions(
                model,
                [(c.source_division, c.consumer_division) for c in cands],
                src.division,
                dst.division,
                lit,
            )
            model.AddImplication(lit, self.vars[layout].in_buffer)
            served[layout] = lit
            self._serves.setdefault(layout, []).append(lit)
        pairs = self._pairs(edge)
        if not served:
            _gate_divisions(model, pairs, src.division, dst.division, src.in_buffer)
            return
        self.served.setdefault(edge, []).extend(served.items())
        match = model.BoolVar(f"match_{edge.src.name}__{edge.dst.name}")
        _gate_divisions(model, pairs, src.division, dst.division, match)
        model.AddBoolOr([match, *served.values()]).OnlyEnforceIf(src.in_buffer)

    # ---- 6. relayouts ----

    def encode_relayout(self, layout: RelayoutLayout, model: Model) -> None:
        """A resident copy needs its producer resident under a division it
        was priced for, and must serve some read. One whose producer is not
        in the solve, or chooses no division, never fires."""
        v = self.vars[layout]
        source = layout.producer
        if source is None or not self.chooses_split(source):
            model.Add(v.in_buffer == 0)
            return
        sv = self.vars[source]
        model.AddImplication(v.in_buffer, sv.in_buffer)
        priced = [sv.division_is(i) for i in self._priced[layout]]
        model.AddBoolOr(priced).OnlyEnforceIf(v.in_buffer)
        serves = self._serves.get(layout, [])
        if serves:
            model.AddBoolOr(serves).OnlyEnforceIf(v.in_buffer)
        else:
            model.Add(v.in_buffer == 0)

    # ---- 7. coarse tiling ----

    def encode_tiling_vars(self, item: Item, model: Model) -> None:
        """The item's loop nest id (tied to its division like ``eff_size``)
        and whether it is tiled."""
        if not self._tiling_on:
            return
        v = self.vars[item]
        ids = [self._nest_ids[cd.tiling.level_counts] for cd in self._candidates(item)]
        var = model.IntVar(0, max(self._nest_ids.values()), f"loop_id_{item.name}")
        model.AddElement(v.division, ids, var)
        tiled = model.BoolVar(f"tiled_{item.name}")
        model.Add(var != 0).OnlyEnforceIf(tiled)
        model.Add(var == 0).OnlyEnforceIf(tiled.Not())
        self._loop_id[item] = var
        self._tiled[item] = tiled

    def tileable(self, node: ResourceNode) -> bool:
        return self.chooses_split(node) and any(
            not cd.tiling.is_untiled for cd in self._candidates(node)
        )

    def tiling_segments(self, graph: SpyreOptimizationGraph) -> dict[ResourceNode, int]:
        """Walk the execution chain over the tileable ops. Consecutive ops
        share a loop nest only on consecutive ticks, so an op one tick after
        the previous tileable op continues its segment and any other starts
        a new one: an untileable op in between always breaks the run, one on
        the same tick (another output of a multi-output op) does not."""
        segments: dict[ResourceNode, int] = {}
        if not self._tiling_on:
            return segments
        prev: Optional[ResourceNode] = None
        count = 0
        node = graph.first
        while node is not None:
            if self.tileable(node):
                assert node.produced_at is not None
                if prev is not None and node.produced_at - 1 == prev.produced_at:
                    segments[node] = segments[prev]
                    self._join_prev[node] = prev
                else:
                    segments[node] = count
                    count += 1
                    self._join_prev[node] = None
                prev = node
            node = node.next.after if node.next else None
        self._chain_len = len(segments)
        return segments

    def encode_chain_head(self, node: ResourceNode, model: Model) -> None:
        """The first op of the chain has no incoming execution edge."""
        self._encode_group(node, model)

    def encode_nest(self, edge: ExecutionEdge, model: Model) -> None:
        """The loop group of ``edge.after``: a fresh 0 at a segment start,
        else its predecessor's group stepped by ``1 - joined``, where
        ``joined`` means both run the same non-empty nest."""
        self._encode_group(edge.after, model)

    def _encode_group(self, node: ResourceNode, model: Model) -> None:
        if node not in self._join_prev:
            return
        prev = self._join_prev[node]
        if prev is None:
            self._group[node] = model.Constant(0)
            return
        a, b = prev.name, node.name
        same = model.BoolVar(f"samenest_{a}_{b}")
        model.Add(self._loop_id[prev] == self._loop_id[node]).OnlyEnforceIf(same)
        model.Add(self._loop_id[prev] != self._loop_id[node]).OnlyEnforceIf(same.Not())
        joined = model.BoolVar(f"joined_{a}_{b}")
        model.AddBoolAnd([same, self._tiled[node]]).OnlyEnforceIf(joined)
        model.AddBoolOr([same.Not(), self._tiled[node].Not()]).OnlyEnforceIf(
            joined.Not()
        )
        var = model.IntVar(0, self._chain_len, f"tile_group_{b}")
        model.Add(var == self._group[prev] + 1 - joined)
        self._group[node] = var

    def encode_cuts(
        self, item: Item, segments: Mapping[ResourceNode, int], model: Model
    ) -> None:
        """The item's cut literal: tiled, and its value must be published to
        HBM because a consumer sits outside its loop nest. A consumer outside
        its segment, or a graph output, makes it unshareable; a consumer in
        its segment gets an ``apart`` literal, and sharing the nest requires
        their divisions to slice it alike."""
        if item not in self._tiled:
            return
        is_tiled = self._tiled[item]
        diffs: list[Var] = []
        unshareable = getattr(item, "boundary", None) == BufferType.Output
        seg = segments.get(item) if isinstance(item, ResourceNode) else None
        outputs = item.outputs if isinstance(item, ResourceNode) else []
        for edge, child in ((e, e.dst) for e in outputs for _ in range(e.reads)):
            if not self.chooses_split(child):
                continue
            if seg is None or segments.get(child) != seg:
                unshareable = True
                continue
            d = model.BoolVar(f"apart_{item.name}_{child.name}")
            model.Add(self._group[item] != self._group[child]).OnlyEnforceIf(d)
            model.Add(self._group[item] == self._group[child]).OnlyEnforceIf(d.Not())
            _gate_divisions(
                model,
                self._pairs(edge),
                self.vars[item].division,
                self.vars[child].division,
                d.Not(),
            )
            diffs.append(d)
        cut = model.BoolVar(f"cut_{item.name}")
        if unshareable or not diffs:
            model.Add(cut == is_tiled)
        else:
            any_diff = model.BoolVar(f"anydiff_{item.name}")
            model.AddMaxEquality(any_diff, diffs)
            model.AddBoolAnd([is_tiled, any_diff]).OnlyEnforceIf(cut)
            model.AddBoolOr([is_tiled.Not(), any_diff.Not()]).OnlyEnforceIf(cut.Not())
        self.cut_terms.append(cut)

    def encode_tile_counts(self, item: Item, model: Model) -> None:
        """The tile count of the chosen tiling, when the candidates differ."""
        if self.open_choices(item) < 2:
            return
        counts = [cd.tiling.tile_count for cd in self._candidates(item)]
        if min(counts) == max(counts):
            return
        var = model.IntVar(min(counts), max(counts), f"tile_count_{item.name}")
        model.AddElement(self.vars[item].division, counts, var)
        self.tile_terms.append(var)

    # ---- 8. clones ----

    def encode_clone(self, node: ResourceNode, model: Model) -> None:
        """A graph input's relation to its consumers: ordinary edge tables."""

    # ---- objective inputs and extraction ----

    def objective_terms(self, item: Item) -> tuple[Optional[Var], Optional[Var]]:
        """``(cores, core_cost)`` for the parallelism and shape stages, from an
        open work split only: a fixed one is a constant that cannot move the
        optimum."""
        if self.open_choices(item) < 2:
            return None, None
        v = self.vars[item]
        return v.cores, v.core_cost

    def cost_bindings(self) -> tuple[dict, dict]:
        """``sym_map`` / ``buffer_map`` for lowering the cost expression: each
        cost-model symbol to its var, and the ``_division_of_*`` entries that
        let a table over candidates select by identity."""
        sym_map: dict = {}
        buffer_map: dict = {}
        for item, v in self.vars.items():
            sym_map[item.sym_is_lx.name] = v.in_buffer
            if v.buffer is None:
                continue
            sym_map[v.buffer.sym_division.name] = v.division
            sym_map[f"_division_of_{v.buffer.sym_division.name}"] = v
            product = []
            for key, symbol in v.buffer.sym_core_divs.items():
                sym_map[symbol.name] = v.splits[key]
                buffer_map[symbol.name] = (v, v.splits_raw[key])
                product.append(symbol.name)
            product.sort()
            symbol = sympy.Symbol("_product_" + "_".join(product))
            sym_map[symbol.name] = v.cores
            buffer_map[symbol.name] = (v, v.cores_used)
            _, sym_tile_counts = self._syms(item)
            for key, symbol in sym_tile_counts.items():
                sym_map[symbol.name] = v.tile_counts[key]
                buffer_map[symbol.name] = (v, v.tile_counts_raw[key])
        return sym_map, buffer_map

    def chosen_split(self, item: Item, solver) -> WorkSplit:
        v = self.vars[item]
        if v.division is None:
            return WorkSplit.COMMITTED
        i = self.solved[item] = solver.Value(v.division)
        return WorkSplit.of(self._candidates(item)[i])

    def footprint(self, item: Item, solver) -> int:
        v = self.vars[item]
        return item.size if v.division is None else v.per_core[solver.Value(v.division)]

    def record_fired(self, edge: DependencyEdge, solver) -> None:
        """The relayout candidate the read on ``edge`` fired through: the one
        on its copy for this consumer under the solved division pair."""
        assert edge.chosen is not None
        i = solver.Value(self.vars[edge.src].division)
        j = solver.Value(self.vars[edge.dst].division)
        (self.fired[edge],) = [
            c
            for c in self._candidates_of[edge.chosen]
            if c.consumer == edge.dst.name
            and c.source_division == i
            and c.consumer_division == j
        ]


@dataclass(frozen=True)
class EnumerationFactory:
    """Built from configuration only: how each axis is enumerated. The
    defaults reproduce the table model."""

    division: AxisSetting = "table"
    tiling: AxisSetting = "table"
    relayout: RelayoutSetting = "data"

    def __post_init__(self) -> None:
        for axis in ("division", "tiling"):
            if getattr(self, axis) not in ("table", "lazy", "pruned"):
                raise ValueError(f"unknown {axis} enumeration {getattr(self, axis)!r}")
        if self.relayout not in ("data", "pruned", "lazy"):
            raise ValueError(f"unknown relayout enumeration {self.relayout!r}")

    def candidates(self, allocator: CoOptimizingAllocator) -> TableCandidates:
        """What generates the table candidates for ``allocator``'s solves."""
        from torch_spyre._inductor.scratchpad.table_candidates import (
            TableCandidates,
        )

        return TableCandidates(allocator)

    def strategy(self, **candidates) -> EncodingStrategy:
        if self.division == "table" and self.tiling == "table":
            return TableStrategy(
                division=self.division,
                tiling=self.tiling,
                relayout=self.relayout,
                **candidates,
            )
        raise NotImplementedError(
            f"no strategy for division={self.division} tiling={self.tiling}"
        )
