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

"""What ``SpyreOptimizationGraph.solve`` runs on: the ``Decomposed`` backend
and the graph solvers. Device-free."""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("ortools")

from torch_spyre._inductor.scratchpad.backend import CpSat, Decomposed  # noqa: E402
from torch_spyre._inductor.scratchpad.buffer_graph import (  # noqa: E402
    SolveSettings,
    build_graph,
)
from torch_spyre._inductor.scratchpad.encoding_strategy import (  # noqa: E402
    TableStrategy,
)
from torch_spyre._inductor.scratchpad.graph_solvers import (  # noqa: E402
    DecomposedPlacement,
    PlacementSolver,
    solve_buffers,
)
from torch_spyre._inductor.scratchpad.greedy_solver import (  # noqa: E402
    GreedyLayoutSolver,
)
from torch_spyre._inductor.scratchpad.plan_solver import (  # noqa: E402
    CoreDivision,
    CoreDivisionBuffer,
    LifetimeBoundBuffer,
)


def _settings() -> SolveSettings:
    return SolveSettings(
        time_limit_seconds=None,
        bottom_justify=False,
        record=lambda solver, model, objective=False: solver.Solve(model),
    )


def _menu_buffers() -> list[CoreDivisionBuffer]:
    """Three buffers that fit in LX together only when two of them split."""
    menu = [CoreDivision(), CoreDivision(splits={1: 2})]
    every = [(i, j) for i in range(2) for j in range(2)]

    def buf(name, uses, parents=(), **kw):
        return CoreDivisionBuffer(
            name=name,
            size=4096,
            uses=list(uses),
            core_divisions=list(menu),
            parents=list(parents),
            **kw,
        )

    return [
        buf("a", [0, 1, 2]),
        buf("b", [1, 2], ["a"], cd_parent_matches={"a": every}),
        buf("c", [2, 3], ["b"], cd_parent_matches={"b": every}),
    ]


def _spilled_bytes(graph) -> int:
    return sum(n.nbytes for n in graph.get_nodes() if not n.resident)


def test_decomposed_reaches_the_cpsat_optimum():
    results = {}
    for backend in (CpSat, Decomposed):
        buffers = _menu_buffers()
        graph = build_graph(buffers, 128, 48, _settings())
        graph.set_solver(backend)
        graph.solve(None, None, TableStrategy.from_buffers(buffers))
        results[backend.__name__] = _spilled_bytes(graph)
    assert results["Decomposed"] == results["CpSat"]


def test_decomposed_refuses_too_many_leaves():
    from torch_spyre._inductor.scratchpad.backend import TooBig

    buffers = _menu_buffers()
    graph = build_graph(buffers, 128, 48, _settings())
    with pytest.raises(TooBig):
        graph.solve(Decomposed(max_leaves=2), None, TableStrategy.from_buffers(buffers))


def test_a_graph_solver_that_refuses_the_strategy_is_not_run():
    class Refuses:
        def accepts(self, strategy):
            return False

        def solve(self, graph, strategy, cost_expr):
            raise AssertionError("not reached")

    graph = build_graph([LifetimeBoundBuffer("x", 128, [0, 1])], 128, 16, _settings())
    graph.set_solver(Refuses())
    with pytest.raises(ValueError, match="cannot solve"):
        graph.solve()


def _placement_buffers() -> list[LifetimeBoundBuffer]:
    source = LifetimeBoundBuffer("source", 256, [1, 2, 6])
    dest = LifetimeBoundBuffer("dest", 128, [2, 3, 5])
    source.paired_with.append(dest)
    return [
        dest,
        source,
        LifetimeBoundBuffer("other", 256, [0, 4]),
        LifetimeBoundBuffer("barred", 128, [0, 7], residency_reason="op not allowed"),
    ]


def test_placement_solver_packs_the_views_as_the_solver_packs_buffers():
    direct = _placement_buffers()
    GreedyLayoutSolver(direct, 512).plan_layout()
    buffers = _placement_buffers()
    graph = build_graph(
        buffers, 128, 4, _settings(), {b.name: b.residency_reason for b in buffers}
    )
    # A committed relayout's destination is a transform pinned on its source.
    assert [t.name for t in graph.nodes["source"].pinned_transforms] == ["dest"]
    assert "dest" not in graph.nodes
    graph.set_solver(PlacementSolver(GreedyLayoutSolver, 512))
    graph.solve()
    placed = {
        item.name: item.offset
        for node in graph.get_nodes()
        for item in (node, *node.pinned_transforms)
    }
    assert placed == {
        b.name: None if b.address is None else b.address // 128 for b in direct
    }


def test_solve_buffers_writes_the_placement_back():
    direct = _placement_buffers()
    GreedyLayoutSolver(direct, 512).plan_layout()
    buffers = _placement_buffers()
    solve_buffers(GreedyLayoutSolver(buffers, 512))
    assert [(b.name, b.address, b.spill_reason) for b in buffers] == [
        (b.name, b.address, b.spill_reason) for b in direct
    ]


def test_decomposed_placement_keeps_the_leaf_with_least_hbm_traffic():
    buffers = _menu_buffers()
    graph = build_graph(buffers, 128, 48, _settings())
    strategy = TableStrategy.from_buffers(buffers)
    graph.set_solver(DecomposedPlacement(GreedyLayoutSolver, 48 * 128))
    graph.solve(None, None, strategy)
    best = _spilled_bytes(graph)
    # Every other leaf spills at least as much.
    import itertools

    for leaf in itertools.product(range(2), repeat=3):
        trial = copy.deepcopy(_menu_buffers())
        for b, i in zip(trial, leaf):
            b.core_divisions = [b.core_divisions[i]]
        leaf_graph = build_graph(trial, 128, 48, _settings())
        leaf_graph.set_solver(DecomposedPlacement(GreedyLayoutSolver, 48 * 128))
        leaf_graph.solve(None, None, TableStrategy.from_buffers(trial))
        assert _spilled_bytes(leaf_graph) >= best
