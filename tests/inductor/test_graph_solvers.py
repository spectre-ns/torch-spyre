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
