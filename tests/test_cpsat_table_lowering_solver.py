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
"""The table lowering inside the CP-SAT layout solver, which lowers its cost objective
from tables. No device: each test plans two buffers. The last test needs no ortools,
so this file has no module-level skip."""

import contextlib
import importlib.util
import subprocess
import sys
import textwrap
from unittest import mock

import pytest
import sympy

from torch_spyre._inductor import config
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    solved_bindings,
)

needs_ortools = pytest.mark.skipif(
    importlib.util.find_spec("ortools") is None,
    reason="a solve needs ortools, which is not installed on s390x or ppc64le",
)


def _buffers() -> list[CoreDivisionBuffer]:
    """A buffer whose two divisions use 8 and 32 cores, and one whose residency the
    objective prices."""
    return [
        CoreDivisionBuffer(
            "reduction_out",
            128,
            [0, 1],
            core_divisions=[
                CoreDivision(splits={"b": 1, "h": 4, "q": 2}),
                CoreDivision(splits={"b": 2, "h": 4, "q": 4}),
            ],
            residency_reason="no consumer reads it from LX",
        ),
        CoreDivisionBuffer("b1", 128, [1, 3], core_divisions=[CoreDivision()]),
    ]


def _cost(buffers: list[CoreDivisionBuffer]) -> sympy.Expr:
    first, second = buffers
    return 32_000 / first.sym_cores + 4000 * (1 - second.sym_is_lx)


@needs_ortools
def test_the_solve_lowers_its_objective_from_tables_at_its_optimum():
    """OPTIMAL with the objective used: 32 cores for the reduction and the second
    buffer resident, the plan of least exact cost."""
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import CpSatLayoutSolver

    buffers = _buffers()
    cost = _cost(buffers)
    solver = CpSatLayoutSolver(buffers, 1 << 20, alignment=1)
    result = solver.plan_layout_and_core_divisions(cost)
    stats = solver.last_solve_stats
    assert (stats["status"], stats["objective_used"]) == ("OPTIMAL", True)
    assert float(cost.xreplace(solved_bindings(result))) == 1000.0


@needs_ortools
@pytest.mark.parametrize("failure", ["refusal", "ValueError"])
def test_a_failure_in_the_lowering_leaves_the_solve(failure):
    """As TableLoweringError, which the solver's except around the objective does not
    catch into its fallback objective, even where _cpsat_warn_on_cost_expr lets the
    printer's failures fall back: a node of a class the numpy core has no function
    for, and a ValueError raised inside the lowering."""
    from torch_spyre._inductor.scratchpad import cpsat_table_lowering
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import CpSatLayoutSolver

    buffers = _buffers()
    cost = _cost(buffers)
    patch: contextlib.AbstractContextManager = contextlib.nullcontext()
    if failure == "refusal":
        cost = cost + sympy.Function("NoSuchNode")(buffers[1].sym_is_lx)
    else:
        patch = mock.patch.object(
            cpsat_table_lowering.TableLowering, "print", side_effect=ValueError("boom")
        )
    solver = CpSatLayoutSolver(buffers, 1 << 20, alignment=1)
    with config.patch({"_cpsat_warn_on_cost_expr": True}), patch:
        with pytest.raises(cpsat_table_lowering.TableLoweringError):
            solver.plan_layout_and_core_divisions(cost)
    assert solver.last_solve_stats == {}  # no fallback solve followed


def test_with_auto_coarse_tiling_the_allocator_hands_the_solver_its_cost():
    """Where the solver chooses tilings too: the tables lower the cost model's tiling
    terms, so the allocator withholds the expression from no solver. An empty graph
    costs 0, which is not None."""
    from types import SimpleNamespace

    import torch_spyre._inductor.scratchpad.allocator as allocator_module

    solver = mock.create_autospec(
        allocator_module.CoreDivisionLayoutSolver, instance=True
    )
    solver.buffers = []
    solver.chooses_tilings.return_value = True
    solver.plan_layout_and_core_divisions.return_value = []
    allocator = allocator_module.CoOptimizingAllocator(
        # Never built: the test hands _solve the mock directly.
        layout_planning=allocator_module.ExhaustiveSearchSolver.wrapping(
            allocator_module.GreedyLayoutSolver
        ),
        size=0,
    )
    graph = SimpleNamespace(operations=[], get_output_names=lambda: [])
    with (
        config.patch({"auto_coarse_tiling": True}),
        mock.patch.object(allocator_module, "mem_usage_by_buf", return_value={}),
    ):
        allocator._solve(solver, graph)
    solver.plan_layout_and_core_divisions.assert_called_once_with(0)


def test_the_table_lowering_and_the_solver_import_without_ortools():
    """In a fresh interpreter that cannot import ortools, as on s390x and ppc64le, the
    table module imports with cp_model None, and so does ilp_solver_ortools."""
    code = textwrap.dedent(
        """
        import sys
        sys.modules["ortools"] = None
        import torch
        from torch_spyre._inductor.scratchpad import cpsat_table_lowering
        from torch_spyre._inductor.scratchpad import ilp_solver_ortools
        assert cpsat_table_lowering.cp_model is None
        assert ilp_solver_ortools.cp_model is None
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stderr[-3000:]
