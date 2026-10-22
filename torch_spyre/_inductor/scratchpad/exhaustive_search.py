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

import functools
from collections.abc import Sequence
from typing import Any, ClassVar, cast

from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivisionBuffer,
    CoreDivisionLayoutSolver,
    LifetimeBoundBuffer,
    MemoryPlanSolver,
)

__all__ = ["ExhaustiveSearchSolver"]


class ExhaustiveSearchSolver(CoreDivisionLayoutSolver):
    """Joint core-division + placement by exhaustive search: the solver
    facade of :class:`graph_solvers.DecomposedPlacement`, which enumerates
    every open buffer's division menu and places each leaf with a fresh
    placement-only solver, keeping the leaf with the least HBM traffic.

    The inner solver is a class parameter, so that the wrapper is a class the
    allocator can ask about capabilities: build one with :meth:`wrapping`.
    """

    _inner_factory: ClassVar[type[MemoryPlanSolver]]

    @classmethod
    @functools.cache
    def wrapping(cls, inner: type[MemoryPlanSolver]) -> type["ExhaustiveSearchSolver"]:
        """The exhaustive search over placements by ``inner``; one class per
        ``inner``."""
        name = f"{cls.__name__}[{inner.__name__}]"
        namespace = {
            "_inner_factory": inner,
            # type() would take these from the frame creating the class,
            # ABCMeta.__new__ in abc.py.
            "__module__": cls.__module__,
            "__qualname__": name,
        }
        return cast("type[ExhaustiveSearchSolver]", type(name, (cls,), namespace))

    @classmethod
    def supports_paired_buffers(cls) -> bool:
        return False

    @classmethod
    def decides_lx_relayouts(cls) -> bool:
        return False

    @classmethod
    def generates_divisions(cls) -> bool:
        return False

    @classmethod
    def chooses_tilings(cls) -> bool:
        return False

    @classmethod
    def tilings_from_menu(cls) -> bool:
        return False

    @classmethod
    def replans_after_tiling(cls) -> bool:
        return False

    @classmethod
    def linear_cost_only(cls) -> bool:
        return False

    def __init__(
        self,
        buffers: Sequence[LifetimeBoundBuffer],
        size: int,
        alignment: int = 128,
    ):
        if not hasattr(type(self), "_inner_factory"):
            raise TypeError(
                "ExhaustiveSearchSolver needs an inner solver: use "
                "ExhaustiveSearchSolver.wrapping(inner)"
            )
        super().__init__(buffers, size, alignment)

    # ------------------------------------------------------------------
    # MemoryPlanSolver contract
    # ------------------------------------------------------------------

    def plan_layout(self, log_lx_usage: bool = False) -> list[LifetimeBoundBuffer]:
        # Dispatch is per buffer, not per class (mirrors CpSatLayoutSolver): a
        # buffer with no candidate divisions to choose among places here just
        # as well as through the joint entry point.
        return cast(
            "list[LifetimeBoundBuffer]",
            self.plan_layout_and_core_divisions(),
        )

    # ------------------------------------------------------------------
    # CoreDivisionLayoutSolver contract
    # ------------------------------------------------------------------

    def graph_solver(self) -> Any:
        """The search over the graph: each leaf of the division menus placed
        by the inner solver."""
        from torch_spyre._inductor.scratchpad.graph_solvers import (
            DecomposedPlacement,
        )

        return DecomposedPlacement(self._inner_factory, self.limit)

    def plan_layout_and_core_divisions(
        self, cost_expr=None
    ) -> list[CoreDivisionBuffer]:
        from torch_spyre._inductor.scratchpad.graph_solvers import solve_buffers

        return cast(
            "list[CoreDivisionBuffer]", solve_buffers(self, cost_expr, joint=True)
        )
