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

from abc import ABC, abstractmethod

from torch._inductor.graph import GraphLowering
from torch._inductor.ops_handler import WrapperHandler

from torch_spyre._inductor.work_division import (
    cost_model_matmul_division,
    span_reduction,
    work_distribution,
)


class ScratchpadOptimizationPass(ABC):
    """
    Abstract class for optimization passes which are implemented to improve
    a graph's overall scratchpad memory utilization and/or memory latency.
    """

    @abstractmethod
    def apply_pass(self, graph: GraphLowering):
        """
        Accepts a candidate graph to be optimized and evaluated for scratchpad memory allocation.
        `graph` will be mutated according in an implementation defined way. The order and
        number of nodes in the graph may change as a result of an optimization pass.

        Args:
            graph (GraphLowering): The graph to be optimized for scratchpad memory allocation
        """
        pass


class SpanReductionPass(ScratchpadOptimizationPass):
    """Commit the minimum per-op splits required by ``MAX_SPAN_BYTES``.

    Mandatory: an op left unsplit whose per-core span exceeds the hardware
    limit only logs CRITICAL (``warn_if_per_core_overflow``) and fails later in
    the backend, so this must run on every compile -- including when
    ``config.lx_planning`` is off, which is why the allocator runs its
    pre-optimization passes before that gate.

    NOT idempotent; see :class:`WorkDistributionPass`.
    """

    def apply_pass(self, graph: GraphLowering):
        span_reduction(graph)


class WorkDistributionPass(ScratchpadOptimizationPass):
    """Spend the remaining cores across ops to maximize parallelism.

    NOT idempotent, and must run exactly once per compile: both
    ``work_distribution`` and ``cost_model_matmul_division`` read an
    already-committed ``op_it_space_splits`` as a hard *floor*, and
    ``apply_splits`` never clears a stale attribute -- so a second run ratchets
    splits upward rather than reproducing them. It must also run before
    ``_push_allocation`` inserts boundary clones, whose splits are hand-re-keyed
    by ``GraphEditor`` and would be clobbered by a re-division.
    """

    def apply_pass(self, graph: GraphLowering):
        # cost_model_matmul_division claims a subset of ops; work_distribution
        # skips those so every op is divided by exactly one of the two passes.
        preassigned_ops = cost_model_matmul_division(graph)
        work_distribution(graph, preassigned_ops)


class _NameSwapHandler(WrapperHandler):
    def __init__(self, inner, name_map: dict[str, str]):
        super().__init__(inner)
        self._name_map = name_map

    def load(self, name, index):
        return super().load(self._name_map.get(name, name), index)
