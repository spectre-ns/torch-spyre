from collections import defaultdict
import copy
from dataclasses import dataclass
from typing import Callable, Optional, override
from unittest import TestCase, expectedFailure
import os
from functools import wraps
from torch_spyre._inductor.layout_backend import (
    Buffer,
    Operation,
    Component,
    Allocation,
    AllocationResult,
    calculate_liveness,
    SimulatedAnnealingAllocationStrategy
)


from torch_spyre._inductor.scratchpad import (
    scratchpad_planning,
    AllocationStrategy,
    DefaultAllocationStrategy,
    SpyreLxOptimizationPass,
    GreedyLayoutSolver,
    InputBufferOptimization,
    LayoutSolver
)
from torch_spyre._inductor import config

# From scratchpad.py
AVAILABLE_LX_SIZE = int((2 << 20) * (1.0 - config.dxp_lx_frac_avail))

if os.environ.get("SCRATCHPAD_PATTERN_BYPASS_XFAIL", "0") == "1":
    # Define usuallyExpectedFailure as a no-op. This should show failures indicating that the
    # current allocation uses more HBM than the good allocation, not anything else.
    def usuallyExpectedFailure(test_item: Callable) -> Callable:
        @wraps(test_item)
        def wrapper(*args, **kwargs):
            return test_item(*args, **kwargs)

        return wrapper
else:
    usuallyExpectedFailure = expectedFailure


@dataclass
class Pattern:
    buffers: dict[str, Buffer]
    operations: list[Operation]
    # A "good" allocation pattern that we want to compare to. The test verifies that this pattern
    # is valid and that the current result is at least as good -- that is, the HBM usage of the
    # current result is no more than that of the good pattern.
    good_allocation: AllocationResult

    def determine_inputs_outputs(self) -> tuple[list[str], list[str]]:
        # A buffer is an input if it is read before it is written. A buffer is an output if it is
        # only written to.
        bufs_written_to = set()
        bufs_read_from = set()
        inputs = set()

        for op in self.operations:
            bufs_read_from.update(op.inputs)
            for buf in op.inputs:
                if buf not in bufs_written_to:
                    inputs.add(buf)
            bufs_written_to.update(op.outputs)

        outputs = list(bufs_written_to.difference(bufs_read_from))
        return (list(inputs), outputs)


def make_buffer_registry(names_sizes: dict[str, int]) -> dict[str, Buffer]:
    return {name: Buffer(name=name, size=size) for (name, size) in names_sizes.items()}


def make_allocation_result(lists: list[list[Allocation]]) -> AllocationResult:
    return [{alloc.buffer: alloc for alloc in lst} for lst in lists]


class IdentityOptimizationPass(SpyreLxOptimizationPass):
    def apply_pass(self, operations: list[Operation]) -> list[Operation]:
        return operations

class MockAllocationStrategy(DefaultAllocationStrategy):
    def __init__(
            self,
            optimization_passes: list[SpyreLxOptimizationPass] | None = None,
            layout_planning: list[LayoutSolver] | None = None
    ):
        super().__init__(optimization_passes, layout_planning)
        self.allocations = []

    @override
    def push_allocation(self, allocation: AllocationResult):
        self.allocations = allocation

class InstrumentedLayoutSolver(GreedyLayoutSolver):
    def __init__(self, pattern: Pattern):
        super().__init__()
        self.inputs, self.outputs = pattern.determine_inputs_outputs()

    @override
    def op_output_good_for_lx_reuse(self, org_op_name: str) -> bool:
        return True

    @override
    def mem_usage_by_op(self, op: Operation) -> dict[str, dict[str, bool | int]]:
        # Returns a dict mapping each buffer name to a dict with keys "is_input" and "size".
        # is_input is True if the buffer is an input to the op, and False otherwise. size is the
        # size of the buffer.
        result = {}
        for tensor_name in op.inputs:
            result[tensor_name] = {
                "is_input": True,
                "size": op._buffer_registry[tensor_name].size,
            }
        for tensor_name in op.outputs:
            result[tensor_name] = {
                "is_input": False,
                "size": op._buffer_registry[tensor_name].size,
            }
        return result

    @override
    def get_output_names(self) -> list[str]:
        return self.outputs

    @override
    def is_graph_input(self, buffer: str) -> bool:
        return buffer in self.inputs


class MockGraphLowering:
    """This class impersonates V.graph."""

    def __init__(self, pattern: Pattern):
        self.graph_input_names = pattern.determine_inputs_outputs()[0]
        self.buffers = pattern.buffers

    def get_buffer(self, buf: str) -> Buffer:
        return self.buffers[buf]


class InstrumentedInputBufferOptimization(InputBufferOptimization):
    def __init__(self, pattern: Pattern):
        super().__init__(MockGraphLowering(pattern))
        self.buffers = pattern.buffers
        self.operations = pattern.operations

    @override
    def should_consider_op(self, op: Operation) -> bool:
        return True

    def new_name(self, prefix: str, current_names: set[str]) -> str:
        candidate = prefix
        i = 0
        while candidate in current_names:
            candidate = f"{prefix}_{i}"
            i += 1
        return candidate

    @override
    def insert_op_after(
        self,
        buf: Buffer,
        lowering_func: Callable,
        buf_users: dict,
        operations: list[Operation],
    ) -> None:
        buf_index = [i for i, op in enumerate(operations) if buf.name in op.inputs]
        if not buf_index:
            raise ValueError(
                f"Was asked to insert after {buf.name}, but couldn't find it"
            )

        buffer_name = self.new_name("copy_buf", {buf for buf in self.buffers})
        self.buffers[buffer_name] = Buffer(buffer_name, buf.size)

        op_name = self.new_name("copy_op", {op.name for op in self.operations})
        new_op = Operation(
            op_name,
            inputs=[buf.name],
            outputs=[buffer_name],
            _buffer_registry=self.buffers,
        )

        # The expected order in the list of operations is actually *before* the first operation
        # that uses buf.
        self.operations.insert(buf_index[0], new_op)

        for op in self.operations[buf_index[0] + 1 :]:
            op.inputs = [
                buffer_name if buf.name == input else input for input in op.inputs
            ]
            op.outputs = [
                buffer_name if buf.name == output else output for output in op.outputs
            ]


class TestExamplePattern(TestCase):
    def find_single_use_buffers(
        self,
        operations: list[Operation],
        *,
        see_later: Optional[Callable[[Operation, str], None]] = None,
        see_first: Optional[Callable[[Operation, str], None]] = None,
    ) -> set[str]:
        """Returns the set of buffers that are used only once in the list of operations. see_first
        is called the first time any buffer is seen, and see_later is called any other time any
        buffer is seen."""
        single_use_buffers = set()
        seen_buffers = set()
        for op in operations:
            for buffer_name in op.inputs + op.outputs:
                if buffer_name in seen_buffers:
                    if see_later is not None:
                        see_later(op, buffer_name)
                    single_use_buffers.discard(buffer_name)
                else:
                    if see_first is not None:
                        see_first(op, buffer_name)
                    single_use_buffers.add(buffer_name)
                    seen_buffers.add(buffer_name)

        return single_use_buffers

    def verify_pattern(self, pattern: Pattern, *, inplace: bool = False):
        allocation = pattern.good_allocation
        operations = pattern.operations
        self.assertEqual(
            len(allocation),
            len(operations),
            f"Good allocation should have the same number of entries as the number of operations, "
            f"but found {len(allocation)} allocations and {len(operations)} operations.",
        )
        for alloc in allocation:
            for a in alloc.values():
                self.assertEqual(
                    a.address is not None,
                    a.component == Component.LX,
                    f"Buffers should have an address iff they are allocated in LX, but found {a}.",
                )

        # Buffers that are used only once need not be allocated in the scratchpad, because it
        # doesn't help reduce HBM transfers. In the meantime, verify that we didn't write any
        # operations that write to a buffer, except possibly the first time we see that buffer.
        def no_output(op: Operation, buffer_name: str):
            self.assertNotIn(
                buffer_name,
                op.outputs,
                f"Buffer {buffer_name} is written to in operation {op.name}, but accessed before "
                f"that operation. However, this test is case is not marked as in-place, so we "
                f"avoid in-place operations.",
            )

        single_use_buffers = self.find_single_use_buffers(
            operations, see_later=None if inplace else no_output
        )

        for i, op in enumerate(operations):
            # Check that each buffer that is used is allocated.
            for buffer_name in op.inputs + op.outputs:
                if buffer_name not in single_use_buffers:
                    self.assertTrue(
                        any(
                            alloc.buffer == buffer_name
                            for alloc in allocation[i].values()
                        ),
                        f"Buffer {buffer_name} used by operation {op.name} is not allocated at "
                        f"this point in the good allocation pattern, but it is used more than once.",
                    )

            # Check that there is at least one output.
            self.assertGreater(
                len(op.outputs),
                0,
                f"Operation {op.name} should have at least one output.",
            )

            # Check that allocated buffers do not overlap.
            allocated_buffers = [
                alloc for alloc in allocation[i].values() if alloc.address is not None
            ]
            if allocated_buffers:
                # Sort by address:
                sorted_allocations = sorted(
                    list(allocated_buffers),
                    key=lambda x: x.address,  # pyright: ignore[reportCallIssue, reportArgumentType]
                )
                for j in range(len(sorted_allocations) - 1):
                    buffer_name_j = sorted_allocations[j].buffer
                    addr_j = sorted_allocations[j].address
                    buffer_name_next = sorted_allocations[j + 1].buffer
                    addr_next = sorted_allocations[j + 1].address
                    size_j = op._buffer_registry[buffer_name_j].size
                    self.assertLessEqual(
                        addr_j + size_j,
                        addr_next,
                        f"Buffers {buffer_name_j} and {buffer_name_next} overlap during operation "
                        f"{op.name}",
                    )

                self.assertLessEqual(
                    sorted_allocations[-1].address
                    + op._buffer_registry[sorted_allocations[-1].buffer].size,
                    AVAILABLE_LX_SIZE,
                    f"Buffer {sorted_allocations[-1].buffer} exceeds scratch pad size during "
                    f"operation {op.name}",
                )

    def verify_actual_run(self, pattern: Pattern, alloc):
        (liveness_start, liveness_end) = calculate_liveness(pattern.operations)
        
        # Sanity check -- every buffer should have a start and an end to its liveness.
        self.assertTrue(set(liveness_start.keys()) == set(liveness_end.keys()))

        allocate_at = defaultdict(list)
        deallocate_at = defaultdict(list)
        for buffer_name in liveness_start:
            if buffer_name in [a.buffer for a in alloc.allocations]:
                allocate_at[liveness_start[buffer_name]].append(buffer_name)
                deallocate_at[liveness_end[buffer_name] + 1].append(buffer_name)

        live_buffers = set()
        for i, op in enumerate(pattern.operations):
            live_buffers.update(allocate_at[i])
            for buffer_name in op.inputs + op.outputs:
                if buffer_name not in [a.buffer for a in alloc.allocations]:
                    # This buffer resides in HBM.
                    continue

                # Verify that buffer_name does not overlap with any allocated buffers at this point.
                allocation = next(x for x in alloc.allocations if x.buffer == buffer_name)
                addr = allocation.address
                size = op._buffer_registry[buffer_name].size
                self.assertLessEqual(
                    addr + size,
                    AVAILABLE_LX_SIZE,
                    f"Buffer {buffer_name} exceeds scratch pad size during operation {op.name}",
                )
                for other_buffer_name in live_buffers:
                    if (
                        other_buffer_name == buffer_name
                        or other_buffer_name not in alloc.allocations
                    ):
                        continue
                    other_allocation = next(x for x in alloc.allocations if x.buffer == other_buffer_name)
                    other_addr = other_allocation.address
                    other_size = op._buffer_registry[other_buffer_name].size
                    if addr <= other_addr:
                        self.assertLessEqual(
                            addr + size,
                            other_addr,
                            f"Buffers {buffer_name} and {other_buffer_name} overlap during "
                            f"operation {op.name}",
                        )
                    else:
                        self.assertLessEqual(
                            other_addr + other_size,
                            addr,
                            f"Buffers {buffer_name} and {other_buffer_name} overlap during "
                            f"operation {op.name}",
                        )
            live_buffers.difference_update(deallocate_at[i + 1])

    def hbm_usage_for_good_allocation(
        self, allocation: AllocationResult, operations: list[Operation]
    ) -> int:
        if not operations:
            return 0
        registry = operations[0]._buffer_registry

        single_use_buffers = self.find_single_use_buffers(operations)
        hbm_usage = sum(
            registry[buffer_name].size for buffer_name in single_use_buffers
        )

        for i, op in enumerate(operations):
            for buffer_name in op.inputs:
                if buffer_name not in single_use_buffers and (
                    i == 0 or buffer_name not in allocation[i - 1]
                ):
                    # This buffer is not allocated in the scratch pad before this operation, so it
                    # must be loaded from HBM.
                    hbm_usage += registry[buffer_name].size
            for buffer_name in op.outputs:
                if buffer_name not in single_use_buffers and (
                    i == len(operations) - 1 or buffer_name not in allocation[i + 1]
                ):
                    # This buffer is not allocated in the scratch pad after this operation, so it
                    # must be stored to HBM.
                    hbm_usage += registry[buffer_name].size
        return hbm_usage

    def hbm_usage_for_actual_run(
        self, operations: list[Operation], alloc
    ) -> int:
        if not operations:
            return 0

        hbm_usage = 0

        # Count all usage for buffers not allocated in the scratchpad.
        for i, op in enumerate(operations):
            for buffer_name in op.inputs + op.outputs:
                if buffer_name not in [a.buffer for a in alloc.allocations]:
                    # This buffer is not allocated in the scratch pad before this operation, so it
                    # must be loaded from HBM.
                    hbm_usage += op._buffer_registry[buffer_name].size

        # All buffers allocated in the scratchpad are counted only once each.
        for buffer in alloc.allocations:
            hbm_usage += operations[0]._buffer_registry[buffer.buffer].size

        return hbm_usage

    def run_pattern(self, pattern: Pattern):
        # The scratchpad_planning operation may modify the pattern (adding operations), and then
        # examining the "good" allocation will run into trouble.
        pattern_copy = copy.deepcopy(pattern)
        # alloc = InstrumentedAllocator(pattern_copy)
        # strategy = InstrumentedGreedyAllocationStrategy(pattern_copy, alloc)

        strategy = MockAllocationStrategy(
            [InstrumentedInputBufferOptimization(pattern_copy)],
            [InstrumentedLayoutSolver(pattern_copy)]
        )

        scratchpad_planning(pattern_copy.operations, strategy)

        # Verify that the currently implemented allocation is indeed valid
        self.verify_actual_run(pattern_copy, strategy)

        # Verify that the currently implemented allocation is at least as good as the "good
        # allocation" in terms of HBM usage.
        current_hbm_usage = self.hbm_usage_for_actual_run(
            pattern_copy.operations, strategy
        )
        good_hbm_usage = self.hbm_usage_for_good_allocation(
            pattern.good_allocation, pattern.operations
        )
        self.assertLessEqual(
            current_hbm_usage,
            good_hbm_usage,
            f"Current allocation uses more HBM ({current_hbm_usage} bytes) than the good allocation ({good_hbm_usage} bytes). ",
        )

    def make_simple_fragmentation_pattern(self) -> Pattern:
        """Allocate two buffers A and B that are each a third of the available scratchpad size,
        where A can be freed after the second operation. Then allocate a third buffer C
        that is two thirds of the scratchpad size. This can only fit if B was allocated at the start
        or end of the scratchpad, leaving a contiguous region for C."""
        third_scratchpad_size = AVAILABLE_LX_SIZE // 3
        third_scratchpad_size = (
            third_scratchpad_size // 128
        ) * 128  # round down to a multiple of the stick size
        buffers = make_buffer_registry(
            {
                "A": third_scratchpad_size,
                "B": third_scratchpad_size,
                "C": 2 * third_scratchpad_size,
                "D": third_scratchpad_size,
                "E": third_scratchpad_size,
            }
        )

        op1 = Operation("op1", inputs=["A"], outputs=["B"], _buffer_registry=buffers)
        op2 = Operation(
            "op2", inputs=["A", "B"], outputs=["D"], _buffer_registry=buffers
        )
        op3 = Operation("op3", inputs=["B"], outputs=["C"], _buffer_registry=buffers)
        op4 = Operation(
            "op4", inputs=["B", "C"], outputs=["E"], _buffer_registry=buffers
        )

        # A is used only during op1 and op2, so we allocate it after B. This way we can
        # evict it after op2 and have enough space for C during op3.
        alloc_A = Allocation(buffer="A", address=third_scratchpad_size)
        alloc_B = Allocation(buffer="B", address=0)
        alloc_C = Allocation(buffer="C", address=third_scratchpad_size)
        alloc_D = Allocation(buffer="D", component=Component.HBM)
        alloc_E = Allocation(buffer="E", component=Component.HBM)
        return Pattern(
            buffers,
            [op1, op2, op3, op4],
            good_allocation=make_allocation_result(
                [
                    [alloc_A, alloc_B],
                    [alloc_A, alloc_B, alloc_D],
                    [alloc_B, alloc_C],
                    [alloc_B, alloc_C, alloc_E],
                ]
            ),
        )

    def test_verify_simple_fragmentation_pattern(self):
        self.verify_pattern(self.make_simple_fragmentation_pattern())

    @usuallyExpectedFailure
    def test_simple_fragmentation_pattern(self):
        self.run_pattern(self.make_simple_fragmentation_pattern())

    def make_staircase_pattern(self) -> Pattern:
        """Allocate N*2 buffers of sizes k, k, 2*k, 2*k, 3*k, 3*k, ..., N*k, N*k. After an
        even-numbered buffer is allocated, free the previous odd-numbered buffer. This creates a
        "staircase" pattern of allocations that can only be fit if the allocator is smart about
        fragmentation. In that case, the maximum scratchpad usage is
        k + 2*k + ... + N*k + N*k = k * N * (N + 1) / 2 + N * k = k * N * (N + 3) / 2, so we choose
        k such that this is just less than the available scratchpad size.

        The greedy allocator will always allocate the next buffer just after all other buffers,
        because no gap is big enough for the current size. So it uses
        2 * (k + 2*k + ... + N*k) = k * N * (N + 1) or roughly 2/3 times more."""
        N = 7
        k = (2 * AVAILABLE_LX_SIZE) // (N * (N + 3))
        k = (k // 128) * 128  # round down to a multiple of the stick size

        # This only works if the greedy allocator uses more than fits in the scratchpad, so we
        # assert that here.
        self.assertGreater(k * N * (N + 1), AVAILABLE_LX_SIZE)

        buffers = make_buffer_registry(
            {f"{letter}{i}": i * k for i in range(1, N + 1) for letter in ["A", "B"]}
            | {f"C{i}": k for i in range(1, N + 2)}
        )

        def op_pair(i: int) -> tuple[Operation, Operation]:
            return (
                Operation(
                    f"op{i}_0",
                    inputs=[f"A{i}"],
                    outputs=[f"B{i}"],
                    _buffer_registry=buffers,
                ),
                Operation(
                    f"op{i}_1",
                    inputs=[f"A{i}", f"B{i}"],
                    outputs=[f"C{i}"],
                    _buffer_registry=buffers,
                ),
            )

        ops = [op for i in range(1, N + 1) for op in op_pair(i)] + [
            Operation(
                "op_final",
                inputs=[f"B{i}" for i in range(1, N + 1)],
                outputs=[f"C{N + 1}"],
                _buffer_registry=buffers,
            )
        ]

        def good_allocation_pair(i: int) -> tuple[list[Allocation], list[Allocation]]:
            # Allocate A{i} at 0 and B{j} for j <= i at N*k, (N+1)*k, (N+3)*k, (N+6)*k, ...,
            # (N+i*(i-1)/2)*k.
            alloc0 = [
                Allocation(buffer=f"B{j}", address=(N + j * (j - 1) // 2) * k)
                for j in range(1, i + 1)
            ]
            alloc0.append(Allocation(buffer=f"A{i}", address=0))
            alloc1 = alloc0 + [Allocation(buffer=f"C{i}", component=Component.HBM)]
            return (alloc0, alloc1)

        good_allocations = [
            alloc for i in range(1, N + 1) for alloc in good_allocation_pair(i)
        ]
        good_allocations.append(
            [
                Allocation(buffer=f"B{i}", address=(N + i * (i - 1) // 2) * k)
                for i in range(1, N + 1)
            ]
            + [Allocation(buffer=f"C{N + 1}", component=Component.HBM)]
        )

        pattern = Pattern(
            buffers, ops, good_allocation=make_allocation_result(good_allocations)
        )
        return pattern

    def test_verify_staircase_pattern(self):
        self.verify_pattern(self.make_staircase_pattern())

    @usuallyExpectedFailure
    def test_staircase_pattern(self):
        self.run_pattern(self.make_staircase_pattern())

    def make_downward_staircase_pattern(self) -> Pattern:
        """Allocate 1+N*2 buffers of sizes k, N*k, N*k, (N-1)*k, (N-1)*k, ..., 2*k, 2*k, k, k.
        After an odd-numbered buffer (>1) is allocated, free the previous even-numbered buffer.
        This creates an easier "staircase" pattern of allocations than in
        `make_staircase_pattern`. Still, the greedy allocator will prefer to allocate
        buffers at the end if it can't allocate them at address 0. So we first allocate one small
        buffer at the start which will block address 0. In the optimal case, the maximum scratchpad
        usage is k + N*k + (N-1)*k + ... + 2*k + k + k = k * (4 + N * (N + 1)) / 2, so we choose k
        such that this is just less than the available scratchpad size.

        The greedy allocator will always allocate the next buffer just after all other buffers,
        up until the point where it reaches the top of available memory and starts looking for gaps.
        The total usage is less clear to analyze."""
        N = 5
        k = (2 * AVAILABLE_LX_SIZE) // (4 + N * (N + 1))
        k = (k // 128) * 128  # round down to a multiple of the stick size

        buffers = make_buffer_registry(
            {
                f"{letter}{i}": (N + 1 - i) * k
                for i in range(1, N + 1)
                for letter in ["A", "B"]
            }
            | {"Z": k}
            | {f"C{i}": k for i in range(N + 2)}
        )

        def op_pair(i: int) -> tuple[Operation, Operation]:
            return (
                Operation(
                    f"op{i}_0",
                    inputs=[f"A{i}"],
                    outputs=[f"B{i}"],
                    _buffer_registry=buffers,
                ),
                Operation(
                    f"op{i}_1",
                    inputs=[f"A{i}", f"B{i}"],
                    outputs=[f"C{i}"],
                    _buffer_registry=buffers,
                ),
            )

        ops = (
            [
                Operation(
                    "op_start",
                    inputs=["Z"],
                    outputs=["C0"],
                    _buffer_registry=buffers,
                )
            ]
            + [op for i in range(1, N + 1) for op in op_pair(i)]
            + [
                Operation(
                    "op_final",
                    inputs=["Z"] + [f"B{i}" for i in range(1, N + 1)],
                    outputs=[f"C{N + 1}"],
                    _buffer_registry=buffers,
                )
            ]
        )

        def good_allocation_pair(i: int) -> tuple[list[Allocation], list[Allocation]]:
            # Allocate Z at 0, A{i} at k, and B{j} for j <= i as follows: B{N} at 2*k, B{N-1} at
            # 3*k, B{N-2} at 5*k, B{N-3} at 8*k, ..., B{N-j} at (j*(j+1)/2 + 2)*k. The gap
            # between A{i} and B{i} = B{N+1-(N+1-i)} is (N+1-i)*(N+2-i)/2*k, which is big enough
            # for A{i+1} of size (N-i)*k.
            alloc0 = [
                Allocation(buffer=f"B{N - j}", address=(j * (j + 1) // 2 + 2) * k)
                for j in range(N - i, N)
            ]
            alloc0.append(Allocation(buffer="Z", address=0))
            alloc0.append(Allocation(buffer=f"A{i}", address=k))
            alloc1 = alloc0 + [Allocation(buffer=f"C{i}", component=Component.HBM)]
            return (alloc0, alloc1)

        good_allocations = [
            [
                Allocation(buffer="Z", address=0),
                Allocation(buffer="C0", component=Component.HBM),
            ]
        ]
        good_allocations.extend(
            [alloc for i in range(1, N + 1) for alloc in good_allocation_pair(i)]
        )
        last_allocation = good_allocation_pair(N)[1]
        last_allocation = [
            alloc for alloc in last_allocation if alloc.buffer != f"A{N}"
        ]
        good_allocations.append(last_allocation)

        pattern = Pattern(
            buffers, ops, good_allocation=make_allocation_result(good_allocations)
        )
        return pattern

    def test_verify_downward_staircase_pattern(self):
        self.verify_pattern(self.make_downward_staircase_pattern())

    def test_downward_staircase_pattern(self):
        self.run_pattern(self.make_downward_staircase_pattern())

    def make_simple_eviction_pattern(self) -> Pattern:
        """This pattern requires allocating a buffer, evicting it, and then reallocating it later.

        We use two buffers A and B that are each exactly the available LX size. We have six
        operations. The first two use A, the next two use B, and the last two use A again. Optimal
        use would allocate A and B for two ops each at alternate times."""
        buffers = make_buffer_registry(
            {"A": AVAILABLE_LX_SIZE, "B": AVAILABLE_LX_SIZE}
            | {f"C{i}": AVAILABLE_LX_SIZE for i in range(1, 7)}
        )
        ops = [
            Operation("op1", inputs=["A"], outputs=["C1"], _buffer_registry=buffers),
            Operation("op2", inputs=["A"], outputs=["C2"], _buffer_registry=buffers),
            Operation("op3", inputs=["B"], outputs=["C3"], _buffer_registry=buffers),
            Operation("op4", inputs=["B"], outputs=["C4"], _buffer_registry=buffers),
            Operation("op5", inputs=["A"], outputs=["C5"], _buffer_registry=buffers),
            Operation("op6", inputs=["A"], outputs=["C6"], _buffer_registry=buffers),
        ]

        good_allocation = [
            [
                Allocation(buffer="A", address=0),
                Allocation(buffer="C1", component=Component.HBM),
            ],
            [
                Allocation(buffer="A", address=0),
                Allocation(buffer="C2", component=Component.HBM),
            ],
            [
                Allocation(buffer="B", address=0),
                Allocation(buffer="C3", component=Component.HBM),
            ],
            [
                Allocation(buffer="B", address=0),
                Allocation(buffer="C4", component=Component.HBM),
            ],
            [
                Allocation(buffer="A", address=0),
                Allocation(buffer="C5", component=Component.HBM),
            ],
            [
                Allocation(buffer="A", address=0),
                Allocation(buffer="C6", component=Component.HBM),
            ],
        ]

        pattern = Pattern(
            buffers, ops, good_allocation=make_allocation_result(good_allocation)
        )
        return pattern

    def test_verify_simple_eviction_pattern(self):
        self.verify_pattern(self.make_simple_eviction_pattern())

    #@usuallyExpectedFailure
    def test_simple_eviction_pattern(self):
        self.run_pattern(self.make_simple_eviction_pattern())

    def make_eviction_reallocation_pattern(self) -> Pattern:
        """This pattern requires allocating a buffer, evicting it, and then reallocating it later
        at a different address to achieve optimality.

        We use four buffers total: A0, A1, A2 of size 1/3 the available size, and B of size twice
        that. We first ensure that A0, A1, and A2 must be allocated together, then A0 and B, then
        A1 and B, and finally A2 and B. Because B can fit only with one of the A buffers at the top
        or the bottom, whichever one was allocated in the middle must be moved.

        We ensure that any set is allocated together in an optimal allocation by using four ops
        in a row that use them all as input. This means that, whatever was in the scratchpad before
        and whatever is in it after, we can complete that phase with one full scratchpad worth of
        HBM transfers. On the other hand, if not everything is allocated on the scratchpad, then we
        have to stream at least one buffer four times, which entails at least 4/3 of the scratchpad
        size in HBM transfers."""
        A_size = AVAILABLE_LX_SIZE // 3
        A_size = (A_size // 128) * 128  # round down to a multiple of the stick size
        B_size = 2 * A_size

        # This will work if 4 * A_size > AVAILABLE_LX_SIZE.
        self.assertGreater(4 * A_size, AVAILABLE_LX_SIZE)

        buffers = make_buffer_registry(
            {f"A{i}": A_size for i in range(3)}
            | {"B": B_size}
            | {f"C{i}_{j}": A_size for i in range(4) for j in range(4)}
        )

        pattern = [["A0", "A1", "A2"], ["A0", "B"], ["A1", "B"], ["A2", "B"]]
        ops = [
            Operation(
                f"op{i}_{j}",
                inputs=group,
                outputs=[f"C{i}_{j}"],
                _buffer_registry=buffers,
            )
            for i, group in enumerate(pattern)
            for j in range(4)
        ]

        addresses_per_group = [
            {"A0": 0, "A1": A_size, "A2": 2 * A_size},
            {"A0": 0, "B": A_size},
            {"A1": 0, "B": A_size},
            {"A2": 0, "B": A_size},
        ]
        good_allocations = [
            (
                [
                    Allocation(buffer=buffer, address=addresses_per_group[i][buffer])
                    for buffer in group
                ]
                + [Allocation(buffer=f"C{i}_{j}", component=Component.HBM)]
            )
            for i, group in enumerate(pattern)
            for j in range(4)
        ]

        pattern = Pattern(
            buffers, ops, good_allocation=make_allocation_result(good_allocations)
        )
        return pattern

    def test_verify_eviction_reallocation_pattern(self):
        self.verify_pattern(self.make_eviction_reallocation_pattern())

    @usuallyExpectedFailure
    def test_eviction_reallocation_pattern(self):
        self.run_pattern(self.make_eviction_reallocation_pattern())


if __name__ == "__main__":
    import unittest

    unittest.main()
