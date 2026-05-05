# Copyright 2025 The Torch-Spyre Authors.
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

from typing import List, Dict
from torch.utils._ordered_set import OrderedSet
from dataclasses import dataclass
from abc import abstractmethod
from enum import Enum

class BufferDeviceLayout:
    """This class mimics the FixedTiledLayout.device_layout field."""

    def __init__(self, size: int):
        self.device_size = [(size + 127) // 128, 128]


class BufferLayout:
    """This class mimics the TensorBox.layout field (a FixedTiledLayout)."""

    def __init__(self, size: int):
        self.device_layout = BufferDeviceLayout(size)
        self.size = size
        self.allocation = {}


class Buffer:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.layout = BufferLayout(size)
        self.data = self  # This helps 'scratchpad'


@dataclass
class ReadWrites:
    reads: OrderedSet[Buffer]
    writes: OrderedSet[Buffer]


@dataclass
class Operation:
    name: str
    inputs: List[str]
    outputs: List[str]
    _buffer_registry: Dict[str, Buffer]

    # To make scratchpad.py work, we add origin_node and target fields that point to the op itself,
    # a field _opname that is the same as name, and a field op_it_space_splits that is used in core
    # division. (If the value of op_it_space_splits is different for operations in a sequence, that
    # blocks LX allocation, so we make sure it is always the same.)
    op_it_space_splits = None
    origin_node = None
    target = None
    _opname = None

    def __post_init__(self):
        self.op_it_space_splits = []
        self.origin_node = self
        self.target = self
        self._opname = self.name

    def get_read_writes(self) -> ReadWrites:
        # Returns a List of (buffer_name, "read" or "write") for all buffers used by this operation.
        reads = OrderedSet(
            self._buffer_registry[buffer_name] for buffer_name in self.inputs
        )
        writes = OrderedSet(
            self._buffer_registry[buffer_name] for buffer_name in self.outputs
        )
        return ReadWrites(reads=reads, writes=writes)

    def get_read_names(self):
        return self.inputs

class Component(Enum):
    LX = "LX"
    HBM = "HBM"


class LxAllocator:
    def __init__(self, graph):
        self.graph = graph

    @abstractmethod
    def plan_allocation(self, operations: List[Operation]):
        """
        Interface method meant as a placeholder for the interface of LX
        allocator classes.

        Args:
            operations (List[Operation]): List of operations to be considered
                for LX planning.
        """        
        pass