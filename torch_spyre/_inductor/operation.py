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

from typing import List, Dict, Protocol, Self
from torch.utils._ordered_set import OrderedSet


class BufferDeviceLayout(Protocol):
    """This class mimics the FixedTiledLayout.device_layout field."""

    device_size: int


class BufferLayout(Protocol):
    """This class mimics the TensorBox.layout field (a FixedTiledLayout)."""

    device_layout: BufferDeviceLayout
    size: int
    allocation: Dict


class Buffer(Protocol):
    name: str
    size: int
    layout: BufferLayout
    data: Self  # This helps 'scratchpad'


class ReadWrites(Protocol):
    reads: OrderedSet[Buffer]
    writes: OrderedSet[Buffer]


class Operation(Protocol):
    name: str
    inputs: List[str]
    outputs: List[str]
    _buffer_registry: Dict[str, Buffer]
    op_it_space_splits: List
    origin_node: Self
    target: Self
    _opname: str

    def __post_init__(self) -> None: ...

    def get_read_writes(self) -> ReadWrites: ...

    def get_read_names(self) -> List[str]: ...
