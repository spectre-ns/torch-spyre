from typing import List, Dict, Protocol, Self
from torch.utils._ordered_set import OrderedSet
from dataclasses import dataclass
from enum import Enum


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
    data = Self  # This helps 'scratchpad'


@dataclass
class ReadWrites(Protocol):
    reads: OrderedSet[Buffer]
    writes: OrderedSet[Buffer]


@dataclass
class Operation(Protocol):
    name: str
    inputs: List[str]
    outputs: List[str]
    _buffer_registry: Dict[str, Buffer]
    op_it_space_splits: List
    origin_node: Self
    target: Self
    _opname: str

    def __post_init__(self) -> None:
        ...

    def get_read_writes(self) -> ReadWrites:
        ...

    def get_read_names(self) -> List[str]:
        ...

class Component(Enum):
    LX = "LX"
    HBM = "HBM"