"""Type definitions for TopoLink."""

from typing import Iterable, Collection, Mapping

NodeInput = Iterable[str]
EdgeInput = Iterable[tuple[str, str]]

NodeView = Collection[str]
EdgeView = Collection[tuple[str, str]]
AdjView = Mapping[str, Mapping[str, float]]

from typing import TypedDict


class NeighborInfo(TypedDict):
    name: str
    endpoint: str
    weight: float


from typing import NamedTuple
from zmq import SyncSocket


class Neighbor(NamedTuple):
    name: str
    weight: float
    endpoint: str
    in_socket: SyncSocket

    def __eq__(self, name: str) -> bool:
        return self.name == name
