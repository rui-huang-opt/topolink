from typing import Iterable, Mapping

NodeInput = Iterable[str]
EdgeInput = Iterable[tuple[str, str]]

NodeView = Iterable[str]
EdgeView = Iterable[tuple[str, str]]
AdjView = Mapping[str, Mapping[str, float]]

from typing import TypedDict


class NeighborInfo(TypedDict):
    name: str
    address: str
    weight: float
