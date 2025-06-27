from typing import Iterable, Tuple, Mapping

NodeInput = Iterable[str]
EdgeInput = Iterable[Tuple[str, str]] | Iterable[Tuple[str, str, Mapping[str, float]]]

NodeView = Iterable[str]
EdgeView = Iterable[Tuple[str, str, Mapping[str, float]]]
AdjView = Mapping[str, Mapping[str, float]]

from typing import TypedDict


class NeighborInfo(TypedDict):
    name: str
    address: str
    weight: float
