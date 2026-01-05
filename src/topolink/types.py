"""Type definitions for TopoLink."""

from typing import Iterable, Collection, Mapping

NodeInput = Iterable[str]
EdgeInput = Iterable[tuple[str, str]]

NodeView = Collection[str]
EdgeView = Collection[tuple[str, str]]
AdjView = Mapping[str, Mapping[str, float]]

from typing import TypedDict


class NeighborInfo(TypedDict):
    endpoint: str
    weight: float
