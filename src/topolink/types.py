"""Type definitions for TopoLink."""

from typing import TypedDict


class NeighborInfo(TypedDict):
    endpoint: str
    weight: float
