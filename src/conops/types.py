"""Type definitions for ConOps."""

from typing import TypedDict


class NeighborInfo(TypedDict):
    endpoint: str
    weight: float
