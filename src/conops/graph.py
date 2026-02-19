from logging import getLogger
from typing import Literal, KeysView
from collections import deque

import numpy as np
from numpy.typing import NDArray

from .types import NeighborInfo
from .utils import is_symmetric_doubly_stochastic, normalize_transport

logger = getLogger("conops.graph")


class Graph:
    """
    Represents a network topology graph for distributed nodes.

    Parameters
    ----------
    nodes : list[str] | None, optional
        An iterable of node names to initialize the graph. If None, the graph starts with no nodes.

    edges : list[tuple[str, str]] | None, optional
        An iterable of edges represented as tuples of node names (u, v). If None, the graph starts with no edges.

    name : str, optional
        The name of the graph service. Default is "default". When deploying multiple graphs, ensure each has a unique name.

    transport : Literal["tcp", "ipc"], optional
        The transport type for ZeroMQ communication (default is "tcp").
        "ipc" is recommended for local multi-process communication on the same machine for better performance.

    Attributes
    ----------
    name : str
        The name of the graph service.

    nodes : KeysView[str]
        A view of the node names in the graph.

    edges : list[tuple[str, str]]
        A list of edges in the graph represented as tuples of node names (u, v).

    number_of_nodes : int
        The total number of nodes in the graph.

    transport : str
        The transport type used for communication.
        Note that "ipc" is only supported for Linux systems.

    adjacency : dict[str, dict[str, NeighborInfo]]
        The adjacency representation of the graph, where each key is a node name, and the value is a dictionary mapping neighboring node names to their NeighborInfo.

    Notes
    -----
    - Use the `bootstrap` function to start the bootstrap service for node attachment and network formation.
    - The `from_mixing_matrix` method provides a convenient way to create a graph from a mixing matrix, ensuring the matrix is symmetric and double-stochastic.
    """

    def __init__(
        self,
        nodes: list[str] | None = None,
        edges: list[tuple[str, str]] | None = None,
        name: str = "default",
        transport: Literal["tcp", "ipc"] = "tcp",
    ) -> None:
        nodes_ = nodes or []
        edges_ = edges or []
        self._adj: dict[str, dict[str, NeighborInfo]] = {u: {} for u in nodes_}

        for u, v in edges_:
            if u not in self._adj:
                self._adj[u] = {}
            if v not in self._adj:
                self._adj[v] = {}

            self._adj[u][v] = NeighborInfo(weight=1.0, endpoint="")
            self._adj[v][u] = NeighborInfo(weight=1.0, endpoint="")

        self.name = name
        self._transport = normalize_transport(transport)

    @classmethod
    def from_mixing_matrix(
        cls,
        mixing_matrix: NDArray[np.float64],
        nodes: list[str] | None = None,
        name: str = "default",
        transport: Literal["tcp", "ipc"] = "tcp",
    ) -> "Graph":
        """
        Create a Graph instance from a mixing matrix.
        The mixing matrix must be symmetric doubly stochastic.

        Parameters
        ----------
        mixing_matrix : NDArray[float64]
            A square matrix representing the mixing coefficients between nodes.

        nodes : list[str], optional
            A list of self defined node names. If not provided, nodes will be named sequentially as "1", "2", ..., "n".
        """
        if not is_symmetric_doubly_stochastic(mixing_matrix):
            err_msg = "The mixing matrix must be symmetric doubly stochastic."
            logger.error(err_msg)
            raise ValueError(err_msg)

        n = len(mixing_matrix)

        if nodes is None:
            nodes = [str(i + 1) for i in range(n)]
        elif len(nodes) != n:
            err_msg = "The length of nodes must match the size of the mixing matrix."
            logger.error(err_msg)
            raise ValueError(err_msg)

        adj: dict[str, dict[str, NeighborInfo]] = {u: {} for u in nodes}

        nz_rows, nz_cols = np.nonzero(mixing_matrix)
        mask = nz_rows < nz_cols

        for i, j in zip(nz_rows[mask], nz_cols[mask]):
            weight = mixing_matrix[i, j].item()
            u = nodes[i]
            v = nodes[j]
            adj[u][v] = NeighborInfo(weight=weight, endpoint="")
            adj[v][u] = NeighborInfo(weight=weight, endpoint="")

        graph = cls(name=name, transport=transport)
        graph._adj = adj

        return graph

    @property
    def nodes(self) -> KeysView[str]:
        return self._adj.keys()

    @property
    def edges(self) -> list[tuple[str, str]]:
        edges = []
        for u in self._adj:
            for v in self._adj[u]:
                if (v, u) not in edges:
                    edges.append((u, v))
        return edges

    @property
    def number_of_nodes(self) -> int:
        return len(self._adj)

    @property
    def transport(self) -> str:
        return self._transport

    @transport.setter
    def transport(self, value: Literal["tcp", "ipc"]) -> None:
        self._transport = normalize_transport(value)

    @property
    def adjacency(self) -> dict[str, dict[str, NeighborInfo]]:
        return self._adj

    def is_connected(self) -> bool:
        """
        Check if the graph is connected.

        Returns
        -------
        bool
            True if the graph is connected, False otherwise.
        """
        if not self._adj:
            return True

        start = next(iter(self._adj))
        visited = {start}

        q = deque([start])

        while q:
            u = q.popleft()
            for v in self._adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)

        return len(visited) == len(self._adj)
