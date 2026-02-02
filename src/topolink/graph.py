from logging import getLogger
from json import dumps
from typing import Literal, KeysView
from collections import deque

import zmq
import numpy as np
from numpy.typing import NDArray

from .types import NeighborInfo
from .utils import get_local_ip, is_symmetric_doubly_stochastic, normalize_transport
from .discovery import GraphAdvertiser

logger = getLogger("topolink.graph")


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
    nodes : KeysView[str]
        A view of the node names in the graph.

    number_of_nodes : int
        The total number of nodes in the graph.

    adjacency : dict[str, dict[str, float]]
        The adjacency representation of the graph, where each key is a node name and the value is a dictionary of neighboring nodes and their edge weights.

    Notes
    -----
    - The graph must be connected before deployment.
    - The `from_mixing_matrix` method provides a convenient way to create a graph from a mixing matrix, ensuring the matrix is symmetric and double-stochastic.
    - Throughout the code, we use 'i' to denote the current node and 'j' to denote neighbor nodes, following common conventions in graph theory and distributed algorithms.
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

        self._context = zmq.Context()

        self._name = name
        self._transport = normalize_transport(transport)

        self._router = self._context.socket(zmq.ROUTER)
        self._setup_router()

        # idx: (rid: the router identity, endpoint: ip and port of the node)
        # We do not use the node index as the ROUTER identity, so that multiple
        # connections with the same idx can be distinguished during registration
        # (e.g., to properly reject or replace duplicate nodes).
        self._node_registry: dict[str, tuple[bytes, str]] = {}

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
    def number_of_nodes(self) -> int:
        return len(self._adj)

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

    def _setup_router(self) -> None:
        if self._transport == "tcp":
            self._graph_advertiser = GraphAdvertiser(self._name)
            ip_address = get_local_ip()
            port = self._router.bind_to_random_port("tcp://*")
            logger.info(f"Graph '{self._name}' running on: {ip_address}:{port}")
            self._graph_advertiser.register(ip_address, port)
        elif self._transport == "ipc":
            self._router.bind(f"ipc://@topolink-graph-{self._name}")
        else:
            err_msg = f"Unsupported transport type: {self._transport}"
            logger.error(err_msg)
            raise ValueError(err_msg)

    def _register_nodes(self) -> None:
        while len(self._node_registry) < self.number_of_nodes:
            rid, idx_bytes, endpoint_bytes = self._router.recv_multipart()
            idx = idx_bytes.decode()

            if idx not in self.nodes:
                self._router.send_multipart([rid, b"Error: Undefined node"])
                continue

            if idx in self._node_registry:
                old_rid, _ = self._node_registry[idx]
                self._router.send_multipart([old_rid, b"Error: Node replaced"])

            endpoint = endpoint_bytes.decode()
            self._node_registry[idx] = (rid, endpoint)
            logger.info(f"Node '{idx}' joined graph '{self._name}' from {endpoint}.")

        for idx in self._node_registry:
            rid, _ = self._node_registry[idx]
            self._router.send_multipart([rid, b"OK"])

        logger.info(f"Graph '{self._name}' registration complete.")

    def _notify_nodes_of_neighbors(self) -> None:
        for i in self._adj:
            neighbors = self._adj[i]
            for j in neighbors:
                _, endpoint = self._node_registry[j]
                self._adj[i][j]["endpoint"] = endpoint

        for i in self.nodes:
            neighbors = self._adj[i]
            messages = dumps(neighbors).encode()
            rid, _ = self._node_registry[i]
            self._router.send_multipart([rid, messages])

        logger.info(f"Sent neighbor information to all nodes in graph '{self._name}'.")

    def deploy(self) -> None:
        """
        Deploy the network topology.
        This method handles the registration of nodes, sends them their neighbor information,
        and ensures proper cleanup of resources after deployment.

        This process is only responsible for deploying the network topology.
        It does not take part in the actual communication between nodes.
        Once all nodes are registered and notified, the process will shut down automatically.

        Raises
        ------
        ValueError
            If the graph is not connected.
        """
        try:
            if not self.is_connected():
                err_msg = "The graph is not connected."
                logger.error(err_msg)
                raise ValueError(err_msg)

            self._register_nodes()
            self._notify_nodes_of_neighbors()
            # TODO: More deployment logic can be added here if needed.
        finally:
            self._router.close()
            self._context.term()
            if self._transport == "tcp":
                self._graph_advertiser.unregister()
