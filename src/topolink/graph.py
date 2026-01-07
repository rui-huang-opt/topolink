from logging import getLogger
from json import dumps

import numpy as np
import networkx as nx
import zmq
from numpy.typing import NDArray

from .types import NodeInput, EdgeInput, NodeView, EdgeView, AdjView, NeighborInfo
from .utils import get_local_ip, is_symmetric_doubly_stochastic
from .discovery import GraphAdvertiser

logger = getLogger("topolink.graph")


class Graph:
    """
    Represents a network topology using a NetworkX graph and provides methods for managing nodes, edges, and network deployment.

    Parameters
    ----------
    nodes : NodeInput | None, optional
        An iterable of node names to initialize the graph. If None, the graph starts empty.

    edges : EdgeInput | None, optional
        An iterable of edges (tuples of node names) to initialize the graph. If None, the graph starts with no edges.

    name : str, optional
        The name of the graph service. Default is "default". When deploying multiple graphs, ensure each has a unique name.

    Attributes
    ----------
    nodes : NodeView
        A view of the nodes in the graph.

    edges : EdgeView
        A view of the edges in the graph.

    number_of_nodes : int
        The number of nodes in the graph.

    number_of_edges : int
        The number of edges in the graph.

    is_connected : bool
        Indicates whether the graph is connected.

    Notes
    -----
    - The graph must be connected before deployment.
    - The `from_mixing_matrix` method provides a convenient way to create a graph from a mixing matrix, ensuring the matrix is symmetric and double-stochastic.
    - Throughout the code, we use 'i' to denote the current node and 'j' to denote neighbor nodes, following common conventions in graph theory and distributed algorithms.
    """

    def __init__(
        self,
        nodes: NodeInput | None = None,
        edges: EdgeInput | None = None,
        name: str = "default",
    ) -> None:
        self._nx_graph = nx.Graph()
        self._nx_graph.add_nodes_from(nodes or [])
        self._nx_graph.add_edges_from(edges or [])

        self._context = zmq.Context()

        self._name = name

        self._graph_advertiser = GraphAdvertiser(name)
        self._ip_address = get_local_ip()
        self._router, self._port = self._setup_router()
        self._graph_advertiser.register(self._ip_address, self._port)

        self._node_registry: dict[str, str] = {}

    @classmethod
    def from_mixing_matrix(
        cls,
        mixing_matrix: NDArray[np.float64],
        nodelist: list[str] | None = None,
        name: str = "default",
    ) -> "Graph":
        """
        Create a Graph instance from a mixing matrix.
        The mixing matrix must be symmetric doubly stochastic.

        Parameters
        ----------
        mixing_matrix : NDArray[float64]
            A square matrix representing the mixing coefficients between nodes.

        nodelist : list[str], optional
            A list of self defined node names. If not provided, nodes will be named sequentially as "1", "2", ..., "n".
        """
        if not is_symmetric_doubly_stochastic(mixing_matrix):
            err_msg = "The mixing matrix must be symmetric doubly stochastic."
            logger.error(err_msg)
            raise ValueError(err_msg)

        mixing_matrix_no_self = mixing_matrix.copy()
        np.fill_diagonal(mixing_matrix_no_self, 0.0)
        nx_graph = nx.from_numpy_array(mixing_matrix_no_self, create_using=nx.Graph)

        if nodelist is None:
            n = mixing_matrix.shape[0]
            mapping = {i: str(i + 1) for i in range(n)}
        elif len(nodelist) == mixing_matrix.shape[0]:
            mapping = {i: nodelist[i] for i in range(mixing_matrix.shape[0])}
        else:
            err_msg = "Length of nodelist must match the size of the mixing matrix."
            logger.error(err_msg)
            raise ValueError(err_msg)

        nx_graph = nx.relabel_nodes(nx_graph, mapping)

        graph = cls(name=name)
        graph._nx_graph = nx_graph

        return graph

    @property
    def nodes(self) -> NodeView:
        return self._nx_graph.nodes

    @property
    def edges(self) -> EdgeView:
        return self._nx_graph.edges

    @property
    def number_of_nodes(self) -> int:
        return self._nx_graph.number_of_nodes()

    @property
    def number_of_edges(self) -> int:
        return self._nx_graph.number_of_edges()

    @property
    def is_connected(self) -> bool:
        return nx.is_connected(self._nx_graph)

    def add_nodes(self, nodes: NodeInput) -> None:
        """
        Add nodes to the graph.

        Parameters
        ----------
        nodes : NodeInput
            An iterable of node names to add to the graph.
        """
        self._nx_graph.add_nodes_from(nodes)

    def add_edges(self, edges: EdgeInput) -> None:
        """
        Add edges to the graph.

        Parameters
        ----------
        edges : EdgeInput
            An iterable of edges (tuples of node names) to add to the graph.
        """
        self._nx_graph.add_edges_from(edges)

    def adjacency(self, i: str) -> AdjView:
        """
        Get the adjacency view of a specific node i.

        Parameters
        ----------
        i : str
            The name of the node.

        Returns
        -------
        AdjView
            The adjacency view of the specified node.
        """
        return self._nx_graph[i]

    def _get_neighbor_info_dict(self, i: str) -> dict[str, NeighborInfo]:
        # When the assert fails, it indicates a bug in the deployment logic.
        assert i in self.nodes, f"Node '{i}' is not defined in the graph."

        neighbor_info_dict: dict[str, NeighborInfo] = {}
        n_i = self.adjacency(i)
        for j in n_i:
            endpoint = self._node_registry.get(j, "")

            # When the assert fails, it indicates a bug in the deployment logic.
            assert endpoint, f"Node '{j}' has not registered."

            weight = n_i[j].get("weight", 1.0)
            info = NeighborInfo(endpoint=endpoint, weight=weight)
            neighbor_info_dict[j] = info

        return neighbor_info_dict

    def _setup_router(self) -> tuple[zmq.Socket, int]:
        router = self._context.socket(zmq.ROUTER)
        port = router.bind_to_random_port("tcp://*")
        logger.info(f"Graph '{self._name}' running on: {self._ip_address}:{port}")

        return router, port

    def _register_nodes(self) -> None:
        while len(self._node_registry) < self.number_of_nodes:
            idx_bytes, endpoint_bytes = self._router.recv_multipart()
            idx = idx_bytes.decode()

            if idx not in self.nodes:
                self._router.send_multipart([idx_bytes, b"Error: Undefined node"])
                continue

            endpoint = endpoint_bytes.decode()
            self._node_registry[idx] = endpoint
            logger.info(f"Node '{idx}' joined graph '{self._name}' from {endpoint}.")
            self._router.send_multipart([idx_bytes, b"OK"])

        logger.info(f"Graph '{self._name}' registration complete.")

    def _notify_nodes_of_neighbors(self) -> None:
        for i in self.nodes:
            neighbor_info_dict = self._get_neighbor_info_dict(i)
            messages = dumps(neighbor_info_dict).encode()
            self._router.send_multipart([i.encode(), messages])

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
            if not self.is_connected:
                err_msg = "The graph is not connected."
                logger.error(err_msg)
                raise ValueError(err_msg)

            self._register_nodes()
            self._notify_nodes_of_neighbors()
            # TODO: More deployment logic can be added here if needed.
        finally:
            self._router.close()
            self._context.term()
            self._graph_advertiser.unregister()
