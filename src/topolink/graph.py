from logging import getLogger

logger = getLogger("topolink.graph")

import networkx as nx
from json import dumps
from typing import Any
from zmq import Context, SyncSocket, ROUTER
from numpy import float64
from numpy.typing import NDArray
from .exceptions import ConnectivityError, InvalidWeightedMatrixError
from .types import NodeInput, EdgeInput, NodeView, EdgeView, AdjView, NeighborInfo
from .utils import get_local_ip, is_symmetric_double_stochastic
from .discovery import GraphAdvertiser


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
    *args : Any
        Additional positional arguments (not used).
    **kwargs : Any
        Additional keyword arguments. If 'nx_graph' is provided, it will be used directly as the internal graph representation.

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

    Methods
    -------
    from_mixing_matrix(mixing_matrix, nodes=None) -> "Graph"
        Alternative constructor to create a Graph from a symmetric, double-stochastic mixing matrix.
    add_nodes(nodes) -> None
        Adds nodes to the graph.
    add_edges(edges) -> None
        Adds edges to the graph.
    adjacency(node) -> AdjView
        Returns the adjacency view for a specified node.
    deploy() -> None
        Deploys the network topology by registering nodes, notifying them of their neighbors, and unregistering nodes.

    Notes
    -----
    - Uses ZeroMQ for communication between nodes.
    - The graph must be connected before deployment.
    - The `nx_graph` keyword argument is intended for internal use only and should not be set from outside this package.
    - The `from_mixing_matrix` method provides a convenient way to create a graph from a mixing matrix, ensuring the matrix is symmetric and double-stochastic.
    """

    def __init__(
        self,
        nodes: NodeInput | None = None,
        edges: EdgeInput | None = None,
        name: str = "default",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # If a NetworkX graph is provided, use it directly.
        # Note: The `nx_graph` attribute is intended for internal use only and should not be set from outside this package.
        # It is primarily used by alternative constructors (such as from_mixing_matrix) to initialize the graph structure efficiently.
        if "nx_graph" in kwargs:
            self._nx_graph: nx.Graph = kwargs["nx_graph"]
        else:
            self._nx_graph = nx.Graph()
            self._nx_graph.add_nodes_from(nodes or [])
            self._nx_graph.add_edges_from(edges or [])

        self._context = Context()

        self._name = name

        self._graph_advertiser = GraphAdvertiser(name)
        self._ip_address = get_local_ip()
        self._router, self._port = self._setup_router()
        self._graph_advertiser.register(self._ip_address, self._port)

        self._registered_nodes: dict[str, str] = {}

    @classmethod
    def from_mixing_matrix(
        cls, mixing_matrix: NDArray[float64], nodes: list[str] | None = None
    ) -> "Graph":
        """
        Create a Graph instance from a mixing matrix.
        The mixing matrix must be symmetric and double-stochastic.

        Parameters
        ----------
        mixing_matrix : NDArray[float64]
            A square matrix representing the mixing coefficients between nodes.
        nodes : list[str], optional
            A list of self defined node names. If not provided, nodes will be named sequentially as "1", "2", ..., "n".
        """
        if not is_symmetric_double_stochastic(mixing_matrix):
            raise InvalidWeightedMatrixError(
                "The mixing matrix must be symmetric and double-stochastic."
            )

        nodelist = nodes or [f"{i + 1}" for i in range(mixing_matrix.shape[0])]

        nx_graph: nx.Graph = nx.from_numpy_array(
            mixing_matrix, create_using=nx.Graph, nodelist=nodelist  # type: ignore[arg-type]
        )
        nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))

        return cls(nx_graph=nx_graph)

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
        self._nx_graph.add_nodes_from(nodes)

    def add_edges(self, edges: EdgeInput) -> None:
        self._nx_graph.add_edges_from(edges)

    def adjacency(self, node: str) -> AdjView:
        """
        Returns the adjacency view of the specified node in the graph.

        Parameters:
            node (str): The identifier of the node whose adjacency is to be retrieved.

        Returns:
            AdjView: A view of the adjacent nodes and edge data for the given node.

        Raises:
            KeyError: If the specified node does not exist in the graph.
        """
        return self._nx_graph[node]

    def _get_neighbor_info_list(self, node: str) -> list[NeighborInfo]:
        # When the assert fails, it indicates a bug in the deployment logic.
        assert node in self.nodes, f"Node '{node}' is not defined in the graph."

        neighbor_info_list = []
        adjacency = self.adjacency(node)
        for neighbor in adjacency:
            endpoint = self._registered_nodes.get(neighbor, "")

            # When the assert fails, it indicates a bug in the deployment logic.
            assert endpoint, f"Node '{neighbor}' has not registered."

            weight = adjacency[neighbor].get("weight", 1.0)
            n_info = NeighborInfo(name=neighbor, endpoint=endpoint, weight=weight)
            neighbor_info_list.append(n_info)

        return neighbor_info_list

    def _setup_router(self) -> tuple[SyncSocket, int]:
        router = self._context.socket(ROUTER)
        port = router.bind_to_random_port("tcp://*")
        logger.info(f"Graph '{self._name}' running on: {self._ip_address}:{port}")

        return router, port

    def _register_nodes(self) -> None:
        while len(self._registered_nodes) < self.number_of_nodes:
            name_bytes, _, endpoint_bytes = self._router.recv_multipart()
            name = name_bytes.decode()

            if name not in self.nodes:
                self._router.send_multipart([name_bytes, b"", b"Error: Undefined node"])
                continue

            endpoint = endpoint_bytes.decode()
            self._registered_nodes[name] = endpoint
            logger.info(f"Node '{name}' joined graph '{self._name}' from {endpoint}.")

        logger.info(f"Graph '{self._name}' registration complete.")

    def _notify_nodes_of_neighbors(self) -> None:
        for i in self.nodes:
            neighbor_info_list = self._get_neighbor_info_list(i)
            messages = [dumps(n_info).encode() for n_info in neighbor_info_list]

            self._router.send_multipart([i.encode(), b"", *messages])

        logger.info(f"Sent neighbor information to all nodes in graph '{self._name}'.")

    def deploy(self) -> None:
        """
        Deploys the network topology by registering nodes, notifying them of their neighbors,
        and then unregistering the nodes. Ensures that the topology is connected before deployment.
        Cleans up resources by closing the router and terminating the context after deployment.

        Raises:
            ConnectivityError: If the graph is not fully connected.
        """
        try:
            if not self.is_connected:
                raise ConnectivityError("The graph is not fully connected.")

            self._register_nodes()
            self._notify_nodes_of_neighbors()
            # TODO: More deployment logic can be added here if needed.
        finally:
            self._router.close()
            self._context.term()
            self._graph_advertiser.unregister()
