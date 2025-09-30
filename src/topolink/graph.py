import networkx as nx
from json import dumps
from logging import getLogger
from functools import cached_property
from typing import Any
from zmq import Context, SyncSocket, ROUTER
from matplotlib.axes import Axes
from numpy import float64, allclose, ones
from numpy import sum as np_sum
from numpy.typing import NDArray
from .types import NodeInput, EdgeInput, NodeView, EdgeView, AdjView, NeighborInfo
from .utils import get_local_ip


class Graph:
    """
    Represents a network topology using a NetworkX graph and provides methods for managing nodes, edges, and network deployment.

    Parameters
    ----------
    nodes : NodeInput | None, optional
        Initial nodes to add to the graph. Can be any iterable accepted by NetworkX.
    edges : EdgeInput | None, optional
        Initial edges to add to the graph. Can be any iterable accepted by NetworkX.
    address : str | None, optional
        Address to bind the server socket for deployment. If not provided, a random port is used.
    *args : Any
        Additional positional arguments.
    **kwargs : Any
        Additional keyword arguments. If 'nx_graph' is provided, it is used directly as the internal graph.

    Attributes
    ----------
    nodes : NodeView
        View of the nodes in the graph.
    edges : EdgeView
        View of the edges in the graph, including edge data.
    number_of_nodes : int
        Number of nodes in the graph.
    number_of_edges : int
        Number of edges in the graph.
    is_connected : bool
        Indicates whether the graph is connected.

    Methods
    -------
    from_mixing_matrix(mixing_matrix, address=None, nodes=None) -> "Graph"
        Alternative constructor to create a Graph from a symmetric, double-stochastic mixing matrix.
    add_nodes(nodes) -> None
        Adds nodes to the graph.
    add_edges(edges) -> None
        Adds edges to the graph.
    draw(ax, **kwargs) -> None
        Draws the graph on a given matplotlib Axes.
    adjacency(node) -> AdjView
        Returns the adjacency view for a specified node.
    deploy() -> None
        Deploys the network topology by registering nodes, notifying them of their neighbors, and unregistering nodes.

    Notes
    -----
    - The internal NetworkX graph can be initialized directly via the 'nx_graph' keyword argument.
    - Deployment methods use ZeroMQ sockets for node registration and communication.
    - The class ensures the topology is connected before deployment.
    """

    def __init__(
        self,
        nodes: NodeInput | None = None,
        edges: EdgeInput | None = None,
        address: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self._address = address

        # If a NetworkX graph is provided, use it directly.
        # Note: The `nx_graph` attribute is intended for internal use only and should not be set from outside this package.
        # It is primarily used by alternative constructors (such as from_mixing_matrix) to initialize the graph structure efficiently.
        if "nx_graph" in kwargs:
            self._nx_graph: nx.Graph = kwargs["nx_graph"]
        else:
            self._nx_graph = nx.Graph()
            self._nx_graph.add_nodes_from(nodes or [])
            self._nx_graph.add_edges_from(edges or [])

        self._registered_addresses: dict[str, str] = {}
        self._logger = getLogger("topolink.Graph")
        self._local_ip = get_local_ip()
        self._context = Context()

    @classmethod
    def from_mixing_matrix(
        cls,
        mixing_matrix: NDArray[float64],
        address: str | None = None,
        nodes: list[str] | None = None,
    ) -> "Graph":
        """
        Create a Graph instance from a mixing matrix.
        The mixing matrix must be symmetric and double-stochastic.

        Parameters
        ----------
        mixing_matrix : NDArray[float64]
            A square matrix representing the mixing coefficients between nodes.
        address : str, optional
            The address to bind the server socket to. If not provided, a random port is used.
        nodes : list[str], optional
            A list of self defined node names. If not provided, nodes will be named sequentially as "1", "2", ..., "n".
        """
        if not allclose(mixing_matrix, mixing_matrix.T):
            raise ValueError("The mixing matrix must be symmetric.")

        ones_vec = ones(mixing_matrix.shape[0])
        if not allclose(np_sum(mixing_matrix, axis=1), ones_vec):
            raise ValueError("The mixing matrix must be double-stochastic.")

        nodelist = nodes or [f"{i + 1}" for i in range(mixing_matrix.shape[0])]

        nx_graph: nx.Graph = nx.from_numpy_array(
            mixing_matrix, create_using=nx.Graph, nodelist=nodelist  # type: ignore[arg-type]
        )
        nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))

        return cls(nx_graph=nx_graph, address=address)

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

    @cached_property
    def _router(self) -> SyncSocket:
        router = self._context.socket(ROUTER)

        if self._address is None:
            port = router.bind_to_random_port("tcp://*")
            self._address = f"{self._local_ip}:{port}"
        else:
            router.bind(f"tcp://{self._address}")

        self._logger.info(f"Server running on {self._address}")

        return router

    @property
    def is_connected(self) -> bool:
        return nx.is_connected(self._nx_graph)

    def add_nodes(self, nodes: NodeInput) -> None:
        self._nx_graph.add_nodes_from(nodes)

    def add_edges(self, edges: EdgeInput) -> None:
        self._nx_graph.add_edges_from(edges)

    def draw(self, ax: Axes, **kwargs) -> None:
        """
        Draws the graph using NetworkX's drawing functionality on the provided matplotlib Axes.

        Parameters:
            ax (Axes): A matplotlib Axes object where the graph will be drawn.
            **kwargs: Additional keyword arguments passed to `networkx.draw` for customizing the drawing.

        Returns:
            None
        """
        nx_graph = self._nx_graph

        pos = nx.spring_layout(nx_graph)

        options = {"pos": pos, "ax": ax, "with_labels": True}
        options.update(kwargs)

        nx.draw(nx_graph, **options)

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
        if node not in self.nodes:
            raise ValueError(f"Node {node} is not part of the graph.")

        neighbor_info_list = []
        adjacency = self.adjacency(node)
        for neighbor in adjacency:
            address = self._registered_addresses.get(neighbor, "")

            if not address:
                raise ValueError(f"Node {neighbor} is not registered.")

            weight = adjacency[neighbor].get("weight", 1.0)
            n_info = NeighborInfo(name=neighbor, address=address, weight=weight)
            neighbor_info_list.append(n_info)

        return neighbor_info_list

    def deploy(self) -> None:
        """
        Deploys the network topology by registering nodes, notifying them of their neighbors,
        and then unregistering the nodes. Ensures that the topology is connected before deployment.
        Cleans up resources by closing the router and terminating the context after deployment.

        Raises:
            ValueError: If the topology is not connected.
        """
        try:
            if not self.is_connected:
                raise ValueError("The provided topology must be connected.")

            self._register_nodes()
            self._notify_nodes_their_neighbors()
            # TODO: More deployment logic can be added here if needed.
            self._unregister_nodes()
        finally:
            self._router.close()
            self._context.term()

    def _register_nodes(self) -> None:
        while len(self._registered_addresses) < self.number_of_nodes:
            name_bytes, _, address_bytes = self._router.recv_multipart()
            name = name_bytes.decode()

            if name not in self.nodes:
                self._logger.info(f"Unknown node {name} tried to register")
                self._router.send_multipart([name_bytes, b"", b"Error: Unknown node"])
                continue

            address = address_bytes.decode()
            self._registered_addresses[name] = address
            self._logger.info(f"Node {name} registered with address {address}")

        self._logger.info("All nodes registered. Server is now ready.")

    def _notify_nodes_their_neighbors(self) -> None:
        for i in self.nodes:
            neighbor_info_list = self._get_neighbor_info_list(i)
            messages = [dumps(n_info).encode() for n_info in neighbor_info_list]

            self._router.send_multipart([i.encode(), b"", *messages])

        self._logger.info("Sent neighbor information to all nodes.")

    def _unregister_nodes(self) -> None:
        nodes_unregistered: list[str] = []
        while len(nodes_unregistered) < self.number_of_nodes:
            name_bytes, _, message = self._router.recv_multipart()
            name = name_bytes.decode()

            if name not in self.nodes:
                self._logger.info(f"Unknown node {name}. Cannot unregister.")
                continue

            if message == b"unregister":
                nodes_unregistered.append(name)
                self._router.send_multipart([name.encode(), b"", b"OK"])
                self._logger.info(f"Node {name} has unregistered.")
            else:
                self._logger.info(f"Received message from {name}: {message.decode()}")
