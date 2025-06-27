import networkx as nx
from json import dumps
from logging import getLogger
from functools import cached_property
from zmq import Context, SyncSocket, ROUTER
from matplotlib.axes import Axes
from numpy import float64, allclose, ones
from numpy import sum as np_sum
from numpy.typing import NDArray
from .types import NodeInput, EdgeInput, NodeView, EdgeView, AdjView, NeighborInfo
from .utils import get_local_ip


class Graph:
    def __init__(
        self, nx_graph: nx.Graph | None = None, address: str | None = None
    ) -> None:
        super().__init__()

        self._address = address
        self._nx_graph = nx.Graph() if nx_graph is None else nx_graph

        self._registered_addresses: dict[str, str] = {}
        self._logger = getLogger("topolink.Graph")
        self._local_ip = get_local_ip()
        self._context = Context()

    @property
    def nodes(self) -> NodeView:
        return self._nx_graph.nodes

    @property
    def edges(self) -> EdgeView:
        return self._nx_graph.edges(data=True)

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
        else:
            router.bind(f"tcp://{self._address}")
            _, port = self._address.split(":")

        self._address = f"{self._local_ip}:{port}"
        self._logger.info(f"Server running on {self._address}")

        return router

    @property
    def is_connected(self) -> bool:
        return nx.is_connected(self._nx_graph)

    def add_nodes(self, nodes: NodeInput) -> None:
        self._nx_graph.add_nodes_from(nodes)

    def add_edges(self, edges: EdgeInput) -> None:
        self._nx_graph.add_edges_from(edges)

    @classmethod
    def from_nodes_and_edges(
        cls,
        nodes: NodeInput,
        edges: EdgeInput,
        address: str | None = None,
    ) -> "Graph":
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(nodes)
        nx_graph.add_edges_from(edges)

        return cls(nx_graph=nx_graph, address=address)

    @classmethod
    def from_mixing_matrix(
        cls,
        mixing_matrix: NDArray[float64],
        address: str | None = None,
        nodes: list[str] | None = None,
    ) -> "Graph":
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

    def draw(self, ax: Axes, **kwargs) -> None:
        nx_graph = self._nx_graph

        pos = nx.spring_layout(nx_graph)

        options = {"pos": pos, "ax": ax, "with_labels": True}
        options.update(kwargs)

        nx.draw(nx_graph, **options)

    def adjacency(self, node: str) -> AdjView:
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

        self._logger.info("Sent neighbor addresses to all nodes.")

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
