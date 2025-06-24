import networkx as nx
from json import dumps
from logging import getLogger
from functools import cached_property
from zmq import Context, SyncSocket, ROUTER
from matplotlib.axes import Axes
from numpy import float64, allclose
from numpy.typing import NDArray
from .types import NeighborInfo
from .utils import get_local_ip


class Graph:
    def __init__(
        self,
        nodes: list[str],
        edges: list[tuple[str, str]] | dict[tuple[str, str], float],
        address: str | None = None,
    ) -> None:
        super().__init__()

        self._nodes = nodes

        if isinstance(edges, list):
            self._edges = {(u, v): 1.0 for u, v in edges}
        elif isinstance(edges, dict):
            self._edges = edges
        else:
            raise TypeError("Edges must be a list of tuples or a dictionary.")

        self._address = address

        if not self.is_connected:
            raise ValueError("The provided topology must be connected.")

        self._logger = getLogger("topolink.Graph")
        self._local_ip = get_local_ip()

        self._node_addresses: dict[str, str] = {}
        self._context = Context()

    @cached_property
    def neighbor_info_by_node(self) -> dict[str, list[NeighborInfo]]:
        neighbor_info_by_node: dict[str, list[NeighborInfo]] = {
            i: [] for i in self._nodes
        }
        for (u, v), weight in self._edges.items():
            ni_u = NeighborInfo(name=v, address=self._node_addresses[v], weight=weight)
            ni_v = NeighborInfo(name=u, address=self._node_addresses[u], weight=weight)
            neighbor_info_by_node[u].append(ni_u)
            neighbor_info_by_node[v].append(ni_v)

        return neighbor_info_by_node

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

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

    @cached_property
    def is_connected(self) -> bool:
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(self._nodes)
        nx_graph.add_edges_from(self._edges)

        is_connected = nx.is_connected(nx_graph)

        return is_connected

    def draw(self, ax: Axes, **kwargs) -> None:
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(self._nodes)
        nx_graph.add_edges_from(self._edges)

        options = {"pos": nx.spring_layout(nx_graph), "ax": ax, "with_labels": True}

        options.update(kwargs)

        nx.draw(nx_graph, **options)

    def deploy(self) -> None:
        try:
            self._register_nodes()
            self._notify_nodes_their_neighbors()
            # TODO: More deployment logic can be added here if needed.
            self._unregister_nodes()
        finally:
            self._router.close()
            self._context.term()

    def _register_nodes(self) -> None:
        while len(self._node_addresses) < self.num_nodes:
            name_bytes, _, address_bytes = self._router.recv_multipart()
            name = name_bytes.decode()

            if name not in self._nodes:
                self._logger.info(f"Unknown node {name} tried to register")
                self._router.send_multipart([name_bytes, b"", b"Error: Unknown node"])
                continue

            address = address_bytes.decode()
            self._node_addresses[name] = address
            self._logger.info(f"Node {name} registered with address {address}")

        self._logger.info("All nodes registered. Server is now ready.")

    def _notify_nodes_their_neighbors(self) -> None:
        for i in self._nodes:
            neighbor_info = self.neighbor_info_by_node[i]
            messages = [dumps(info).encode() for info in neighbor_info]

            self._router.send_multipart([i.encode(), b"", *messages])

        self._logger.info("Sent neighbor addresses to all nodes.")

    def _unregister_nodes(self) -> None:
        nodes_unregistered: list[str] = []
        while len(nodes_unregistered) < self.num_nodes:
            name_bytes, _, message = self._router.recv_multipart()
            name = name_bytes.decode()

            if name not in self._nodes:
                self._logger.info(f"Unknown node {name}. Cannot unregister.")
                continue

            if message == b"unregister":
                nodes_unregistered.append(name)
                self._router.send_multipart([name.encode(), b"", b"OK"])
                self._logger.info(f"Node {name} has unregistered.")
            else:
                self._logger.info(f"Received message from {name}: {message.decode()}")
