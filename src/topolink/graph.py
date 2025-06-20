import networkx as nx
from logging import info
from functools import cached_property
from zmq import Context, ROUTER
from .utils import get_local_ip


class Graph:
    def __init__(
        self,
        node_names: list[str],
        edge_pairs: list[tuple[str, str]],
        address: str | None = None,
    ) -> None:
        super().__init__()

        self._node_names = node_names
        self._edge_pairs = edge_pairs

        if not self.is_connected:
            raise ValueError("The provided topology must be connected.")

        self._local_ip = get_local_ip()

        self._address_map: dict[str, str] = {}

        self._context = Context()
        self._router = self._context.socket(ROUTER)
        if address is None:
            port = self._router.bind_to_random_port("tcp://*")
            self._address = f"{self._local_ip}:{port}"
        else:
            self._router.bind(f"tcp://{address}")
            self._address = address

        info(f"Server running on {self._address}")

    @cached_property
    def num_nodes(self) -> int:
        return len(self._node_names)

    @cached_property
    def num_edges(self) -> int:
        return len(self._edge_pairs)

    @cached_property
    def neighbor_map(self) -> dict[str, list[str]]:
        neighbor_map: dict[str, list[str]] = {name: [] for name in self._node_names}
        for u, v in self._edge_pairs:
            neighbor_map[u].append(v)
            neighbor_map[v].append(u)

        return neighbor_map

    @cached_property
    def is_connected(self) -> bool:
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(self._node_names)
        nx_graph.add_edges_from(self._edge_pairs)

        is_connected = nx.is_connected(nx_graph)

        return is_connected

    def plot(self, **kwargs) -> None:
        import matplotlib.pyplot as plt

        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(self._node_names)
        nx_graph.add_edges_from(self._edge_pairs)

        options = {"pos": nx.spring_layout(nx_graph), "with_labels": True}

        options.update(kwargs)

        nx.draw(nx_graph, **options)
        plt.title("Network Topology")
        plt.show()

    def deploy(self) -> None:
        try:
            self._register_nodes()
            self._notify_nodes_of_neighbors()
            # More deployment logic can be added here if needed.
            self._unregister_nodes()
        finally:
            self._router.close()
            self._context.term()

    def _register_nodes(self) -> None:
        while len(self._address_map) < self.num_nodes:
            name_bytes, _, address_bytes = self._router.recv_multipart()
            name = name_bytes.decode()

            if name not in self._node_names:
                info(f"Unknown node {name} tried to register")
                self._router.send_multipart([name_bytes, b"", b"Error: Unknown node"])
                continue

            address = address_bytes.decode()
            self._address_map[name] = address
            info(f"Node {name} registered with address {address}")

        info("All nodes registered. Server is now ready.")

    def _notify_nodes_of_neighbors(self) -> None:
        a_map = self._address_map
        n_map = self.neighbor_map
        neighbor_addresses: dict[str, list[str]] = {}
        for i in self._node_names:
            n_addr_i = [neighbor + ", " + a_map[neighbor] for neighbor in n_map[i]]
            neighbor_addresses[i] = n_addr_i

        for i in self._node_names:
            message = [addr.encode() for addr in neighbor_addresses[i]]
            self._router.send_multipart([i.encode(), b"", *message])

        info("Sent neighbor addresses to all nodes.")

    def _unregister_nodes(self) -> None:
        nodes_unregistered: set[str] = set()
        while len(nodes_unregistered) < self.num_nodes:
            name_bytes, _, message = self._router.recv_multipart()
            name = name_bytes.decode()

            if name not in self._node_names:
                info(f"Unknown node {name} sent a message {message.decode()}")
                continue

            if message == b"unregister":
                nodes_unregistered.add(name)
                self._router.send_multipart([name.encode(), b"", b"OK"])
                info(f"Node {name} has unregistered.")
            else:
                info(f"Received message from {name}: {message.decode()}")
