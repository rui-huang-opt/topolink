from zmq import Context, ROUTER
from functools import cached_property
from .utils import get_local_ip


class Topology:
    def __init__(
        self,
        node_names: list[str],
        edge_pairs: list[tuple[str, str]],
        address: str | None = None,
    ) -> None:
        super().__init__()

        self._node_names = node_names
        self._edge_pairs = edge_pairs

        self._neighbor_map: dict[str, list[str]] = {name: [] for name in node_names}
        for u, v in edge_pairs:
            self._neighbor_map[u].append(v)
            self._neighbor_map[v].append(u)

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

        print(f"Server running on {self._address}")

    def deploy(self) -> None:
        while len(self._address_map) < len(self._node_names):
            name_bytes, _, address_bytes = self._router.recv_multipart()
            name = name_bytes.decode()

            if name not in self._node_names:
                print(f"Unknown node {name} tried to register")
                self._router.send_multipart(
                    [name_bytes, b"", b"Error: Unknown node"]
                )
                continue

            address = address_bytes.decode()
            self._address_map[name] = address
            print(f"Node {name} registered with address {address}")

        print("All nodes registered. Server is now ready.")

        neighbor_addresses: dict[str, list[str]] = {}
        for name in self._node_names:
            neighbor_addresses[name] = [
                neighbor + ", " + self._address_map[neighbor]
                for neighbor in self._neighbor_map[name]
            ]

        for name in self._node_names:
            message = [address.encode() for address in neighbor_addresses[name]]
            self._router.send_multipart([name.encode(), b"", *message])

        print("Sent neighbor addresses to all nodes.")
