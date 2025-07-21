from logging import getLogger
from json import loads
from typing import KeysView
from zmq import Context, REQ, ROUTER, DEALER, IDENTITY, SyncSocket
from numpy import float64, frombuffer
from numpy.typing import NDArray
from .types import NeighborInfo
from .utils import get_local_ip


class NodeHandle:
    def __init__(self, name: str, server_address: str | None = None) -> None:
        self._name = name
        self._server_address = server_address

        self._logger = getLogger(f"topolink.NodeHandle")
        self._local_ip = get_local_ip()

        self._context = Context()
        self._req = self._context.socket(REQ)
        self._req.setsockopt(IDENTITY, self._name.encode())

        self._router = self._context.socket(ROUTER)
        self._port = self._router.bind_to_random_port("tcp://*")

        self._weight = 1.0
        self._neighbor_addresses: dict[str, str] = {}
        self._neighbor_weights: dict[str, float] = {}
        self._dealers: dict[str, SyncSocket] = {}

        self._register()
        self._connect_to_neighbors()

    def __del__(self) -> None:
        self._unregister()

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_neighbors(self) -> int:
        return len(self._neighbor_addresses)

    @property
    def neighbor_names(self) -> KeysView[str]:
        return self._neighbor_addresses.keys()

    def _register(self) -> None:
        if self._server_address is None:
            self._server_address = input(
                "Please enter the server address (IP:Port):"
            ).strip()

        self._req.connect(f"tcp://{self._server_address}")
        self._req.send(self._local_ip.encode() + b":" + str(self._port).encode())
        reply = self._req.recv_multipart()

        if reply[0] == b"Error: Unknown node":
            raise ValueError("Unknown node. Please check the node name.")

        for part in reply:
            neighbor_info: NeighborInfo = loads(part.decode())
            name = neighbor_info["name"]
            self._neighbor_addresses[name] = neighbor_info["address"]
            self._neighbor_weights[name] = neighbor_info["weight"]

        self._weight = 1 - sum(self._neighbor_weights.values())

        self._logger.info(f"Registered node {self._name} at {self._server_address}")
        self._logger.info(f"Neighbor addresses: {self._neighbor_addresses}")
        self._logger.info(f"Node address: {self._local_ip}:{self._port}")

    def _unregister(self) -> None:
        self._req.send(b"unregister")
        reply = self._req.recv()
        if reply != b"OK":
            raise RuntimeError("Failed to unregister node.")

        self._logger.info(f"Node {self._name} unregistered from server.")

    def _connect_to_neighbors(self) -> None:
        for neighbor_name, address in self._neighbor_addresses.items():
            dealer = self._context.socket(DEALER)
            dealer.setsockopt(IDENTITY, self._name.encode())
            dealer.connect(f"tcp://{address}")
            self._dealers[neighbor_name] = dealer

        for dealer in self._dealers.values():
            dealer.send(b"")

        connected = set()
        while len(connected) < self.num_neighbors:
            client_id, _ = self._router.recv_multipart()
            neighbor_name = client_id.decode()
            if neighbor_name in self._neighbor_addresses:
                connected.add(neighbor_name)

        self._logger.info("Connected to all neighbors.")

    def send(self, neighbor: str, state: NDArray[float64]) -> None:
        if neighbor not in self._neighbor_addresses:
            raise ValueError(f"Neighbor {neighbor} is not registered.")

        state_bytes = state.tobytes()
        self._router.send_multipart([neighbor.encode(), state_bytes])

    def recv(self, neighbor: str) -> NDArray[float64]:
        if neighbor not in self._dealers:
            raise ValueError(f"Dealer for neighbor {neighbor} is not registered.")

        neighbor_state_bytes = self._dealers[neighbor].recv()

        return frombuffer(neighbor_state_bytes, dtype=float64)

    def broadcast(self, state: NDArray[float64]) -> None:
        state_bytes = state.tobytes()
        for neighbor in self._neighbor_addresses:
            self._router.send_multipart([neighbor.encode(), state_bytes])

    def gather(self) -> list[NDArray[float64]]:
        return [
            frombuffer(dealer.recv(), dtype=float64)
            for dealer in self._dealers.values()
        ]

    def laplacian(self, state: NDArray[float64]) -> NDArray[float64]:
        state_bytes = state.tobytes()
        for neighbor in self._neighbor_addresses:
            self._router.send_multipart([neighbor.encode(), state_bytes])

        neighbor_states = [
            frombuffer(dealer.recv(), dtype=float64)
            for dealer in self._dealers.values()
        ]

        laplacian = state * len(neighbor_states) - sum(neighbor_states)

        return laplacian

    def weighted_mix(self, state: NDArray[float64]) -> NDArray[float64]:
        state_bytes = state.tobytes()
        for neighbor in self._neighbor_addresses:
            self._router.send_multipart([neighbor.encode(), state_bytes])

        weighted_neighbor_states = [
            frombuffer(dealer.recv(), dtype=float64) * self._neighbor_weights[neighbor]
            for neighbor, dealer in self._dealers.items()
        ]

        mixed_state = state * self._weight + sum(weighted_neighbor_states)

        return mixed_state
