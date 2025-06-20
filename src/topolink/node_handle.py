from logging import info
from typing import KeysView
from functools import cached_property
from zmq import Context, REQ, ROUTER, DEALER, IDENTITY, SyncSocket
from numpy import float64, frombuffer
from numpy.typing import NDArray
from .utils import get_local_ip


class NodeHandle:
    def __init__(self, name: str, server_address: str | None = None) -> None:
        self._name = name
        self._server_address = server_address

        self._local_ip = get_local_ip()

        self._context = Context()
        self._req = self._context.socket(REQ)
        self._req.setsockopt(IDENTITY, self._name.encode())

        self._router = self._context.socket(ROUTER)
        self._port = self._router.bind_to_random_port("tcp://*")

        self._neighbor_addresses: dict[str, str] = {}
        self._dealers: dict[str, SyncSocket] = {}

        self._register()
        self._connect_to_neighbors()

    def __del__(self) -> None:
        self._unregister()

    @property
    def name(self) -> str:
        return self._name

    @cached_property
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
        for part in reply:
            if part == b"Error: Unknown node":
                raise ValueError(
                    "Unknown node. Please check the server address and node name."
                )

            neighbor_name, neighbor_address = part.decode().split(", ")
            self._neighbor_addresses[neighbor_name] = neighbor_address

        info(f"Node {self._name} registered with server at {self._server_address}")
        info(f"Neighbor addresses: {self._neighbor_addresses}")
        info(f"Node address: {self._local_ip}:{self._port}")

    def _unregister(self) -> None:
        self._req.send(b"unregister")
        reply = self._req.recv()
        if reply != b"OK":
            raise RuntimeError("Failed to unregister node.")

        info(f"Node {self._name} unregistered from server.")

    def _connect_to_neighbors(self) -> None:
        for neighbor_name, address in self._neighbor_addresses.items():
            dealer = self._context.socket(DEALER)
            dealer.setsockopt(IDENTITY, self._name.encode())
            dealer.connect(f"tcp://{address}")
            self._dealers[neighbor_name] = dealer

        for dealer in self._dealers.values():
            dealer.send(b"")

        connected = set()
        while len(connected) < len(self._neighbor_addresses):
            client_id, _ = self._router.recv_multipart()
            neighbor_name = client_id.decode()
            if neighbor_name in self._neighbor_addresses:
                connected.add(neighbor_name)

        info(f"Connected to neighbors: {', '.join(self._neighbor_addresses)}")

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
        neighbor_states: list[NDArray[float64]] = []
        for dealer in self._dealers.values():
            neighbor_state_bytes = dealer.recv()
            neighbor_states.append(frombuffer(neighbor_state_bytes, dtype=float64))
        return neighbor_states

    def laplacian(self, state: NDArray[float64]) -> NDArray[float64]:
        state_bytes = state.tobytes()
        for neighbor in self._neighbor_addresses:
            self._router.send_multipart([neighbor.encode(), state_bytes])

        neighbor_states: list[NDArray[float64]] = []
        for dealer in self._dealers.values():
            n_state_bytes = dealer.recv()
            neighbor_states.append(frombuffer(n_state_bytes, dtype=float64))

        bias = state * len(neighbor_states) - sum(neighbor_states)

        return bias
