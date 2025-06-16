import time
import zmq
from multiprocessing import Process
from functools import cached_property
from numpy import float64, frombuffer
from numpy.typing import NDArray
from numpy.random import seed, uniform


class NameServer(Process):
    def __init__(self, n_names: list[str], e_pairs: list[tuple[str, str]]) -> None:
        super().__init__()

        self._n_names = n_names
        self._e_pairs = e_pairs

        self._neighbor_map: dict[str, list[str]] = {name: [] for name in n_names}
        for u, v in e_pairs:
            self._neighbor_map[u].append(v)
            self._neighbor_map[v].append(u)

        self._context = zmq.Context()
        self._address_map: dict[str, str] = {}

    def run(self) -> None:
        router = self._context.socket(zmq.ROUTER)
        port = router.bind_to_random_port("tcp://*")
        print(f"Name server running on port {port}")

        while len(self._address_map) < len(self._neighbor_map):
            client_id, _, address_bytes = router.recv_multipart()
            name = client_id.decode()

            if name in self._neighbor_map and name not in self._address_map:
                self._address_map[name] = address_bytes.decode()
                print(f"Registered {name} at {self._address_map[name]}")


class Peer(Process):
    def __init__(
        self,
        name: str,
        address: str,
        server_address: str,
        state: NDArray[float64],
        max_iter: int = 1000,
    ) -> None:
        super().__init__()

        self._name = name
        self._address = address
        self._server_address = server_address
        self._state = state.astype(float64)
        self._max_iter = max_iter

        self._context = zmq.Context()
        self._neighbors: list[str] = []

    @property
    def degree(self) -> int:
        return len(self._neighbors)

    @cached_property
    def name_bytes(self) -> bytes:
        return self._name.encode()

    def register(self) -> None:
        req = self._context.socket(zmq.REQ)
        req.setsockopt(zmq.IDENTITY, self.name_bytes)
        req.connect(self._server_address)

        req.send(self._address.encode())

        while True:
            reply = req.recv()

            if reply == b"OK":
                break

            self._neighbors.append(reply.decode())

    def run(self) -> None:
        router = self._context.socket(zmq.ROUTER)
        router.bind(address_map[self._name])

        dealers: dict[str, zmq.SyncSocket] = {}
        for neighbor in self._neighbors:
            dealer = self._context.socket(zmq.DEALER)
            dealer.setsockopt(zmq.IDENTITY, self.name_bytes)
            dealer.connect(address_map[neighbor])
            dealers[neighbor] = dealer

        for dealer in dealers.values():
            dealer.send(b"")

        n_names: set[str] = set()
        while n_names != self._neighbors:
            client_id, _ = router.recv_multipart()
            n_name = client_id.decode()
            n_names.add(n_name)

        begin = time.perf_counter()

        for k in range(self._max_iter):
            state_bytes = self._state.tobytes()
            for neighbor in self._neighbors:
                router.send_multipart([neighbor.encode(), state_bytes])

            neighbor_states: list[NDArray[float64]] = []
            for neighbor in self._neighbors:
                n_state_bytes = dealers[neighbor].recv()
                neighbor_states.append(frombuffer(n_state_bytes, dtype=float64))

            bias = self._state * self.degree - sum(neighbor_states)
            self._state -= bias * 0.45

        end = time.perf_counter()
        print(f"Node {self._name} finished in {end - begin:.6f} seconds")


def get_local_ip() -> str:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"


if __name__ == "__main__":
    node_names = ["1", "2", "3", "4", "5"]
    edge_pairs = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

    seed(0)
    states = {name: uniform(-10.0, 10.0, size=3) for name in node_names}

    server = NameServer(node_names, edge_pairs)

    address_map: dict[str, str] = {
        "1": "tcp://localhost:5551",
        "2": "tcp://localhost:5552",
        "3": "tcp://localhost:5553",
        "4": "tcp://localhost:5554",
        "5": "tcp://localhost:5555",
    }

    print("My IP:", get_local_ip())
