from logging import getLogger
from json import loads
from typing import KeysView
from zmq import Context, REQ, ROUTER, DEALER, IDENTITY, SyncSocket
from numpy import float64, frombuffer
from numpy.typing import NDArray
from .types import NeighborInfo
from .utils import get_local_ip
from .discovery import get_registry_info


class NodeHandle:
    """
    NodeHandle manages communication and state exchange between a node and its neighbors in a distributed network.

    This class handles:
    - Registration with a central registry to obtain neighbor information.
    - Establishing ZeroMQ sockets for communication with neighbors.
    - Sending and receiving state information to/from neighbors.
    - Performing operations like broadcasting, gathering, computing Laplacians, and weighted mixing.

    Parameters
    ----------
    name : str
        Unique identifier for the node.
    graph_name : str, optional
        Name of the graph to connect to (default is "default"). This should match the name used during graph creation.

    Attributes
    ----------
    name : str
        Name of the node.
    num_neighbors : int
        Number of neighbor nodes.
    neighbor_names : KeysView[str]
        Names of all neighbor nodes.

    Methods
    -------
    send_to_all(data_to_send: dict[str, NDArray[float64]]) -> None
        Sends data to all specified neighbor nodes.
    broadcast(state: NDArray[float64]) -> None
        Broadcasts the given state to all neighbor nodes.
    gather() -> list[NDArray[float64]]
        Receives and collects data from all neighbors.
    weighted_gather() -> list[NDArray[float64]]
        Gathers data from all neighbors, applying corresponding weights.
    laplacian(state: NDArray[float64]) -> NDArray[float64]
        Computes the Laplacian of the given state vector based on the states of neighboring nodes.
    weighted_mix(state: NDArray[float64]) -> NDArray[float64]
        Performs the weighted mixing operation for distributed optimization using the weight matrix W.

    Notes
    -----
    - Uses ZeroMQ sockets for communication.
    - State information is exchanged as NumPy arrays of float64, serialized to bytes.
    - Laplacian and weighted mixing operations are useful for consensus and distributed optimization algorithms.
    """

    def __init__(self, name: str, graph_name: str = "default") -> None:
        self._name = name

        self._registry_ip_addr, self._registry_port = get_registry_info(graph_name)

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
        self._req.connect(f"tcp://{self._registry_ip_addr}:{self._registry_port}")
        self._req.send(self._local_ip.encode() + b":" + str(self._port).encode())
        reply = self._req.recv_multipart()

        if reply[0] == b"Error: Unknown node":
            raise ValueError("Unknown node. Please check the node name.")

        for part in reply:
            neighbor_info: NeighborInfo = loads(part.decode())
            name = neighbor_info["name"]
            self._neighbor_addresses[name] = neighbor_info["address"]
            self._neighbor_weights[name] = neighbor_info["weight"]

        self._weight = 1.0 - sum(self._neighbor_weights.values())

        registry_address = f"{self._registry_ip_addr}:{self._registry_port}"
        self._logger.info(f"Registered node {self._name} at {registry_address}")
        self._logger.info(f"Neighbor addresses: {self._neighbor_addresses}")
        self._logger.info(f"Node address: {self._local_ip}:{self._port}")

    def _unregister(self) -> None:
        self._req.send(b"unregister")
        reply = self._req.recv()

        if reply == b"Error: Unknown node":
            self._logger.error("Node was not registered.")
        elif reply == b"OK":
            self._logger.info(f"Node {self._name} unregistered from server.")
        else:
            raise ValueError("Received unknown reply from server.")

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

    def send_to_all(self, data_to_send: dict[str, NDArray[float64]]) -> None:
        """
        Sends data to all specified neighbor nodes.

        Args:
            data_to_send (dict[str, NDArray[float64]]): A dictionary mapping neighbor names to the data arrays to send.

        Returns:
            None
        """
        for neighbor, state in data_to_send.items():
            if neighbor not in self._neighbor_addresses:
                raise ValueError(f"Neighbor {neighbor} not found.")

            state_bytes = state.tobytes()
            self._router.send_multipart([neighbor.encode(), state_bytes])

    def broadcast(self, state: NDArray[float64]) -> None:
        """
        Broadcasts the given state to all neighbor nodes.

        Args:
            state (NDArray[float64]): The state array to broadcast to neighbors.

        Returns:
            None
        """
        state_bytes = state.tobytes()
        for neighbor in self._neighbor_addresses:
            self._router.send_multipart([neighbor.encode(), state_bytes])

    def gather(self) -> dict[str, NDArray[float64]]:
        """
        Receives and collects data from all neighbors.

        Returns:
            dict[str, NDArray[float64]]: A dictionary mapping neighbor names to their received data arrays.
        """
        return {
            neighbor: frombuffer(dealer.recv(), dtype=float64)
            for neighbor, dealer in self._dealers.items()
        }

    def weighted_gather(self) -> dict[str, NDArray[float64]]:
        """
        Gathers data from all neighbors, applying corresponding weights to each received array.

        Returns:
            dict[str, NDArray[float64]]: A dictionary mapping neighbor names to their weighted data arrays.
            and multiplied by its associated weight.
        """
        return {
            neighbor: frombuffer(dealer.recv(), dtype=float64)
            * self._neighbor_weights[neighbor]
            for neighbor, dealer in self._dealers.items()
        }

    def laplacian(self, state: NDArray[float64]) -> NDArray[float64]:
        """
        Computes the Laplacian of the given state vector based on the states of neighboring nodes.

        The Laplacian is calculated as:

            laplacian = state * number_of_neighbors - sum_of_neighbor_states

        Args:
            state (NDArray[float64]): The state vector of the current node.

        Returns:
            NDArray[float64]: The Laplacian vector representing the difference between the current state and the average state of its neighbors.
        """
        self.broadcast(state)
        neighbor_states = self.gather()

        laplacian = state * len(neighbor_states) - sum(neighbor_states.values())

        return laplacian

    def weighted_mix(self, state: NDArray[float64]) -> NDArray[float64]:
        """
        Performs the weighted mixing operation for distributed optimization using the weight matrix W.

        For a given node i, the mixed state is computed as the i-th row of Wx, where x is the stacked state vector of all nodes.
        If x_i is multi-dimensional, the operation is applied element-wise.
        Specifically:

            mixed_state = W_ii * state + sum_j(W_ij * neighbor_state_j)

        where W_ii is self._weight and W_ij are the weights in self._neighbor_weights.

        Args:
            state (NDArray[float64]): The current state vector of node i.

        Returns:
            NDArray[float64]: The mixed state vector corresponding to the i-th row of Wx.
        """
        self.broadcast(state)
        weighted_neighbor_states = self.weighted_gather()

        mixed_state = state * self._weight + sum(weighted_neighbor_states.values())

        return mixed_state
