from logging import getLogger
from json import loads
from typing import KeysView
from zmq import Context, REQ, ROUTER, DEALER, IDENTITY, SyncSocket
from numpy import float64, frombuffer
from numpy.typing import NDArray
from .types import NeighborInfo
from .utils import get_local_ip


class NodeHandle:
    """
    NodeHandle manages communication and state exchange between a node and its neighbors in a distributed network.

    This class handles:
        - Registration with a central server
        - Connection setup with neighbor nodes
        - Methods for sending, receiving, broadcasting, and gathering state information
        - Distributed optimization operations such as Laplacian and weighted mixing computations

    Parameters
    ----------
    name : str
        Unique identifier for the node.
    server_address : str | None, optional
        Address of the central server (IP:Port). If not provided, it will be prompted interactively.

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
    send(neighbor: str, state: NDArray[float64]) -> None
        Sends state information to a specified neighbor node.
    recv(neighbor: str) -> NDArray[float64]
        Receives state data from a specified neighbor.
    broadcast(state: NDArray[float64]) -> None
        Broadcasts the given state to all neighbor nodes.
    gather() -> list[NDArray[float64]]
        Receives and collects data from all neighbors.
    weighted_gather() -> list[NDArray[float64]]
        Gathers data from all neighbors, applying corresponding weights.
    gather_with_name() -> dict[str, NDArray[float64]]
        Collects data from all neighbor nodes and returns a dictionary mapping neighbor names to their received data arrays.
    laplacian(state: NDArray[float64]) -> NDArray[float64]
        Computes the Laplacian of the given state vector based on the states of neighboring nodes.
    weighted_mix(state: NDArray[float64]) -> NDArray[float64]
        Performs the weighted mixing operation for distributed optimization using the weight matrix W.

    Notes
    -----
    - Uses ZeroMQ sockets for communication.
    - State information is exchanged as NumPy arrays of float64, serialized to bytes.
    - Node must be registered with the server before interacting with neighbors.
    - Laplacian and weighted mixing operations are useful for consensus and distributed optimization algorithms.
    """

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

        self._weight = 1.0 - sum(self._neighbor_weights.values())

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
        """
        Sends the state information to a specified neighbor node.

        Parameters
        ----------
        neighbor : str
            The identifier of the neighbor node to send the state to. Must be registered in `_neighbor_addresses`.
        state : NDArray[float64]
            The state data to be sent, represented as a NumPy array of float64 values.

        Raises
        ------
        ValueError
            If the specified neighbor is not registered in `_neighbor_addresses`.

        Notes
        -----
        The state is serialized to bytes before transmission. The message is sent using the router's
        `send_multipart` method, with the neighbor's address and the serialized state.
        """
        if neighbor not in self._neighbor_addresses:
            raise ValueError(f"Neighbor {neighbor} is not registered.")

        state_bytes = state.tobytes()
        self._router.send_multipart([neighbor.encode(), state_bytes])

    def recv(self, neighbor: str) -> NDArray[float64]:
        """
        Receives the state data from a specified neighbor.

        Parameters
        ----------
        neighbor : str
            The identifier of the neighbor node to receive data from.

        Returns
        -------
        NDArray[float64]
            The state data received from the neighbor, as a NumPy array of float64.

        Raises
        ------
        ValueError
            If the dealer for the specified neighbor is not registered.
        """
        if neighbor not in self._dealers:
            raise ValueError(f"Dealer for neighbor {neighbor} is not registered.")

        neighbor_state_bytes = self._dealers[neighbor].recv()

        return frombuffer(neighbor_state_bytes, dtype=float64)

    def broadcast(self, state: NDArray[float64]) -> None:
        """
        Broadcasts the given state to all neighbor nodes.

        Converts the provided state array to bytes and sends it to each neighbor address
        using the router's send_multipart method.

        Args:
            state (NDArray[float64]): The state array to broadcast to neighbors.

        Returns:
            None
        """
        state_bytes = state.tobytes()
        for neighbor in self._neighbor_addresses:
            self._router.send_multipart([neighbor.encode(), state_bytes])

    def gather(self) -> list[NDArray[float64]]:
        """
        Receives and collects data from all dealers.

        Iterates over all dealer objects in self._dealers, receives raw byte data from each dealer,
        and converts the received bytes into NumPy arrays of type float64.

        Returns:
            list[numpy.ndarray]: A list of NumPy arrays containing the received data from each dealer.
        """
        return [
            frombuffer(dealer.recv(), dtype=float64)
            for dealer in self._dealers.values()
        ]

    def weighted_gather(self) -> list[NDArray[float64]]:
        """
        Gathers data from all neighbors, applying corresponding weights to each received array.

        Returns:
            list[NDArray[float64]]: A list of weighted arrays, where each array is received from a neighbor
            and multiplied by its associated weight.
        """
        return [
            frombuffer(dealer.recv(), dtype=float64) * self._neighbor_weights[neighbor]
            for neighbor, dealer in self._dealers.items()
        ]

    def gather_with_name(self) -> dict[str, NDArray[float64]]:
        """
        Collects data from all neighbor nodes and returns a dictionary mapping neighbor names to their received data arrays.

        Returns:
            dict[str, NDArray[float64]]: A dictionary where each key is a neighbor's name and each value is a NumPy array
            of float64 values received from the corresponding neighbor.
        """
        return {
            neighbor: frombuffer(dealer.recv(), dtype=float64)
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

        laplacian = state * len(neighbor_states) - sum(neighbor_states)

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

        mixed_state = state * self._weight + sum(weighted_neighbor_states)

        return mixed_state
