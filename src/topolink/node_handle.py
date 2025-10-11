from logging import getLogger

logger = getLogger(f"topolink.node_handle")

from json import loads
from zmq import REQ, ROUTER, DEALER, SNDTIMEO, RCVTIMEO, IDENTITY
from zmq import Context, Again
from numpy import float64, frombuffer
from numpy.typing import NDArray
from .exceptions import NodeUndefinedError, NodeJoinTimeoutError, NodeDiscoveryError
from .types import NeighborInfo, Neighbor
from .utils import get_local_ip
from .discovery import discover_graph_endpoint


class NodeHandle:
    """
    NodeHandle manages communication and state exchange between a node and its neighbors in a distributed network.

    This class handles:
    - Registration with a graph service to obtain neighbor information.
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
        endpoint = discover_graph_endpoint(graph_name)

        if endpoint is None:
            err_msg = f"Timeout: Node '{name}' can't discover graph '{graph_name}'."
            logger.error(err_msg)
            raise NodeDiscoveryError(err_msg)

        self._name = name
        self._graph_ip_addr, self._graph_port = endpoint
        self._graph_name = graph_name

        self._context = Context()
        self._req = self._context.socket(REQ)
        self._req.setsockopt(SNDTIMEO, 5000)
        self._req.setsockopt(RCVTIMEO, 5000)
        self._req.setsockopt(IDENTITY, self._name.encode())

        self._weight = 1.0
        self._out_socket = self._context.socket(ROUTER)
        self._ip_address = get_local_ip()
        self._port = self._out_socket.bind_to_random_port("tcp://*")

        self._neighbors: list[Neighbor] = []

        self._register_to_graph()
        self._connect_to_neighbors()

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_neighbors(self) -> int:
        return len(self._neighbors)

    @property
    def neighbor_names(self) -> list[str]:
        return [neighbor.name for neighbor in self._neighbors]

    def _register_to_graph(self) -> None:
        try:
            self._req.connect(f"tcp://{self._graph_ip_addr}:{self._graph_port}")
            self._req.send(self._ip_address.encode() + b":" + str(self._port).encode())
            reply = self._req.recv_multipart()
        except Again:
            err_msg = f"Timeout: Node '{self._name}' can't join the graph."
            logger.error(err_msg)
            raise NodeJoinTimeoutError(err_msg)

        if reply[0] == b"Error: Undefined node":
            err_msg = f"Undefined node '{self._name}' tried to join graph '{self._graph_name}'."
            logger.error(err_msg)
            raise NodeUndefinedError(err_msg)

        for part in reply:
            neighbor_info: NeighborInfo = loads(part.decode())
            in_socket = self._context.socket(DEALER)
            self._neighbors.append(Neighbor(in_socket=in_socket, **neighbor_info))

        self._weight = 1.0 - sum(neighbor.weight for neighbor in self._neighbors)
        logger.info(f"Node '{self._name}' joined graph '{self._graph_name}'.")

    def _connect_to_neighbors(self) -> None:
        for neighbor in self._neighbors:
            neighbor.in_socket.setsockopt(IDENTITY, self._name.encode())
            neighbor.in_socket.connect(f"tcp://{neighbor.endpoint}")

        for neighbor in self._neighbors:
            neighbor.in_socket.send(b"")

        connected = set()
        while len(connected) < self.num_neighbors:
            client_id, _ = self._out_socket.recv_multipart()
            received_name = client_id.decode()
            if received_name in self._neighbors:
                connected.add(received_name)

        logger.info(f"Node '{self._name}' connected to all neighbors.")

    def send_to_all(self, data_by_neighbor: dict[str, NDArray[float64]]) -> None:
        """
        Sends data to all specified neighbor nodes.

        Args:
            data_by_neighbor (dict[str, NDArray[float64]]): A dictionary mapping neighbor names to the data arrays to send.

        Returns:
            None
        """
        for n_name, data in data_by_neighbor.items():
            assert n_name in self.neighbor_names, f"Neighbor {n_name} not found."

            data_bytes = data.tobytes()
            self._out_socket.send_multipart([n_name.encode(), data_bytes])

    def broadcast(self, state: NDArray[float64]) -> None:
        """
        Broadcasts the given state to all neighbor nodes.

        Args:
            state (NDArray[float64]): The state array to broadcast to neighbors.

        Returns:
            None
        """
        state_bytes = state.tobytes()
        for neighbor in self._neighbors:
            self._out_socket.send_multipart([neighbor.name.encode(), state_bytes])

    def gather(self) -> dict[str, NDArray[float64]]:
        """
        Receives and collects data from all neighbors.

        Returns:
            dict[str, NDArray[float64]]: A dictionary mapping neighbor names to their received data arrays.
        """
        return {
            neighbor.name: frombuffer(neighbor.in_socket.recv(), dtype=float64)
            for neighbor in self._neighbors
        }

    def weighted_gather(self) -> dict[str, NDArray[float64]]:
        """
        Gathers data from all neighbors, applying corresponding weights to each received array.

        Returns:
            dict[str, NDArray[float64]]: A dictionary mapping neighbor names to their weighted data arrays.
            and multiplied by its associated weight.
        """
        return {
            neighbor.name: frombuffer(neighbor.in_socket.recv(), dtype=float64)
            * neighbor.weight
            for neighbor in self._neighbors
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
