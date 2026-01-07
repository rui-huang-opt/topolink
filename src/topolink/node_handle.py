from logging import getLogger
from json import loads
from typing import Callable, KeysView
from dataclasses import dataclass

import numpy as np
import zmq
from numpy.typing import NDArray

from .types import NeighborInfo
from .utils import get_local_ip
from .discovery import discover_graph


logger = getLogger(f"topolink.node_handle")


@dataclass(slots=True)
class NeighborContext:
    weight: float
    endpoint: str
    in_socket: zmq.Socket


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
    idx : str
        The index of the node within the graph.

    graph_name : str, optional
        Name of the graph to connect to (default is "default"). This should match the name used during graph creation.

    mask : Callable[[NDArray[float64]], NDArray[float64]], optional
        A function to apply to the state before sending it to neighbors (default is identity function).
        This can be used to add noise, apply privacy mechanisms, or modify the state in other ways.

    Attributes
    ----------
    idx : str
        The index of the node within the graph.

    degree : int
        The number of neighbors connected to this node.

    neighbors : KeysView[str]
        A view of the names of the neighbor nodes.

    weights : dict[str, float]
        A dictionary mapping neighbor names to their corresponding weights.

    Notes
    -----
    - Throughout the code, we use 'i' to denote the current node and 'j' to denote neighbor nodes, following common conventions in graph theory and distributed algorithms.
    """

    def __init__(
        self,
        idx: str,
        graph_name: str = "default",
        mask: Callable[[NDArray[np.float64]], NDArray[np.float64]] = lambda x: x,
    ) -> None:
        self._idx = idx
        self._graph_name = graph_name
        self._mask = mask

        endpoint = discover_graph(graph_name)

        if endpoint is None:
            err_msg = f"Timeout: Node '{idx}' can't discover graph '{graph_name}'."
            logger.error(err_msg)
            raise ConnectionError(err_msg)

        self._graph_ip_addr, self._graph_port = endpoint

        self._context = zmq.Context()
        self._reg = self._context.socket(zmq.DEALER)
        self._reg.setsockopt(zmq.IDENTITY, idx.encode())

        self._weight = 1.0
        self._out_socket = self._context.socket(zmq.ROUTER)
        self._ip_address = get_local_ip()
        self._port = self._out_socket.bind_to_random_port("tcp://*")

        self._neighbor_contexts: dict[str, NeighborContext] = {}

        self._register_to_graph()
        self._setup_neighbors()
        self._connect_to_neighbors()

        # Cache neighbor indices as bytes for ZeroMQ communication.
        # NOTE: Neighbor set is currently static.
        # If dynamic neighbor updates are introduced, this cache must be refreshed.
        self._neighbor_idx_bytes = [j.encode() for j in self._neighbor_contexts]

    @property
    def idx(self) -> str:
        return self._idx

    @property
    def degree(self) -> int:
        return len(self._neighbor_contexts)

    @property
    def neighbors(self) -> KeysView[str]:
        return self._neighbor_contexts.keys()

    @property
    def weights(self) -> dict[str, float]:
        return {j: nc.weight for j, nc in self._neighbor_contexts.items()}

    def _register_to_graph(self) -> None:
        self._reg.connect(f"tcp://{self._graph_ip_addr}:{self._graph_port}")
        self._reg.send(self._ip_address.encode() + b":" + str(self._port).encode())
        reply = self._reg.recv()

        if reply == b"OK":
            logger.info(f"Node '{self._idx}' joined graph '{self._graph_name}'.")
        elif reply == b"Error: Undefined node":
            err_msg = f"Undefined node '{self._idx}' tried to join graph '{self._graph_name}'."
            logger.error(err_msg)
            raise KeyError(err_msg)
        else:
            err_msg = f"Node '{self._idx}' failed to join graph '{self._graph_name}': {reply.decode()}."
            logger.error(err_msg)
            raise ConnectionError(err_msg)

    def _setup_neighbors(self) -> None:
        message = self._reg.recv()
        neighbor_info_dict: dict[str, NeighborInfo] = loads(message.decode())

        for j, info in neighbor_info_dict.items():
            in_socket = self._context.socket(zmq.DEALER)
            nc = NeighborContext(in_socket=in_socket, **info)
            self._neighbor_contexts[j] = nc

        self._weight = 1.0 - sum(nc.weight for nc in self._neighbor_contexts.values())
        logger.info(f"Node '{self._idx}' received neighbor info and set up sockets.")

    def _connect_to_neighbors(self) -> None:
        for nc in self._neighbor_contexts.values():
            nc.in_socket.setsockopt(zmq.IDENTITY, self._idx.encode())
            nc.in_socket.connect(f"tcp://{nc.endpoint}")

        for nc in self._neighbor_contexts.values():
            nc.in_socket.send(b"")

        connected = set()
        while len(connected) < self.degree:
            client_id, _ = self._out_socket.recv_multipart()
            received_name = client_id.decode()
            if received_name in self._neighbor_contexts:
                connected.add(received_name)

        logger.info(f"Node '{self._idx}' connected to all neighbors.")

    def exchange_map(
        self, state_map: dict[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        """
        Exchanges the given state map with all neighbor nodes.

        This method broadcasts the state map to all neighbors and then gathers their states.

        Args:
            state_map (dict[str, NDArray[np.float64]]): The state map to exchange with neighbors.

        Returns:
            dict[str, NDArray[np.float64]]: A dictionary mapping neighbor names to their received state maps.
        """
        if state_map.keys() != self._neighbor_contexts.keys():
            missing = self._neighbor_contexts.keys() - state_map.keys()
            extra = state_map.keys() - self._neighbor_contexts.keys()
            err_msg = f"State dictionary keys do not match neighbor names. Missing: {missing}, Extra: {extra}."
            logger.error(err_msg)
            raise ValueError(err_msg)

        for j in self._neighbor_contexts:
            state = state_map[j]
            masked_state = self._mask(state).astype(np.float64, copy=False)
            masked_state = np.ascontiguousarray(masked_state)
            self._out_socket.send_multipart([j.encode(), masked_state], copy=False)

        return {
            j: np.frombuffer(nc.in_socket.recv(), dtype=np.float64) * nc.weight
            for j, nc in self._neighbor_contexts.items()
        }

    def exchange(self, state: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        """
        Exchanges the given state with all neighbor nodes.

        This method broadcasts the state to all neighbors and then gathers their states.

        Args:
            state (NDArray[np.float64]): The state array to exchange with neighbors.

        Returns:
            dict[str, NDArray[np.float64]]: A dictionary mapping neighbor names to their received state arrays.
        """
        masked_state = self._mask(state).astype(np.float64, copy=False)
        masked_state = np.ascontiguousarray(masked_state)
        for j_bytes in self._neighbor_idx_bytes:
            self._out_socket.send_multipart([j_bytes, masked_state], copy=False)

        return {
            j: np.frombuffer(nc.in_socket.recv(), dtype=np.float64)
            for j, nc in self._neighbor_contexts.items()
        }

    def exchange_as_array(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Exchanges the given state with all neighbor nodes and returns their states as a stacked array.

        Note: Using this method will add an extra copy of the neighbor states in memory compared to the `exchange` method.
        This is because the states are first received as individual memory buffers and then stacked into a single array.

        Args:
            state (NDArray[np.float64]): The state array to exchange with neighbors.

        Returns:
            NDArray[np.float64]: A 2D array where each row corresponds to a neighbor's received state array.
        """
        masked_state = self._mask(state).astype(np.float64, copy=False)
        masked_state = np.ascontiguousarray(masked_state)
        for j_bytes in self._neighbor_idx_bytes:
            self._out_socket.send_multipart([j_bytes, masked_state], copy=False)

        neighbor_states = [
            np.frombuffer(nc.in_socket.recv(), dtype=np.float64)
            for nc in self._neighbor_contexts.values()
        ]

        return np.stack(neighbor_states, axis=0)

    def laplacian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes the Laplacian of the given state vector based on the states of neighboring nodes.

        The Laplacian is calculated as:

            laplacian = state * number_of_neighbors - sum_of_neighbor_states

        Args:
            state (NDArray[float64]): The state vector of the current node.

        Returns:
            NDArray[float64]: The Laplacian vector representing the difference between the current state and the average state of its neighbors.
        """
        neighbor_states = self.exchange(state)
        laplacian = state * len(neighbor_states) - sum(neighbor_states.values())

        return laplacian

    def weighted_mix(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Performs the weighted mixing operation for distributed optimization using the weight matrix W.

        For a given node i, the mixed state is computed as the i-th row of Wx, where x is the stacked state vector of all nodes.
        If x_i is multi-dimensional, the operation is applied element-wise.
        Specifically:

            mixed_state = W_ii * state + sum_j(W_ij * neighbor_state_j)

        where W_ii is self._weight and W_ij are the weights in self._neighbor_weights.

        Args:
            state (NDArray[np.float64]): The current state vector of node i.

        Returns:
            NDArray[float64]: The mixed state vector corresponding to the i-th row of Wx.
        """
        neighbor_states = self.exchange(state)
        nc = self._neighbor_contexts
        mixed_state = state * self._weight + sum(
            neighbor_state * nc[j].weight
            for j, neighbor_state in neighbor_states.items()
        )

        return mixed_state
