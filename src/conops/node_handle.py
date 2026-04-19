from logging import getLogger
from json import loads
from typing import KeysView, Literal
from dataclasses import dataclass

import zmq
import pyre
import numpy as np
from numpy.typing import NDArray

from .utils import get_local_ip, normalize_transport
from .transform import Transform, Identity

logger = getLogger(f"conops.node_handle")


@dataclass(slots=True)
class Neighbor:
    weight: float
    endpoint: str
    in_socket: zmq.Socket


class NodeHandle:
    """
    NodeHandle manages communication and state exchange between a node and its neighbors in a distributed network.

    This class handles:
    - Discovery of neighbor nodes using Pyre for service discovery.
    - Establishing ZeroMQ connections to neighbors based on the discovered endpoints.
    - Exchanging state information with neighbors using a specified Transform for encoding and decoding.
    - Providing utility methods for common operations like computing the Laplacian and performing weighted mixing.

    Parameters
    ----------
    idx : str
        The index of the node within the graph.

    namespace : str, optional
        The namespace for Pyre service discovery (default is "default"). Nodes will only discover neighbors that join the same namespace.

    transform : Transform | None, optional
        An optional Transform instance for encoding and decoding state data during communication.
        If None, the Identity transform is used, which performs no transformation.
        The Transform interface consists of `encode` and `decode` methods for serializing and deserializing state data.
        Typical transforms may include quantization, perturbation for privacy.

    transport : Literal["tcp", "ipc"], optional
        The transport type for ZeroMQ communication (default is "tcp").
        "ipc" is recommended for local multi-process communication on the same machine for better performance.

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
    - Throughout the code, we use 'i' to denote the current node and 'j' to denote neighbor nodes,
      following common conventions in graph theory and distributed algorithms.
    """

    def __init__(
        self,
        idx: str,
        neighbors: dict[str, float],
        namespace: str = "default",
        transform: Transform | None = None,
        transport: Literal["tcp", "ipc"] = "tcp",
    ) -> None:
        self._idx = idx

        self._context = zmq.Context()
        self._neighbors: dict[str, Neighbor] = {}
        for j, weight in neighbors.items():
            in_socket = self._context.socket(zmq.DEALER)
            in_socket.setsockopt(zmq.IDENTITY, self._idx.encode())
            self._neighbors[j] = Neighbor(
                weight=weight, endpoint="", in_socket=in_socket
            )

        self._namespace = namespace
        self._transform = transform or Identity()
        self._transport = normalize_transport(transport)
        self._endpoint, self._out_socket = self._create_and_bind_endpoint()
        self._weight = 1.0 - sum(neighbors.values())

        self._discover_neighbors()
        self._connect_to_neighbors()

        # Cache neighbor indices as bytes for ZeroMQ communication.
        # NOTE: Neighbor set is currently static.
        # If dynamic neighbor updates are introduced, this cache must be refreshed.
        self._neighbor_idx_bytes = [j.encode() for j in self._neighbors]

    @property
    def idx(self) -> str:
        return self._idx

    @property
    def degree(self) -> int:
        return len(self._neighbors)

    @property
    def neighbors(self) -> KeysView[str]:
        return self._neighbors.keys()

    @property
    def weights(self) -> dict[str, float]:
        return {j: nbr.weight for j, nbr in self._neighbors.items()}

    def __enter__(self) -> "NodeHandle":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def _create_and_bind_endpoint(self) -> tuple[str, zmq.SyncSocket]:
        out_socket = self._context.socket(zmq.ROUTER)
        if self._transport == "tcp":
            ip_address = get_local_ip()
            port = out_socket.bind_to_random_port(f"tcp://{ip_address}")
            endpoint = f"{ip_address}:{port}"
        elif self._transport == "ipc":
            out_socket.bind(f"ipc://@conops-{self._namespace}-{self._idx}")
            endpoint = f"@conops-{self._namespace}-{self._idx}"
        else:
            err_msg = f"Unsupported transport type: {self._transport}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        return endpoint, out_socket

    def _discover_neighbors(self) -> None:
        if self._transport == "tcp":
            node = pyre.Pyre(self._idx)
            node.set_header("endpoint", self._endpoint)
            node.join(self._namespace)
            node.start()

            discovered: set[str] = set()
            expected = set(self._neighbors.keys())
            pending: dict[bytes, tuple[str, str]] = {}

            while discovered != expected:
                msg = node.recv()
                event = msg[0].decode()
                nbr_uuid = msg[1]

                if event == "ENTER":
                    nbr_name = msg[2].decode()
                    headers: dict[str, str] = loads(msg[3].decode())
                    pending[nbr_uuid] = (nbr_name, headers["endpoint"])
                    continue

                if event != "JOIN":
                    continue

                namespace = msg[3].decode()
                if namespace != self._namespace:
                    continue

                info = pending.pop(nbr_uuid, None)
                if info is None:
                    continue

                nbr_name, nbr_endpoint = info
                if nbr_name not in expected:
                    continue

                self._neighbors[nbr_name].endpoint = nbr_endpoint
                discovered.add(nbr_name)
                logger.info(
                    "Neighbor discovered: "
                    f"node='{self._idx}', "
                    f"neighbor='{nbr_name}', "
                    f"endpoint='{self._neighbors[nbr_name].endpoint}'"
                )

            node.stop()

        elif self._transport == "ipc":
            # For IPC transport, we can directly construct the neighbor endpoints without discovery.
            for j in self._neighbors:
                self._neighbors[j].endpoint = f"@conops-{self._namespace}-{j}"

        else:
            err_msg = f"Unsupported transport type: {self._transport}"
            logger.error(err_msg)
            raise ValueError(err_msg)

    def _connect_to_neighbors(self) -> None:
        for nbr in self._neighbors.values():
            nbr.in_socket.connect(f"{self._transport}://{nbr.endpoint}")

        for nbr in self._neighbors.values():
            nbr.in_socket.send(b"")

        connected = set()
        expected = set(self._neighbors.keys())
        while connected != expected:
            client_id, _ = self._out_socket.recv_multipart()
            received_name = client_id.decode()
            if received_name in self._neighbors:
                connected.add(received_name)

        logger.info(f"Node '{self._idx}' connected to all neighbors.")

    def close(self) -> None:
        """
        Explicitly closes all sockets and terminates the ZeroMQ context.
        """
        self._out_socket.close(linger=0)
        for nbr in self._neighbors.values():
            nbr.in_socket.close(linger=0)
        self._context.term()
        logger.info(f"Node '{self._idx}' closed all sockets.")

    def neighborwise_exchange(
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
        if state_map.keys() != self._neighbors.keys():
            missing = self._neighbors.keys() - state_map.keys()
            extra = state_map.keys() - self._neighbors.keys()
            err_msg = f"State dictionary keys do not match neighbor names. Missing: {missing}, Extra: {extra}."
            logger.error(err_msg)
            raise ValueError(err_msg)

        for j in self._neighbors:
            state = state_map[j]
            meta, payload = self._transform.encode(state)
            payload = np.ascontiguousarray(payload)
            msgs = [j.encode(), meta, payload]
            self._out_socket.send_multipart(msgs, copy=False)

        neighbor_states: dict[str, NDArray[np.float64]] = {}
        for j, nbr in self._neighbors.items():
            meta, payload = nbr.in_socket.recv_multipart()
            neighbor_states[j] = self._transform.decode(meta, payload)

        return neighbor_states

    def exchange(self, state: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        """
        Exchanges the given state with all neighbor nodes.

        This method broadcasts the state to all neighbors and then gathers their states.

        Args:
            state (NDArray[np.float64]): The state array to exchange with neighbors.

        Returns:
            dict[str, NDArray[np.float64]]: A dictionary mapping neighbor names to their received state arrays.
        """
        meta, payload = self._transform.encode(state)
        payload = np.ascontiguousarray(payload)
        for j_bytes in self._neighbor_idx_bytes:
            msgs = [j_bytes, meta, payload]
            self._out_socket.send_multipart(msgs, copy=False)

        neighbor_states: dict[str, NDArray[np.float64]] = {}
        for j, nbr in self._neighbors.items():
            meta, payload = nbr.in_socket.recv_multipart()
            neighbor_states[j] = self._transform.decode(meta, payload)

        return neighbor_states

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
        meta, payload = self._transform.encode(state)
        payload = np.ascontiguousarray(payload)
        for j_bytes in self._neighbor_idx_bytes:
            msgs = [j_bytes, meta, payload]
            self._out_socket.send_multipart(msgs, copy=False)

        neighbor_states: list[NDArray[np.float64]] = []
        for nbr in self._neighbors.values():
            meta, payload = nbr.in_socket.recv_multipart()
            neighbor_states.append(self._transform.decode(meta, payload))

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
        nbrs = self._neighbors
        mixed_state = state * self._weight + sum(
            n_state * nbrs[j].weight for j, n_state in neighbor_states.items()
        )

        return mixed_state
