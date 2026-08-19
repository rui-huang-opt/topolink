import typing
import threading
import time
import uuid
import logging
from dataclasses import dataclass, field

import zmq
import pyre
import msgspec
import numpy as np
import numpy.typing as npt

from .transform import Transform, Identity

logger = logging.getLogger(__name__)


class State(msgspec.Struct, tag="STATE", tag_field="type", frozen=True):
    round_id: int
    name: str
    meta: bytes
    payload: bytes


class Request(msgspec.Struct, tag="REQUEST", tag_field="type", frozen=True):
    round_id: int
    name: str


class Sync(msgspec.Struct, tag="SYNC", tag_field="type", frozen=True):
    round_id: int
    name: str
    meta: bytes
    payload: bytes


Message = State | Request | Sync


@dataclass(slots=True)
class PeerInfo:
    pyre_uuid: uuid.UUID | None = None
    last_seen: float = 0.0
    reachable: bool = False
    pending: list[State] = field(default_factory=list)


class PeerRegistry:
    """
    Maintains runtime information for known peers.
    """

    def __init__(self) -> None:
        self._peers: dict[str, PeerInfo] = {}
        self._uuid_to_node_id: dict[uuid.UUID, str] = {}

    def get_or_create_peer(self, node_id: str) -> PeerInfo:
        peer = self._peers.get(node_id)

        if peer is None:
            peer = PeerInfo()
            self._peers[node_id] = peer

        return peer

    def get_peer_id_by_uuid(self, pyre_uuid: uuid.UUID) -> str | None:
        return self._uuid_to_node_id.get(pyre_uuid)

    def get_uuid_by_peer_id(self, node_id: str) -> uuid.UUID | None:
        peer = self._peers.get(node_id)

        if peer is None or not peer.reachable or peer.pyre_uuid is None:
            return None

        return peer.pyre_uuid

    def mark_reachable(
        self, node_id: str, pyre_uuid: uuid.UUID, last_seen: float
    ) -> PeerInfo:
        peer = self.get_or_create_peer(node_id)

        previous_uuid = peer.pyre_uuid

        if previous_uuid is not None and previous_uuid != pyre_uuid:
            self._uuid_to_node_id.pop(previous_uuid, None)

        peer.pyre_uuid = pyre_uuid
        peer.last_seen = last_seen
        peer.reachable = True

        self._uuid_to_node_id[pyre_uuid] = node_id

        return peer

    def mark_unreachable(
        self, pyre_uuid: uuid.UUID, last_seen: float
    ) -> PeerInfo | None:
        node_id = self._uuid_to_node_id.get(pyre_uuid)

        if node_id is None:
            return None

        peer = self._peers.get(node_id)

        if peer is None or peer.pyre_uuid != pyre_uuid:
            return None

        peer.reachable = False
        peer.last_seen = last_seen

        return peer


class Exchange:
    CACHE_ROUNDS = 2

    def __init__(self) -> None:
        self._lock = threading.Lock()

        self._round_id = 0
        self._name: str | None = None

        self._sent: dict[int, dict[str, State]] = {}
        self._received: dict[str, dict[tuple[int, str], State]] = {}
        self._delivered: dict[str, set[tuple[int, str]]] = {}
        self._waiting: dict[str, tuple[int, str]] = {}

    @property
    def round_id(self) -> int:
        with self._lock:
            return self._round_id

    @property
    def key(self) -> tuple[int, str | None]:
        with self._lock:
            return self._round_id, self._name

    def begin(self, name: str) -> tuple[int, str]:
        with self._lock:
            if self._name is not None:
                raise RuntimeError(f"Exchange already active: {self._name}")

            self._name = name

            return self._round_id, name

    def end(self, name: str) -> None:
        with self._lock:
            if self._name != name:
                raise RuntimeError(f"Unexpected exchange end: {name}")

            self._name = None

    def advance_round(self) -> int:
        with self._lock:
            if self._name is not None:
                raise RuntimeError("Cannot advance round during an active exchange")

            self._round_id += 1
            self._prune()

            return self._round_id

    def sync_round(self, round_id: int, name: str) -> None:
        with self._lock:
            if self._name != name:
                raise RuntimeError(f"Exchange mismatch: {name} != {self._name}")

            if round_id < self._round_id:
                return

            self._round_id = round_id
            self._prune()

    def cache_sent(self, msg: State) -> None:
        with self._lock:
            round_id = msg.round_id
            if round_id != self._round_id:
                raise RuntimeError(f"Round mismatch: {round_id} != {self._round_id}")

            self._sent.setdefault(round_id, {})[msg.name] = msg

    def get_sent(self, round_id: int, name: str) -> State | None:
        with self._lock:
            messages = self._sent.get(round_id)

            if messages is None:
                return None

            return messages.get(name)

    def cache_received(self, peer_id: str, msg: State) -> None:
        with self._lock:
            key = (msg.round_id, msg.name)

            # A duplicate of something already delivered is useless.
            if key in self._delivered.get(peer_id, set()):
                return

            self._received.setdefault(peer_id, {})[key] = msg

    def take_received(self, peer_id: str) -> State | None:
        """
        Returns a previously received state matching the current exchange.

        If a state is returned, it is atomically marked as delivered.
        """
        with self._lock:
            if self._name is None:
                return None

            key = (self._round_id, self._name)

            messages = self._received.get(peer_id)

            if messages is None:
                return None

            # If this exchange has already been delivered, discard any
            # duplicate cached copy.
            delivered = self._delivered.get(peer_id)

            if delivered is not None and key in delivered:
                messages.pop(key, None)
                return None

            msg = messages.pop(key, None)

            if msg is None:
                return None

            self._delivered.setdefault(peer_id, set()).add(key)

            return msg

    def claim_delivery(self, peer_id: str, msg: State) -> bool:
        """
        Claims a received state for delivery to the frontend.

        Returns True only when the state matches the current exchange and
        has not already been delivered.
        """
        with self._lock:
            if self._name is None:
                return False

            key = (msg.round_id, msg.name)

            current_key = (self._round_id, self._name)

            if key != current_key:
                return False

            delivered = self._delivered.setdefault(peer_id, set())

            if key in delivered:
                return False

            delivered.add(key)

            # Remove an equivalent cached copy if one exists.
            messages = self._received.get(peer_id)

            if messages is not None:
                messages.pop(key, None)

            return True

    def begin_wait(self, peer_id: str) -> None:
        with self._lock:
            if self._name is None:
                raise RuntimeError("No active exchange")

            self._waiting[peer_id] = (self._round_id, self._name)

    def is_waiting(self, peer_id: str, msg: State) -> bool:
        with self._lock:
            return self._waiting.get(peer_id) == (msg.round_id, msg.name)

    def finish_wait(self, peer_id: str, msg: State) -> None:
        with self._lock:
            key = (msg.round_id, msg.name)

            if self._waiting.get(peer_id) == key:
                self._waiting.pop(peer_id, None)

            self._delivered.setdefault(peer_id, set()).add(key)

    def _prune(self) -> None:
        min_round = max(0, self._round_id - self.CACHE_ROUNDS + 1)

        self._sent = {
            round_id: messages
            for round_id, messages in self._sent.items()
            if round_id >= min_round
        }

        for peer_id in list(self._received):
            received = {
                key: msg
                for key, msg in self._received[peer_id].items()
                if key[0] >= min_round
            }

            if received:
                self._received[peer_id] = received
            else:
                self._received.pop(peer_id, None)

        for peer_id in list(self._delivered):
            delivered = {key for key in self._delivered[peer_id] if key[0] >= min_round}

            if delivered:
                self._delivered[peer_id] = delivered
            else:
                self._delivered.pop(peer_id, None)


class NetworkBackend:
    def __init__(
        self,
        node_id: str,
        namespace: str,
        exchange: Exchange,
        node: pyre.Pyre,
        router: zmq.SyncSocket,
    ) -> None:
        self._node_id = node_id
        self._namespace = namespace
        self._exchange = exchange
        self._node = node
        self._router = router

        self._peers = PeerRegistry()

    def handle_pyre_event(self, event: list[bytes]) -> None:
        if not event:
            return

        try:
            event_type = event[0].decode("utf-8")
        except UnicodeDecodeError:
            return

        timestamp = time.time()

        if event_type == "ENTER":
            self._handle_enter(event, timestamp)

        elif event_type == "EXIT":
            self._handle_exit(event, timestamp)

        elif event_type == "WHISPER":
            self._handle_whisper(event)

        else:
            logger.debug("[%s] Ignoring Pyre event: %s", self._node_id, event_type)

    def handle_frontend_message(self, frames: list[bytes]) -> None:
        if len(frames) != 3:
            return

        peer_bytes, meta, payload = frames

        try:
            peer_id = peer_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return

        round_id, name = self._exchange.key

        if name is None:
            logger.warning("[%s] No active exchange", self._node_id)
            return

        msg = State(round_id=round_id, name=name, meta=meta, payload=payload)

        self._exchange.cache_sent(msg)

        self._exchange.begin_wait(peer_id)

        peer = self._peers.get_or_create_peer(peer_id)

        received = self._exchange.take_received(peer_id)

        if received is not None:
            self._forward_to_frontend(peer_id, received)
            self._exchange.finish_wait(peer_id, received)

        if not peer.reachable or peer.pyre_uuid is None:
            peer.pending.append(msg)
            return

        self._send_message(peer.pyre_uuid, msg)

    def _handle_enter(self, event: list[bytes], timestamp: float) -> None:
        pyre_uuid = self._extract_uuid(event)

        if pyre_uuid is None:
            return

        peer_id = self._extract_node_id(event)

        if peer_id is None or peer_id == self._node_id:
            return

        namespace = self._extract_namespace(event)

        if namespace != self._namespace:
            return

        peer = self._peers.mark_reachable(
            node_id=peer_id,
            pyre_uuid=pyre_uuid,
            last_seen=timestamp,
        )

        for msg in peer.pending:
            self._send_message(pyre_uuid, msg)

        peer.pending.clear()

    def _handle_exit(self, event: list[bytes], timestamp: float) -> None:
        pyre_uuid = self._extract_uuid(event)

        if pyre_uuid is None:
            return

        self._peers.mark_unreachable(pyre_uuid=pyre_uuid, last_seen=timestamp)

    def _handle_whisper(self, event: list[bytes]) -> None:
        peer_id = self._extract_node_id(event)
        msg = self._extract_message(event)

        if peer_id is None or msg is None:
            return

        if isinstance(msg, State):
            self._handle_state(peer_id, msg)

        elif isinstance(msg, Request):
            self._handle_request(peer_id, msg)

        elif isinstance(msg, Sync):
            self._handle_sync(peer_id, msg)

    def _handle_state(self, peer_id: str, msg: State) -> None:
        local_round, local_name = self._exchange.key

        if local_name is None:
            self._exchange.cache_received(peer_id, msg)
            return

        # 1. 正常当前消息
        if msg.round_id == local_round and msg.name == local_name:
            if self._exchange.is_waiting(peer_id, msg):
                self._forward_to_frontend(peer_id, msg)
                self._exchange.finish_wait(peer_id, msg)
            else:
                self._exchange.cache_received(peer_id, msg)
            return

        # 2. peer 落后：restart / catch-up
        if msg.round_id < local_round:
            promoted = State(
                round_id=local_round,
                name=local_name,
                meta=msg.meta,
                payload=msg.payload,
            )

            if self._exchange.is_waiting(peer_id, promoted):
                self._forward_to_frontend(peer_id, promoted)
                self._exchange.finish_wait(peer_id, promoted)
            else:
                self._exchange.cache_received(peer_id, promoted)

            current = self._exchange.get_sent(round_id=local_round, name=local_name)

            if current is None:
                return

            pyre_uuid = self._peers.get_uuid_by_peer_id(peer_id)

            if pyre_uuid is None:
                return

            sync = Sync(
                round_id=local_round,
                name=local_name,
                meta=current.meta,
                payload=current.payload,
            )

            self._send_message(pyre_uuid, sync)
            return

        # 3. peer ahead / 同轮不同 exchange
        self._exchange.cache_received(peer_id, msg)

    def _handle_request(self, peer_id: str, request: Request) -> None:
        sent = self._exchange.get_sent(round_id=request.round_id, name=request.name)

        if sent is None:
            return

        pyre_uuid = self._peers.get_uuid_by_peer_id(peer_id)

        if pyre_uuid is None:
            return

        self._send_message(pyre_uuid, sent)

    def _handle_sync(self, peer_id: str, sync: Sync) -> None:
        local_round, local_name = self._exchange.key

        if local_name != sync.name:
            return

        if sync.round_id < local_round:
            return

        if sync.round_id > local_round:
            self._exchange.sync_round(
                round_id=sync.round_id,
                name=sync.name,
            )

        msg = State(
            round_id=sync.round_id,
            name=sync.name,
            meta=sync.meta,
            payload=sync.payload,
        )

        if self._exchange.claim_delivery(peer_id, msg):
            self._forward_to_frontend(peer_id, msg)

    def _handle_behind_peer(self, peer_id: str, msg: State) -> None:
        local_round, local_name = self._exchange.key

        if local_name is None:
            return

        # 第一版只对相同 exchange 做直接 rejoin。
        if msg.name != local_name:
            self._exchange.cache_received(peer_id, msg)
            self._request_current_state(peer_id)
            return

        # 把重启节点的旧状态提升为“当前 round 的状态”。
        promoted = State(
            round_id=local_round, name=local_name, meta=msg.meta, payload=msg.payload
        )

        if self._exchange.claim_delivery(peer_id, promoted):
            self._forward_to_frontend(peer_id, promoted)

        # 把我自己当前 round 的状态发给重启节点，
        # 告诉它直接同步到这里。
        current = self._exchange.get_sent(round_id=local_round, name=local_name)

        if current is None:
            return

        pyre_uuid = self._peers.get_uuid_by_peer_id(peer_id)

        if pyre_uuid is None:
            return

        sync = Sync(
            round_id=local_round,
            name=local_name,
            meta=current.meta,
            payload=current.payload,
        )

        self._send_message(pyre_uuid, sync)

    def _request_current_state(self, peer_id: str) -> None:
        round_id, name = self._exchange.key

        if name is None:
            return

        pyre_uuid = self._peers.get_uuid_by_peer_id(peer_id)

        if pyre_uuid is None:
            return

        request = Request(round_id=round_id, name=name)

        self._send_message(pyre_uuid, request)

    def _forward_to_frontend(self, peer_id: str, msg: State) -> None:
        self._router.send_multipart([peer_id.encode("utf-8"), msg.meta, msg.payload])

    def _send_message(self, pyre_uuid: uuid.UUID, msg: Message) -> None:
        self._node.whisper(pyre_uuid, msgspec.msgpack.encode(msg))

    @staticmethod
    def _extract_uuid(event: list[bytes]) -> uuid.UUID | None:
        if len(event) < 2:
            return None

        value = event[1]

        if not isinstance(value, bytes) or len(value) != 16:
            return None

        return uuid.UUID(bytes=value)

    @staticmethod
    def _extract_node_id(event: list[bytes]) -> str | None:
        if len(event) < 3:
            return None

        value = event[2]

        if not isinstance(value, bytes):
            return None

        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return None

    @staticmethod
    def _extract_namespace(event: list[bytes]) -> str | None:
        if len(event) < 4:
            return None

        candidate = event[3]

        if not isinstance(candidate, bytes):
            return None

        try:
            headers = msgspec.json.decode(candidate, type=dict[str, str])
            return headers.get("namespace")
        except msgspec.DecodeError:
            return None

    @staticmethod
    def _extract_message_type(event: list[bytes]) -> str | None:
        if len(event) < 4:
            return None

        value = event[3]

        if not isinstance(value, bytes):
            return None

        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return None

    @staticmethod
    def _extract_message(event: list[bytes]) -> Message | None:
        if len(event) < 4:
            return None

        try:
            return msgspec.msgpack.decode(event[3], type=Message)
        except msgspec.DecodeError:
            return None


class Network:
    def __init__(
        self,
        node_id: str,
        neighbors: dict[str, float],
        *,
        namespace: str = "default",
        transform: Transform | None = None,
        recv_timeout_ms: int = 200,
        stop_timeout: float = 1.0,
        context: zmq.SyncContext | None = None,
    ) -> None:
        self._node_id = node_id
        self._neighbors = neighbors
        self._namespace = namespace
        self._transform = transform if transform is not None else Identity()
        self._recv_timeout_ms = recv_timeout_ms
        self._stop_timeout = stop_timeout

        if context is None:
            self._context: zmq.SyncContext = zmq.Context.instance()
        else:
            self._context = context

        self._exchange = Exchange()

        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._dealers: dict[str, zmq.SyncSocket] = {}

        backend_uuid = uuid.uuid4().hex
        self._endpoint = f"inproc://{self._namespace}:{self._node_id}:{backend_uuid}"

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def neighbors(self) -> typing.KeysView[str]:
        return self._neighbors.keys()

    @property
    def round_id(self) -> int:
        round_id, _ = self._exchange.key
        return round_id

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"[{self._node_id}] Network already started")

        if self._dealers:
            raise RuntimeError(f"[{self._node_id}] Stale dealer sockets")

        self._running.set()
        thread = threading.Thread(target=self._run, daemon=True)
        self._thread = thread
        thread.start()

        try:
            for neighbor_id in self._neighbors:
                dealer = self._context.socket(zmq.DEALER)
                dealer.setsockopt(zmq.LINGER, 0)
                dealer.setsockopt(zmq.IDENTITY, neighbor_id.encode("utf-8"))
                dealer.connect(self._endpoint)

                self._dealers[neighbor_id] = dealer

        except Exception:
            self._running.clear()

            for dealer in self._dealers.values():
                dealer.close()
            self._dealers.clear()

            thread.join(self._stop_timeout)

            if not thread.is_alive():
                self._thread = None

            raise

    def stop(self) -> None:
        self._running.clear()

        for dealer in self._dealers.values():
            dealer.close()
        self._dealers.clear()

        thread = self._thread

        if thread is None:
            return

        thread.join(self._stop_timeout)

        if thread.is_alive():
            logger.warning("[%s] Network stop timed out", self._node_id)
            return

        self._thread = None

    def next_round(self) -> int:
        return self._exchange.advance_round()

    def neighborwise_exchange(
        self, name: str, state_map: dict[str, npt.NDArray]
    ) -> dict[str, npt.NDArray]:
        """
        Exchanges the given state map with all neighbor nodes.

        This method broadcasts the state map to all neighbors and then gathers their states.

        Args:
            name (str): The name of the variable to be exchanged.

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

        self._exchange.begin(name)

        for j in self._neighbors:
            state = state_map[j]
            meta, payload = self._transform.encode(state)
            payload = np.ascontiguousarray(payload)
            dealer = self._dealers[j]
            dealer.send_multipart([meta, payload])

        neighbor_states: dict[str, npt.NDArray] = {}
        for j in self._neighbors:
            dealer = self._dealers[j]
            meta, payload = dealer.recv_multipart()
            neighbor_states[j] = self._transform.decode(meta, payload)

        self._exchange.end(name)

        return neighbor_states

    def exchange(self, name: str, state: npt.NDArray) -> dict[str, npt.NDArray]:
        """
        Exchanges the given state with all neighbor nodes.

        This method broadcasts the state to all neighbors and then gathers their states.

        Args:
            name (str): The name of the variable to be exchanged.

            state (NDArray[np.float64]): The state array to exchange with neighbors.

        Returns:
            dict[str, NDArray[np.float64]]: A dictionary mapping neighbor names to their received state arrays.
        """
        self._exchange.begin(name)

        meta_bytes, payload = self._transform.encode(state)
        payload = np.ascontiguousarray(payload)
        for j in self._neighbors:
            dealer = self._dealers[j]
            dealer.send_multipart([meta_bytes, payload])

        neighbor_states: dict[str, npt.NDArray] = {}
        for j in self._neighbors:
            dealer = self._dealers[j]
            meta_bytes, payload = dealer.recv_multipart()
            value = self._transform.decode(meta_bytes, payload)
            neighbor_states[j] = value

        self._exchange.end(name)

        return neighbor_states

    def exchange_as_array(self, name: str, state: npt.NDArray) -> npt.NDArray:
        """
        Exchanges the given state with all neighbor nodes and returns their states as a stacked array.

        Note: Using this method will add an extra copy of the neighbor states in memory compared to the `exchange` method.
        This is because the states are first received as individual memory buffers and then stacked into a single array.

        Args:
            name (str): The name of the variable to be exchanged.

            state (NDArray[np.float64]): The state array to exchange with neighbors.

        Returns:
            NDArray[np.float64]: A 2D array where each row corresponds to a neighbor's received state array.
        """
        self._exchange.begin(name)

        meta_bytes, payload = self._transform.encode(state)
        payload = np.ascontiguousarray(payload)
        for j in self._neighbors:
            dealer = self._dealers[j]
            dealer.send_multipart([meta_bytes, payload])

        neighbor_states: list[npt.NDArray] = []
        for j in self._neighbors:
            dealer = self._dealers[j]
            meta_bytes, payload = dealer.recv_multipart()
            value = self._transform.decode(meta_bytes, payload)
            neighbor_states.append(value)

        self._exchange.end(name)

        return np.stack(neighbor_states, axis=0)

    def laplacian(self, name: str, state: npt.NDArray) -> npt.NDArray:
        """
        Computes the Laplacian of the given state vector based on the states of neighboring nodes.

        The Laplacian is calculated as:

            laplacian = state * number_of_neighbors - sum_of_neighbor_states

        Args:
            name (str): The name of the variable for which the Laplacian is being computed.

            state (NDArray[float64]): The state vector of the current node.

        Returns:
            NDArray[float64]: The Laplacian vector representing the difference between the current state and the average state of its neighbors.
        """
        neighbor_states = self.exchange(name, state)
        laplacian = state * len(neighbor_states) - sum(neighbor_states.values())

        return laplacian

    def weighted_mix(self, name: str, state: npt.NDArray) -> npt.NDArray:
        """
        Performs the weighted mixing operation for distributed optimization using the weight matrix W.

        For a given node i, the mixed state is computed as the i-th row of Wx, where x is the stacked state vector of all nodes.
        If x_i is multi-dimensional, the operation is applied element-wise.
        Specifically:

            mixed_state = W_ii * state + sum_j(W_ij * neighbor_state_j)

        where W_ii is self._weight and W_ij are the weights in self._neighbor_weights.

        Args:
            name (str): The name of the variable being mixed.

            state (NDArray[np.float64]): The current state vector of node i.

        Returns:
            NDArray[float64]: The mixed state vector corresponding to the i-th row of Wx.
        """
        neighbor_states = self.exchange(name, state)
        nbrs = self._neighbors
        neighbor_mix = sum(neighbor_states[j] * w for j, w in nbrs.items())

        return (1.0 - sum(nbrs.values())) * state + neighbor_mix

    def _run(self) -> None:
        """
        The main event loop for the network backend.
        """
        node = pyre.Pyre(self._node_id, ctx=self._context)
        node.set_header("namespace", self._namespace)
        pyre_socket = node.socket()

        router = self._context.socket(zmq.ROUTER)
        router.setsockopt(zmq.LINGER, 0)
        router.bind(self._endpoint)

        poller = zmq.Poller()
        poller.register(pyre_socket, zmq.POLLIN)
        poller.register(router, zmq.POLLIN)

        backend = NetworkBackend(
            node_id=self._node_id,
            namespace=self._namespace,
            exchange=self._exchange,
            node=node,
            router=router,
        )

        try:
            node.start()

            while self._running.is_set():
                events = dict(poller.poll(timeout=self._recv_timeout_ms))

                if pyre_socket in events:
                    event = node.recv()
                    backend.handle_pyre_event(event)

                if router in events:
                    frames = router.recv_multipart()
                    backend.handle_frontend_message(frames)

        except Exception:
            if self._running.is_set():
                logger.exception("[%s] Network event loop failed", self._node_id)

        finally:
            poller.unregister(pyre_socket)
            poller.unregister(router)

            try:
                node.stop()
            except Exception:
                logger.exception("[%s] Error stopping Pyre node", self._node_id)

            router.close()

            logger.info("[%s] Network thread stopped", self._node_id)
