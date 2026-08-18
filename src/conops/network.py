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


class StateMeta(msgspec.Struct, frozen=True):
    dtype: str
    shape: tuple[int, ...]


class StateMessage(msgspec.Struct, frozen=True):
    round_id: int
    name: str
    meta: bytes
    payload: bytes


@dataclass
class PeerInfo:
    pyre_uuid: uuid.UUID | None = None
    last_seen: float = 0.0
    reachable: bool = False
    pending: list[StateMessage] = field(default_factory=list)


class PeerRegistry:
    """
    Maintains runtime information for known peers, including
    Pyre identities, reachability, and pending messages.
    """

    def __init__(self) -> None:
        self._peers: dict[str, PeerInfo] = {}
        self._uuid_to_node_id: dict[uuid.UUID, str] = {}

    def get_peer(self, node_id: str) -> PeerInfo | None:
        return self._peers.get(node_id)

    def get_peer_id_by_uuid(self, pyre_uuid: uuid.UUID) -> str | None:
        return self._uuid_to_node_id.get(pyre_uuid)

    def get_or_create_peer(self, node_id: str) -> PeerInfo:
        peer = self._peers.get(node_id)

        if peer is None:
            peer = PeerInfo()
            self._peers[node_id] = peer

        return peer

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


class ProtocolState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._round_id = 0
        self._exchange_name: str | None = None

    @property
    def exchange_key(self) -> tuple[int, str | None]:
        with self._lock:
            return self._round_id, self._exchange_name

    def begin_exchange(self, name: str) -> tuple[int, str]:
        with self._lock:
            if self._exchange_name is not None:
                raise RuntimeError(f"Exchange already active: {self._exchange_name}")

            self._exchange_name = name
            return self._round_id, name

    def end_exchange(self, name: str) -> None:
        with self._lock:
            if self._exchange_name != name:
                raise RuntimeError(f"Unexpected exchange end: {name}")

            self._exchange_name = None

    def advance_round(self) -> None:
        with self._lock:
            if self._exchange_name is not None:
                raise RuntimeError("Cannot advance round during an active exchange")

            self._round_id += 1

    def set_round(self, round_id: int) -> None:
        with self._lock:
            if self._exchange_name is not None:
                raise RuntimeError("Cannot set round during an active exchange")

            self._round_id = round_id


class NetworkBackend:
    def __init__(
        self,
        node_id: str,
        state: ProtocolState,
        node: pyre.Pyre,
        router: zmq.SyncSocket,
    ) -> None:
        self._node_id = node_id
        self._state = state
        self._node = node
        self._router = router

        self._peers = PeerRegistry()

        self._msg_cache: StateMessage | None = None

    def handle_pyre_event(self, event: list[bytes]) -> None:
        if not event:
            return

        event_type = event[0].decode("utf-8")
        timestamp = time.time()

        match event_type:
            case "ENTER":
                self._handle_enter(event, timestamp)
            case "EXIT":
                self._handle_exit(event, timestamp)
            case "WHISPER":
                self._handle_message_event(event)
            case _:
                logger.debug("[%s] Ignoring Pyre event: %s", self._node_id, event_type)

    def handle_frontend_message(self, frames: list[bytes]) -> None:
        if len(frames) < 3:
            return

        peer_node_id_bytes, meta, payload = frames[:3]
        peer_node_id = peer_node_id_bytes.decode("utf-8")

        peer = self._peers.get_or_create_peer(peer_node_id)

        round_id, name = self._state.exchange_key

        if name is None:
            logger.warning(
                "[%s] Received message from %s outside of an active exchange",
                self._node_id,
                peer_node_id,
            )
            return

        msg = StateMessage(round_id=round_id, name=name, meta=meta, payload=payload)
        self._msg_cache = msg

        if not peer.reachable or peer.pyre_uuid is None:
            peer.pending.append(msg)
            return

        self._node.whisper(peer.pyre_uuid, msgspec.msgpack.encode(msg))

    def _handle_enter(self, event: list[bytes], timestamp: float) -> None:
        pyre_uuid = self._extract_uuid(event)
        if pyre_uuid is None:
            return

        peer_node_id = self._extract_node_id(event)
        if peer_node_id is None or peer_node_id == self._node_id:
            return

        peer = self._peers.mark_reachable(
            node_id=peer_node_id,
            pyre_uuid=pyre_uuid,
            last_seen=timestamp,
        )

        for msg in peer.pending:
            self._node.whisper(pyre_uuid, msgspec.msgpack.encode(msg))

        peer.pending.clear()

    def _handle_exit(self, event: list[bytes], timestamp: float) -> None:
        pyre_uuid = self._extract_uuid(event)
        if pyre_uuid is None:
            return

        self._peers.mark_unreachable(pyre_uuid=pyre_uuid, last_seen=timestamp)

    def _handle_message_event(self, event: list[bytes]) -> None:
        peer_node_id = self._extract_node_id(event)
        msg = self._extract_state_message(event)

        if peer_node_id is None or msg is None:
            return

        round_id, name = self._state.exchange_key

        if msg.name != name:
            return

        if msg.round_id < round_id:
            self._state.set_round(msg.round_id)
        elif msg.round_id > round_id:
            return

        self._router.send_multipart(
            [peer_node_id.encode("utf-8"), msg.meta, msg.payload]
        )

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

        node_id_bytes = event[2]

        if not isinstance(node_id_bytes, bytes):
            return None

        try:
            return node_id_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return None

    @staticmethod
    def _extract_state_message(event: list[bytes]) -> StateMessage | None:
        if len(event) < 4:
            return None

        try:
            return msgspec.msgpack.decode(event[3], type=StateMessage)
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

        self._state = ProtocolState()

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
        round_id, _ = self._state.exchange_key
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

    def next_round(self) -> None:
        self._state.advance_round()

    def exchange(self, name: str, state: npt.NDArray) -> dict[str, npt.NDArray]:
        """
        Exchanges the given state with all neighbor nodes.

        This method broadcasts the state to all neighbors and then gathers their states.

        Args:
            state (NDArray[np.float64]): The state array to exchange with neighbors.

        Returns:
            dict[str, NDArray[np.float64]]: A dictionary mapping neighbor names to their received state arrays.
        """
        self._state.begin_exchange(name)

        meta = StateMeta(dtype=str(state.dtype), shape=state.shape)
        meta_bytes = msgspec.msgpack.encode(meta)
        payload = np.ascontiguousarray(state)
        for j in self._neighbors:
            dealer = self._dealers[j]
            msgs = [meta_bytes, payload]
            dealer.send_multipart(msgs)

        neighbor_states: dict[str, npt.NDArray] = {}
        for j in self._neighbors:
            dealer = self._dealers[j]
            meta_bytes, payload = dealer.recv_multipart()

            meta = msgspec.msgpack.decode(meta_bytes, type=StateMeta)
            value = np.frombuffer(payload, dtype=meta.dtype).reshape(meta.shape)
            neighbor_states[j] = value

        self._state.end_exchange(name)

        return neighbor_states

    def laplacian(self, name: str, state: npt.NDArray) -> npt.NDArray:
        """
        Computes the Laplacian of the given state vector based on the states of neighboring nodes.

        The Laplacian is calculated as:

            laplacian = state * number_of_neighbors - sum_of_neighbor_states

        Args:
            state (NDArray[float64]): The state vector of the current node.

        Returns:
            NDArray[float64]: The Laplacian vector representing the difference between the current state and the average state of its neighbors.
        """
        neighbor_states = self.exchange(name, state)
        laplacian = state * len(neighbor_states) - sum(neighbor_states.values())

        return laplacian

    def _run(self) -> None:
        node = pyre.Pyre(self._node_id, ctx=self._context)
        pyre_socket = node.socket()

        router = self._context.socket(zmq.ROUTER)
        router.setsockopt(zmq.LINGER, 0)
        router.bind(self._endpoint)

        poller = zmq.Poller()
        poller.register(pyre_socket, zmq.POLLIN)
        poller.register(router, zmq.POLLIN)

        backend = NetworkBackend(self._node_id, self._state, node, router)

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
