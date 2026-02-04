from logging import getLogger
from json import dumps
from threading import Thread

import zmq

from .graph import Graph
from .utils import get_local_ip
from .discovery import GraphAdvertiser

logger = getLogger("topolink.graph")


class BootstrapService:
    """
    Bootstrap service to manage node registration and neighbor notification.
    """

    def __init__(self, context: zmq.Context[zmq.Socket], graph: Graph) -> None:
        self._context = context
        self._graph = graph
        # idx: (rid: the router identity, endpoint: ip and port of the node)
        # We do not use the node index as the ROUTER identity, so that multiple
        # connections with the same idx can be distinguished during registration
        # (e.g., to properly reject or replace duplicate nodes).
        self._node_registry: dict[str, tuple[bytes, str]] = {}

    def _setup_router(self, router: zmq.Socket) -> None:
        if self._graph.transport == "tcp":
            self._graph_advertiser = GraphAdvertiser(self._graph.name)
            ip_address = get_local_ip()
            port = router.bind_to_random_port(f"tcp://{ip_address}")
            logger.info(f"Graph '{self._graph.name}' running on: {ip_address}:{port}")
            self._graph_advertiser.register(ip_address, port)

        elif self._graph.transport == "ipc":
            router.bind(f"ipc://@topolink-graph-{self._graph.name}")

        else:
            err_msg = f"Unsupported transport type: {self._graph.transport}"
            logger.error(err_msg)
            raise ValueError(err_msg)

    def _register_nodes(self, router: zmq.Socket) -> None:
        n_nodes = self._graph.number_of_nodes
        adj = self._graph.adjacency

        while len(self._node_registry) < n_nodes:
            rid, idx_bytes, endpoint_bytes = router.recv_multipart()
            idx = idx_bytes.decode()

            if idx not in adj:
                router.send_multipart([rid, b"Error: Undefined node"])
                continue

            if idx in self._node_registry:
                old_rid, _ = self._node_registry[idx]
                router.send_multipart([old_rid, b"Error: Node replaced"])

            endpoint = endpoint_bytes.decode()
            self._node_registry[idx] = (rid, endpoint)
            logger.info(
                f"Node '{idx}' joined graph '{self._graph.name}' from {endpoint}."
            )

        for idx in self._node_registry:
            rid, _ = self._node_registry[idx]
            router.send_multipart([rid, b"OK"])

        logger.info(f"Graph '{self._graph.name}' registration complete.")

    def _notify_nodes_of_neighbors(self, router: zmq.Socket) -> None:
        adj = self._graph.adjacency

        for i, neighbors in adj.items():
            for j in neighbors:
                _, endpoint = self._node_registry[j]
                adj[i][j]["endpoint"] = endpoint

        for i, neighbors in adj.items():
            messages = dumps(neighbors).encode()
            rid, _ = self._node_registry[i]
            router.send_multipart([rid, messages])

        logger.info(f"Sent neighbor info to all nodes in graph '{self._graph.name}'.")

    def apply(self) -> None:
        try:
            if not self._graph.is_connected():
                err_msg = "The graph is not connected."
                logger.error(err_msg)
                raise ValueError(err_msg)

            router = self._context.socket(zmq.ROUTER)
            self._setup_router(router)
            self._register_nodes(router)
            self._notify_nodes_of_neighbors(router)
            # TODO: More deployment logic can be added here if needed.

        finally:
            router.close(linger=0)

            if self._graph.transport == "tcp":
                self._graph_advertiser.unregister()


def bootstrap(*graphs: Graph) -> None:
    """
    Bootstrap the given graph by starting the bootstrap service in a separate thread.
    The first graph is bootstrapped in the main thread, while additional graphs
    are bootstrapped in separate threads.

    Parameters
    ----------
    graph : Graph
        The graph to bootstrap.

    Raises
    ------
    ValueError
        If no graphs are provided.
    """
    if not graphs:
        err_msg = "At least one graph must be provided for bootstrapping."
        logger.error(err_msg)
        raise ValueError(err_msg)

    context = zmq.Context().instance()

    services = [BootstrapService(context, graph) for graph in graphs]
    bootstrap_threads = [Thread(target=s.apply, daemon=True) for s in services]
    for t in bootstrap_threads:
        t.start()
