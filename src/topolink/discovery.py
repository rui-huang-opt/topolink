from logging import getLogger

logger = getLogger("topolink.discovery")

import socket
import threading
from zeroconf import Zeroconf, ServiceInfo, ServiceBrowser, ServiceListener
from .exceptions import GraphDiscoveryError

SERVICE_TYPE = "_topolink._tcp.local."


class GraphAdvertiser:
    def __init__(self, name: str):
        self._name = name
        self._zeroconf = Zeroconf()
        self._service_info: ServiceInfo | None = None

    def register(self, ip_addr: str, port: int):
        self._service_info = ServiceInfo(
            SERVICE_TYPE,
            self._name + "." + SERVICE_TYPE,
            addresses=[socket.inet_aton(ip_addr)],
            port=port,
            properties={
                "role": "graph",
                "version": "0.1.0",
                "description": "Topolink Graph Service",
            },
            server=socket.gethostname() + ".local.",
        )
        self._zeroconf.register_service(self._service_info)
        logger.info(f"Registered graph service with name {self._name}")

    def unregister(self):
        if self._service_info:
            self._zeroconf.unregister_service(self._service_info)
        self._zeroconf.close()
        logger.info(f"Unregistered graph service with name {self._name}")


class GraphListener(ServiceListener):
    def __init__(self, service_found: threading.Event):
        self.services = {}
        self.service_found = service_found
        self.service_found.clear()

    def add_service(self, zeroconf_: Zeroconf, service_type: str, name: str) -> None:
        info = zeroconf_.get_service_info(service_type, name)
        if info:
            ip_address = socket.inet_ntoa(info.addresses[0])
            port = info.port
            properties = {
                k.decode(): v.decode()
                for k, v in info.properties.items()
                if v is not None
            }
            self.services[name] = (ip_address, port, properties)
            self.service_found.set()
            logger.info(
                f"Graph service {name} added, service info: {self.services[name]}"
            )

    def remove_service(self, zeroconf_: Zeroconf, service_type: str, name: str) -> None:
        if name in self.services:
            del self.services[name]
            logger.info(f"Graph service {name} removed")


def get_graph_info(name: str) -> tuple[str, int]:
    service_name = name + "." + SERVICE_TYPE
    zeroconf_ = Zeroconf()
    service_found = threading.Event()

    try:
        listener = GraphListener(service_found)
        browser = ServiceBrowser(zeroconf_, SERVICE_TYPE, listener)

        if not service_found.wait(timeout=5):
            err_msg = f"Timeout: Graph service '{service_name}' not found."
            logger.error(err_msg)
            raise GraphDiscoveryError(err_msg)

        service_found.clear()

        if service_name not in listener.services:
            err_msg = f"Graph service '{service_name}' not found."
            logger.error(err_msg)
            raise GraphDiscoveryError(err_msg)

        graph_ip_addr: str = listener.services[service_name][0]
        graph_port: int = listener.services[service_name][1]

        return graph_ip_addr, graph_port
    finally:
        zeroconf_.close()
