from logging import getLogger

logger = getLogger("topolink.discovery")

import socket
import threading
from zeroconf import Zeroconf, ServiceInfo, ServiceBrowser, ServiceListener

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
        logger.info(f"Registered graph service with name '{self._name}'")

    def unregister(self):
        if self._service_info:
            self._zeroconf.unregister_service(self._service_info)
        self._zeroconf.close()
        logger.info(f"Unregistered graph service with name '{self._name}'")


class GraphListener(ServiceListener):
    def __init__(self, name):
        self.services = {}
        self._name = name
        self.service_found = threading.Event()

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
            if name == self._name + "." + SERVICE_TYPE:
                self.service_found.set()
                logger.info(f"Graph service '{name}' added")

    def remove_service(self, zeroconf_: Zeroconf, service_type: str, name: str) -> None:
        if name in self.services:
            del self.services[name]
            self.service_found.clear()
            logger.info(f"Graph service '{name}' removed")


def discover_graph(graph_name: str) -> tuple[str, int] | None:
    service_name = graph_name + "." + SERVICE_TYPE
    zeroconf_ = Zeroconf()

    try:
        listener = GraphListener(graph_name)
        browser = ServiceBrowser(zeroconf_, SERVICE_TYPE, listener)

        if not listener.service_found.wait(timeout=5):
            return None

        graph_ip_addr: str = listener.services[service_name][0]
        graph_port: int = listener.services[service_name][1]

        return graph_ip_addr, graph_port
    finally:
        browser.cancel()
        zeroconf_.close()
