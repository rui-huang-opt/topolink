import socket
from zeroconf import Zeroconf, ServiceInfo, ServiceBrowser, ServiceListener

SERVICE_TYPE = "_topolink._tcp.local."
SERVICE_NAME = "Topolink Registry Service._topolink._tcp.local."


class RegistryAdvertiser:
    def __init__(self):
        self._zeroconf = Zeroconf()
        self._service_info: ServiceInfo | None = None

    def register(self, ip_addr: str, port: int):
        self._service_info = ServiceInfo(
            SERVICE_TYPE,
            SERVICE_NAME,
            addresses=[socket.inet_aton(ip_addr)],
            port=port,
            properties={
                "role": "registry",
                "version": "0.1.0",
                "description": "Topolink Registry Service",
            },
            server=socket.gethostname() + ".local.",
        )
        self._zeroconf.register_service(self._service_info)

    def unregister(self):
        if self._service_info:
            self._zeroconf.unregister_service(self._service_info)
        self._zeroconf.close()


class RegistryListener(ServiceListener):
    def __init__(self):
        self.services = {}

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

    def remove_service(self, zeroconf_: Zeroconf, service_type: str, name: str) -> None:
        if name in self.services:
            del self.services[name]
            print(f"Service {name} removed")


def get_registry_info() -> tuple[str, int]:
    zeroconf_ = Zeroconf()
    try:
        listener = RegistryListener()
        browser = ServiceBrowser(zeroconf_, SERVICE_TYPE, listener)

        while SERVICE_NAME not in listener.services:
            pass

        registry_ip_addr: str = listener.services[SERVICE_NAME][0]
        registry_port: int = listener.services[SERVICE_NAME][1]

        return registry_ip_addr, registry_port
    finally:
        zeroconf_.close()
