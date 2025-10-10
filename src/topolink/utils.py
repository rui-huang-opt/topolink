"""Utility functions for TopoLink."""

from socket import socket, AF_INET, SOCK_DGRAM


def get_local_ip() -> str:
    with socket(AF_INET, SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
