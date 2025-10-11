"""Utility functions for TopoLink."""

from socket import socket, AF_INET, SOCK_DGRAM


def get_local_ip() -> str:
    """Get the local IP address of the machine."""
    with socket(AF_INET, SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"


from numpy import sum as np_sum
from numpy import allclose, float64
from numpy.typing import NDArray


def is_symmetric_double_stochastic(matrix: NDArray[float64]) -> bool:
    """Check if a matrix is symmetric and double-stochastic."""
    if not allclose(matrix, matrix.T):
        return False

    return allclose(np_sum(matrix, axis=0), 1.0)
