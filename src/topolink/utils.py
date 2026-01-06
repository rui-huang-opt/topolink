"""Utility functions for TopoLink."""

from socket import socket, AF_INET, SOCK_DGRAM


def get_local_ip() -> str:
    """
    Get the local IP address of the machine.

    Returns
    -------
    str
        The local IP address.
    """
    with socket(AF_INET, SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"


import numpy as np
from numpy import allclose, float64
from numpy.typing import NDArray


def is_symmetric_doubly_stochastic(matrix: NDArray[float64]) -> bool:
    """
    Check if a matrix is symmetric doubly stochastic.

    Parameters
    ----------
    matrix : NDArray[float64]
        The matrix to check.

    Returns
    -------
    bool
        True if the matrix is symmetric doubly stochastic, False otherwise.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    if not allclose(matrix, matrix.T):
        return False
    if np.any(matrix < 0):
        return False

    return allclose(np.sum(matrix, axis=0), 1.0)
