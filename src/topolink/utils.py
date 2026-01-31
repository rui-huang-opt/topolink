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


import sys
import warnings
from typing import Literal


def normalize_transport(transport: Literal["tcp", "ipc"]) -> Literal["tcp", "ipc"]:
    """
    Normalize the transport type based on the operating system.

    Parameters
    ----------
    transport : Literal["tcp", "ipc"]
        The desired transport type.

    Returns
    -------
    Literal["tcp", "ipc"]
        The normalized transport type.

    Raises
    ------
    ValueError
        If the transport type is unsupported.
    """
    supported_transports = {"tcp", "ipc"}
    if transport not in supported_transports:
        err_msg = f"Unsupported transport type: {transport}. Supported types are: {supported_transports}"
        raise ValueError(err_msg)

    is_linux = sys.platform.startswith("linux")
    if is_linux:
        return transport
    else:
        warn_msg = (
            "IPC transport is only supported on Linux systems. "
            "Falling back to TCP transport."
        )
        warnings.warn(warn_msg)
        return "tcp"
