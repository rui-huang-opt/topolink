"""Custom exceptions for TopoLink."""


class UndefinedNodeError(Exception):
    """Exception raised when a node is not defined in the graph."""

    pass


class ConnectivityError(Exception):
    """Exception raised when the graph is not fully connected."""

    pass


class InvalidWeightedMatrixError(Exception):
    """Exception raised when the graph is defined with a non-symmetric or non-double-stochastic matrix."""

    pass


class GraphDiscoveryError(Exception):
    """Exception raised when the graph service cannot be discovered."""

    pass


class GraphJoinError(Exception):
    """Exception raised when a node fails to join the graph."""

    pass
