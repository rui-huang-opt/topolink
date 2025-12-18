"""Custom exceptions for TopoLink."""


class GraphError(Exception):
    """Base exception for graph side errors."""

    pass


class ConnectivityError(GraphError):
    """Exception raised when the graph is not fully connected."""

    pass


class InvalidWeightedMatrixError(GraphError):
    """Exception raised when the graph is defined with a non-symmetric or non-double-stochastic matrix."""

    pass


class NodeError(Exception):
    """Base exception for node side errors."""

    pass


class NodeDiscoveryError(NodeError):
    """Exception raised when the graph service cannot be discovered by the node."""

    pass


class NodeUndefinedError(NodeError):
    """Exception raised when a node is not defined in the graph but tried to join the graph."""

    pass
