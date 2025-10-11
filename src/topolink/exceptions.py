"""Custom exceptions for TopoLink."""


class GraphError(Exception):
    """Base exception for graph side errors."""

    pass


class GraphInitializationError(GraphError):
    """Exception raised during graph initialization."""

    pass


class ConnectivityError(GraphInitializationError):
    """Exception raised when the graph is not fully connected."""

    pass


class InvalidWeightedMatrixError(GraphInitializationError):
    """Exception raised when the graph is defined with a non-symmetric or non-double-stochastic matrix."""

    pass


class NodeError(Exception):
    """Base exception for node side errors."""

    pass


class NodeJoinError(NodeError):
    """Exception raised when a node fails to join the graph."""

    pass


class NodeDiscoveryError(NodeJoinError):
    """Exception raised when the graph service cannot be discovered by the node."""

    pass


class NodeUndefinedError(NodeJoinError):
    """Exception raised when a node is not defined in the graph but tried to join the graph."""

    pass


class NodeJoinTimeoutError(NodeJoinError):
    """Exception raised when a node fails to join the graph due to a timeout."""

    pass
