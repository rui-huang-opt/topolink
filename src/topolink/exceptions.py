class UnknownNodeError(Exception):
    """Exception raised when a node is not defined in the graph."""

    pass


class UnknownReplyError(Exception):
    """Exception raised when an unknown reply is received from the registry."""

    pass


class ConnectivityError(Exception):
    """Exception raised when the graph is not fully connected."""

    pass
