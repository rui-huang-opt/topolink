import sys
import numpy as np
from numpy.random import seed, uniform
from topolink import NodeHandle


def sync_check(node_handle: NodeHandle) -> None:
    for k in range(10):
        node_handle.broadcast(np.array(k, dtype=np.float64))
        iter_arr = np.hstack(node_handle.gather())

        sync = np.all(iter_arr == k)

        print(
            f"Node {node_name} at iteration {k}: "
            f"gathered states: {iter_arr}, sync: {sync}"
        )


def gossip_check(node_handle: NodeHandle) -> None:
    state = uniform(-100.0, 100.0, size=3)

    for _ in range(100):
        laplacian = node_handle.compute_laplacian(state)
        state -= laplacian * 0.45

        print(f"Node {node_name} updated state: {state}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        node_name = "".join(sys.argv[1:])
    else:
        print("Usage: python node.py <node_name>")
        sys.exit(1)

    node_name = sys.argv[1]
    nh = NodeHandle(node_name, server_address="localhost:5555")

    input("Press Enter to start the node...")

    sync_check(nh)
    # gossip_check(nh)
