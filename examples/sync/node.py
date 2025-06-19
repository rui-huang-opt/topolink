import sys
from logging import basicConfig, INFO
from topolink import NodeHandle


def sync_check(node_handle: NodeHandle, n_iter: int = 10) -> None:
    from numpy import float64, array, hstack

    for k in range(n_iter):
        node_handle.broadcast(array(k, dtype=float64))
        iter_arr = hstack(node_handle.gather())

        sync = all(iter_arr == k)

        print(
            f"Node {node_handle.name} at iteration {k}: "
            f"gathered iterations: {iter_arr}, sync: {sync}"
        )


basicConfig(level=INFO)

if len(sys.argv) > 1:
    node_name = "".join(sys.argv[1:])
else:
    print("Usage: python node.py <node_name>")
    sys.exit(1)

nh = NodeHandle(node_name)

print("Synchronization functionality check...")

sync_check(nh)
