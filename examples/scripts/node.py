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


def consensus(node_handle: NodeHandle, n_state: int = 3, n_iter: int = 100) -> None:
    from numpy import zeros
    from numpy.random import seed, uniform

    name = node_handle.name

    states = zeros((n_iter, n_state))
    seed(int(name))  # Ensure reproducibility for each node
    states[0] = uniform(-100.0, 100.0, n_state)

    print(f"Node {name} initial state: {states[0]}")

    for k in range(n_iter - 1):
        lap_state = nh.laplacian(states[k])
        states[k + 1] = states[k] - 0.45 * lap_state

        print(f"Node {name} at iteration {k + 1}: state: {states[k + 1]}")


basicConfig(level=INFO)

if len(sys.argv) > 1:
    node_name = "".join(sys.argv[1:])
else:
    print("Usage: python node.py <node_name>")
    sys.exit(1)

nh = NodeHandle.create(node_name)

print("Synchronization functionality check...")

sync_check(nh)

input("Press Enter to continue to consensus test...")

consensus(nh)
