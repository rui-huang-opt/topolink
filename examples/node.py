import sys
from numpy.random import seed, uniform
from topolink import NodeHandle

if __name__ == "__main__":
    if len(sys.argv) > 1:
        node_name = "".join(sys.argv[1:])
    else:
        print("Usage: python node.py <node_name>")
        sys.exit(1)

    node_name = sys.argv[1]
    nh = NodeHandle(node_name, server_address="localhost:5555")

    state = uniform(-100.0, 100.0, size=3)

    input("Press Enter to start the node...")

    for _ in range(100):
        laplacian = nh.compute_laplacian(state)
        state -= laplacian * 0.45

        print(f"Node {node_name} updated state: {state}")
