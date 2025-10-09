import sys
from logging import basicConfig, INFO

basicConfig(level=INFO)

if len(sys.argv) > 1:
    name = "".join(sys.argv[1:])
else:
    print("Usage: python node.py <node_name>")
    sys.exit(1)

from numpy import zeros
from numpy.random import seed, uniform
from topolink import NodeHandle

n_state = 3
n_iter = 100
nh = NodeHandle(name)

states = zeros((n_iter, n_state))
seed(int(name))  # Ensure reproducibility for each node
states[0] = uniform(-100.0, 100.0, n_state)

print(f"Node {name} initial state: {states[0]}")

for k in range(n_iter - 1):
    lap_state = nh.laplacian(states[k])
    states[k + 1] = states[k] - 0.45 * lap_state

    print(f"Node {name} at iteration {k + 1}: state: {states[k + 1]}")
