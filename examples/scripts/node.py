from logging import basicConfig, INFO

basicConfig(level=INFO)

import argparse


class Args(argparse.Namespace):
    name: str
    n_state: int
    n_iter: int
    alpha: float


parser = argparse.ArgumentParser(description="A simple consensus test using TopoLink.")
parser.add_argument("name", type=str, help="Name of the node.")
parser.add_argument("--n_state", type=int, default=3, help="Dimension of the state.")
parser.add_argument("--n_iter", type=int, default=100, help="Number of iterations.")
parser.add_argument("--alpha", type=float, default=0.45, help="Step size alpha.")

args = parser.parse_args(namespace=Args())

from numpy import zeros
from numpy.random import seed, uniform
from topolink import NodeHandle

nh = NodeHandle(args.name)

states = zeros((args.n_iter, args.n_state))
seed(int(args.name))  # Ensure reproducibility for each node
states[0] = uniform(-100.0, 100.0, args.n_state)

print(f"Node {args.name} initial state: {states[0]}")

for k in range(args.n_iter - 1):
    lap_state = nh.laplacian(states[k])
    states[k + 1] = states[k] - args.alpha * lap_state

    print(f"Node {args.name} at iteration {k + 1}: state: {states[k + 1]}")
