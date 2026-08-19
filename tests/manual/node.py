import argparse
import time

import numpy as np
import numpy.random as npr

from conops.network import Network
from conops.graph import Graph

NODES = ["1", "2", "3", "4", "5"]
EDGES = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

graph = Graph(NODES, EDGES)


class Args(argparse.Namespace):
    node_id: str
    n_iter: int
    n_state: int
    interval: float = 0.1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("node_id", type=str, choices=NODES)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--n_state", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args(namespace=Args())

    network = Network(node_id=args.node_id, neighbors=graph[args.node_id])

    network.start()

    try:
        x = np.zeros((args.n_iter, args.n_state))
        y = np.zeros((args.n_iter, args.n_state))
        npr.seed(int(args.node_id))  # Ensure reproducibility for each node
        x[0] = npr.uniform(-100.0, 100.0, args.n_state)
        y[0] = npr.uniform(-100.0, 100.0, args.n_state)

        k = network.round_id

        while k < args.n_iter - 1:
            print(f"[{args.node_id}] - Round {k}: x = {x[k]}, y = {y[k]}", flush=True)

            x[k + 1] = x[k] - network.laplacian("x", x[k]) * 0.45
            y[k + 1] = y[k] - network.laplacian("y", y[k]) * 0.2

            k = network.next_round()

            time.sleep(args.interval)

        print(f"[{args.node_id}] - Round {k}: x = {x[k]}, y = {y[k]}", flush=True)

    finally:
        network.stop()


if __name__ == "__main__":
    main()
