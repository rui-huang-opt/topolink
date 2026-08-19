import logging

import ray

import numpy as np
import numpy.random as npr
import numpy.typing as npt
import matplotlib.pyplot as plt

from conops import Graph, Network

logging.basicConfig(level=logging.INFO)

N_STATE = 3

NODES = ["1", "2", "3", "4", "5"]
N_NODES = len(NODES)

RING_EDGES = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]
STAR_EDGES = [("1", "2"), ("1", "3"), ("1", "4"), ("1", "5")]

ring = Graph(NODES, RING_EDGES)
star = Graph(NODES, STAR_EDGES)


@ray.remote
def consensus(
    idx: str,
    neighbors: dict[str, float],
    n_state: int,
    alpha1: float,
    alpha2: float,
    n_iter: int = 50,
) -> tuple[npt.NDArray, npt.NDArray]:
    network = Network(node_id=idx, neighbors=neighbors)

    x = np.zeros((n_iter, n_state))
    y = np.zeros((n_iter, n_state))
    npr.seed(int(idx))  # Ensure reproducibility for each node
    x[0] = npr.uniform(-100.0, 100.0, n_state)
    y[0] = npr.uniform(-100.0, 100.0, n_state)

    network.start()

    try:
        k = network.round_id

        while k < n_iter - 1:
            x[k + 1] = x[k] - network.laplacian("x", x[k]) * alpha1
            y[k + 1] = y[k] - network.laplacian("y", y[k]) * alpha2

            k = network.next_round()

    finally:
        network.stop()

    return x, y


def main() -> None:
    ray.init()

    try:
        tasks = [
            consensus.remote(str(i + 1), ring[str(i + 1)], N_STATE, 0.35, 0.1)
            for i in range(N_NODES)
        ]

        results = [ray.get(task) for task in tasks]
    finally:
        ray.shutdown()

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    _, ax1 = plt.subplots()

    for i in range(N_NODES):
        x = results[i][0]
        for j in range(N_STATE):
            (line,) = ax1.plot(x[:, j], color=colors[i])
            line.set_label(f"Node {i + 1}") if j == 0 else None

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("x Value")
    ax1.grid()
    ax1.legend()

    _, ax2 = plt.subplots()

    for i in range(N_NODES):
        y = results[i][1]
        for j in range(N_STATE):
            (line,) = ax2.plot(y[:, j], color=colors[i])
            line.set_label(f"Node {i + 1}") if j == 0 else None

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("y Value")
    ax2.legend()
    ax2.grid()

    plt.show()


if __name__ == "__main__":
    main()
