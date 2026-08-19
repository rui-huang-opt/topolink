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
    namespace: str,
    n_state: int,
    alpha: float,
    n_iter: int = 50,
) -> npt.NDArray[np.float64]:
    network = Network(node_id=idx, neighbors=neighbors, namespace=namespace)

    states = np.zeros((n_iter, n_state))
    npr.seed(int(idx))  # Ensure reproducibility for each node
    states[0] = npr.uniform(-100.0, 100.0, n_state)

    network.start()

    try:
        k = network.round_id

        while k < n_iter - 1:
            states[k + 1] = states[k] - network.laplacian(states[k]) * alpha

            k = network.next_round()

    finally:
        network.stop()

    return states


def main() -> None:
    ray.init()

    try:
        ring_tasks = [
            consensus.remote(str(i + 1), ring[str(i + 1)], "ring", N_STATE, 0.35)
            for i in range(N_NODES)
        ]
        star_tasks = [
            consensus.remote(str(i + 1), star[str(i + 1)], "star", N_STATE, 0.35)
            for i in range(N_NODES)
        ]

        ring_states = [ray.get(task) for task in ring_tasks]
        star_states = [ray.get(task) for task in star_tasks]
    finally:
        ray.shutdown()

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    _, ax1 = plt.subplots()

    for i in range(N_NODES):
        states = ring_states[i]
        for j in range(N_STATE):
            (line,) = ax1.plot(states[:, j], color=colors[i])
            line.set_label(f"Node {i + 1}") if j == 0 else None

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("State Value")
    ax1.grid()
    ax1.legend()
    ax1.set_title(f"Ring Graph Consensus (Nodes={N_NODES}, State Dim={N_STATE})")

    _, ax2 = plt.subplots()

    for i in range(N_NODES):
        states = star_states[i]
        for j in range(N_STATE):
            (line,) = ax2.plot(states[:, j], color=colors[i])
            line.set_label(f"Node {i + 1}") if j == 0 else None

    ax2.set_title(f"Star Graph Consensus (Nodes={N_NODES}, State Dim={N_STATE})")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("State Value")
    ax2.legend()
    ax2.grid()

    plt.show()


if __name__ == "__main__":
    main()
