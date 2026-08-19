import numpy as np
import numpy.typing as npt
import numpy.random as npr
import matplotlib.pyplot as plt
import ray

from conops import Graph
from conops.network import Network

N_STATE = 3

NODES = ["1", "2", "3", "4", "5"]
EDGES = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

graph = Graph(NODES, EDGES)


@ray.remote(num_cpus=1)
def laplacian_consensus(
    idx: str,
    neighbors: dict[str, float],
    n_state: int,
    n_iter: int = 50,
) -> "npt.NDArray[np.float64]":
    network = Network(
        node_id=idx,
        neighbors=neighbors,
    )

    network.start()

    try:
        x = np.zeros((n_iter, n_state))

        npr.seed(int(idx))
        x[0] = npr.uniform(-100.0, 100.0, n_state)

        k = network.round_id

        while k < n_iter - 1:
            x[k + 1] = x[k] - network.laplacian("x", x[k]) * 0.45

            k = network.next_round()

    finally:
        network.stop()

    return x


def main() -> None:
    ray.init()

    try:
        tasks = [
            laplacian_consensus.remote(str(i + 1), graph[str(i + 1)], N_STATE)
            for i in range(graph.num_nodes)
        ]

        node_states = ray.get(tasks)

    finally:
        ray.shutdown()

    _, ax = plt.subplots()

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i in range(graph.num_nodes):
        states = node_states[i]

        for j in range(N_STATE):
            (line,) = ax.plot(states[:, j], color=colors[i])

            if j == 0:
                line.set_label(f"Node {i + 1}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("State Value")
    ax.set_title(f"Laplacian Consensus for {graph.num_nodes} Nodes")
    ax.legend()
    ax.grid()

    plt.show()


if __name__ == "__main__":
    main()
