import numpy as np
import ray

from conops import Graph, Network
from conops.transform import DPMechanism

N_STATE = 3

L = np.array(
    [
        [2, -1, 0, 0, -1],
        [-1, 2, -1, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 0, -1, 2, -1],
        [-1, 0, 0, -1, 2],
    ]
)
N_NODES = L.shape[0]
W = np.eye(N_NODES) - L * 0.45

graph = Graph.from_mixing_matrix(W)

import numpy.typing as npt
import numpy.random as npr


@ray.remote
def consensus(
    idx: str, neighbors: dict[str, float], n_state: int, n_iter: int = 50
) -> npt.NDArray[np.float64]:
    x = np.zeros((n_iter, n_state))
    npr.seed(int(idx))  # Ensure reproducibility for each node
    x[0] = npr.uniform(-100.0, 100.0, n_state)

    transform = DPMechanism(epsilon=1.0, sensitivity=1.0)
    network = Network(node_id=idx, neighbors=neighbors, transform=transform)

    network.start()

    try:
        k = network.round_id

        while k < n_iter - 1:
            x[k + 1] = x[k] - network.laplacian(x[k]) * 0.45

            k = network.next_round()

    finally:
        network.stop()

    return x


def main() -> None:
    ray.init()

    try:
        tasks = [
            consensus.remote(str(i + 1), graph[str(i + 1)], N_STATE)
            for i in range(N_NODES)
        ]

        node_states = [ray.get(task) for task in tasks]
    finally:
        ray.shutdown()

    import matplotlib.pyplot as plt

    _, ax = plt.subplots()

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i in range(N_NODES):
        states = node_states[i]
        for j in range(N_STATE):
            (line,) = ax.plot(states[:, j], color=colors[i])
            line.set_label(f"Node {i + 1}") if j == 0 else None

    ax.set_xlabel("Iteration")
    ax.set_ylabel("State Value")
    ax.set_title(f"Gossip Algorithm States for {N_NODES} Nodes")
    ax.legend()
    ax.grid()

    plt.show()


if __name__ == "__main__":
    main()
