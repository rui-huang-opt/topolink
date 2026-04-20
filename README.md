# ConOps (Consensus Operations)
This Python package **conops** facilitates distributed consensus operations over graph topologies.

## Installation
Install via pip:

```bash
pip install git+https://github.com/rui-huang-opt/conops.git
```

Or, for development:

```bash
git clone https://github.com/rui-huang-opt/conops.git
cd conops
pip install -e .
```

## Undirected Graphs
An **undirected graph** represents pairwise connections between objects without directional constraints. Formally defined as:

`G = (V, E)` where:  
- `V`: Set of vertices/nodes  
- `E`: Set of unordered edges `{u,v}` (connections have no direction)

### Key Properties:
1. **Symmetric Relationships**  
   If `(u,v)` exists, `(v,u)` is the same edge

2. **Neighbor Communication**  
   For any edge `(u, v)` in `E`, nodes `u` and `v` can directly communicate with each other.

### Undirected Graph Example: Ring Topology

#### **Graph Definition**
- `V = {1, 2, 3, 4, 5}`
- `E = {(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)}`

#### **Define and Visualize the Graph**

```python
nodes = ["1", "2", "3", "4", "5"]
edges = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

# Create the graph object
import networkx as nx

ring = nx.Graph()
ring.add_nodes_from(nodes)
ring.add_edges_from(edges)

# Visualize the topology
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
nx.draw(ring, ax=ax, with_labels=True)
plt.show()
```

#### **Visualization**

![Ring Topology Example](docs/images/ring_topology.png)

## Deploying an Undirected Graph Network

To deploy an undirected graph network using `conops`, follow these steps:

1. **Prepare local topology information**  
   Each node should know:
   - its own index `idx`
   - its neighbors as a dictionary `{idx: weight}`

2. **Initialize the node locally**  
   On each machine or process, create a `NodeHandle` with the node's local graph information.

3. **Start distributed communication**  
   Each node communicates directly with its neighbors.

> **Note:**  
> `conops` is fully distributed and does not use a central server.  
> Each node is initialized independently using its own `idx` and neighbor dictionary.

The `NodeHandle` class provides an interface for neighbor communication and common graph operators, such as the Laplacian operator, for distributed computation over the network topology.

### NodeHandle Example: Laplacian Consensus

The **consensus algorithm** is widely used in distributed systems to ensure that all nodes gradually reach agreement on their states through local communication. For an undirected graph, the state update of each node can be represented using the Laplacian matrix $L$:

Let $x_i(k)$ denote the state of node $i$ at iteration $k$, and $x(k) = [x_1(k), x_2(k), \dots, x_n(k)]^\top$ be the vector of all node states. The Laplacian matrix $L$ is defined as:

$$
L_{ij} = 
\begin{cases}
\text{deg}(i), & i = j \\
-1, & (i, j) \in E \\
0, & \text{otherwise}
\end{cases}
$$

The consensus iteration formula is:

$$
x(k+1) = x(k) - \alpha L x(k)
$$

where $\alpha > 0$ is the step size parameter. In each iteration, nodes only exchange information with their neighbors. Eventually, all $x_i$ converge to the same value (e.g., the initial average).

```python
# On each node machine/process
from conops import NodeHandle

node_idx = "1"  # Change this for each node (e.g., "2", "3", ...)
neighbors = {"2": 0.45, "5": 0.45} # Change this for each node

# Create the node handle
nh = NodeHandle(node_idx, neighbors)

# Achieve state convergence across all nodes through neighbor communication
import numpy as np
import numpy.random as npr

npr.seed(int(node_idx))
state = npr.uniform(-100.0, 100.0, 3)

alpha = 0.45

print(f"Node {node_idx} initial state: {state}")

for k in range(50):
   state = state - alpha * nh.laplacian(state)

print(f"Node {node_idx} final state: {state}")

nh.close()
```

```
Node 1 initial state: [-16.59559906  44.06489869 -99.97712504]
Node 1 final state: [  3.71351278  14.89452413 -19.19659572]
```

```
Node 2 initial state: [-12.80101957 -94.81475363   9.93249558]
Node 2 final state: [  3.71351278  14.89452413 -19.19659572]
```

```
Node 3 initial state: [ 10.15958051  41.62956452 -41.81905222]
Node 3 final state: [  3.71351278  14.89452413 -19.19659573]
```

```
Node 4 initial state: [93.4059678   9.44644984 94.53687199]
Node 4 final state: [  3.71351278  14.89452413 -19.19659572]
```

```
Node 5 initial state: [-55.60136578  74.14646124 -58.65616893]
Node 5 final state: [  3.71351278  14.89452413 -19.19659573]
```

#### **Results plot**
![Consensus](docs/images/consensus.png)