from topolink import Topology

if __name__ == "__main__":
    node_names = ["1", "2", "3"]
    edge_pairs = [("1", "2"), ("2", "3"), ("3", "1")]
    topology = Topology(node_names, edge_pairs, address="localhost:5555")
    topology.deploy()
