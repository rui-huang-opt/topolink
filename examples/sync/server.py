from logging import basicConfig, INFO
from topolink import Graph

basicConfig(level=INFO)

node_names = ["1", "2", "3", "4", "5"]
edge_pairs = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

graph = Graph(node_names, edge_pairs, address="localhost:5555")
graph.deploy()
