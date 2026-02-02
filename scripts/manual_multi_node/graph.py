from logging import basicConfig, INFO
from topolink import Graph, bootstrap

basicConfig(level=INFO)

nodes = ["1", "2", "3", "4", "5"]
edges = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

graph = Graph(nodes, edges)
bootstrap(graph)
