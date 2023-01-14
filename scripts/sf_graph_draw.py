import sys
import os

sys.path.append("../classes/")

from SFGraph import *
from HomGraph import *
import networkx as nx
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Make graph
    n = 200
    gam = 3.0

    graph = SFGraph(n, gam)

    ## Todo: transpose this because nx takes args the other way
    mat = graph.npgraph.todense()
    mat = np.transpose(mat)
    outdegree = graph.out_degree()
    print(outdegree)
    nx_graph = nx.DiGraph(mat)
    nx.write_gexf(nx_graph, "/Users/Sam/Desktop/out_graph.gexf")
#    layout = nx.layout.spectral_layout(nx_graph)
#    nx.draw(nx_graph, pos=layout, node_size = outdegree*100, node_color="blue")

#    plt.show()
