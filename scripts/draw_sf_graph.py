""" draw_sf_graph.py

Draw sf graph

"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from resonant_learning import graphs
import networkx as nx

if __name__ == "__main__":
    res_dir = Path("results/network_sample_outputs")
    res_dir.mkdir(exist_ok=True)
    n = 200
    gam = 3.0
    graph = graphs.SFGraph(n, gam)

    mat = graph.npgraph.todense()
    mat = np.transpose(mat)
    out_degree = graph.out_degree()
    print(f"Out degree of network: {out_degree}")
    nx_graph = nx.DiGraph(mat)
    nx.write_gexf(nx_graph, res_dir / "out_graph.gexf")

    # layout = nx.layout.spectral_layout(nx_graph)
    # nx.draw(nx_graph, pos=layout, node_size=out_degree*100,
    #        node_color="blue")
    # plt.show()
