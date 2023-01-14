import sys
import os

sys.path.append("../classes")
from HomGraph import *
from SFGraph import *
from sighelp import *

# from AttractorGraph import *
from AttractorGraphWithPhase import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

import time

from matplotlib import rcParams

rcParams["font.size"] = 20
rcParams["figure.figsize"] = (20, 10)


graph_types = ["hom_graph", "sf_graph"]

n = 1000  # 50
gammas = np.linspace(1.72, 3.5, 30)  # np.arange(1,14)
r = 20
per1 = 4
per2 = 6
trials = 20
t_start = 50
t_max = 20000
attr_cutoff = 20
hub_index = 0
graph_type = graph_types[1]  # 0: hom_graph; 1: sf_graph
remove_overlap = True

params = {
    "n": n,
    "gammas": gammas,
    "r": r,
    "per1": per1,
    "per2": per2,
    "trials": trials,
    "t_start": t_start,
    "t_max": t_max,
    "attr_cutoff": attr_cutoff,
    "hub_index": hub_index,
    "graph_type": graph_type,
    "remove_overlap": remove_overlap,
}

per1_to_per2 = []
per2_to_per1 = []
per1_to_per2_var = []
per2_to_per1_var = []

for gam in gammas:
    print("gamma...", gam)
    t = t_start
    temp_ar_1 = []
    temp_ar_2 = []
    for trial in range(r):
        if graph_type == graph_types[0]:
            test_graph1 = HomGraph(n, gam)  # SFGraph(n, gam)
            test_graph2 = HomGraph(n, gam)  # SFGraph(n, gam)
        else:
            test_graph1 = SFGraph(n, gam)
            test_graph2 = SFGraph(n, gam)

        hub_node1 = test_graph1.find_hub(hub_number=hub_index)
        hub_node2 = test_graph2.find_hub(hub_number=hub_index)
        while True:
            try:
                print("Trial, ", trial)
                attr_graph1 = AttractorGraph(graph=test_graph1, oscil_node=hub_node1)
                attr_graph2 = AttractorGraph(graph=test_graph2, oscil_node=hub_node2)
                AttractorGraph.cross_compare_attrs_ctrl(
                    attr_graph1=attr_graph1,
                    attr_graph2=attr_graph2,
                    per1=per1,
                    per2=per2,
                    t=t,
                    trials=trials,
                    explore_null=False,
                    attr_cutoff=attr_cutoff,
                )
                temp_ar_1.append(
                    np.mean(
                        AttractorGraph.check_attr_parsimony_ctrl(
                            attr_graph1,
                            attr_graph2,
                            per1,
                            per2,
                            remove_overlap=remove_overlap,
                        )
                    )
                )
                temp_ar_2.append(
                    np.mean(
                        AttractorGraph.check_attr_parsimony_ctrl(
                            attr_graph2,
                            attr_graph1,
                            per2,
                            per1,
                            remove_overlap=remove_overlap,
                        )
                    )
                )
                break
            except Exception as e:
                print(e)
                t *= 2
                if t > t_max:
                    print(f"No attr found in {t} time steps")
                    print("Resetting t value and finding new example")
                    t = t_start
                    trial -= 1
                    break
                print("new t value:", t)

    per1_to_per2.append(np.mean(temp_ar_1))
    per2_to_per1.append(np.mean(temp_ar_2))
    per1_to_per2_var.append(np.var(temp_ar_1))
    per2_to_per1_var.append(np.var(temp_ar_2))

    print("Temp ars: ")
    print(temp_ar_1)
    print(temp_ar_2)
    print("------\nCurrent Results:")
    print("Mean")
    print(per1_to_per2, per2_to_per1)
    print("Var")
    print(per1_to_per2_var, per2_to_per1_var)


results = {
    "per1_to_per2": per1_to_per2,
    "per2_to_per1": per2_to_per1,
    "per1_to_per2_var": per1_to_per2_var,
    "per2_to_per1_var": per2_to_per1_var,
}

with open(
    f"mapping_{graph_type}_{per1}_{per2}_hub_{hub_index}_overlap_{remove_overlap}_CTRL.p",
    "wb",
) as fp:
    pickle.dump((params, results), fp)
