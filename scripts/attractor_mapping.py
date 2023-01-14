import sys
import os

sys.path.append("../Classes")
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

n = 1000  # 50#
gammas = np.linspace(1.75, 3.5, 15)  # np.linspace(1.72, 3.5, 30) #np.arange(1,14)
r = 20
per1 = 4
per2 = 6
trials = 20
t_start = 50
t_max = 20000
attr_cutoff = 20
hub_index = 0  # 50
graph_type = graph_types[1]  # 0: hom_graph; 1: sf_graph
remove_overlap = False

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


# %%capture capt
start_time = time.time()
per1_to_per2_agreement = []
per2_to_per1_agreement = []
per1_to_per2_agreement_var = []
per2_to_per1_agreement_var = []
per1_to_per2_overlap = []
per2_to_per1_overlap = []
per1_to_per2_overlap_var = []
per2_to_per1_overlap_var = []

for gam in gammas:
    print("gamma...", gam)
    t = t_start
    temp_ar_1_agreement = []
    temp_ar_2_agreement = []
    temp_ar_1_overlap = []
    temp_ar_2_overlap = []
    for trial in range(r):
        if graph_type == graph_types[0]:
            test_graph = HomGraph(n, gam)
        else:
            test_graph = SFGraph(n, gam)

        hub_node = test_graph.find_hub(hub_number=hub_index)
        while True:
            try:
                print("Trial ", trial)
                attr_graph = AttractorGraph(graph=test_graph, oscil_node=hub_node)
                attr_graph.cross_compare_attrs(
                    per1=per1,
                    per2=per2,
                    t=t,
                    trials=trials,
                    explore_null=False,
                    attr_cutoff=attr_cutoff,
                )

                res1_agreement, res1_overlap = attr_graph.check_attr_parsimony(
                    per1, per2, remove_overlap=remove_overlap
                )
                res2_agreement, res2_overlap = attr_graph.check_attr_parsimony(
                    per2, per1, remove_overlap=remove_overlap
                )
                temp_ar_1_agreement.append(np.mean(res1_agreement))
                temp_ar_2_agreement.append(np.mean(res2_agreement))
                temp_ar_1_overlap.append(np.mean(res1_overlap))
                temp_ar_2_overlap.append(np.mean(res2_overlap))

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

    per1_to_per2_agreement.append(np.mean(temp_ar_1_agreement))
    per2_to_per1_agreement.append(np.mean(temp_ar_2_agreement))
    per1_to_per2_agreement_var.append(np.var(temp_ar_1_agreement))
    per2_to_per1_agreement_var.append(np.var(temp_ar_2_agreement))

    per1_to_per2_overlap.append(np.mean(temp_ar_1_overlap))
    per2_to_per1_overlap.append(np.mean(temp_ar_2_overlap))
    per1_to_per2_overlap_var.append(np.var(temp_ar_1_overlap))
    per2_to_per1_overlap_var.append(np.var(temp_ar_2_overlap))

    print("Temp ars: ")
    print(temp_ar_1_agreement)
    print(temp_ar_2_agreement)
    print("------\nCurrent Results (Agreement):")
    print("Mean")
    print(per1_to_per2_agreement, per2_to_per1_agreement)
    print("Var")
    print(per1_to_per2_agreement_var, per2_to_per1_agreement_var)

    print("Temp ars: ")
    print(temp_ar_1_overlap)
    print(temp_ar_2_overlap)
    print("------\nCurrent Results (Overlap):")
    print("Mean")
    print(per1_to_per2_overlap, per2_to_per1_overlap)
    print("Var")
    print(per1_to_per2_overlap_var, per2_to_per1_overlap_var)


print("End time: ", time.time() - start_time, " seconds")

results = {
    "per1_to_per2_agreement": per1_to_per2_agreement,
    "per2_to_per1_agreement": per2_to_per1_agreement,
    "per1_to_per2_agreement_var": per1_to_per2_agreement_var,
    "per2_to_per1_agreement_var": per2_to_per1_agreement_var,
    "per1_to_per2_overlap": per1_to_per2_overlap,
    "per2_to_per1_overlap": per2_to_per1_overlap,
    "per1_to_per2_overlap_var": per1_to_per2_overlap_var,
    "per2_to_per1_overlap_var": per2_to_per1_overlap_var,
}

with open(
    f"mapping_{graph_type}_{per1}_{per2}_hub_{hub_index}_overlap_{remove_overlap}.p",
    "wb",
) as fp:
    pickle.dump((params, results), fp)
