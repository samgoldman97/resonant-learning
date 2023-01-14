"""
    phase_robustness.py 

    June 25, 2018 

    Script to probe the number of attractors that each node maps to. 
    Because we are imposing oscillations on the network for example with sequence [1,1,1,1,0,0,0,0,1,1,1,1], 
    we could conceivably have many different trajectories from the same network state.  
    We ask what fraction of the time do these network states converge on the same attractor.  

    Sample "trials" network states, and calculate the fraction of those that lead to multiple attractors 
    based upon the phase.  Repeat for "r" realizations and average.
    All of this is for a given period and tested over multiple gammas
"""


import sys
import os

sys.path.append("../classes")
from HomGraph import *
from SFGraph import *
from sighelp import *

from AttractorGraph import *
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
gammas = np.linspace(1.7, 3.5, 35)  # np.arange(1,14)
r = 40
per = 4
trials = 100
t_start = 50
t_max = 20000
hub_index = 0  # 50
graph_type = graph_types[1]  # 0: hom_graph; 1: sf_graph

start_time = time.time()


params = {
    "n": n,
    "control_params": gammas,
    "r": r,
    "per": per,
    "trials": trials,
    "t_start": t_start,
    "t_max": t_max,
    "hub_index": hub_index,
    "graph_type": graph_type,
}


scores_ar = []
scores_var = []

for gam in gammas:
    print("gamma...", gam)
    t = t_start
    temp_scores = []
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
                temp_scores.append(
                    attr_graph.calculate_trajectory_score(per, trials, t=t)
                )
                break
            except Exception as e:

                t *= 2
                if t > t_max:
                    print(f"No attr found in {t} time steps")
                    print("Resetting t value and finding new example")
                    t = t_start
                    trial -= 1
                    break
                print("new t value:", t)

    scores_ar.append(np.mean(temp_scores))
    scores_var.append(np.var(temp_scores))
    print("Temp ars: ")
    print(temp_scores)
    print("------\nCurrent Results:")
    print("Mean")
    print(scores_ar)
    print("Var")
    print(scores_var)

print("End time: ", time.time() - start_time, " seconds")

results = {"score_results": scores_ar, "score_variances": scores_var}

with open(f"phase_robustness_{graph_type}_{per}_hub_{hub_index}.p", "wb") as fp:
    pickle.dump((params, results), fp)
