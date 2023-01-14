"""
Compute network potentials
"""
import sys
import os

sys.path.append("../classes")
from HomGraph import *
from SFGraph import *
from sighelp import *
from AttractorGraphWithPhase import *
import numpy as np
import pandas as pd
from scipy import sparse
import scipy as scp
from scipy.sparse import csgraph

from functools import reduce

import pickle


# starting_time = timer.time()
if len(sys.argv) >= 3:
    filestamp = sys.argv[1]
    gamma_val = float(sys.argv[2])
elif len(sys.argv) >= 2:
    filestamp = sys.argv[1]
    gamma_val = None
else:
    filestamp = None
    gamma_val = None

n = 1000
num_trials = 1000
periods = np.array([4, 8, 20, 40])
if gamma_val:
    gammas = np.array([gamma_val])
else:
    gammas = [2.0]  # np.array([1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 3.0])

num_networks = 2  # 10
t = 1000
out_file_location = f"output_csvs/results_csv_{filestamp}.csv"


csv_rows = []
failed_nets = {gamma: 0 for gamma in gammas}
for gamma in gammas:
    print(gamma)
    if gamma < 1.8:
        t = 8000
    elif gamma < 2.0:
        t = 2000
    else:
        t = 1000
    for network_num in range(num_networks):
        print(network_num)
        my_graph = SFGraph(n, gamma)
        oscil_node = my_graph.find_hub()
        test_graph = AttractorGraph(graph=my_graph, oscil_node=oscil_node)

        avg_depth_temp = {}
        avg_frac_overlap = {}
        # Check base..
        # wrap the whole thing in a try catch so if this network bombs, we just go to next
        try:
            for _ in range(num_trials):
                test_graph.graph.random_config()
                test_graph.explore_state_block(
                    block_nodes=oscil_node,
                    block_state=test_graph.graph.get_config()[oscil_node],
                    t=t,
                )
                test_graph.graph.random_config()
                test_graph.explore_state_free(t=t)

            test_graph.explore_periods(periods, num_trials, t=t, explore_null=False)

            num_block_ctrl_attractors = len(test_graph.per_map["0"].keys())
            num_free_ctrl_attractors = len(test_graph.per_map["-1"].keys())
            avg_block_ctrl_attractor_size = test_graph.get_avg_attractor_size("0")
            avg_free_ctrl_attractor_size = test_graph.get_avg_attractor_size("-1")
            # silly control
            avg_frac_overlap_block_block = test_graph.attr_state_fraction(
                period_1=str(0), period_2=str(0)
            )

            for per in periods:
                avg_depth_temp[per] = test_graph.avg_depth_period(per=str(per), t=t)
                avg_frac_overlap[per] = test_graph.attr_state_fraction(
                    period_1=str(per), period_2=str(0)
                )

        except Exception as e:
            failed_nets[gamma] += 1
            print(
                e,
                f"no attr found for gamma {gamma}, network number failed: ",
                failed_nets[gamma],
            )
            #### Repeat this trial
            network_num -= 1
            continue

        out_deg = my_graph.out_degree()[oscil_node]

        blocked_entry = {
            "n": n,
            "gamma": gamma,
            "period": "blocked_ctrl",
            "hub_out": out_deg,
            "avg_number": num_block_ctrl_attractors,
            "avg_depth": 0,
            "avg_size": avg_block_ctrl_attractor_size,
            "trials": num_trials,
            "avg_overlap_with_block": avg_frac_overlap_block_block,
        }

        free_entry = {
            "n": n,
            "gamma": gamma,
            "period": "free_ctrl",
            "hub_out": out_deg,
            "avg_number": num_free_ctrl_attractors,
            "avg_depth": 0,
            "avg_size": avg_free_ctrl_attractor_size,
            "trials": num_trials,
            "avg_overlap_with_block": 0,
        }
        csv_rows.append(blocked_entry)
        csv_rows.append(free_entry)

        for per in periods:
            csv_rows.append(
                {
                    "n": n,
                    "gamma": gamma,
                    "period": per,
                    "hub_out": out_deg,
                    "avg_number": len(test_graph.per_map[str(per)].keys()),
                    "avg_depth": avg_depth_temp[per],
                    "avg_size": test_graph.get_avg_attractor_size(per),
                    "trials": num_trials,
                    "avg_overlap_with_block": avg_frac_overlap[per],
                }
            )


df = pd.DataFrame(csv_rows)
print(df)


df.to_csv(out_file_location)
