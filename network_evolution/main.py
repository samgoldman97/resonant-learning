#!/usr/bin/env python3

"""
        Main.py
        Call cython module to test network evolution
"""
import numpy as np

# Cython module
import scoring
import time as timer
from scipy.sparse import csr_matrix
from scipy.signal import periodogram
import sys
import os
import pickle
import argparse

# Call functions to generate matrices and for easy updates..
sys.path.append("../Classes")
from HomGraph import *
from SFGraph import *
from Graph import *

get_in_degree = lambda x: np.array((x != 0).sum(1)).flatten()
get_out_degree = lambda x: np.array((x != 0).sum(0)).flatten()


def compare_in_out_mat(original_mats, networks_matrix):
    """
    Fn to investigate distributions after learning
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    for index, mat in enumerate(original_mats):
        in_deg = get_in_degree(mat)
        sns.distplot(
            np.log(in_deg) + 1, label=f"{index}", kde=False, hist_kws={"log": True}
        )
    plt.xlabel("Log(In degree)")
    plt.ylabel("Freq")

    plt.savefig("/Users/Sam/Desktop/in_degree_pre.png")
    # plt.show()

    plt.figure()
    for index, mat in enumerate(original_mats):
        out_deg = get_out_degree(mat)
        sns.distplot(
            np.log(out_deg + 1), label=f"{index}", kde=False, hist_kws={"log": True}
        )

    plt.xlabel("Log(Out degree)")
    plt.ylabel("Freq")

    plt.savefig("/Users/Sam/Desktop/out_degree_pre.png")

    plt.figure()
    for index, mat in enumerate(networks_matrix):
        in_deg = get_in_degree(mat)
        sns.distplot(
            np.log(in_deg + 1), label=f"{index}", kde=False, hist_kws={"log": True}
        )
    plt.xlabel("Log(In degree)")
    plt.ylabel("Freq")

    plt.savefig("/Users/Sam/Desktop/in_degree_post.png")
    # plt.show()

    plt.figure()
    for index, mat in enumerate(networks_matrix):
        out_deg = get_out_degree(mat)
        sns.distplot(
            np.log(out_deg + 1), label=f"{index}", kde=False, hist_kws={"log": True}
        )

    plt.xlabel("Log(Out degree)")
    plt.ylabel("Freq")

    plt.savefig("/Users/Sam/Desktop/out_degree_post.png")

    plt.show()


def create_sf_in_matrix(n, gam):
    """
    Create a network that has an in degree with a scale free distribution
    """

    def _power_law_dist(n, param, lower_bound, upper_bound):
        degree_options = np.arange(lower_bound, upper_bound + 1)
        prop_to = degree_options ** (-1 * param)
        probabilities = prop_to / np.sum(prop_to)
        return np.random.choice(a=degree_options, size=n, replace=True, p=probabilities)

    possible_connections = range(n)
    degree_seqs = _power_law_dist(n=n, param=gam, upper_bound=n, lower_bound=1)
    adj_mat = np.zeros(shape=(n, n))

    for index in range(n):
        in_degree = degree_seqs[index]
        connections = np.random.choice(possible_connections, in_degree, replace=False)
        for j in connections:
            adj_mat[index, j] = np.random.uniform(-1, 1)
    return adj_mat


def create_sf_matrix(n, gam):
    """
    Use the configuration model to make a scale free network...
    """

    def _power_law_dist(n, param, lower_bound, upper_bound):
        degree_options = np.arange(lower_bound, upper_bound + 1)
        prop_to = degree_options ** (-1 * param)
        probabilities = prop_to / np.sum(prop_to)
        return np.random.choice(a=degree_options, size=n, replace=True, p=probabilities)

    in_degree = _power_law_dist(n, gam, 1, n)
    # IMPOSE UCM MODEL
    in_degree[in_degree > np.sqrt(np.mean(in_degree) * n)] = np.sqrt(
        np.mean(in_degree) * n
    )
    out_degree = np.copy(in_degree)

    np.random.shuffle(in_degree)
    np.random.shuffle(out_degree)
    # np.nonzero()

    # Sorted copy of the array...
    out_ordering = np.flip(np.argsort(out_degree), axis=0)
    adj_mat = np.zeros(shape=(n, n))
    for index in out_ordering:
        # print(index, out_degree[index])
        out_edges = np.random.choice(
            np.nonzero(in_degree)[0], size=out_degree[index], replace=False
        )
        for out_target in out_edges:
            # Decrement
            in_degree[out_target] -= 1
            adj_mat[out_target, index] = np.random.uniform(-1, 1)

    # plt.hist(np.array((adj_mat != 0).sum(1)).flatten(), bins=50, label='indegree') #
    # plt.hist(np.array((adj_mat != 0).sum(0)).flatten(), bins=50, label='outdegree') # outdegree
    # plt.legend()
    # plt.show()
    return adj_mat


def generate_network_types(
    n, control_param, network_type, population_number, networks_matrix, node_indices
):
    """
    Generate  a number of networks and their associated hub nodes
    Return both the np array of the networks in adj matrix and a second np array of hub nodes

    n: num nodes
    control_param: gamma or k
    population_number: number of networks:
    networks_matrix: 3D np array to store networks
    node_indices: 1D np array to store network hub nodes
    """

    for i in range(population_number):
        if network_type == 0:
            my_graph = HomGraph(n, control_param)
            # For the cython implementation...
            adj_mat_cpy = np.copy(my_graph.npgraph.todense())
        elif network_type == 1:
            my_graph = SFGraph(n, control_param)
            # For the cython implementation...
            adj_mat_cpy = np.copy(my_graph.npgraph.todense())
        elif network_type == 2:
            ## Dummy graph to keep pipeline in check
            adj_mat_cpy = create_sf_matrix(n, control_param)
        elif network_type == 3:
            ## Dummy graph to keep pipeline in check
            adj_mat_cpy = create_sf_in_matrix(n, control_param)

        networks_matrix[i] = adj_mat_cpy

        # Find hub node
        out_degree = get_out_degree(adj_mat_cpy)
        indices = np.flip(np.argsort(out_degree), axis=0)
        node_indices[i] = indices[0]


def test_output():
    """
    Helper function to do sanity checks on the updating rules
    """
    n = 1000
    k = 3
    time = 800
    my_graph = HomGraph(n, k)
    adj_mat_cpy = np.copy(my_graph.npgraph.todense())
    state_cpy = np.copy(my_graph.npstate).reshape(n).astype(np.int32)

    tbl = my_graph.update_return_table(time=time)
    start, end = Graph.find_attractor(tbl)
    print("Attractor in python: ", Graph.find_attractor(tbl))
    ret_tbl = np.zeros(shape=(time, n)).astype(np.int32)
    scoring.update_wrapper(adj_mat_cpy, state_cpy, ret_tbl)

    print(f"These are equal: {np.all(ret_tbl[0:end] == tbl[0:end])}")

    period = 4
    node_index = my_graph.find_hub()
    my_graph.random_config()
    state_cpy = np.copy(my_graph.npstate).reshape(n).astype(np.int32)
    tbl = my_graph.oscillate_update(
        node_index, period, time, force_start_on=False, oscil_shift=0
    )
    start, end = Graph.find_attractor_in_oscillations(tbl, period)
    print("Attractor in python: ", Graph.find_attractor_in_oscillations(tbl, period))
    ret_tbl = np.zeros(shape=(time, n)).astype(np.int32)
    scoring.update_oscillation_wrapper(
        adj_mat_cpy, state_cpy, ret_tbl, node_index, period
    )

    print(f"These are equal: {np.all(ret_tbl[0:end] == tbl[0:end])}")


if __name__ == "__main__":
    """
                In order to parallelize this easily across a super computer,
                if an argument is given to the script, this will be the unique identifier for the outfile
                This will replace the date for ease of use
                A shell script can call this passing in the date


                Now accepts args

                Usage: 
                    python3 main.py --timestamp [timestamp] --n [network size] --time [time to run]
                        --pop_size [pop size] --replicates [replicates] --network_type [type of net (1 to 4)]
                        --control_param [float gamma val or k val] --simulation_type [simulation type int]
                        --target_lengths [2 4 6 8 10] --target_node [-1] --periods [2 4 6 8 10]
                        --resonance_period [4] --prob_perturb [0] --constant_hub [0] --generations [5000]
                        --switch_time [-1] [--steretyped]

                    python3 main.py --timestamp 20170110 --n 500 --time 1000 \
                        --pop_size 50 --replicates 3 --network_type 1 \
                        --control_param 1.9 --simulation_type 4 \
                        --target_lengths 2 4 6 --target_node -1 --periods 2 4 6 \
                        --resonance_period -1 --prob_perturb 0 --constant_hub 0 --generations 10 \
                        --switch_time -1 


        """

    parser = argparse.ArgumentParser()

    parser.add_argument("--timestamp", help="set timestmap, file output prefix")
    parser.add_argument("--n", help="network size", type=int)
    parser.add_argument("--time", help="time to run each net for attr", type=int)
    parser.add_argument("--pop_size", help="population size", type=int)
    parser.add_argument("--replicates", help="number of network replicates", type=int)
    parser.add_argument(
        "--network_type", help="type of network (integer index)", type=int
    )
    parser.add_argument("--control_param", help="network parameter", type=float)
    parser.add_argument(
        "--simulation_type", help="simulation type (integer index)", type=int
    )
    parser.add_argument(
        "--target_lengths", nargs="+", help="simulation targets", type=int
    )
    parser.add_argument("--target_node", help="target_node index", type=int)
    parser.add_argument("--periods", nargs="+", help="simulation periods", type=int)
    parser.add_argument("--resonance_period", help="resonance period", type=int)
    parser.add_argument(
        "--prob_perturb", help="probability of perturbation for sim 8", type=float
    )
    parser.add_argument(
        "--constant_hub", help="if 0, set the hub to constant else free", type=int
    )
    parser.add_argument("--generations", help="generations for simulation", type=int)
    parser.add_argument(
        "--switch_time", help="Time to switch functions, -1 if no switch", type=int
    )
    parser.add_argument(
        "--stereotyped",
        help="Whether or not to use stereotyped functions",
        action="store_true",
    )

    args = parser.parse_args()

    # Get local time regardless
    starting_time = timer.time()
    t = timer.localtime()
    localtime = timer.strftime("%b-%d-%Y_%H%M", t)

    network_types = ["hom_graph", "sf_graph", "both_sf", "sf_in_degree"]
    simulation_types = [
        "evole_free",
        "evolve_random",
        "evolve_oscil",
        "evolve_multiplex_free",
        "evolve_multiplex_oscil",
        "block_then_simul",
        "random_noise_no_freq",
        "freq_node_select",
        "fixed_start_states",
    ]

    timestamp = args.timestamp
    n = args.n if args.n is not None else 500
    time = args.time if args.time is not None else 1000
    pop_size = args.pop_size if args.pop_size is not None else 50
    num_replicates = args.replicates if args.replicates is not None else 3
    network_type = args.network_type if args.network_type is not None else 1
    control_param = args.control_param if args.control_param is not None else 1.9
    simulation_type = args.simulation_type if args.simulation_type is not None else 4
    target_lengths = args.target_lengths if args.target_lengths is not None else [2]
    target_lengths = np.array(target_lengths).astype(np.int32)

    # -1 if we don't want it to be set...
    target_node = args.target_node if args.target_node is not None else -1

    periods = args.periods if args.periods is not None else [2]
    periods = np.array(periods).astype(np.int32)

    # For simulation type 7, we want to select for nodes that have this resonance period
    resonance_period = args.resonance_period if args.resonance_period is not None else 4

    # Simulation type 8: Probability of a perturbation for fixed start states
    prob_perturb = args.prob_perturb if args.prob_perturb is not None else 0

    # Don't use constant hub, 1 if we should use constant hub
    constant_hub = args.constant_hub if args.constant_hub is not None else 1

    generations = (
        args.generations * len(target_lengths)
        if args.generations is not None
        else 50 * len(target_lengths)
    )

    # for simulation type 4, we may want to switch targets at an arbitrary time... this determines that time
    # -1 is the default, no swithc behavior
    switch_time = (
        args.switch_time if args.switch_time is not None else -1
    )  # generations / 2 #-1

    # if flag is set, use stereotyped functions
    stereotyped = args.stereotyped

    # Don't sort first; score them THEN sort
    sort_first = 0
    MSE = True
    selection_delay = len(target_lengths)
    learning_blocks = 1  # 200

    # If we want to specify exactly how this thing will learn
    learning_rotation = None
    # learning_rotation = np.array([0, 1] * 200).astype(np.int32)

    # If 0, use random seed
    # If not 0, use the seed
    r_seed = 0

    experimental_trials = 1  # 0

    all_results = False
    record_mutations = False

    check_mini_cycles = True
    include_input = False

    # If not including input, set mini cycles flag on
    if not include_input:
        check_mini_cycles = True

    description = ""

    experimental_results = np.zeros((experimental_trials, generations))
    record_all = (
        np.zeros((experimental_trials, generations, len(target_lengths)))
        if all_results
        else None
    )
    start = timer.time()
    for trial in range(experimental_trials):
        # print(f"-----TRIAL {trial}----")
        networks_matrix = np.zeros((pop_size, n, n))
        node_indices = np.zeros(pop_size)

        generate_network_types(
            n, control_param, network_type, pop_size, networks_matrix, node_indices
        )

        # To save a run from a previous run
        # temp_file = "save_net.p"
        # with open(temp_file, "wb") as fp:
        #         pickle.dump((networks_matrix, node_indices), fp)
        # with open(temp_file, "rb") as fp:
        #         networks_matrix, node_indices = pickle.load(fp)

        original_mats = np.copy(networks_matrix)

        # Generate random state
        samples = np.array([0, 1])
        state_cpy = np.random.choice(samples, (n, 1)).reshape(n).astype(np.int32)
        scores = np.zeros(generations)
        temp_record_all = (
            np.zeros(shape=(generations, len(target_lengths))) if all_results else None
        )

        # Make sure int
        node_indices = node_indices.astype(np.int32)

        if simulation_type == 0:
            scoring.evolve_free(
                networks_matrix,
                scores,
                time,
                target_lengths[0],
                target_node,
                sort_first,
                num_replicates,
                record_mutations,
                r_seed,
                check_mini_cycles,
                include_input,
            )
        elif simulation_type == 1:
            scoring.evolve_random(
                networks_matrix,
                scores,
                time,
                node_indices,
                target_length,
                target_node,
                sort_first,
                num_replicates,
                record_mutations,
                r_seed,
                check_mini_cycles,
                include_input,
            )
        elif simulation_type == 2:
            scoring.evolve_oscillation(
                networks_matrix,
                scores,
                time,
                node_indices,
                periods[0],
                target_lengths[0],
                target_node,
                sort_first,
                stereotyped,
                num_replicates,
                record_mutations,
                r_seed,
                check_mini_cycles,
                include_input,
            )
        elif simulation_type == 3:
            scoring.multiplex_free(
                networks_matrix,
                scores,
                time,
                target_lengths,
                target_node,
                selection_delay,
                sort_first,
                MSE,
                learning_blocks,
                learning_rotation,
                temp_record_all,
                num_replicates,
                record_mutations,
                r_seed,
                check_mini_cycles,
                include_input,
            )
        elif simulation_type == 4:
            scoring.multiplex_oscil(
                networks_matrix,
                scores,
                time,
                node_indices,
                periods,
                target_lengths,
                target_node,
                selection_delay,
                sort_first,
                MSE,
                learning_blocks,
                learning_rotation,
                stereotyped,
                temp_record_all,
                num_replicates,
                record_mutations,
                r_seed,
                check_mini_cycles,
                include_input,
                switch_time,
            )
        elif simulation_type == 5:
            # First learn by block...
            # No selection delay
            scoring.multiplex_oscil(
                networks_matrix,
                scores,
                time,
                node_indices,
                periods,
                target_lengths,
                target_node,
                1,
                sort_first,
                MSE,
                learning_blocks,
                learning_rotation,
                stereotyped,
                temp_record_all,
                num_replicates,
                record_mutations,
                r_seed,
                check_mini_cycles,
                include_input,
                -1,
            )

            ## Remember to make sure out networks get stored back in networks matrix in C code
            new_scores = np.zeros(generations)

            # Now learn simul with no blocks...
            ## PROBLEM: Need to be able to pass back and forth target functions and target nodes..
            ## Hack fix: Just set a random seed in the C file.
            scoring.multiplex_oscil(
                networks_matrix,
                new_scores,
                time,
                node_indices,
                periods,
                target_lengths,
                target_node,
                selection_delay,
                sort_first,
                MSE,
                1,
                learning_rotation,
                stereotyped,
                temp_record_all,
                num_replicates,
                record_mutations,
                r_seed,
                check_mini_cycles,
                include_input,
                -1,
            )

            # Update the experimental results storage..
            scores = np.concatenate([scores, new_scores])
            if np.all(
                experimental_results == np.zeros((experimental_trials, generations))
            ):
                experimental_results = np.zeros((experimental_trials, generations * 2))
        elif simulation_type == 6:
            scoring.multiplex_no_freq(
                networks_matrix,
                scores,
                time,
                node_indices,
                target_lengths,
                target_node,
                selection_delay,
                sort_first,
                MSE,
                learning_blocks,
                learning_rotation,
                temp_record_all,
                num_replicates,
                record_mutations,
                r_seed,
                check_mini_cycles,
                include_input,
            )

        elif simulation_type == 7:
            """Choose by freq"""

            assert len(periods) == 1

            ## Duplicate the input matrix
            networks_matrix[slice(1, pop_size), :, :] = networks_matrix[0]
            node_indices[slice(1, pop_size)] = node_indices[0]

            # Construct a graph.
            my_graph = HomGraph(n, 1)
            my_graph.npgraph = csr_matrix(networks_matrix[0])
            my_graph.random_config()

            # only do this for the first peirod
            tbl = my_graph.oscillate_update(
                force_start_on=False,
                node_index=node_indices[0],
                oscil_shift=0,
                period=periods[0],
                time=time,
            )

            # Ensure that the hub node is not counted
            tbl[:, node_indices[0]] = 0
            start_, stop_ = Graph.find_attractor(tbl)

            # Take spectra of 3 cycles
            freqs, power_spectra = periodogram(
                x=tbl[
                    start_ : stop_ + (stop_ - start_) * 3 :,
                ].T
            )
            # power_spectra dim: nodes, freqs

            index = np.where(freqs == 1 / resonance_period)
            # print(index)
            if len(index) == 0:
                raise Exception("Could not find freq w/ desired target")

            # Now descending order of power for target_per = 4
            arg_sorted = np.flip(
                np.argsort(power_spectra[:, index], axis=0).squeeze(1), axis=0
            )

            target_node = arg_sorted[0]

            scoring.multiplex_oscil(
                networks_matrix,
                scores,
                time,
                node_indices,
                periods,
                target_lengths,
                target_node,
                selection_delay,
                sort_first,
                MSE,
                learning_blocks,
                learning_rotation,
                stereotyped,
                temp_record_all,
                num_replicates,
                record_mutations,
                r_seed,
                check_mini_cycles,
                include_input,
            )

        elif simulation_type == 8:
            scoring.multiplex_fixed_start(
                networks_matrix,
                scores,
                time,
                node_indices,
                target_lengths,
                target_node,
                selection_delay,
                sort_first,
                MSE,
                learning_blocks,
                learning_rotation,
                temp_record_all,
                num_replicates,
                record_mutations,
                r_seed,
                include_input,
                prob_perturb,
                constant_hub,
            )

        experimental_results[trial] = scores
        if all_results:
            record_all[trial] = temp_record_all

    end = timer.time()
    print(f"Seconds for {generations}: {end - start}")

    if simulation_type == 0:
        output = {
            "n": n,
            "control_param": control_param,
            "generations": generations,
            "pop_size": pop_size,
            "num_replicates": num_replicates,
            "network_type": network_types[network_type],
            "scores": experimental_results,
            "description": description,
            "target_length": target_lengths[0],
            "trials": experimental_trials,
            "simulation_type": simulation_types[simulation_type],
            "sort_first": False if sort_first == 0 else True,
            "DateTime": localtime,
            "r_seed": None if r_seed == 0 else r_seed,
            "check_mini_cycles": check_mini_cycles,
            "include_input": include_input,
            "target_node": target_node,
        }
        file_name = (
            f"free_evolution_run_{n}_{control_param}_{network_types[network_type]}.p"
        )

    elif simulation_type == 1:
        output = {
            "n": n,
            "control_param": control_param,
            "generations": generations,
            "pop_size": pop_size,
            "num_replicates": num_replicates,
            "network_type": network_types[network_type],
            "scores": experimental_results,
            "description": description,
            "target_length": target_lengths[0],
            "trials": experimental_trials,
            "simulation_type": simulation_types[simulation_type],
            "sort_first": False if sort_first == 0 else True,
            "DateTime": localtime,
            "r_seed": None if r_seed == 0 else r_seed,
            "check_mini_cycles": check_mini_cycles,
            "include_input": include_input,
            "target_node": target_node,
        }
        file_name = (
            f"random_evolution_run_{n}_{control_param}_{network_types[network_type]}.p"
        )

    elif simulation_type == 2:
        output = {
            "n": n,
            "control_param": control_param,
            "generations": generations,
            "pop_size": pop_size,
            "num_replicates": num_replicates,
            "network_type": network_types[network_type],
            "scores": experimental_results,
            "description": description,
            "period": periods[0],
            "target_length": target_lengths[0],
            "trials": experimental_trials,
            "simulation_type": simulation_types[simulation_type],
            "sort_first": False if sort_first == 0 else True,
            "DateTime": localtime,
            "Stereotyped": stereotyped,
            "check_mini_cycles": check_mini_cycles,
            "include_input": include_input,
            "target_node": target_node,
        }
        file_name = (
            f"oscil_evolution_run_{n}_{control_param}_{network_types[network_type]}.p"
        )
    elif simulation_type == 3:
        output = {
            "n": n,
            "control_param": control_param,
            "generations": generations,
            "pop_size": pop_size,
            "num_replicates": num_replicates,
            "network_type": network_types[network_type],
            "scores": experimental_results,
            "description": description,
            "target_lengths": target_lengths,
            "trials": experimental_trials,
            "simulation_type": simulation_types[simulation_type],
            "sort_first": False if sort_first == 0 else True,
            "selection_delay": selection_delay,
            "Mean Squared Error": MSE,
            "Learning Blocks": learning_blocks,
            "learning_rotation": learning_rotation,
            "DateTime": localtime,
            "all_results": record_all,
            "r_seed": None if r_seed == 0 else r_seed,
            "check_mini_cycles": check_mini_cycles,
            "include_input": include_input,
            "target_node": target_node,
        }
        file_name = f"free_multiplex_evolution_run_{n}_{control_param}_{network_types[network_type]}_{len(target_lengths)}_targets.p"
    elif simulation_type == 4:
        output = {
            "n": n,
            "control_param": control_param,
            "generations": generations,
            "pop_size": pop_size,
            "num_replicates": num_replicates,
            "network_type": network_types[network_type],
            "scores": experimental_results,
            "description": description,
            "periods": periods,
            "target_lengths": target_lengths,
            "trials": experimental_trials,
            "simulation_type": simulation_types[simulation_type],
            "sort_first": False if sort_first == 0 else True,
            "selection_delay": selection_delay,
            "Mean Squared Error": MSE,
            "Learning Blocks": learning_blocks,
            "learning_rotation": learning_rotation,
            "DateTime": localtime,
            "Stereotyped": stereotyped,
            "all_results": record_all,
            "r_seed": None if r_seed == 0 else r_seed,
            "check_mini_cycles": check_mini_cycles,
            "include_input": include_input,
            "target_node": target_node,
            "switch_time": switch_time,
        }

        file_name = f"oscil_multiplex_evolution_run_{n}_{control_param}_{network_types[network_type]}_{len(target_lengths)}_targets.p"

    elif simulation_type == 5:
        output = {
            "n": n,
            "control_param": control_param,
            "generations": generations * 2,
            "pop_size": pop_size,
            "num_replicates": num_replicates,
            "network_type": network_types[network_type],
            "scores": experimental_results,
            "description": description,
            "periods": periods,
            "target_lengths": target_lengths,
            "trials": experimental_trials,
            "simulation_type": simulation_types[simulation_type],
            "sort_first": False if sort_first == 0 else True,
            "selection_delay": selection_delay,
            "Mean Squared Error": MSE,
            "Learning Blocks": learning_blocks,
            "learning_rotation": learning_rotation,
            "DateTime": localtime,
            "Stereotyped": stereotyped,
            "all_results": record_all,
            "r_seed": None if r_seed == 0 else r_seed,
            "check_mini_cycles": check_mini_cycles,
            "include_input": include_input,
            "target_node": target_node,
        }

        file_name = f"oscil_multiplex_mixed_strategy_run_{n}_{control_param}_{network_types[network_type]}_{len(target_lengths)}_targets.p"
    elif simulation_type == 6:
        output = {
            "n": n,
            "control_param": control_param,
            "generations": generations,
            "pop_size": pop_size,
            "num_replicates": num_replicates,
            "network_type": network_types[network_type],
            "scores": experimental_results,
            "description": description,
            "target_lengths": target_lengths,
            "trials": experimental_trials,
            "simulation_type": simulation_types[simulation_type],
            "sort_first": False if sort_first == 0 else True,
            "selection_delay": selection_delay,
            "Mean Squared Error": MSE,
            "Learning Blocks": learning_blocks,
            "learning_rotation": learning_rotation,
            "DateTime": localtime,
            "all_results": record_all,
            "r_seed": None if r_seed == 0 else r_seed,
            "check_mini_cycles": check_mini_cycles,
            "include_input": include_input,
            "target_node": target_node,
        }

        file_name = f"no_frequency_input_multiplex_evolution_run_{n}_{control_param}_{network_types[network_type]}_{len(target_lengths)}_targets.p"

    elif simulation_type == 7:
        output = {
            "n": n,
            "control_param": control_param,
            "generations": generations,
            "pop_size": pop_size,
            "num_replicates": num_replicates,
            "network_type": network_types[network_type],
            "scores": experimental_results,
            "description": description,
            "periods": periods,
            "target_lengths": target_lengths,
            "trials": experimental_trials,
            "simulation_type": simulation_types[simulation_type],
            "sort_first": False if sort_first == 0 else True,
            "selection_delay": selection_delay,
            "Mean Squared Error": MSE,
            "Learning Blocks": learning_blocks,
            "learning_rotation": learning_rotation,
            "DateTime": localtime,
            "Stereotyped": stereotyped,
            "all_results": record_all,
            "r_seed": None if r_seed == 0 else r_seed,
            "check_mini_cycles": check_mini_cycles,
            "include_input": include_input,
            "target_node": target_node,
            "resonance_period": resonance_period,
        }

        file_name = f"oscil_multiplex_evolution_run_freq_sel_{n}_{control_param}_{network_types[network_type]}_{len(target_lengths)}_targets.p"
    elif simulation_type == 8:
        # Start with difefrent random states or slight perturbations of the ones selected
        output = {
            "n": n,
            "control_param": control_param,
            "generations": generations,
            "pop_size": pop_size,
            "num_replicates": num_replicates,
            "network_type": network_types[network_type],
            "scores": experimental_results,
            "description": description,
            "target_lengths": target_lengths,
            "trials": experimental_trials,
            "simulation_type": simulation_types[simulation_type],
            "sort_first": False if sort_first == 0 else True,
            "selection_delay": selection_delay,
            "Mean Squared Error": MSE,
            "Learning Blocks": learning_blocks,
            "learning_rotation": learning_rotation,
            "DateTime": localtime,
            "Stereotyped": stereotyped,
            "all_results": record_all,
            "r_seed": None if r_seed == 0 else r_seed,
            "include_input": include_input,
            "target_node": target_node,
            "prob_perturb": prob_perturb,
            "constant_hub": constant_hub,
        }
        if constant_hub == 1:
            const_name = "const"
        else:
            const_name = "free"

        file_name = f"fixed_start_{const_name}_multiplex_evolution_run_{n}_{control_param}_{network_types[network_type]}_{len(target_lengths)}_targets.p"

    ## If this was not created with odyssey
    if not timestamp:
        timestamp = localtime

    with open(timestamp + "_" + file_name, "wb") as out_file:
        pickle.dump(output, out_file)

    print("Time taken for script:", timer.time() - starting_time)
    # plt.plot(1 - scores[selection_delay -1::selection_delay])
    # plt.plot(1 - scores)
    # plt.show()
