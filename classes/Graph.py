#!/usr/bin/env python
# Thursday, November 16 2016
# 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
import random
import seaborn as sns
import pandas as pd
import time
from scipy import sparse as scp
from functools import reduce


class Graph(object):
    """A parent class for our networks

    Attributes:
            graph: The nx object to represent this
            config_prob: The probability that in a random configuration any node will be on
            layout: The constant layout used to draw the given network; a spring layout is used

    """

    ON = np.int8(1)
    OFF = np.int8(0)
    THRESH = 0.0

    def __init__(self, size, config_prob=0.5, config=None):
        """Constructor function

        Args:
                size: Number of nodes in the network
                config_prob: The probability that in a random configuration, a state is on
                config: A given input configuration
        """
        self.npgraph = np.zeros((size, size))
        self.config_prob = config_prob
        self.size = size
        self.nodes = np.arange(size)
        # if no configuration is provided, randomly generate one
        if config is not None:
            self.set_config(config)
        else:
            self.random_config(self.config_prob)

    def random_config(self, prob=0.5):
        """Randomly configure the boolean network

        Args:
                p: The probability that any node will be a 1.
        """

        # self.npstate = np.random.randint(0, 2, (self.size, 1))
        samples = np.array([self.OFF, self.ON])
        self.npstate = np.random.choice(samples, (self.size, 1))

    def set_config(self, config):
        """Configure the boolean network

        Args:
                config: The configuration to change the network to; numpy array with shape [size, 1]
        """

        new_config = np.copy(config).reshape(self.size, 1)
        self.npstate = new_config

    def get_config(self):
        """Returns a list of the current state of each node"""
        return np.copy(self.npstate)

    def update(self):
        """Update states using matrix multiplication and numpy implementation"""
        # This means that every node is determined by the INCOMING edges as we defined an incoming edge
        dotProd = self.npgraph.dot(self.npstate)
        self.npstate[dotProd > self.THRESH] = self.ON
        self.npstate[dotProd < self.THRESH] = self.OFF
        # Test
        # self.npstate[dotProd == self.THRESH] = self.ON #np.random.choice([self.OFF, self.ON])

    def perturb_at_index(self, node_index, new_state):
        """Auxilary method to change the state of the network at a node_index

        Args:
                node_index: the index of the node we want to specifically change.
                new_state: the new state of the node we're changing
        """
        if new_state != self.ON and new_state != self.OFF:
            raise "State must be self.ON or self.OFF"

        self.npstate[node_index, 0] = np.int8(new_state)

    def controlled_update(self, node_index, new_state):
        """Updates the given network using update method, but changes a node first

        Note:
                We use a hardcoded threshold of 0

                This is no longer used for oscillate update because it duplicates perturb at index call

        Args:
                node_index: the index of the node we want to specifically change.
                new_state: the new state of the node we're changing
                thresh: the threshold to determine the new state of a node.
        """
        self.perturb_at_index(node_index, new_state)
        self.update()
        # QUICK FIX: Because update() is not coded to ignore the given state,
        # it will update the state we control
        # To fix, we just set that state here
        self.perturb_at_index(node_index, new_state)

    def update_return_table(self, time):
        """Evolves the network in a normal fashion, returns a np array with the evolution to time

        Note:
                We use a hardcoded threshold of 0

        Args:
                time: The time we'd like to run this simulation

        Return: A 2d array containing a column for every node and a row for each time
        """
        myTable = np.zeros((time, self.size))  # could be empty for speed but marginal
        for time in range(time):
            current_sequence = self.get_config()
            myTable[time] = current_sequence.T
            self.update()
        return myTable

    def update_return_table_blocked(self, time, node, state):
        """
        Get the update return table blocking a given node in one state
        """

        if state == 1:
            new_state = self.ON
        elif state == 0:
            new_state = self.OFF
        else:
            raise "Bad state argument! Must be ON (1) or OFF (0)"

        # Froce the node intot he corrected block state to start!
        self.perturb_at_index(node, new_state)
        myTable = np.zeros((time, self.size))  # could be empty for speed but marginal
        for time in range(time):
            current_sequence = self.get_config()
            myTable[time] = current_sequence.T
            # Block update!!!
            self.controlled_update(node, new_state)
        return myTable

    def frozen_nodes(self, start_config, t1, t2):
        """Returns a list of frozen nodes that are frozen upon updating in the interval of t1 to t2

        Note:
                This requires the normal procedure of updating

        Return: List of size 'nodes'.  If 1, the node is frozen at ON.  If -1, the node is frozen at OFF, If 0, the node is not frozen
        """
        removeList = []
        self.set_config(start_config)
        # set of all the frozen nodes; assume all frozen
        frozen_nodes = set(self.nodes)
        for i in range(t2):
            if i == t1:
                t1_config = np.ndarray.copy(self.get_config())
            self.update()
            if i >= t1:
                new_config = self.get_config()
                # Hacky approach
                for j in frozen_nodes:
                    if t1_config[j, 0] != new_config[j, 0]:
                        removeList.append(j)
                for q in removeList:
                    frozen_nodes.remove(q)
                removeList = []

        return_ar = []
        for i in self.nodes:
            if i in frozen_nodes:
                if new_config[i] == self.OFF:
                    return_ar.append(-1)
                else:
                    return_ar.append(1)
            else:
                return_ar.append(0)
        self.set_config(start_config)
        return return_ar

    def oscillate_update(
        self, node_index, period, time, pulse=False, force_start_on=True, oscil_shift=0
    ):
        """Evolves the network, oscillating a given node at a certain period/frequency

        Note:
                We use a hardcoded threshold of 0

        Args:
                node_index: the index of the node we want to specifically change.
                period: the period that we'd like to oscilate the given node at
                        Period can now be a list, and the freq sequences will be multiplied by eachother
                time: The time we'd like to run this simulation
                thresh: the threshold to determine the new state of a node.
                pulse: If we want the oscillations to be pulses of 1's rather than square waves
                force_start_on: If we want to force the network to start in the on state... default true.
                        If toggled, the network will start with 0 or 1 based on what the initial config is
                oscil_shift: If we want to probe "robustness to phase", we want to start the network in the
                        same state with a slightly different phase... e.g. starting from state X, we can run with sequence [1,1,1,0,0,0,1,1,1] on the hub node
                        Or we can use an oscil shift of 1 and end up starting from state X with oscillating node following [1,1,0,0,0,1,1,1,...]


        Return: A pandas data frame containing a column for every node and a row for each time
        """
        myTable = np.zeros((time, self.size))
        if type(period) == list or type(period) == np.ndarray:
            oscillate_sequence = Graph.combine_frequencies(per_list=period, t=time)
        else:
            oscillate_sequence = Graph.freq_sequence(period, time, pulse)

        if not force_start_on:
            # Get node index state
            start_val = self.get_config()[node_index][0]

            # Invert the list...
            if start_val != self.ON:
                # print("INVERTING THE SEQUENCE")
                oscillate_sequence = [
                    self.OFF if i == self.ON else self.ON for i in oscillate_sequence
                ]

        ### Preform phase shift!
        oscillate_sequence = oscillate_sequence[oscil_shift:]
        new_cutoff = time - oscil_shift
        for time, new_state in enumerate(oscillate_sequence):
            self.perturb_at_index(node_index, new_state)
            current_sequence = self.get_config()
            myTable[time] = current_sequence.T
            # self.controlled_update(node_index, new_state)
            self.update()

        return myTable[:new_cutoff]

    @classmethod
    def freq_sequence(cls, period, length, pulse=False):
        """Returns an array with a given frequency of changes in 1's and 0's

        In the past, this has been returning a pulsing.  Instead, it now returns a square wave
        If the pulse flag is set to true, it can still return a

        Args:
                period: Must be an integer value; period is 1 / frequency
                length: lenght of the list
        """
        if period <= 1:
            raise "Period must be > 1"

        period = int(period)
        if pulse:
            return [
                np.int8(Graph.ON) if i % period == 0 else np.int8(Graph.OFF)
                for i in range(length)
            ]
        else:
            period = int(period / 2)
            return [
                np.int8(Graph.ON) if (i // period) % 2 == 0 else np.int8(Graph.OFF)
                for i in range(length)
            ]

    @classmethod
    def combine_frequencies(cls, per_list, t):
        new_series = np.ones(t)
        for per in per_list:
            new_series *= Graph.freq_sequence(length=t, pulse=False, period=per)
        return new_series

    @classmethod
    def get_lcm(cls, per_list):
        """
        Helper function to get the new timescale of a list of periods
        """

        def lcm(a, b):
            prod = a * b
            while b:
                a, b = b, a % b
            return prod / a

        return int(reduce(lcm, per_list))

    def out_degree(self):
        """Return a numpy array containing the out degrees of the nodes in the graph"""
        dense_graph = self.npgraph.todense()
        # sum (0) sums along columns
        return np.array((dense_graph != 0).sum(0)).flatten()

    def in_degree(self):
        """Return a numpy array containing the in degrees of the nodes in the graph"""
        # sum (1) sums along rows
        dense_graph = self.npgraph.todense()
        return np.array((dense_graph != 0).sum(1)).flatten()

    def find_hub(self, hub_number=0):
        """Return the index of the node with the highest number of outgoing connections"""

        out_degree = self.out_degree()
        indices = np.flip(np.argsort(out_degree), axis=0)
        correct_index = indices[hub_number]
        return correct_index

    def find_cycle(self, upper_bound=None, block=False, hub=None):
        """Finds the attractor for the graph with its current configuration
        upperbound is the last time to check

        Return: A tuple containing the first time point at which an attractor is found and the time when that state is repeated
        """
        initial_state = np.copy(self.get_config())
        if block:
            block_state = initial_state[hub, 0]
        found = False
        state_array = {}
        ctr = 0
        state_array[self.get_config().tostring()] = ctr
        while not found:
            if upper_bound is not None and ctr > upper_bound:
                # print "No attractor was found before time %d" % (upper_bound)
                return None
            if not block:
                self.update()
            else:
                self.controlled_update(hub, block_state)
            ctr += 1
            current_state = self.get_config().tostring()
            val = state_array.get(current_state)
            if val != None:
                # Reset state
                self.set_config(initial_state)
                return val, ctr
                found = True
            state_array[current_state] = ctr

    def perturb_network(self, p):
        """Randomly chooses a fraction p of the nodes to perturb and change the value of

        Args:
        p : fraction of nodes to perturb
        """
        nodes = self.nodes
        perturb_number = int(self.size * p)
        perturb = np.random.choice(nodes, perturb_number, replace=False)
        state = self.get_config()
        for i in np.nditer(perturb):
            if state[i, 0] == self.OFF:
                self.npstate[i, 0] = self.ON
            elif state[i, 0] == self.ON:
                self.npstate[i, 0] = self.OFF

    def plot_in_degree_distribution(self, title="In degree distribution"):
        """Plots the in degree distribution of an network

        Args:
        title : title
        """
        in_degree = self.in_degree()
        sns.distplot(in_degree)  # Add more bins
        plt.title(title)
        plt.ylabel("Frequency")
        plt.xlabel("In_degree distribution")
        plt.xlim(
            0,
        )
        return plt.gca()

    def plot_out_degree_distribution(self, title="Out degree distribution"):
        """Plots the out degree distribution of an network

        Args:
        title : title
        """
        out_degree = self.out_degree()
        sns.distplot(out_degree)  # Add more bins
        plt.title(title)
        plt.ylabel("Frequency")
        plt.xlabel("Out_degree distribution")
        plt.xlim(
            0,
        )
        return plt.gca()

    def get_prob_vector(self, trials, time):
        """Gets an array with the probability P(ON) for each node
        Args:
        trials: The number of initial conditions to try
        time: The time to let each trial run

        """
        results = np.zeros((trials, self.nodes.size))
        ic = np.copy(self.get_config())
        is_on = np.vectorize(lambda x: x == self.ON)
        for i in range(trials):
            self.random_config()
            table = self.update_return_table(time=time)
            table_mod = is_on(table)
            sums = np.apply_along_axis(np.sum, 0, table_mod)
            results[i, :] = sums
        results = np.apply_along_axis(np.sum, 0, results) * 1.0 / (trials * time)
        self.set_config(ic)
        return results

    def get_weights_df(self, trials=50, time=1000):
        """Returns a dataframe with the sum of weights for each node,
        the probability of being on for each node,
        and  the dot product between the two

        Args:
        trials: The number of trials to calculate P(ON)
        time: The itme to calculate P(ON)
        """

        results = self.get_prob_vector(trials=trials, time=time)
        dense_graph = self.npgraph.todense()
        sum_of_weights = [
            self._get_sum_weights(dense_graph, i) for i in range(len(results))
        ]
        dot_prods = [
            self._get_dot_prod(dense_graph, i, results)[0, 0]
            for i in range(len(results))
        ]
        df = pd.DataFrame(
            {
                "P(ON)": results,
                "sum(weight)": sum_of_weights,
                "weights.dot(P(ON))": dot_prods,
            }
        )
        return df

    def _get_sum_weights(self, dense_matrix, row_index):
        """Gets the sum of the incoming edge weights
        args:
        dense_matrix: Dense matrix of weights
        row_index: Node we're interested in
        """

        flattened_row = dense_matrix[
            row_index,
        ].flatten()
        return flattened_row[flattened_row != 0].sum()

    def _get_dot_prod(self, dense_matrix, row_index, results):
        """Gets the dot product between the sum of incoming weights for a node and a probability vector
        args:
        dense_matrix: Dense matrix of weights
        row_index: Node we're interested in
        results: Probability vector (P(ON))
        """
        return dense_matrix[row_index, :].dot(results)

    def save_graph(self, file_name_prefix):
        """
        Outputs the graph using file_name_prefix
        """
        graph_file = file_name_prefix + "_graph"
        state_file = file_name_prefix + "_state"

        np.save(file=graph_file, arr=self.npgraph.todense())
        np.save(file=state_file, arr=self.npstate)

    def find_attractors(self, trials, upper_bound=None):
        """
        Find multiple attractors and transient lengths for a graph
        """
        initial_config = np.copy(self.get_config())
        transient_ar = []
        size_ar = []
        for i in range(trials):
            self.random_config()
            start, end = self.find_cycle()
            transient_ar.append(start)
            size_ar.append(end - start)
        self.set_config(initial_config)
        return (transient_ar, size_ar)

    def avg_activity_table(self, time, trials):
        """
        Get the average activity of each node as a function of time over some number of trials
        """
        start_config = np.copy(self.get_config())
        nodes = self.nodes
        return_table = self.update_return_table(time) * 1.0 / trials
        for i in range(trials - 1):
            self.random_config()
            update_table = self.update_return_table(time) * 1.0 / trials
            return_table += update_table
        self.set_config(start_config)
        return return_table

    def avg_activity_table_oscil(self, time, trials, node, period, pulse=False):
        """
        Get the average activity of each node as a function of time over some number of trials WITH oscillations
        """
        start_config = np.copy(self.get_config())
        nodes = self.nodes
        return_table = (
            self.oscillate_update(
                node_index=node, period=period, time=time, pulse=pulse
            )
            * 1.0
            / trials
        )
        for i in range(trials - 1):
            self.random_config()
            update_table = (
                self.oscillate_update(
                    node_index=node, period=period, time=time, pulse=pulse
                )
                * 1.0
                / trials
            )
            return_table += update_table
        self.set_config(start_config)
        return return_table

    def get_shortest_path(self):
        shortest_path = scp.csgraph.shortest_path(
            csgraph=self.npgraph, method="FW", directed=True, unweighted=True
        )
        return shortest_path

    @classmethod
    def load_graph(cls, graph_file, state_file):
        """
        Creates a new graph with these parameters
        ## SHOULD NOT BE CALLED FROM SFGRAPH OR HOMGRAPH
        ## This method does not rely on the connectivity parameter
        """
        npgraph = np.load(file=graph_file)
        states = np.load(file=state_file)
        n = states.size

        graph = cls(size=n, config=states)
        graph.npgraph = scp.csr_matrix(npgraph)
        return graph

    @classmethod
    def hamming_distance(cls, list1, list2):
        """Auxilary function to find the hamming distance between two configuration arrays

        Args:
        list1: first NUMPY ARRAY containing the states of the nodes
        list2: second NUMPY ARRAY containing the states of the nodes

        Return: A positive hamming distance between the two nodes; -1 if the length of the lists is different
        """
        if type(list1) is not np.ndarray:
            list1 = np.asarray(list1)
        if type(list2) is not np.ndarray:
            list2 = np.asarray(list2)

        if list1.size != list2.size:
            return -1
        else:
            return (list1 != list2).sum() * 1.0 / list1.size

    @classmethod
    def hamming_single_perturb(cls, graph1, perturb_frac, time):
        """Parent class method to find the hamming distance between a graph and a perturbed configuration

        Args:
        graph1: the first graph
        perturb_frac: the number of nodes to perturb
        time: the time to run the simulation and calculate the hamming distance

        Return: a list containing the hamming_distance for a given time
        """
        original_state = np.copy(graph1.get_config())
        tbl1 = graph1.update_return_table(time)

        graph1.set_config(np.copy(original_state))
        graph1.perturb_network(perturb_frac)
        tbl2 = graph1.update_return_table(time)
        return Graph.hamming_distance_table(tbl1, tbl2)
        # # graph2 = copy.deepcopy(graph1)
        # graph2.perturb_network(perturb_frac)
        # hamming_list = []
        # for i in range(time):
        # 	hamming_list.append(cls.hamming_distance(graph1.get_config(), graph2.get_config()))
        # 	graph1.update()
        # 	graph2.update()
        # return hamming_list

    @classmethod
    def avg_hamming_ic(cls, graph1, config_prob, perturb_frac, time, trials):
        """Parent class method to find the average hamming distance for a given number of initial conditions (trials)

        Args:
        graph1: the network realization with the topology we are investigating
        perturb_frac: the number of nodes to perturb
        time: the time to run the simulation and calculate the hamming distance
        trials: the number of different initial conditions to perturb this graph with


        return: A list of length 'time' with the average hamming distance at each time for the network with a number of times
        """
        current = np.zeros(time)
        # current = Graph.hamming_single_perturb(graph1, perturb_frac, time)
        for i in range(trials):
            # re-shuffle the initial conditions for this given graph
            graph1.random_config(config_prob)
            current += Graph.hamming_single_perturb(graph1, perturb_frac, time)

            # current = [x + y for x, y in zip(current, new)]
        # Divid each by the number of trials
        return current * 1.0 / trials

    ################# GRAPHING ######################

    @classmethod
    def plot_hamming(
        cls,
        ar,
        title="Hamming Distance",
        xtitle="Time (t)",
        ytitle="H(t)",
        xticks=None,
        color=None,
        label=None,
        savefig=None,
    ):
        """Plot a given list of hamming distances

        Args:
        ar: The list of hamming distances
        title: title of graph
        xtitle: X axis title
        ytitle: Y axis title
        color: optional parameter for the color of the plot if we want to plot multiple graphs

        """
        plt.ion()
        sns.set_style("darkgrid")
        if color is not None:
            plt.plot(ar, color=color, label=label)
        else:
            plt.plot(ar, label=label)
        plt.title(title)
        plt.ylabel(ytitle)
        plt.xlabel(xtitle)
        if xticks is not None:
            ticks = range(0, len(xticks))
            plt.xticks(ticks, xticks, rotation="vertical")
        legend = plt.legend(frameon=True, bbox_to_anchor=(1.1, 1.05))
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")

        if savefig:
            # save the figure then clear the graph
            pylab.savefig(savefig)
            Graph.clear_graph()
        else:
            plt.show()
            _ = raw_input("Press [enter] to continue.")

    @classmethod
    def implot(cls, update_table):
        plt.imshow(df2, cmap="gray")
        plt.xlabel("Node")
        plt.ylabel("Time")
        plt.title("Interpolation Diagram")
        plt.grid(False)

    @classmethod
    def clear_graph(cls):
        """Auxilary function to clear the graph"""
        plt.clf()

    @classmethod
    def animate_update_tbl(
        cls,
        data,
        fps=10,
        interval=200,
        new_shape=(25, 40),
        file_name="simple_animation.mp4",
    ):
        """
        animate_update_tbl:

                Helper function to visualize how these data frames are changing

                Args:
                        data:2x2 array containing the state table to be played in animation
                        new_shape: specifies the rectangular dimensions of the data for plotting each state of the network
                                defaults to 25x40 assuming we use n = 1000
                        file_name: Give file name for the output
                        fps: Frames per second (default 10)
                        interval: Milliseconds (default 200)

        """

        def update(index):
            # global mat
            # global data
            # global new_shape
            row = data[index].reshape(new_shape)
            mat.set_data(row)
            return mat

        def index_gen():
            # global data
            for j in range(data.shape[0]):
                yield j

        fig, ax = plt.subplots()
        plt.axis("off")
        mat = ax.matshow(data[0].reshape(new_shape))
        # plt.colorbar(mat)

        ### Animation args
        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=index_gen, interval=interval
        )

        ani.save(file_name, fps=fps, extra_args=["-vcodec", "libx264"])

    ################# Managing update tables ######################

    @classmethod
    def frozen_from_table(cls, table, t1, t2):
        """Returns a list of frozen nodes that are frozen upon updating in the interval of t1 to t2

        Note:
        This requires the normal procedure of updating

        Return: List of size 'nodes'.  If 1, the node is frozen at ON.  If -1, the node is frozen at OFF, If 0, the node is not frozen
        """
        removeList = []
        rows, columns = table.shape
        nodes = range(columns)
        # set of all the frozen nodes; assume all frozen
        frozen_nodes = set(nodes)
        for i in range(t1, t2):
            if i == t1:
                t1_config = table[i]
            if i >= t1:
                new_config = table[i]
                # Hacky approach
                for j in frozen_nodes:
                    if t1_config[j] != new_config[j]:
                        removeList.append(j)
                for q in removeList:
                    frozen_nodes.remove(q)
                removeList = []
        return_ar = []
        for i in nodes:
            if i in frozen_nodes:
                if new_config[i] == cls.OFF:
                    return_ar.append(-1)
                else:
                    return_ar.append(1)
            else:
                return_ar.append(0)

        return return_ar

    @classmethod
    def get_frozen_nodes(cls, frzn_ar):
        """Auxilary array to actually pick up only the frozen nodes from the frozen node; returns a list containing the
        indices of nodes that are actually frozen
        """
        return [node for node, direction in enumerate(frzn_ar) if direction != 0]

    @classmethod
    def hamming_distance_table(cls, table1, table2):
        """Calculates the hamming distance between table 1 and table 2
        Return should be an array of length equal to the row dimensions of the tables
        """
        hamming_dist = []
        # Switch this to an iter?
        for time_index, row in enumerate(table1):
            hamming_dist.append(Graph.hamming_distance(row, table2[time_index]))
        return hamming_dist

    @classmethod
    def single_ham_table(cls, table):
        """Calculates the hamming distance internally between each time step in table1
        Return should be an array of length of one less than the number of rows of the table
        """
        hamming_dist = []
        # Switch this to an iter?
        for time_index in range(1, len(table)):
            hamming_dist.append(
                Graph.hamming_distance(table[time_index], table[time_index - 1])
            )
        return hamming_dist

    @classmethod
    def find_attractor(cls, state_table):
        """Takes in state table returns tuple of start and end of attractor"""
        # for time_index, row in enumerate(state_table):
        # 	for i in range(time_index):
        # 		if np.array_equal(state_table[i], state_table[time_index]):
        # 			return (i, time_index)
        state_dict = {}
        for time_index, row in enumerate(state_table):
            cur_row = row.tostring()
            val = state_dict.get(cur_row)
            if val != None:
                return val, time_index
            state_dict[cur_row] = time_index
        return None

    @classmethod
    def find_attractor_in_oscillations(cls, state_table, period):
        """Takes in state table and oscillation frequency
        returns tuple of start and end of attractor
        These attractors are only scanned for at intervals of period!

        """
        if type(period) == list or type(period) == np.ndarray:
            period = Graph.get_lcm(period)
        state_dict = {}
        for time_index, row in enumerate(state_table):
            if time_index % period == 0:
                cur_row = row.tostring()
                val = state_dict.get(cur_row)
                if val != None:
                    return val, time_index
                state_dict[cur_row] = time_index
        return None
