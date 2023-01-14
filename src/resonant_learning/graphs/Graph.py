""" Graph.py

Base class to hold graphs 

"""

import numpy as np
import pandas as pd
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
            config_prob: The probability that in a random configuration,
                a state is on
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
            prob: The probability that any node will be a 1.
        """
        samples = np.array([self.OFF, self.ON])
        self.npstate = np.random.choice(samples, (self.size, 1))

    def set_config(self, config):
        """set_config.

        Configure the boolean network

        Args:
            config: The configuration to change the network to;
                numpy array with shape [size, 1]
        """
        new_config = np.copy(config).reshape(self.size, 1)
        self.npstate = new_config

    def get_config(self):
        """Return a list of the current state of each node"""
        return np.copy(self.npstate)

    def update(self):
        """update.
        Update states using matrix multiplication and
        numpy implementation
        """
        # This means that every node is determined by the INCOMING edges as we defined an incoming edge
        dot_prod = self.npgraph.dot(self.npstate)
        self.npstate[dot_prod > self.THRESH] = self.ON
        self.npstate[dot_prod < self.THRESH] = self.OFF

    def perturb_at_index(self, node_index, new_state):
        """perturb_at_index.

        Auxilary method to change the state of the network at a node_index

        Args:
            node_index: the index of the node we want to specifically change.
            new_state: the new state of the node we're changing
        """
        if new_state not in [self.ON, self.OFF]:
            raise ValueError("State must be self.ON or self.OFF")

        self.npstate[node_index, 0] = np.int8(new_state)

    def controlled_update(self, node_index, new_state):
        """controlled_update.

        Updates the given network using update method, but changes a node first

        Args:
            node_index: the index of the node we want to specifically change.
            new_state: the new state of the node we're changing
            thresh: the threshold to determine the new state of a node.
        """
        # Perturb at index, update, and make sure the orig state is set accordingly
        self.perturb_at_index(node_index, new_state)
        self.update()
        self.perturb_at_index(node_index, new_state)

    def update_return_table(self, time):
        """update_return_table.

        Evolves the network in a normal fashion,
        returns a np array with the evolution to time

        Args:
            time: The time we'd like to run this simulation

        Return:
            A 2d array containing a column for every node and a row for each time
        """
        my_table = np.zeros((time, self.size))
        for time in range(time):
            current_sequence = self.get_config()
            my_table[time] = current_sequence.T
            self.update()
        return my_table

    def update_return_table_blocked(self, time, node, state):
        """update_return_table_blocked.

        Get the update return table blocking a given node in one state.

        Args:
            time: time
            node: node
            state: state

        Return:
            update
        """
        new_state = self.ON if state == 1 else self.OFF

        # force the node intot he corrected block state to start!
        self.perturb_at_index(node, new_state)

        # could be empty for speed but marginal
        my_table = np.zeros((time, self.size))
        for time in range(time):
            current_sequence = self.get_config()
            my_table[time] = current_sequence.T
            # Block update
            self.controlled_update(node, new_state)
        return my_table

    def oscillate_update(
        self, node_index, period, time, pulse=False, force_start_on=True, oscil_shift=0
    ):
        """oscillate_update.

        Evolves the network, oscillating a given node at a certain period/frequency

        Args:
            node_index: Node index to oscillate
            period: Periodicity of oscillation
            time: Time to run simulation
            pulse: If true, use pulses instead of square waves
            force_start_on: Force node to start on in ON state if True
            oscil_shift: If true, randomly sample a phase for square wave of oscillation

        Return:
            Pandas data frame containing a column for every node and a row for each time
        """
        my_table = np.zeros((time, self.size))

        if isinstance(period, list) or isinstance(period, np.ndarray):
            oscillate_sequence = Graph.combine_frequencies(per_list=period, t=time)
        else:
            oscillate_sequence = Graph.freq_sequence(period, time, pulse)

        start_val = self.get_config()[node_index][0]
        if not force_start_on and start_val != self.ON:
            # Get node index state
            # invert sequence
            oscillate_sequence = [
                self.OFF if i == self.ON else self.ON for i in oscillate_sequence
            ]

        # Preform phase shift if needed
        oscillate_sequence = oscillate_sequence[oscil_shift:]
        new_cutoff = time - oscil_shift
        for time, new_state in enumerate(oscillate_sequence):
            self.perturb_at_index(node_index, new_state)
            current_sequence = self.get_config()
            my_table[time] = current_sequence.T
            self.update()

        return my_table[:new_cutoff]

    @classmethod
    def freq_sequence(cls, period, length, pulse=False):
        """freq_sequence.

        Returns a square wave array with a given frequency

        Args:
            period: Must be an integer value; period is 1 / frequency
            length: length of the list
        """
        if period <= 1:
            raise ValueError("Period must be > 1")

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
        """combine_frequencies."""
        new_series = np.ones(t)
        for per in per_list:
            new_series *= Graph.freq_sequence(length=t, pulse=False, period=per)
        return new_series

    @classmethod
    def get_lcm(cls, per_list):
        """get_lcm.

        Get LCM of a list of periods

        """

        def lcm(a, b):
            prod = a * b
            while b:
                a, b = b, a % b
            return prod / a

        return int(reduce(lcm, per_list))

    def out_degree(self):
        """out_degree."""
        dense_graph = self.npgraph.todense()
        return np.array((dense_graph != 0).sum(0)).flatten()

    def in_degree(self):
        """in_degree."""
        dense_graph = self.npgraph.todense()
        return np.array((dense_graph != 0).sum(1)).flatten()

    def find_hub(self, hub_number=0):
        """find_hub.
        Return node with largest hub degree
        """

        out_degree = self.out_degree()
        indices = np.flip(np.argsort(out_degree), axis=0)
        correct_index = indices[hub_number]
        return correct_index

    def find_cycle(self, upper_bound=None, block=False, hub=None):
        """find_cycle.

        Finds the attractor for the graph with its current configuration
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
            state_array[current_state] = ctr
        return None

    def perturb_network(self, p):
        """perturb_network.

        Randomly chooses a fraction p of the nodes to perturb
        and change the value

        Args:
        p : fraction of nodes to perturb
        """
        nodes = self.nodes
        perturb_number = int(self.size * p)
        perturb = np.random.choice(nodes, perturb_number, replace=False)
        state = self.get_config()
        for i in perturb:
            if state[i, 0] == self.OFF:
                self.npstate[i, 0] = self.ON
            elif state[i, 0] == self.ON:
                self.npstate[i, 0] = self.OFF

    def save_graph(self, file_name_prefix):
        """
        Outputs the graph using file_name_prefix
        """
        graph_file = file_name_prefix + "_graph"
        state_file = file_name_prefix + "_state"

        np.save(file=graph_file, arr=self.npgraph.todense())
        np.save(file=state_file, arr=self.npstate)

    def find_attractors(self, trials, upper_bound=None):
        """find_attractors."""
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

    @classmethod
    def load_graph(cls, graph_file, state_file):
        """load_graph."""
        npgraph = np.load(file=graph_file)
        states = np.load(file=state_file)
        n = states.size
        graph = cls(size=n, config=states)
        graph.npgraph = scp.csr_matrix(npgraph)
        return graph

    @classmethod
    def hamming_distance(cls, list1, list2):
        """hamming_distance.

        Auxilary function to find the hamming distance between two configuration arrays

        Args:
            list1: first np arraycontaining the states of the nodes
            list2: second np arraycontaining the states of the nodes

        Return:
            A positive hamming distance between the two nodes;
            -1 if the length of the lists is different
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
        """hamming_single_perturb.

        Parent class method to find the hamming distance between a graph and a perturbed configuration

        Args:
            graph1: the first graph
            perturb_frac: the number of nodes to perturb
            time: time to simulate

        Return:
            a list containing the hamming_distance for a given time
        """
        original_state = np.copy(graph1.get_config())
        tbl1 = graph1.update_return_table(time)

        graph1.set_config(np.copy(original_state))
        graph1.perturb_network(perturb_frac)
        tbl2 = graph1.update_return_table(time)
        return Graph.hamming_distance_table(tbl1, tbl2)

    @classmethod
    def avg_hamming_ic(cls, graph1, config_prob, perturb_frac, time, trials):
        """avg_hamming_ic.

        Find the average hamming distance for a given number of initial conditions (trials)

        Args:
            graph1: Nework
            perturb_frac: Nodes to perturb
            time: Time to run
            trials: Number of trials

        return:
            Length 'time' with the average hamming distance at each time for the network with a number of times
        """
        current = np.zeros(time)
        for i in range(trials):
            # re-shuffle the initial conditions for this given graph
            graph1.random_config(config_prob)
            current += Graph.hamming_single_perturb(graph1, perturb_frac, time)

        # Divide each by the number of trials
        return current * 1.0 / trials

    @classmethod
    def hamming_distance_table(cls, table1, table2):
        """hamming_distance_table."""
        hamming_dist = []
        for time_index, row in enumerate(table1):
            hamming_dist.append(Graph.hamming_distance(row, table2[time_index]))
        return hamming_dist

    @classmethod
    def single_ham_table(cls, table):
        """single_ham_table.

            Calculates the hamming distance internally between each time step in table1
        Return:
            Array of length of one less than the number of rows of the table
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
        """find_attractor.

        Find attractor from the given table
        Args:
            state_table: State table
        """

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
        """find_attractor_in_oscillations.

        Take in state table and oscil frequency

        Args:
            state_table: State table to check
            period: Period of oscillation

        Return:
            Start and end of attractor
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
