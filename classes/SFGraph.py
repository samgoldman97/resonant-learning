#!/usr/bin/env python
# Tuesday, November 15 2016


import numpy as np
import matplotlib.pyplot as plt
import random
from Graph import *
from scipy import sparse as scp


class SFGraph(Graph):
    """A class to generate scale free connected boolean networks

    Attributes:
            graph: The nx object to represent this
            config_prob: The probability that in a random configuration any node will be on
            layout: The constant layout used to draw the given network; a spring layout is used

    TODO:
            Fix the sf distribution
            Code a function to perturb the network
            Calculate hamming distance as a function of time

    """

    def __init__(self, size, lam, config_prob=0.5, config=None):
        """Constructor function

        Args:
                size: Number of nodes in the network
                lam: Parameter for the scale free distribution
                config_prob: The probability that in a random configuration, a state is on
                config: A given input configuration

        """
        super(SFGraph, self).__init__(size, config_prob, config)
        self._add_random_connections(size, lam)
        # note, i may need to set the layout after the connections are done
        self.npgraph = scp.csr_matrix(self.npgraph)

    def _add_random_connections(self, size, lam):
        """Add random connections into the network where the output(confirmed by cluzel) for each node follows a scale-free distribution

        Args:
            size: The number of nodes in the network.
            lam: The parameter for the powerlaw_sequence
        Update:
                        Empirically create a power law degree distribution
        """
        possible_connections = range(self.size)
        degree_seqs = self._power_law_degrees(
            n=size, param=lam, upper_bound=size - 1, lower_bound=1
        )

        for index, i in enumerate(np.nditer(self.nodes)):
            out_degree = degree_seqs[index]
            connections = np.random.choice(
                possible_connections, out_degree, replace=False
            )
            for j in connections:
                self.npgraph[j, i] = np.random.uniform(-1, 1)

    def _power_law_degrees(self, n, param, upper_bound, lower_bound=1):
        """
        Uses discrete probabilities!
        n: Number of sequences to return...
        upper_bound: INCLUSIVE
        lower_bound: INCLUSIVE
        param: Gamma
        """
        degree_options = np.arange(lower_bound, upper_bound + 1)
        prop_to = degree_options ** (-1 * param)
        probabilities = prop_to / np.sum(prop_to)
        return np.random.choice(a=degree_options, size=n, replace=True, p=probabilities)

    # def _next_powerlaw(self, param, upper_bound):
    # 	"""Generates a random powerlaw number for the connections lower than upper_bound

    # 		Now using: http://mathworld.wolfram.com/RandomNumber.html
    # 		In combination with Clauset et al. Power Law Distributions in Empiracle Data

    # 		NOTE: Could use: http://www.qucosa.de/fileadmin/data/qucosa/documents/9682/diss.pdf p14 in text, p26 in pdf

    #        Args:
    # 		param: the scale free parameter
    # 		upper_bound: The maximum that this power law number can be; anything higher is thrown out for this estimation
    #        """
    # 	mynum = -1
    # 	while mynum == -1:
    # 		gen = (1 - random.random()) ** (-1.0 / (param - 1))
    # 		if gen < upper_bound:
    # 			return int(gen)

    # Decrease our bounds by 0.5 as according to newman et al this is better

    # x0 = 0.5 # Set lower bound to 1 - 0.5 = 0.5
    # x1 = upper_bound - 0.5
    # lam = param * -1
    # x0pow = np.power(x0, lam+1)
    # first_term = (np.power(x1, lam+1) - x0pow) * np.random.random()
    # rnd_num = np.power(first_term + x0pow, (1.0 / (lam+1)))
    # rnd_num += 0.5
    # return int(np.floor(rnd_num))

    @classmethod
    def hamming_single_perturb(cls, size, lam, config_prob, perturb_frac, time):
        """Find the hamming distance between a network configured with size, lam, and config_prob
                and the same network that has a fraction perturb_frac of its nodes disturbed

        Args:
            cls: the class argument
            size: the size of the network
            lam: the parameter for the power distribution fo the network
            config_prob: The probability that any given node will be turned on or off in initial configuration
            perturb_frac: The fraction of nodes that will be perturbed
            time: the time to let the hamming list be calculated

        """
        graph1 = cls(size, lam, config_prob)
        return super(SFGraph, cls).hamming_single_perturb(graph1, perturb_frac, time)

    @classmethod
    def avg_hamming_ic(cls, size, lam, config_prob, perturb_frac, time, trials):
        """Find the average hamming distance for a network configured with size, lam, and config_prob
                that has a fraction of its nodes, perturb_frac disturbed at each trial


        Args:
            cls: the class argument
            size: the size of the network
            lam: the parameter for the power distribution fo the network
            config_prob: The probability that any given node will be turned on or off in initial configuration
            perturb_frac: The fraction of nodes that will be perturbed
            time: the time to let the hamming list be calculated
            trials: The number of different initial conditions to run this perturbation with

        """
        graph1 = cls(size, lam, config_prob)
        return super(SFGraph, cls).avg_hamming_ic(
            graph1, config_prob, perturb_frac, time, trials
        )

    @classmethod
    def avg_hamming_network(
        cls, size, lam, config_prob, perturb_frac, time, trials, networks
    ):
        """Take the average hamming distance for the scale free topologies with size, lam, config_prob,
                perturbing perturb_frac of the nodes, over some time.  This is done trials number of times for 'networks' realizations


        Args:
            cls: the class argument
            size: the size of the network
            lam: the parameter for the power distribution fo the network
            config_prob: The probability that any given node will be turned on or off in initial configuration
            perturb_frac: The fraction of nodes that will be perturbed
            time: the time to let the hamming list be calculated
            trials: Initial conditions trials for each network
            networks: number of realizations

        """
        # Keep summing the hamming distance arrays for each
        current = np.zeros(time)
        for i in range(networks):
            current += cls.avg_hamming_ic(
                size, lam, config_prob, perturb_frac, time, trials
            )
            # current = [x + y for x, y in zip(current, new)]
        # Divide each by the number of trials
        return current * 1.0 / networks
        # return list(map(lambda x: x * 1.0 / networks, current))
