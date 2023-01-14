""" SFGraph.py

Generate scale free connected network topologies

"""

import numpy as np
from scipy import sparse as scp

from resonant_learning.graphs import Graph


class SFGraph(Graph):
    """A class to generate scale free connected boolean networks

    Attributes:
        graph: The nx object to represent this
        config_prob: The probability that in a random configuration any node will be on
        layout: The constant layout used to draw the given network; a spring layout is used
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
        """_power-law_degrees.

        Sample power law degrees.

        Args:
            n: Number of sequences to return
            upper_bound: inclusive upper bound
            lower_bound: inclusive lower bound
            param: Gamma
        """
        degree_options = np.arange(lower_bound, upper_bound + 1)
        prop_to = degree_options ** (-1 * param)
        probabilities = prop_to / np.sum(prop_to)
        return np.random.choice(a=degree_options, size=n, replace=True, p=probabilities)

    @classmethod
    def hamming_single_perturb(cls, size, lam, config_prob, perturb_frac, time):
        """hamming_single_perturb."""
        graph1 = cls(size, lam, config_prob)
        return super(SFGraph, cls).hamming_single_perturb(graph1, perturb_frac, time)

    @classmethod
    def avg_hamming_ic(cls, size, lam, config_prob, perturb_frac, time, trials):
        """avg_hamming_ic."""
        graph1 = cls(size, lam, config_prob)
        return super(SFGraph, cls).avg_hamming_ic(
            graph1, config_prob, perturb_frac, time, trials
        )

    @classmethod
    def avg_hamming_network(
        cls, size, lam, config_prob, perturb_frac, time, trials, networks
    ):
        """avg_hamming_network."""

        current = np.zeros(time)
        for i in range(networks):
            current += cls.avg_hamming_ic(
                size, lam, config_prob, perturb_frac, time, trials
            )
        return current * 1.0 / networks
