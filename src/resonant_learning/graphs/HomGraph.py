""" HomGraph.py

Generate homogeneously constructed NK networks

"""

import numpy as np
from scipy import sparse as scp

from resonant_learning.graphs import Graph


class HomGraph(Graph):
    """
    A class to generate homogenously connected boolean networks
    """

    def __init__(self, size, k, config_prob=0.5, config=None):
        """__init__.

        Args:
            size: Number of nodes in the network
            connection_prob: The probability of a connection between any two given nodes
            config_prob: The probability that in a random configuration, a state is on
            config: A given input configuration
        """
        if k >= size:
            raise ValueError(
                "The average connectivity cannot be greater than the number of nodes"
            )
        super(HomGraph, self).__init__(size, config_prob, config)
        self._add_random_connections(size, k)
        self.npgraph = scp.csr_matrix(self.npgraph)

    def _add_random_connections(self, size, k):
        """add_random_connections.

        This uses a poisson distribution with no self loops to guarantee that
        networks have homogeneous in degrees.

        Args:
            size: Network size
            k: Average connectivity

        """

        for i in np.nditer(self.nodes):

            # Mean of a poisson is lambda, and we want mean k
            in_degree = np.random.poisson(lam=k)
            # Use this condition so we never try to add more than n connection
            while in_degree >= size:
                in_degree = np.random.poisson(lam=k)

            possible_connections = range(self.size)
            connections = np.random.choice(
                possible_connections, in_degree, replace=False
            )
            for j in connections:
                weight = np.random.uniform(-1, 1)
                # add an incoming edge
                self.npgraph[i, j] = weight

    @classmethod
    def hamming_single_perturb(cls, size, k, config_prob, perturb_frac, time):
        """hamming_single_perturb."""
        graph1 = cls(size, k, config_prob)
        return super(HomGraph, cls).hamming_single_perturb(graph1, perturb_frac, time)

    @classmethod
    def avg_hamming_ic(cls, size, k, config_prob, perturb_frac, time, trials):
        """avg_hamming_ic."""
        graph1 = cls(size, k, config_prob)
        return super(HomGraph, cls).avg_hamming_ic(
            graph1, config_prob, perturb_frac, time, trials
        )

    @classmethod
    def avg_hamming_network(
        cls, size, k, config_prob, perturb_frac, time, trials, networks
    ):
        """avg_hamming_network."""

        # Keep summing the hamming distance arrays for each
        current = np.zeros(time)
        for i in range(networks):
            current += cls.avg_hamming_ic(
                size, k, config_prob, perturb_frac, time, trials
            )
        # Divid each by the number of trials
        return current * 1.0 / networks
