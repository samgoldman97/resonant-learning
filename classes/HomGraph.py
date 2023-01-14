#!/usr/bin/env python
# Thursday, November 10 2016

import numpy as np
import matplotlib.pyplot as plt
import random
from Graph import *
from scipy import sparse as scp


class HomGraph(Graph):
    """A class to generate homogenously connected boolean networks

    Attributes:
            graph: The nx object to represent this
            #### connection_prob: The probability that any two nodes will have a connection
            config_prob: The probability that in a random configuration any node will be on
            layout: The constant layout used to draw the given network; a spring layout is used

            TODO: Confirm that the average connectivity of the network is indeed k
    """

    def __init__(self, size, k, config_prob=0.5, config=None):
        """Constructor function

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
        """Add random connections into the network
        Note:
                        We now use a posison distribution!
                        #Previously, we used the Barabasi binomial method of probability p of a connection between two nodes
                        We have no self-loops for simplification
                        NOTE:: THIS HAS CHANGED TO BE THE INDEGREE
                        We say that an edge, i,j is an edge from J to I

                        Cutting off the distribution seems okay, as the poisson distribution is centered around the means

        Args:
            k: Average connectivity

        """

        for i in np.nditer(self.nodes):
            # Mean of a poisson is lambda, and we want mean k
            in_degree = np.random.poisson(lam=k)
            # Use this condition so we never try to add more than n connection
            while in_degree >= size:
                in_degree = np.random.poisson(lam=k)

            # in_degree = k; 	# change this if we want constant connections!
            possible_connections = range(self.size)
            # del possible_connections[i] # Let's allow self loops !
            connections = np.random.choice(
                possible_connections, in_degree, replace=False
            )
            for j in connections:
                weight = np.random.uniform(-1, 1)
                # add an incoming edge
                self.npgraph[i, j] = weight

    @classmethod
    def hamming_single_perturb(cls, size, k, config_prob, perturb_frac, time):
        """Find the hamming distance between a network configured with size, k, and config_prob
                and the same network that has a fraction perturb_frac disturbed

        Args:
            cls: the class argument
            size: the size of the network
            k: the average connectivity of the network
            config_prob: The probability that any given node will be turned on or off in initial configuration
            perturb_frac: The fraction of nodes that will be perturbed
            time: the time to let the hamming list be calculated

        """
        graph1 = cls(size, k, config_prob)
        return super(HomGraph, cls).hamming_single_perturb(graph1, perturb_frac, time)

    @classmethod
    def avg_hamming_ic(cls, size, k, config_prob, perturb_frac, time, trials):
        """Find the average hamming distance for a network configured with size, k, and config_prob
                that has a fraction of its nodes, perturb_frac disturbed at each trial


        Args:
            cls: the class argument
            size: the size of the network
            k: the average connectivity of the network
            config_prob: The probability that any given node will be turned on or off in initial configuration
            perturb_frac: The fraction of nodes that will be perturbed
            time: the time to let the hamming list be calculated
            trials: The number of different initial conditions to run this perturbation with

        """
        graph1 = cls(size, k, config_prob)
        return super(HomGraph, cls).avg_hamming_ic(
            graph1, config_prob, perturb_frac, time, trials
        )

    @classmethod
    def avg_hamming_network(
        cls, size, k, config_prob, perturb_frac, time, trials, networks
    ):
        """Take the average hamming distance for the Homogeneous  topologies with size, k, config_prob,
                perturbing perturb_frac of the nodes, over some time.  This is done trials number of times for 'networks' realizations


        Args:
            cls: the class argument
            size: the size of the network
            k: the average connectivity of the network
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
                size, k, config_prob, perturb_frac, time, trials
            )
        # Divid each by the number of trials
        return current * 1.0 / networks
