"""
	AttractorGraph.py
	Author: Sam Goldman
	Date: March 5, 2018 

	This is a class to store information about the attractors of a network.
	These attractors can be very interconnected depending on oscillatory period
	This class helps keep track of it 

	Note, this class keeps track of what step in the period each phase is

"""

from HomGraph import *
from SFGraph import *
from sighelp import *
from matplotlib.pyplot import cm
from matplotlib.mlab import frange
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import sparse
import scipy as scp
from scipy.sparse import csgraph
from collections import deque

import networkx as nx
from functools import reduce


class AttractorGraph:

    # If a network is not oscillating, give it per 0
    NO_OSCIL = str(0)
    NO_BLOCK = str(-1)

    def __init__(self, graph, oscil_node=None):
        """
        Args:
                graph: The bool threshold network to be oscillated and probed
                oscil_node: Node to be oscillated or blocked
        Properties:
                self.graph: The bool threshold network to be oscillated and probed
                self.per_map: {period :
                                                        {Attr :
                                                                        {(node, phase) : (outgoing node, outgoing phase)}
                                        }

                self.attr_states: {period :
                                                        {Attr: [attr with phase attached to it...]}}

                ** Temporarily, we say that the control is with the hub node blocked **
        """
        self.graph = graph
        self.per_map = dict()
        self._n = graph.size
        self.attr_states = dict()
        # Set oscil node
        oscil_node = oscil_node if oscil_node is not None else self.graph.find_hub()
        self.set_oscil_nodes(oscil_node)

    ######## Change properties

    def set_oscil_nodes(self, oscil_node):
        self.oscil_node = oscil_node

    ####### Pretty print stats

    def pretty_print(self):
        """
        Pretty print statistics about this node set...
        """
        pers = list(self.per_map.keys())
        print("Periods explored: ", pers)
        for per in pers:
            attrs = self.per_map[per]
            print("Period: ", per)
            print("Number of attractors explored: ", len(attrs))
            attr_sizes = list(
                map(lambda x: len(x), [j for j in self.per_map[per].keys()])
            )
            print("Size of each attractor: ", attr_sizes)
            num_basin = list([len(j) for j in self.per_map[per].values()])
            print(
                "Number of network states (including phase) in each basin: ", num_basin
            )
            print("-----------------")

    ######## Exploration methods

    def explore_state_block(self, block_nodes, block_state, t=2000):
        """
        Explore the network with its given start condition and blocked nodes
        Allows all nodes to oscillate freely (This is a CONTROL)
        """
        edge_hash = dict()
        tbl = self.graph.update_return_table_blocked(
            time=t, node=block_nodes, state=block_state
        )
        tbl = self._pack_tbl(tbl)
        # Get the frozen attractor
        start_stop = Graph.find_attractor(state_table=tbl)
        if start_stop is None:
            raise Exception(f"No attractor could be found within time {t} steps")
        else:
            start, stop = start_stop

        frzn_attr = self._get_frozen_attractor(tbl[start:stop])

        if (AttractorGraph.NO_OSCIL not in self.attr_states) or (
            frzn_attr not in self.attr_states[AttractorGraph.NO_OSCIL]
        ):
            self._add_attr_phase(
                AttractorGraph.NO_OSCIL, frzn_attr, tbl[start:stop], 0, 1
            )
        # For each element,
        for index, row in enumerate(tbl[: stop + 1]):
            row_str = row.tostring()
            if self._was_explored(
                state=row_str, period=AttractorGraph.NO_OSCIL, phase=0
            ):
                break
            else:
                self._add_edge(
                    period=AttractorGraph.NO_OSCIL,
                    s1=row_str,
                    s2=tbl[index + 1].tostring(),
                    attr=frzn_attr,
                    phase1=0,
                    phase2=0,
                )

    def explore_state_free(self, t=2000):
        """
        Explore the network with its given start condition and NO blocked node
        Allows ALLL nodes to move freely (This is a CONTROL)
        """
        edge_hash = dict()
        tbl = self.graph.update_return_table(time=t)
        tbl = self._pack_tbl(tbl)
        # Get the frozen attractor
        start_stop = Graph.find_attractor(state_table=tbl)
        if start_stop is None:
            raise Exception(f"No attractor could be found within time {t} steps")
        else:
            start, stop = start_stop

        frzn_attr = self._get_frozen_attractor(tbl[start:stop])

        if (AttractorGraph.NO_BLOCK not in self.attr_states) or (
            frzn_attr not in self.attr_states[AttractorGraph.NO_BLOCK]
        ):
            self._add_attr_phase(
                AttractorGraph.NO_BLOCK, frzn_attr, tbl[start:stop], 0, 1
            )
        # For each element,
        for index, row in enumerate(tbl[: stop + 1]):
            row_str = row.tostring()
            if self._was_explored(
                state=row_str, period=AttractorGraph.NO_BLOCK, phase=0
            ):
                break
            else:
                self._add_edge(
                    period=AttractorGraph.NO_BLOCK,
                    s1=row_str,
                    s2=tbl[index + 1].tostring(),
                    attr=frzn_attr,
                    phase1=0,
                    phase2=0,
                )

    def explore_oscil(self, per, t=2000, explore_null=True, oscil_shift=0):
        """
        Explore the network with the given oscillations
        NOTE: We do not force the network to change initial conditions when we oscillate...
        We do this to be immune to the phase.
        Return the new attractor
        explore_null is whether or not to explore the null hypothesis
        oscil_shift: If we want to probe "robustness to phase", we want to start the network in the
                same state with a slightly different phase... e.g. starting from state X, we can run with sequence [1,1,1,0,0,0,1,1,1] on the hub node
                Or we can use an oscil shift of 1 and end up starting from state X with oscillating node following [1,1,0,0,0,1,1,1,...]

        """
        # Used to check was_start for phase shift

        effective_period = (
            Graph.get_lcm(per) if type(per) == list or type(per) == np.ndarray else per
        )
        # start_config =

        # Todo: Speed this up....

        tbl = self.graph.oscillate_update(
            node_index=self.oscil_node,
            period=per,
            time=t,
            force_start_on=False,
            oscil_shift=oscil_shift,
        )
        tbl = self._pack_tbl(tbl)

        start_stop = Graph.find_attractor_in_oscillations(tbl, period=per)
        if start_stop is None:
            raise Exception(f"No attractor could be found within time {t} steps")
        else:
            start, stop = start_stop
        oscil_attr = self._get_frozen_attractor(tbl[start:stop])

        if (per not in self.attr_states) or (oscil_attr not in self.attr_states[per]):
            self._add_attr_phase(
                per, oscil_attr, tbl[start:stop], start, effective_period // 2
            )

        next_phase = None
        for index, row in enumerate(tbl[: stop + 1]):
            row_str = row.tostring()

            if next_phase:
                phase = next_phase
            else:
                phase = index % (effective_period // 2)

            next_phase = (index + 1) % (effective_period // 2)

            # If this state hasn't been explored ever, explore
            if explore_null and not self._was_explored(
                state=row_str, period=AttractorGraph.NO_OSCIL, phase=0
            ):
                # unpack state and extract the first n bits
                cur_state = self._unpack_state(row)
                self.graph.set_config(np.copy(cur_state))
                self.explore_state_block(
                    block_nodes=self.oscil_node,
                    block_state=cur_state[self.oscil_node],
                    t=t,
                )
            # If we've already seen this node with this phase!
            if self._was_explored_in_attr(
                state=row_str, period=per, attr=oscil_attr, phase=phase
            ):
                # print("This node has already been seen! Time saved: ", stop - index)
                break

            self._add_edge(
                period=per,
                s1=row_str,
                s2=tbl[index + 1].tostring(),
                attr=oscil_attr,
                phase1=phase,
                phase2=next_phase,
            )

        return oscil_attr

    def explore_periods(self, periods, trials, t=2000, explore_null=True):
        """
        Explore multiple periods of oscillation together with the same intiail conditions
        """

        for _ in range(trials):
            self.graph.random_config()
            start_config = np.copy(self.graph.get_config())
            for per in periods:
                self.graph.set_config(np.copy(start_config))
                self.explore_oscil(per=per, t=t, explore_null=explore_null)

    ###### Compare to original attractors..

    def depth_to_old_attr(self, state, t=2000):
        """
        Calculate the distance from the attractor in the original block state...
        """
        # First check if this state has been explored in the null condition
        if (state, 0) not in self._get_all_nodes(period=AttractorGraph.NO_OSCIL):
            cur_state = self._unpack_state(state)
            self.graph.set_config(np.copy(cur_state))
            self.explore_state_block(
                block_nodes=self.oscil_node, block_state=cur_state[self.oscil_node], t=t
            )

        for attr in self.per_map[AttractorGraph.NO_OSCIL]:
            edges = self.per_map[AttractorGraph.NO_OSCIL][attr]
            if (state, 0) in edges.keys():
                # If this is the right attr... preform depth first search
                cur_node = state
                depth = 0
                while cur_node not in attr:
                    next_node, next_phase = edges[(cur_node, 0)]
                    cur_node = next_node
                    depth += 1
                return depth

    def avg_depth_attr(self, attr, t=2000):
        """
        Get the avg depth from each node in an attractor
        """
        return np.mean([self.depth_to_old_attr(state, t) for state in attr])

    def avg_depth_period(self, per, t=2000):
        """
        Calculate the avg depth of each attractor...
        """

        return np.mean(
            [self.avg_depth_attr(attr, t) for attr in self.per_map[str(per)]]
        )

    ###### Phase exploration

    def calculate_trajectory_score(self, per, trials, t=2000, explore_null=False):
        """
        Return the fraction of times that this network reaches the same attractor when taken with a phase shift
        """
        score = []
        for j in range(trials):
            score.append(
                self.check_attr_trajectory(per=per, t=t, explore_null=explore_null)
            )
        return np.sum(score) / trials

    def check_attr_trajectory(self, per, t=2000, explore_null=False):
        """
        Take a given start state.  Probe it with all possible phase shifts
        Determine if these go to the same place when both started..

        Return: True if same trajectory, false if different
        """
        # Set a random config
        self.graph.random_config()
        start_config = np.copy(self.graph.get_config())

        # Note: Can we delete this line?
        start_node = self._pack_tbl(np.copy(self.graph.get_config()).reshape(1, -1))
        # get baseline
        start_attr = self.explore_oscil(
            per=per, t=t, explore_null=explore_null, oscil_shift=0
        )
        for phase_shift in range(1, int(per / 2)):
            self.graph.set_config(np.copy(start_config))
            new_attr = self.explore_oscil(
                per=per, t=t, explore_null=explore_null, oscil_shift=phase_shift
            )
            if new_attr != start_attr:
                return False
        return True

        # self.graph.random_config()
        # start_node = self._pack_tbl(np.copy(self.graph.get_config()).reshape(1, -1))
        # start_attr = self.explore_oscil(per = per, t = t, explore_null = explore_null)
        # for second_node in self.per_map[str(per)][start_attr][1][start_node.tostring()]:
        # 	second_attr = self.explore_oscil(per = per, t = t, explore_null = explore_null)
        # 	if second_attr != start_attr:
        # 		return False
        # return True

    def phase_explore(self, per, sample_num=None, t=2000):
        """
        Explore all the nodes that have already been explored for this period
        Re-explore with no phase shift at each node..
        Args:
                period: Period to re-explore
                t: Time of exploration
                sample_num: Number of nodes to explore
        """
        # get all the nodes with data
        nodes = self._get_all_nodes(period=per)
        phase_shifted_nodes = [k for k, v in nodes if v != 0]

        if sample_num is not None and sample_num < len(phase_shifted_nodes):
            phase_shifted_nodes = np.random.choice(
                a=phase_shifted_nodes, size=sample_num, replace=False
            )

        for node in phase_shifted_nodes:
            new_state = self._unpack_state(np.fromstring(node, dtype=np.ubyte))
            self.graph.set_config(np.copy(new_state))
            self.explore_oscil(per=per, t=t)

    def phase_explore_attr(self, per, attr, sample_num=None, t=2000):
        """
        Explore all the nodes that have already been explored for this period
        Re-explore with no phase shift at each node..
        Explore only within the given attractor!!
        Args:
                period: Period to re-explore
                t: Time of exploration
                attr: Attractor
                sample_num: Number of ndoes to explore
        """
        # get all the nodes with data
        nodes = self._get_all_nodes_attr(period=per, attr=attr)

        phase_shifted_nodes = [state for (state, phase) in nodes if phase != 0]

        if sample_num is not None and sample_num < len(phase_shifted_nodes):
            phase_shifted_nodes = np.random.choice(
                a=phase_shifted_nodes, size=sample_num, replace=False
            )

        for node in phase_shifted_nodes:
            new_state = self._unpack_state(np.fromstring(node, dtype=np.ubyte))
            self.graph.set_config(np.copy(new_state))
            self.explore_oscil(per=per, t=t)

    def phase_explore_attr_cycle(self, per, attr, sample_num=None, t=2000):
        """
        Explore all the ndoes in the given attr cycle with the given period
        """
        # get all the nodes with data
        nodes = self._get_all_nodes_attr(period=per, attr=attr)

        phase_shifted_nodes = [
            state for (state, phase) in nodes if phase != 0 and state in attr
        ]

        if sample_num is not None and sample_num < len(phase_shifted_nodes):
            phase_shifted_nodes = np.random.choice(
                a=phase_shifted_nodes, size=sample_num, replace=False
            )
        for node in phase_shifted_nodes:
            new_state = self._unpack_state(np.fromstring(node, dtype=np.ubyte))
            self.graph.set_config(np.copy(new_state))
            self.explore_oscil(per=per, t=t)

    ####### Comparing periods

    def explore_new_period(
        self, per, nodes, sample_num=None, t=2000, explore_null=False
    ):
        """
        Explore these nodes listed for a new period given
        Assume we should explore them with phase 0!!!
        """

        # get all the nodes explored
        already_explored = self._get_all_nodes(period=per)

        # Check that we haven't explored the node already
        unexplored_nodes = [k for k in nodes if (k, 0) not in already_explored]

        if sample_num is not None and sample_num < len(unexplored_nodes):
            unexplored_nodes = np.random.choice(
                a=unexplored_nodes, size=sample_num, replace=False
            )

        for node in unexplored_nodes:
            # print(type(node))
            new_state = self._unpack_state(np.fromstring(node, dtype=np.ubyte))
            # if len(new_state) < 1000:
            # 	print("LEN UNEXPLORED", len(unexplored_nodes))
            # 	print("TYPE NEW STATE: ", type(new_state))
            # 	print("NODE: ",  node)
            # 	print("PERIOD", per)
            # 	print("NODES: ", nodes)

            self.graph.set_config(np.copy(new_state))
            self.explore_oscil(per=per, t=t, explore_null=explore_null)

    def cross_compare_attrs(
        self, per1, per2, trials=10, t=2000, explore_null=True, attr_cutoff=2000
    ):
        """
        Make sure each attractor in the one is explored in the other period
        We set a cutoff of exploring at most attr_cutoff attractors in each period for highly connected nets

        Trials: number of initial trials in each period...
        """
        get_nodes_from_attr_list = lambda x: [j for i in x for j in i]

        # Do initial exploration
        self.explore_periods(
            periods=[per1, per2], trials=trials, t=t, explore_null=explore_null
        )

        ## All attractors in period 1
        get_per_1_attrs = lambda: set(self.per_map[str(per1)].keys())

        ## All attractors in period 2
        get_per_2_attrs = lambda: set(self.per_map[str(per2)].keys())

        # Get the new attractors...
        attractors_1 = get_per_1_attrs()
        attractors_2 = get_per_2_attrs()

        # All the attractors from period 1 that have been cross explored in period 2
        cross_explored_1 = set()
        # All the attractors from period 2 that have been cross explored in period 1
        cross_explored_2 = set()

        # Atttractors from period 1 that should be rexplored in period 2
        # to_explore_1 = attractors_1.difference(cross_explored_1)
        to_explore_1 = deque(attractors_1)  # .difference(cross_explored_1)

        # Atttractors from period 2 that should be rexplored in period 1
        # to_explore_2 = attractors_2.difference(cross_explored_2)
        to_explore_2 = deque(attractors_2)  # .difference(cross_explored_2)

        # Keep track of what's on the queue
        queued_1 = set(attractors_1)
        queued_2 = set(attractors_2)

        while (len(to_explore_1) != 0 and len(cross_explored_1) < attr_cutoff) or (
            len(to_explore_2) != 0 and len(cross_explored_2) < attr_cutoff
        ):

            # what to explore from period 1 in period 2
            # All attractors..
            if len(cross_explored_1) + len(to_explore_1) > attr_cutoff:
                to_explore_1_now = {
                    to_explore_1.pop()
                    for _ in range(attr_cutoff - len(cross_explored_1))
                }
                # to_explore_1 = random.sample(population=to_explore_1,
                # k=attr_cutoff - len(cross_explored_1))
            else:
                to_explore_1_now = {
                    to_explore_1.pop() for _ in range(len(to_explore_1))
                }

            # What to explore from period 2 in period 1
            if len(cross_explored_2) + len(to_explore_2) > attr_cutoff:
                to_explore_2_now = {
                    to_explore_2.pop()
                    for _ in range(attr_cutoff - len(cross_explored_2))
                }
                # to_explore_2 = random.sample(population=to_explore_2,
                # 				k=attr_cutoff - len(cross_explored_2))
            else:
                to_explore_2_now = {
                    to_explore_2.pop() for _ in range(len(to_explore_2))
                }

            # print(to_explore_1_now, to_explore_2_now)

            to_explore_1_nodes = get_nodes_from_attr_list(to_explore_1_now)
            to_explore_2_nodes = get_nodes_from_attr_list(to_explore_2_now)

            if len(to_explore_1_nodes) > 0:
                self.explore_new_period(
                    nodes=to_explore_1_nodes, per=per2, explore_null=explore_null
                )

            if len(to_explore_2_nodes) > 0:
                self.explore_new_period(
                    nodes=to_explore_2_nodes, per=per1, explore_null=explore_null
                )

            cross_explored_1 = cross_explored_1.union(to_explore_1_now)
            cross_explored_2 = cross_explored_2.union(to_explore_2_now)

            attractors_1 = get_per_1_attrs()
            attractors_2 = get_per_2_attrs()

            # Atttractors from period 1 that should be rexplored in period 2
            # to_explore_1 = attractors_1.difference(cross_explored_1)
            for new_attr in attractors_1.difference(queued_1):
                to_explore_1.append(new_attr)
            queued_1 = queued_1.union(attractors_1)

            # Atttractors from period 2 that should be rexplored in period 1
            # to_explore_2 = attractors_2.difference(cross_explored_2)
            for new_attr in attractors_2.difference(queued_2):
                to_explore_2.append(new_attr)
            queued_2 = queued_2.union(attractors_2)

        # self.pretty_print()

    @classmethod
    def cross_compare_attrs_ctrl(
        cls,
        attr_graph1,
        attr_graph2,
        per1,
        per2,
        trials=10,
        t=2000,
        explore_null=True,
        attr_cutoff=2000,
    ):
        """
        Make sure each attractor in the one is explored in the other period

        Somewhat of a hack method --

        Note: We set an attr_cutoff meant to guarantee that we don't explore the entire statespace
        we stop cross exploring then...
        """

        get_nodes_from_attr_list = lambda x: [j for i in x for j in i]
        # Do initial exploration
        attr_graph1.explore_periods(
            periods=[per1], trials=trials, t=t, explore_null=explore_null
        )
        attr_graph2.explore_periods(
            periods=[per2], trials=trials, t=t, explore_null=explore_null
        )

        get_per_1_attrs = lambda: set(attr_graph1.per_map[str(per1)].keys())
        get_per_2_attrs = lambda: set(attr_graph2.per_map[str(per2)].keys())

        attractors_1 = get_per_1_attrs()
        attractors_2 = get_per_2_attrs()
        # All the attractors from period 1 that have been cross explored in period 2
        cross_explored_1 = set()
        # All the attractors from period 2 that have been cross explored in period 1
        cross_explored_2 = set()

        # Atttractors from period 1 that should be rexplored in period 2
        # to_explore_1 = attractors_1.difference(cross_explored_1)
        to_explore_1 = deque(attractors_1)  # .difference(cross_explored_1)

        # Atttractors from period 2 that should be rexplored in period 1
        # to_explore_2 = attractors_2.difference(cross_explored_2)
        to_explore_2 = deque(attractors_2)  # .difference(cross_explored_2)

        # Keep track of what's on the queue
        queued_1 = set(attractors_1)
        queued_2 = set(attractors_2)

        while (len(to_explore_1) != 0 and len(cross_explored_1) < attr_cutoff) or (
            len(to_explore_2) != 0 and len(cross_explored_2) < attr_cutoff
        ):

            # what to explore from period 1 in period 2
            if len(cross_explored_1) + len(to_explore_1) > attr_cutoff:
                to_explore_1_now = {
                    to_explore_1.pop()
                    for _ in range(attr_cutoff - len(cross_explored_1))
                }
                # to_explore_1 = random.sample(population=to_explore_1,
                # k=attr_cutoff - len(cross_explored_1))
            else:
                to_explore_1_now = {
                    to_explore_1.pop() for _ in range(len(to_explore_1))
                }

            # What to exlpore from period 2 in period 1
            if len(cross_explored_2) + len(to_explore_2) > attr_cutoff:
                to_explore_2_now = {
                    to_explore_2.pop()
                    for _ in range(attr_cutoff - len(cross_explored_2))
                }
                # to_explore_2 = random.sample(population=to_explore_2,
                # 				k=attr_cutoff - len(cross_explored_2))
            else:
                to_explore_2_now = {
                    to_explore_2.pop() for _ in range(len(to_explore_2))
                }

            # print(to_explore_1_now, to_explore_2_now)

            to_explore_1_nodes = get_nodes_from_attr_list(to_explore_1_now)
            to_explore_2_nodes = get_nodes_from_attr_list(to_explore_2_now)

            if len(to_explore_1_nodes) > 0:
                attr_graph2.explore_new_period(
                    nodes=to_explore_1_nodes, per=per2, explore_null=explore_null
                )

            if len(to_explore_2_nodes) > 0:
                attr_graph1.explore_new_period(
                    nodes=to_explore_2_nodes, per=per1, explore_null=explore_null
                )

            cross_explored_1 = cross_explored_1.union(to_explore_1_now)
            cross_explored_2 = cross_explored_2.union(to_explore_2_now)

            attractors_1 = get_per_1_attrs()
            attractors_2 = get_per_2_attrs()

            # Atttractors from period 1 that should be rexplored in period 2
            # to_explore_1 = attractors_1.difference(cross_explored_1)
            for new_attr in attractors_1.difference(queued_1):
                to_explore_1.append(new_attr)
            queued_1 = queued_1.union(attractors_1)

            # Atttractors from period 2 that should be rexplored in period 1
            # to_explore_2 = attractors_2.difference(cross_explored_2)
            for new_attr in attractors_2.difference(queued_2):
                to_explore_2.append(new_attr)
            queued_2 = queued_2.union(attractors_2)

        # self.pretty_print()

    def cross_compare_periods(
        self, per1, per2, cross_trials=10, initial_trials=10, t=2000, explore_null=True
    ):
        """
        Take two periods
        initially explore them for initial_trials trials
        then, for cross_trials different starting points, explore an unexplored node in each respective graph

        NOTE: When we do np.random.choice, we actually change the type of the underlying option
        use random.choice instead...
        """

        # Do initial exploration
        self.explore_periods(
            periods=[per1, per2], trials=initial_trials, t=t, explore_null=explore_null
        )

        def get_all_nodes(per):
            return {node for (node, phase) in self._get_all_nodes(per) if phase == 0}

        attr1_explored = get_all_nodes(per1)
        attr2_explored = get_all_nodes(per2)

        # while not attr1_n.issubset(attr2_explored) or not attr2_nodes.issubset(attr1_explored):
        for j in range(cross_trials):
            unexplored_per2 = attr1_explored.difference(attr2_explored)
            if len(unexplored_per2) > 0:
                self.explore_new_period(
                    nodes=[random.choice(list(unexplored_per2))],
                    per=per2,
                    explore_null=explore_null,
                )

            unexplored_per1 = attr2_explored.difference(attr1_explored)
            if len(unexplored_per1) > 0:
                self.explore_new_period(
                    nodes=[random.choice(list(unexplored_per1))],
                    per=per1,
                    explore_null=explore_null,
                )

            attr1_explored = get_all_nodes(per1)
            attr2_explored = get_all_nodes(per2)
        # self.pretty_print()

    def check_parsimony(self, test_per, target_per, zero_phase=True):
        """
        Compare test_per to target_per to see if a mapping exists
        Namely:
        First, calculate the overlap in nodes explored with zero phase between the two periods
        Next, loop through each basin of attraction in test_per
                (Only consider nodes part of the overlap above)
                For each attractor, find the attractor in target_per that has the highest fraction overlap
                Record the highest overlap fraction

        Return: A list of the highest fraction of overlap
        """
        # Generally check on all the nodes we have
        per1_nodes = set([i for i, j in self._get_all_nodes(period=test_per) if j == 0])
        per2_nodes = set(
            [i for i, j in self._get_all_nodes(period=target_per) if j == 0]
        )

        # Intersection...
        intersect_nodes = per1_nodes.intersection(per2_nodes)

        # Get the attractors:
        per1_basins = self.get_basins(period=test_per, check_phase=True)
        per2_basins = self.get_basins(period=target_per, check_phase=True)
        results = []
        for test_basin in per1_basins:
            nodes_to_consider = test_basin.intersection(intersect_nodes)
            if len(nodes_to_consider) > 0:
                set_agreement = []
                for j in per2_basins:
                    set_agreement.append(
                        len(nodes_to_consider.intersection(j)) / len(nodes_to_consider)
                    )
                results.append(np.max(set_agreement))
        return results

    def check_attr_parsimony(
        self, test_per, target_per, zero_phase=True, remove_overlap=True
    ):
        """
        Compare test_per to target_per to see if a mapping exists
        Check to see the parsimony between attractor nodes of test_per and basins of target_per
        Namely:
        First, calculate the overlap in nodes explored with zero phase between the two periods
        Next, loop through each attr in test_per
                (Only consider nodes part of the overlap above)
                For each attractor, find the basin in target_per that has the highest fraction overlap
                Record the highest overlap fraction

        -If remove_overlap: we force the comparison such that it does not take into account non-unique attr nodes
                That is, if one network configuration is in both overlaps, we shouldn't count that as a unique mapping since it's redundant

        Return: A list of the highest fraction of overlap
        """

        # Get the attractors and basins:
        ## Per 1 attrs...
        per1_attrs = list(self.per_map[str(test_per)].keys())
        ## List of all nodes in each basin [[node1,node2], [node1,node2] ,etc.]
        per2_basins = self.get_basins(period=target_per, check_phase=True)

        # Per 2 attrs
        per2_attrs = list(self.per_map[str(target_per)].keys())

        # Generally check on all the nodes we have in period 1 attractors and per 2 explored..
        per1_attr_nodes = set(
            [i for specific_attr in per1_attrs for i in specific_attr]
        )
        per2_nodes_explored = set(
            [i for i, j in self._get_all_nodes(period=target_per) if j == 0]
        )

        # Intersection between attractor nodes and explored noes
        intersect_nodes = per1_attr_nodes.intersection(per2_nodes_explored)

        results_agreement = []
        results_overlap = []
        for test_attractor in per1_attrs:
            nodes_to_consider = test_attractor.intersection(intersect_nodes)
            set_agreement = []
            set_overlap = []
            # Make sure we've fully explored this attr we are considering
            # Don't want biased results like division by 1...
            if len(nodes_to_consider) == len(test_attractor):
                for per2_index, j in enumerate(per2_basins):
                    per2_attr = per2_attrs[per2_index]

                    # try removing all nodes common to the second periods attractor!!!
                    # Make sure we're programming a UNIQUE switch event
                    ## Overlap:
                    overlap = len(nodes_to_consider.intersection(per2_attr)) / len(
                        nodes_to_consider
                    )

                    if remove_overlap:
                        agreement = len(
                            nodes_to_consider.intersection(j).difference(per2_attr)
                        ) / len(nodes_to_consider)
                    else:
                        agreement = len(nodes_to_consider.intersection(j)) / len(
                            nodes_to_consider
                        )

                    set_agreement.append(agreement)
                    set_overlap.append(overlap)

                max_index = np.argmax(set_agreement)
                results_agreement.append(set_agreement[max_index])
                results_overlap.append(set_overlap[max_index])
        return results_agreement, results_overlap

    @classmethod
    def check_attr_parsimony_ctrl(
        cls,
        attr_graph1,
        attr_graph2,
        test_per,
        target_per,
        zero_phase=True,
        remove_overlap=True,
    ):
        """
        Control class method to compare test_per to target_per across two different networks
        Check to see the parsimony between attractor nodes of test_per and basins of target_per
        Namely:
        First, calculate the overlap in nodes explored with zero phase between the two periods
        Next, loop through each attr in test_per
                (Only consider nodes part of the overlap above)
                For each attractor, find the basin in target_per that has the highest fraction overlap
                Record the highest overlap fraction

        - In this alternative, we force the comparison such that it does not take into account non-unique attr nodes
                That is, if one network configuration is in both overlaps, we shouldn't count that as a unique mapping since it's redundant

        Return: A list of the highest fraction of overlap
        """

        # Get the attractors and basins:
        per1_attrs = list(attr_graph1.per_map[str(test_per)].keys())
        per2_basins = attr_graph2.get_basins(period=target_per, check_phase=True)

        per2_attrs = list(attr_graph2.per_map[str(target_per)].keys())

        # Generally check on all the nodes we have
        per1_attr_nodes = set(
            [i for specific_attr in per1_attrs for i in specific_attr]
        )
        per2_nodes_explored = set(
            [i for i, j in attr_graph2._get_all_nodes(period=target_per) if j == 0]
        )

        # Intersection...
        intersect_nodes = per1_attr_nodes.intersection(per2_nodes_explored)

        results = []
        for test_attractor in per1_attrs:
            # Only if we have seen all the nodes in the first attrator in the 2nd period
            nodes_to_consider = test_attractor.intersection(intersect_nodes)
            set_agreement = []
            # Make sure we've fully explored this attr we are considering
            # Don't want biased results like division by 1...
            if len(nodes_to_consider) == len(test_attractor):
                for per2_index, j in enumerate(per2_basins):
                    per2_attr = per2_attrs[per2_index]

                    # try removing all nodes common to the second periods attractor!!!
                    # Make sure we're programming a UNIQUE switch event
                    if remove_overlap:
                        agreement = len(
                            nodes_to_consider.intersection(j).difference(per2_attr)
                        ) / len(nodes_to_consider)
                    else:
                        agreement = len(nodes_to_consider.intersection(j)) / len(
                            nodes_to_consider
                        )

                    set_agreement.append(agreement)

                results.append(np.max(set_agreement))
        return results

    ######## Data export methods

    def export_bipartite(self, per1, per2, outfile):
        """
        Create a bipartite graph mapping the attractors in per1 to the appropraite basin in per2 and vice versa
        Edges moving from attractor to nodes in it
        """

        if per1 not in self.per_map or per2 not in self.per_map:
            raise Exception(f"Exception: Must first explore {per1} and {per2}")

        # node, phase
        per1_mapping = self._map_nodes_attr(per1)
        per2_mapping = self._map_nodes_attr(per2)

        ## Only consider the mappings with no phase attached...
        per1_mapping_no_phase = {
            state: attr
            for node, attr in per1_mapping.items()
            for state, phase in node
            if phase == 0
        }
        per2_mapping_no_phase = {
            state: attr
            for node, attr in per2_mapping.items()
            for state, phase in node
            if phase == 0
        }

        all_nodes = set(per1_mapping_no_phase.keys()).union(
            set(per2_mapping_no_phase.keys())
        )

        # Rename the nodes for the output graph..
        new_node_names = {node: index for index, node in enumerate(all_nodes)}

        per1_attr_names = {
            attr: f"Period:{per1},Attractor:{index}"
            for index, attr in enumerate(self.per_map[per1])
        }

        per2_attr_names = {
            attr: f"Period:{per2},Attractor:{index}"
            for index, attr in enumerate(self.per_map[per2])
        }

        edges = []
        all_nodes = []
        # for the attractors in the first period..
        for attrs in self.per_map[per1]:
            # For all the nodes associated with this attractor..
            for node in attrs:
                ### If this node is in per2 mapping.. per2 mapping must have phase, node must not
                if node in per2_mapping_no_phase:
                    node_name = new_node_names[node]
                    all_nodes.append(node_name)

                    source_attr = per1_mapping_no_phase[node]
                    target_attr = per2_mapping_no_phase[node]

                    attr_origin = per1_attr_names[source_attr]
                    attr_dest = per2_attr_names[target_attr]
                    edges.append((attr_origin, node_name))
                    edges.append((node_name, attr_dest))

                    # When it wasn't unique....

                    # Loop over all possibilites of source and destinations!
                    # for source_attr in per1_mapping[node]:
                    # 	for target_attr in per2_mapping[node]:
                    # 		attr_origin =  per1_attr_names[source_attr]
                    # 		attr_dest = per2_attr_names[target_attr]

                    # 		edges.append((attr_origin, node_name))
                    # 		edges.append((node_name, attr_dest))

        # for the attractors in the first period..
        for attrs in self.per_map[per2]:
            # For all the nodes associated with this attractor..
            for node in attrs:
                if node in per1_mapping_no_phase:
                    node_name = new_node_names[node]
                    all_nodes.append(node_name)

                    source_attr = per2_mapping_no_phase[node]
                    target_attr = per1_mapping_no_phase[node]

                    attr_origin = per2_attr_names[source_attr]
                    attr_dest = per1_attr_names[target_attr]
                    edges.append((attr_origin, node_name))
                    edges.append((node_name, attr_dest))

                    # Old...
                    # Loop over all possibilites of source and destinations!
                    # for source_attr in per2_mapping[node]:
                    # 	for target_attr in per1_mapping[node]:
                    # 		attr_origin =  per2_attr_names[source_attr]
                    # 		attr_dest = per1_attr_names[target_attr]

                    # 		edges.append((attr_origin, node_name))
                    # 		edges.append((node_name, attr_dest))

        g = nx.DiGraph()

        g.add_edges_from(edges)

        g.add_nodes_from(all_nodes, is_node=1, size=1)

        all_attrs = list(per1_attr_names.values())
        all_attrs.extend(list(per2_attr_names.values()))
        g.add_nodes_from(all_attrs, is_attr=1, size=5)

        # Adjust the name
        if outfile[-5:] != ".gexf":
            outfile += ".gexf"
        nx.write_gexf(g, outfile)

        return g

    @classmethod
    def export_bipartite_multigraph(cls, graph1, graph2, per1, per2, outfile):
        """
        Create a bipartite graph mapping the attractors in per1 to the appropraite basin in per2 and vice versa
        Edges moving from attractor to nodes in it

        Note:
        """

        if per1 not in graph1.per_map or per2 not in graph2.per_map:
            raise Exception(f"Exception: Must first explore {per1} and {per2}")

        per1_mapping = {
            state: attr
            for node, attr in graph1._map_nodes_attr(per1).items()
            for state, phase in node
            if phase == 0
        }
        per2_mapping = {
            state: attr
            for node, attr in graph2._map_nodes_attr(per2).items()
            for state, phase in node
            if phase == 0
        }

        all_nodes = set(per1_mapping.keys()).union(set(per2_mapping.keys()))

        # Rename the nodes for the output graph..
        new_node_names = {node: index for index, node in enumerate(all_nodes)}

        per1_attr_names = {
            attr: f"Period:{per1},Attractor:{index}"
            for index, attr in enumerate(graph1.per_map[per1])
        }

        per2_attr_names = {
            attr: f"Period:{per2},Attractor:{index}"
            for index, attr in enumerate(graph2.per_map[per2])
        }
        edges = []
        all_nodes = []
        # for the attractors in the first period..
        for attrs in graph1.per_map[per1]:
            # For all the nodes associated with this attractor..
            for node in attrs:
                if node in per2_mapping:
                    node_name = new_node_names[node]
                    all_nodes.append(node_name)

                    source_attr = per1_mapping[node]
                    target_attr = per2_mapping[node]

                    attr_origin = per1_attr_names[source_attr]
                    attr_dest = per2_attr_names[target_attr]
                    edges.append((attr_origin, node_name))
                    edges.append((node_name, attr_dest))

                    # Loop over all possibilites of source and destinations!
                    # for source_attr in per1_mapping[node]:
                    # 	for target_attr in per2_mapping[node]:
                    # 		attr_origin =  per1_attr_names[source_attr]
                    # 		attr_dest = per2_attr_names[target_attr]

                    # 		edges.append((attr_origin, node_name))
                    # 		edges.append((node_name, attr_dest))

        # for the attractors in the first period..
        for attrs in graph2.per_map[per2]:
            # For all the nodes associated with this attractor..
            for node in attrs:
                if node in per1_mapping:
                    node_name = new_node_names[node]
                    all_nodes.append(node_name)
                    # Loop over all possibilites of source and destinations!

                    source_attr = per2_mapping[node]
                    target_attr = per1_mapping[node]

                    attr_origin = per2_attr_names[source_attr]
                    attr_dest = per1_attr_names[target_attr]
                    edges.append((attr_origin, node_name))
                    edges.append((node_name, attr_dest))

                    # for source_attr in per2_mapping[node]:
                    # 	for target_attr in per1_mapping[node]:
                    # 		attr_origin =  per2_attr_names[source_attr]
                    # 		attr_dest = per1_attr_names[target_attr]

                    # 		edges.append((attr_origin, node_name))
                    # 		edges.append((node_name, attr_dest))

        g = nx.DiGraph()

        g.add_edges_from(edges)

        g.add_nodes_from(all_nodes, is_node=1, size=1)

        all_attrs = list(per1_attr_names.values())
        all_attrs.extend(list(per2_attr_names.values()))
        g.add_nodes_from(all_attrs, is_attr=1, size=5)

        # Adjust the name
        if outfile[-5:] != ".gexf":
            outfile += ".gexf"
        nx.write_gexf(g, outfile)

        return g

    def export_graph(self, outfile):
        """
        Write the graph out to a file
        Return the nx graph obj for other modifications


        TOOD: Need consistent way to reference the attractors with the appropriate phase...
        """
        g = nx.DiGraph()

        # Add edges to graph from the control
        all_nodes = self._get_all_nodes(AttractorGraph.NO_OSCIL)
        g.add_nodes_from(all_nodes)
        g.add_edges_from(self._get_all_edges(AttractorGraph.NO_OSCIL), common_edge=True)

        # Counter for each oscillation's attractors...
        oscil_attr_num = {str(per): 0 for per in self.per_map.keys()}

        # Add the edges for each period!
        for index, (period, attr_dict) in enumerate(self.per_map.items()):
            # no need to do this for the contorl...
            if period == AttractorGraph.NO_OSCIL:
                continue

            datum = {f"Period:{period}": True}
            g.add_edges_from(self._get_all_edges(period), **datum)
            for attr in self.attr_states[period].values():
                for state in attr:
                    g.node[state][f"oscil_attractor_{period}"] = True
                    g.node[state][f"oscil_attr_num_{period}"] = oscil_attr_num[
                        str(period)
                    ]
                oscil_attr_num[str(period)] += 1

        # Requires hub node...
        # Color nodes by off or on block
        for state in all_nodes:
            # state is a str repr of the byte array
            np_state = self._unpack_state(np.fromstring(state, dtype=np.ubyte))
            if np_state[self.oscil_node] == 0:
                g.node[state]["HUBSTATE"] = "OFF"
            else:
                g.node[state]["HUBSTATE"] = "ON"

        # Color nodes by their attractor...
        for index, attractor in enumerate(
            self.attr_states[AttractorGraph.NO_OSCIL].values()
        ):
            for state in attractor:
                g.node[state]["Attractor_Num"] = index
                # For size of nodes...
                g.node[state]["HasAttractor"] = 1

        # Say whether each node is an attractor for size reasons!
        for node, a in g.nodes(data=True):
            if "HasAttractor" not in a:
                g.node[node]["HasAttractor"] = 0
            else:
                g.node[node]["HasAttractor"] = 1

            # Make sure other nodes are 0 if they aren't in this oscillation!
            for per in self.per_map.keys():
                if f"oscil_attractor_{per}" in a and a != AttractorGraph.NO_OSCIL:
                    g.node[node][f"oscil_attractor_{per}"] = 1
                else:
                    g.node[node][f"oscil_attractor_{per}"] = 0

        # Adjust the name
        if outfile[-5:] != ".gexf":
            outfile += ".gexf"
        nx.write_gexf(g, outfile)
        return g

    ###### Aux methods

    def attr_state_fraction(self, period_1, period_2):
        """Figure out the fraction of the attr cycle states in this period
        that are attr cycle states in the second period"""
        period_1 = str(period_1)
        period_2 = str(period_2)
        attr_states_1 = self.get_all_attr_states(period_1)
        attr_states_2 = self.get_all_attr_states(period_2)

        return len(attr_states_1.intersection(attr_states_2)) / len(attr_states_1)

    def get_all_attr_states(self, period):
        """Get all states that participate in attractors for a given period"""
        period = str(period)
        return set([state for i in self.per_map[period].keys() for state in i])

    def get_avg_attractor_size(self, period):
        """Calculate avg attractor size for a period"""
        period = str(period)
        return np.mean([len(i) for i in self.per_map[period].keys()])

    def get_basins(self, period, check_phase=False):
        """
        Get a list of all the nodes in each basin
        If check_phase, we will only include nodes that have no phase
        NOTE: Return does not include the phase of each node
        """

        period = str(period)
        return_list = []
        for edge_dict in self.per_map[period].values():
            attr_nodes = set()
            for node, phase in edge_dict:
                # if we only want nodes when they start..
                if check_phase and phase == 0:
                    attr_nodes.add(node)
                elif not check_phase:
                    attr_nodes.add(node)
            return_list.append(attr_nodes)
        return return_list

    def _get_all_nodes(self, period):
        """
        Get all the nodes associated with a given period... return in the form (state, phase)
        """
        # Add edges to graph from the control

        period = str(period)

        if period not in self.per_map:
            return {}  # if datum else []

        all_nodes = set()
        for edge_dict in self.per_map[period].values():
            for node in edge_dict.keys():
                all_nodes.add(node)

        return all_nodes

    def _get_all_nodes_attr(self, period, attr):
        """
        Get all nodes (state, phase) in a given attractor associated with this period
        """

        period = str(period)

        if period not in self.per_map:
            return {}  # if datum else set()

        elif attr not in self.per_map[period]:
            return {}  # if datum else set()

        edge_dict = self.per_map[period][attr]

        return set(edge_dict.keys())

    def _get_all_edges(self, period):
        """
        Get all the edges associated with a given period
        """

        period = str(period)

        if period not in self.per_map:
            return set()

        return {
            k: v
            for edge_dict in self.per_map[period].values()
            for k, v in edge_dict.items()
        }

    def _map_nodes_attr(self, per):
        """
        Create a mapping between (nodes, phase) and attractor for a period

        Assumes each node only appears once
        return: dict(node : attractor... )
        """
        if per not in self.per_map:
            return dict()
        r = dict()
        for attr, edge_dict in self.per_map[per].items():
            for node in edge_dict.keys():
                # if we already saw this node.. raise a warning
                # if node in r:
                # 	print("Warning: This node appears in multiple basins of attraction.\n This case is not handled and it has been overwritten")
                # r[node] = attr

                if node in r:
                    print("This node is in multiple attractors! ERRRORR")
                r[node] = attr
        return r

    def _add_edge(self, period, s1, s2, attr, phase1=0, phase2=0):
        """
        Add this edge to the dictionary to keep track of...
        If it was the first, mark that
        Args:
                period: period of oscillation
                s1, s2: States of the network (strings)
                attr: frozen attractor
                phase1: Denotes the phase in the oscillation pattern of the first node.
                        If we aren't oscillating, default 0
                phase2: Denotes the phase in the oscillation pattern of the target node.
                        If we aren't oscillating, default 0
        """

        # Convert to string repr
        period = str(period)

        if period not in self.per_map:
            self.per_map[period] = dict()
            self.attr_states[period] = dict()

        if attr not in self.per_map[period]:
            self.per_map[period][attr] = dict()

        # Now adjust edges
        if (s1, phase1) not in self.per_map[period][attr]:
            self.per_map[period][attr][(s1, phase1)] = (s2, phase2)
        elif self.per_map[period][attr][(s1, phase1)] != (s2, phase2):
            raise Exception(
                "State mismatch in adding edge.. We should have a different target here"
            )

    def _get_attr_phase(self, attr_table, phase_start, mod_phase):
        """
        Given packed arr of attr_states, the start phase of the attractor,
        and the maximum phase the attractor can go to, return the attr state broken up into tuples of state,phase
        """
        my_set = set()
        for index, row in enumerate(attr_table):
            phase = (phase_start + index) % mod_phase
            my_set.add((row.tostring(), phase))

        return frozenset(my_set)

    def _add_attr_phase(self, period, attr, attr_table, phase_start, mod_phase):
        """
        Given period, attr, and the actual attr_table, the start phase of the attractor (presumably 0)
        and the highest value the phase can take on, add this to the attr to the attr states dictionary
        """

        period = str(period)

        if period not in self.attr_states:
            self.attr_states[period] = dict()

        if attr not in self.attr_states[period]:
            self.attr_states[period][attr] = self._get_attr_phase(
                attr_table, phase_start, mod_phase
            )

    def _get_frozen_attractor(self, arr):
        """
        Arguments:
                arr: A 2d array containing states of the network, specifically the attractor
        Return: A frozen set of the attractor defined in the arr array
        """
        my_set = set()
        add_to_set = lambda x: my_set.add(x.tostring())
        np.apply_along_axis(func1d=add_to_set, axis=1, arr=arr)
        return frozenset(my_set)

    def _was_explored(self, state, period, phase):
        """
        Check if a given network state was explored for a certain period
        """

        explored_list = self._get_all_nodes(period)
        return (state, phase) in explored_list

    def _was_explored_in_attr(self, state, period, attr, phase):
        """
        Determine if the given node was explored for a certain period in a certain attractor
        """
        if type(period) != str:
            period = str(period)

        if period not in self.per_map:
            return False
        elif attr not in self.per_map[period]:
            return False
        elif (state, phase) not in self.per_map[period][attr]:
            return False
        else:
            # Return true if we've already seen this node when there's phase!
            return True

    def _pack_tbl(self, tbl):
        """
        Pack the table into bits
        """
        return np.packbits(tbl.astype(bool), axis=1)

    def _unpack_state(self, packed_state):
        """
        Unpack the state of a given row
        """

        if type(packed_state) == bytes:
            # print("Warning: Converting class bytes to np.ubyte to unpack bits")
            packed_state = np.fromstring(packed_state, dtype=np.ubyte)

        return np.unpackbits(packed_state)[: self._n]
