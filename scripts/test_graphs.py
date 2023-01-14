#!/usr/bin/env python
# Goal: construct a uniform network and play around with it
# Thursday, November 10 2016

from HomGraph import *
from SFGraph import *
import networkx.utils as util
from matplotlib.pyplot import cm
from matplotlib.mlab import frange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import scipy


def find_hom_graph_with_cycle(nodes):
    """
    Purpose to demonstrate that we can find graphs with cycles
    """
    found_cycle = False
    while not found_cycle:
        myGraph = HomGraph(nodes)
        cycle = myGraph.find_cycle()[1]
        if len(cycle) > 2:
            found_cycle = True
            print(myGraph.find_cycle()[1])
            for i in range(len(cycle)):
                myGraph.draw()
                myGraph.update()


def find_SF_graph_with_cycle(nodes, lam):
    """
    Purpose to demonstrate that we can find graphs with cycles
    """
    found_cycle = False
    while not found_cycle:
        myGraph = SFGraph(nodes, lam)
        cycle_return = myGraph.find_cycle()
        timeTaken = cycle_return[0]
        cycle = cycle_return[1]
        print("Time to find cycle: " + str(timeTaken))
        if len(cycle) > 2:
            print("Time to find cycle: " + str(timeTaken))
            found_cycle = True
            print(myGraph.find_cycle()[1])
            for i in range(len(cycle)):
                myGraph.draw()
                myGraph.update()


def plot_hamming_distances():
    """
    Shows single perturbations for homogenous graphs and scale free graphs
    """
    for i in range(10):
        ar1 = HomGraph.hamming_single_perturb(100, 10, 0.2, 0.5, 20)
        ar2 = SFGraph.hamming_single_perturb(100, 2.5, 0.2, 0.5, 20)
        Graph.plot_hamming(ar1, color="b")
        Graph.plot_hamming(ar2, color="r")
        # Graph.clear_graph()


def plot_avg_hamming_distances_hom(n, ic, d, t, trials):
    """
    For a set number of connectivity parameters plot the average hamming distance for some number (trials) of intiial conditions
    """
    k1 = 2
    k2 = 5
    k3 = 7
    k4 = 10
    ar1 = HomGraph.avg_hamming_ic(n, k1, ic, d, t, trials)
    ar2 = HomGraph.avg_hamming_ic(n, k2, ic, d, t, trials)
    ar3 = HomGraph.avg_hamming_ic(n, k3, ic, d, t, trials)
    ar4 = HomGraph.avg_hamming_ic(n, k4, ic, d, t, trials)

    Graph.plot_hamming(
        ar1,
        color="b",
        label="k = %d" % k1,
        title="Average Hamming Distance Hom n=%d, k=%d, d=%f, t=%d, trials=%d"
        % (n, k1, d, t, trials),
    )
    Graph.plot_hamming(
        ar2,
        color="r",
        label="k = %d" % k2,
        title="Average Hamming Distance Hom n=%d, k=%d, d=%f, t=%d, trials=%d"
        % (n, k2, d, t, trials),
    )
    Graph.plot_hamming(
        ar3,
        color="y",
        label="k = %d" % k3,
        title="Average Hamming Distance Hom n=%d, k=%d, d=%f, t=%d, trials=%d"
        % (n, k3, d, t, trials),
    )
    Graph.plot_hamming(
        ar4,
        color="g",
        label="k = %d" % k4,
        title="Average Hamming Distance Hom n=%d, k=%d, d=%f, t=%d, trials=%d"
        % (n, k4, d, t, trials),
    )


def plot_avg_hamming_distances_sf():
    """
    For a set number of connectivity parameters (lam) plot the average hamming distance for some number (trials) of intiial conditions
    """
    ar1 = SFGraph.avg_hamming_ic(100, 2, 0.5, 0.05, 20, 100)
    ar2 = SFGraph.avg_hamming_ic(100, 2.5, 0.5, 0.05, 20, 100)
    ar3 = SFGraph.avg_hamming_ic(100, 3, 0.5, 0.05, 20, 100)
    ar4 = SFGraph.avg_hamming_ic(100, 3.5, 0.5, 0.05, 20, 100)

    Graph.plot_hamming(
        ar1,
        color="b",
        label="lam = 2",
        title="Average Hamming Distance (100 trials) for SF Networks with 100 nodes, lambda connectivity, and 0.05 initially perturbed",
    )
    Graph.plot_hamming(
        ar2,
        color="r",
        label="lam = 2.5",
        title="Average Hamming Distance (100 trials) for SF Networks with 100 nodes, lambda connectivity, and 0.05 initially perturbed",
    )
    Graph.plot_hamming(
        ar3,
        color="y",
        label="lam = 3",
        title="Average Hamming Distance (100 trials) for SF Networks with 100 nodes, lambda connectivity, and 0.05 initially perturbed",
    )
    Graph.plot_hamming(
        ar4,
        color="g",
        label="lam = 3.5",
        title="Average Hamming Distance (100 trials) for SF Networks with 100 nodes, lambda connectivity, and 0.05 initially perturbed",
    )


def plot_avg_ic_hom(k):
    """
    Show that from trial to trial, depending on the network, the average initial conditions vary greatly !
    """
    ar1 = HomGraph.avg_hamming_ic(100, k, 0.5, 0.05, 20, 200)
    ar2 = HomGraph.avg_hamming_ic(100, k, 0.5, 0.05, 20, 200)
    ar3 = HomGraph.avg_hamming_ic(100, k, 0.5, 0.05, 20, 200)
    ar4 = HomGraph.avg_hamming_ic(100, k, 0.5, 0.05, 20, 200)
    ar5 = HomGraph.avg_hamming_ic(100, k, 0.5, 0.05, 20, 200)

    Graph.plot_hamming(
        ar1,
        color="b",
        label="Trial 1: k = %d" % k,
        title="Average Hamming Distance (200 trials) for Homogenous Networks with 100 nodes, k connectivity, and 0.05 initially perturbed",
    )
    Graph.plot_hamming(
        ar2,
        color="r",
        label="Trial 2: k = %d" % k,
        title="Average Hamming Distance (200 trials) for Homogenous Networks with 100 nodes, k connectivity, and 0.05 initially perturbed",
    )
    Graph.plot_hamming(
        ar3,
        color="y",
        label="Trial 3: k = %d" % k,
        title="Average Hamming Distance (200 trials) for Homogenous Networks with 100 nodes, k connectivity, and 0.05 initially perturbed",
    )
    Graph.plot_hamming(
        ar4,
        color="g",
        label="Trial 4: k = %d" % k,
        title="Average Hamming Distance (200 trials) for Homogenous Networks with 100 nodes, k connectivity, and 0.05 initially perturbed",
    )
    Graph.plot_hamming(
        ar5,
        color="black",
        label="Trial 5: k = %d" % k,
        title="Average Hamming Distance (200 trials) for Homogenous Networks with 100 nodes, k connectivity, and 0.05 initially perturbed",
    )


def plot_avg_ic_sf(k):
    """
    Show that from trial to trial, depending on the network, the average initial conditions vary greatly !
    """
    ar1 = SFGraph.avg_hamming_ic(100, k, 0.5, 0.05, 20, 200)
    ar2 = SFGraph.avg_hamming_ic(100, k, 0.5, 0.05, 20, 200)
    ar3 = SFGraph.avg_hamming_ic(100, k, 0.5, 0.05, 20, 200)
    ar4 = SFGraph.avg_hamming_ic(100, k, 0.5, 0.05, 20, 200)
    ar5 = SFGraph.avg_hamming_ic(100, k, 0.5, 0.05, 20, 200)

    Graph.plot_hamming(
        ar1,
        color="b",
        label="Trial 1: lam = %0.1f" % k,
        title="Average Hamming Distance (200 trials) for SF Networks with 100 nodes, k connectivity, and 0.05 initially perturbed",
    )
    Graph.plot_hamming(
        ar2,
        color="r",
        label="Trial 2: lam = %0.1f" % k,
        title="Average Hamming Distance (200 trials) for SF Networks with 100 nodes, k connectivity, and 0.05 initially perturbed",
    )
    Graph.plot_hamming(
        ar3,
        color="y",
        label="Trial 3: lam = %0.1f" % k,
        title="Average Hamming Distance (200 trials) for SF Networks with 100 nodes, k connectivity, and 0.05 initially perturbed",
    )
    Graph.plot_hamming(
        ar4,
        color="g",
        label="Trial 4: lam = %0.1f" % k,
        title="Average Hamming Distance (200 trials) for SF Networks with 100 nodes, k connectivity, and 0.05 initially perturbed",
    )
    Graph.plot_hamming(
        ar5,
        color="black",
        label="Trial 5: lam = %0.1f" % k,
        title="Average Hamming Distance (200 trials) for SF Networks with 100 nodes, k connectivity, and 0.05 initially perturbed",
    )


def plot_avg_network_hom():
    """
    Graph the hamming distances over both initial conditions and realizatoins
    """
    ar1 = HomGraph.avg_hamming_network(100, 2, 0.5, 0.05, 20, 20, 10)
    ar2 = HomGraph.avg_hamming_network(100, 5, 0.5, 0.05, 20, 20, 10)
    ar3 = HomGraph.avg_hamming_network(100, 7, 0.5, 0.05, 20, 20, 10)
    ar4 = HomGraph.avg_hamming_network(100, 10, 0.5, 0.05, 20, 20, 10)
    Graph.plot_hamming(
        ar1,
        color="b",
        label="k = 2",
        title="Average Hamming Distance Hom Network n=100, k connectivity, 10 networks, 20 trials",
    )
    Graph.plot_hamming(
        ar2,
        color="r",
        label="k = 5",
        title="Average Hamming Distance Hom Network n=100, k connectivity, 10 networks, 20 trials",
    )
    Graph.plot_hamming(
        ar3,
        color="y",
        label="k = 7",
        title="Average Hamming Distance Hom Network n=100, k connectivity, 10 networks, 20 trials",
    )
    Graph.plot_hamming(
        ar4,
        color="g",
        label="k = 10",
        title="Average Hamming Distance Hom Network n=100, k connectivity, 10 networks, 20 trials",
    )


def plot_avg_network_sf():
    """
    Graph the hamming distances over both initial conditions and realizatoins
    """
    ar1 = SFGraph.avg_hamming_network(100, 2, 0.5, 0.05, 20, 20, 50)
    ar2 = SFGraph.avg_hamming_network(100, 2.5, 0.5, 0.05, 20, 20, 50)
    ar3 = SFGraph.avg_hamming_network(100, 3, 0.5, 0.05, 20, 20, 50)
    ar4 = SFGraph.avg_hamming_network(100, 3.5, 0.5, 0.05, 20, 20, 50)
    Graph.plot_hamming(
        ar1,
        color="b",
        label="k = 2",
        title="Average Hamming Distance SF Network n=100, k connectivity, 50 networks, 20 trials",
    )
    Graph.plot_hamming(
        ar2,
        color="r",
        label="k = 2.5",
        title="Average Hamming Distance SF Network n=100, k connectivity, 50 networks, 20 trials",
    )
    Graph.plot_hamming(
        ar3,
        color="y",
        label="k = 3",
        title="Average Hamming Distance SF Network n=100, k connectivity, 50 networks, 20 trials",
    )
    Graph.plot_hamming(
        ar4,
        color="g",
        label="k = 3.5",
        title="Average Hamming Distance SF Network n=100, k connectivity, 50 networks, 20 trials",
    )


### 12/1/16 After discussion with Mayra ###
# build 1 network for each k = 2, k = 5, k = 7, k = 10 1000 nodes, 100 time, 10 initial conditions, d = 0.5


def plot_avg_hamming_distances_hom_spectrum(n, ic, d, t, trials):
    """
    For a spectrum of connectivity parameters plot the avearge hamming distance & control parameter graphs
    However, only generate one realization for each conncetivity parameter. Find hamming distnace for this network over several intiial conditions
    """
    color = iter(cm.rainbow(np.linspace(0, 1, 20)))
    arlist = []
    for k in range(1, 21):
        arlist.append(HomGraph.avg_hamming_ic(n, k, ic, d, t, trials))
    for i, ar in enumerate(arlist):
        c = next(color)
        Graph.plot_hamming(
            ar,
            color=c,
            label="k = %d" % (i + 1),
            title="Average Hamming Distance Hom n=%d, d=%0.2f, t=%d, trials=%d"
            % (n, d, t, trials),
        )
    # Also tack on the plot for the average of each
    Graph.clear_graph()
    meanAr = [0]
    for ar in arlist:
        # Only taking the mean after t/2
        meanAr.append(np.mean(ar[t / 2 :]))
    Graph.plot_hamming(
        meanAr,
        color="b",
        label="Average Hamming Distance",
        xtitle="k",
        ytitle="mean(h(t)) from t =0 to t=100",
        title="Mean hamming distance across time",
    )


def plot_avg_network_hamming_distances_hom_spectrum(n, ic, d, t, trials, networks):
    """
    For a spectrum of connectivity parameters plot the avearge hamming distance & control parameter graphs
    Do this for MULTIPLE realizations for each conncetivity parameter. Find hamming distnace for this network over several intiial conditions
    """
    color = iter(cm.rainbow(np.linspace(0, 1, 20)))
    arlist = []
    myRange = range(1, 21)
    for k in myRange:
        arlist.append(HomGraph.avg_hamming_network(n, k, ic, d, t, trials, networks))
    for i, ar in enumerate(arlist):
        c = next(color)
        Graph.plot_hamming(
            ar,
            color=c,
            label="k = %d" % (i + 1),
            title="Average Hamming Distance Hom n=%d, d=%0.2f, t=%d, trials=%d, r=%d"
            % (n, d, t, trials, networks),
        )
    # Also tack on the plot for the average of each
    Graph.clear_graph()
    meanAr = []
    for ar in arlist:
        # Only taking the mean after t/2
        meanAr.append(np.mean(ar[t / 2 :]))
    Graph.plot_hamming(
        meanAr,
        color="b",
        label="Order Parameter vs. Control Parameter",
        xtitle="Control Parameter k",
        xticks=myRange,
        ytitle="Order Parameter",
        title="N = %d, d = %0.2f, t=%d, trials=%d, r=%d" % (n, d, t, trials, networks),
    )


def plot_avg_hamming_distances_sf_spectrum(n, ic, d, t, trials):
    """
    For a spectrum of connectivity parameters plot the avearge hamming distance & control parameter graphs
    However, only generate one realization for each conncetivity parameter. Find hamming distnace for this network over several intiial conditions
    """
    color = iter(cm.rainbow(np.linspace(0, 1, 21)))
    arlist = []
    myRange = list(frange(1.5, 3.5, 0.1))
    for lam in myRange:
        arlist.append(SFGraph.avg_hamming_ic(n, lam, ic, d, t, trials))
    for ar, i in zip(arlist, myRange):
        c = next(color)
        Graph.plot_hamming(
            ar,
            color=c,
            label="lambda = %f" % i,
            title="Average Hamming Distance SF n=%d, d=%0.2f, t=%d, trials=%d"
            % (n, d, t, trials),
        )
    # Also tack on the plot for the average of each
    Graph.clear_graph()
    meanAr = [0]
    for ar in arlist:
        # Only taking the mean after 50
        meanAr.append(np.mean(ar[t / 2 :]))
    Graph.plot_hamming(
        meanAr,
        color="b",
        label="Order Parameter vs. Control Parameter",
        xtitle="Control Parameter Lambda",
        ytitle="Order Parameter",
        title="Mean hamming distance across time",
    )


# Fix X labels
def plot_avg_network_hamming_distances_sf_spectrum(n, ic, d, t, trials, networks):
    """
    For a spectrum of connectivity parameters plot the avearge hamming distance & control parameter graphs
    Do this for MULTIPLE realizations for each conncetivity parameter. Find hamming distnace for this network over several intiial conditions
    """
    color = iter(cm.rainbow(np.linspace(0, 1, 21)))
    arlist = []
    myRange = list(frange(1.5, 3.5, 0.1))
    for lam in myRange:
        arlist.append(SFGraph.avg_hamming_network(n, lam, ic, d, t, trials, networks))
    for ar, i in zip(arlist, myRange):
        c = next(color)
        Graph.plot_hamming(
            ar,
            color=c,
            label="lam = %f" % i,
            title="Average Hamming Distance SF n=%d, d=%0.2f, t=%d, trials=%d, r=%d"
            % (n, d, t, trials, networks),
        )
    # Also tack on the plot for the average of each
    Graph.clear_graph()
    meanAr = []
    for ar in arlist:
        # Only taking the mean after half time
        meanAr.append(np.mean(ar[t / 2 :]))
    Graph.plot_hamming(
        meanAr,
        color="b",
        label="Order Parameter vs. Control Parameter",
        xticks=myRange,
        xtitle="Control Parameter Lamda",
        ytitle="Order Parameter",
        title="N = %d, d = %0.2f, t=%d, trials=%d, r=%d" % (n, d, t, trials, networks),
    )


def graph_distributions():
    """
    Graph the distributions for SF and Homogenous networks
    """
    mySFGraph1 = SFGraph(10000, 1.5)
    mySFGraph2 = SFGraph(10000, 2.5)
    mySFGraph3 = SFGraph(10000, 3.5)

    mySFGraph1.plot_in_degree_distribution("Scale Free Graph n=10000, lambda=1.5")
    mySFGraph2.plot_in_degree_distribution("Scale Free Graph n=10000, lambda=2.5")
    mySFGraph3.plot_in_degree_distribution("Scale Free Graph n=10000, lambda=3.5")

    myHomGraph1 = HomGraph(10000, 5)
    myHomGraph2 = HomGraph(10000, 10)
    myHomGraph3 = HomGraph(10000, 20)

    myHomGraph1.plot_in_degree_distribution("Homogenous Graph n=10000, k=5")
    myHomGraph2.plot_in_degree_distribution("Homogenous Graph n=10000, k=10")
    myHomGraph3.plot_in_degree_distribution("Homogenous Graph n=10000, k=20")


def plot_spectrums():
    """
    Saving a record of some of the graphs i've simulated in the past
    """
    plot_avg_hamming_distances_hom_spectrum(1000, 0.5, 0.05, 1000, 10)

    plot_avg_network_hamming_distances_hom_spectrum(1000, 0.5, 0.05, 1000, 10, 20)

    plot_avg_hamming_distances_sf_spectrum(1000, 0.5, 0.05, 1000, 10)

    plot_avg_network_hamming_distances_sf_spectrum(1000, 0.5, 0.05, 1000, 10, 20)


def plot_hom_graph_examples():
    """
    For presentation, plot some examples of homogenous graphs
    """
    myHomGraph = HomGraph(1000, 20)
    nx.write_gexf(myHomGraph.graph, "HomGraphn1000k20.gexf")

    myHomGraph.plot_in_degree_distribution("Homogenous Graph n=1000, k=20")

    pre_config = myHomGraph.get_config()
    color = iter(cm.rainbow(np.linspace(0, 1, 8)))
    dAr = [0, 0.001, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    for i in dAr:
        d = i
        c = next(color)
        list1 = Graph.hamming_single_perturb(myHomGraph, d, 200)
        Graph.plot_hamming(
            list1, color=c, label="d=%0.3f" % d, title="Homogenous Graph n=1000, k=20"
        )
        myHomGraph.set_config(pre_config)
    Graph.clear_graph()
    homIC = Graph.avg_hamming_ic(myHomGraph, 0.5, 0.5, 200, 20)
    Graph.plot_hamming(
        homIC, title="Homogenous Graph n=1000, k=20, ic=20", ytitle="<H(t)>"
    )


def plot_sf_examples():
    """
    For presentation, plot some examples of SF graphs
    """
    mySFGraph = SFGraph(1000, 2.5)
    nx.write_gexf(mySFGraph.graph, "SFGraphn1000l25.gexf")
    mySFGraph.plot_in_degree_distribution("Scale Free Graph n=1000, lambda=2.5")
    pre_config = mySFGraph.get_config()
    color = iter(cm.rainbow(np.linspace(0, 1, 8)))
    dAr = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
    for i in dAr:
        d = i
        c = next(color)
        list1 = Graph.hamming_single_perturb(mySFGraph, d, 200)
        Graph.plot_hamming(
            list1,
            color=c,
            label="d=%0.3f" % d,
            title="Scale Free Graph n=1000, lambda=2.5",
        )
        mySFGraph.set_config(pre_config)
    Graph.clear_graph()
    sfIC = Graph.avg_hamming_ic(mySFGraph, 0.5, 0.05, 200, 20)
    Graph.plot_hamming(
        sfIC, title="Scale Free Graph n=1000, lambda=2.5, ic=20", ytitle="<H(t)>"
    )


def plot_average_connectivity():
    """
    For presentation, plot the average connectivity of an SF graph
    """
    trials = 1
    degree_ar = []
    degrees = 0
    xticks = np.linspace(1.5, 3.5, 21)
    for i in xticks:
        for j in range(trials):
            myNetwork = SFGraph(1000, i)
            degrees += np.mean(myNetwork.graph.out_degree().values())
        degrees = degrees * 1.0 / trials
        degree_ar.append(degrees)
        degrees = 0
    Graph.plot_hamming(
        degree_ar,
        title="<K> vs Scale Free Parameter",
        xtitle="Scale Free Parameter",
        ytitle="<K>",
        xticks=xticks,
        color="b",
    )


def test_time_to_update():
    myHomGraph = HomGraph(1000, 20)
    for i in range(1000):
        myHomGraph.update()


def test_time_to_make():
    myhomgraph = HomGraph(1000, 20)


def test_np():
    n = 10000
    k = 20
    # myGraph = np.zeros(n,n)
    # myGraph = np.random.uniform(-1, 1, (n,n))
    # myState = np.random.randint(0, 2, (1, n))
    testGraph = HomGraph(1000, 20)
    # myGraph = scipy.sparse.csr_matrix(testGraph.npgraph)
    # for i in range(100):
    # 	dotProd = np.dot(myState, myGraph)
    # 	# myState = [1 if i > 1 else 0 if i < 1 else myState[index]for index, i in np.ndenumerate(dotProd) ]
    # 	myState[dotProd > 0] = 1
    # 	myState[dotProd < 0] = 0
    for i in range(1000):
        testGraph.update()


def timing_functions():
    print(
        timeit.timeit(
            "test_time_to_make()", "from __main__ import test_time_to_make", number=10
        )
    )

    print(
        timeit.timeit(
            "test_time_to_update()",
            "from __main__ import test_time_to_update",
            number=10,
        )
    )

    print(timeit.timeit("test_np()", "from __main__ import test_np", number=1))


# plot_avg_network_hamming_distances_hom_spectrum(1000, 0.5, 0.05, 1000, 10, 20)
# plot_avg_network_hamming_distances_hom_spectrum(100, 0.5, 0.05, 1000, 10, 20)
# plot_avg_network_hamming_distances_sf_spectrum(1000, 0.5, 0.05, 1000, 10, 20)
