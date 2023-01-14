"""
	sighelp.py

	Sam Goldman
	July 24 2017

	Contains functions used to assist in the spectral analysis of boolean threshold network arrays

"""
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import signal
import scipy as scp

from Graph import *

# Thursday, July 6 (Updated 1/8/18)
transient_cutoff = 1000
nperseg = 400
zero_replace = 10**-7


#########################################################
# Auxilary Functions
#########################################################


def get_dist_sort(my_graph, hub_node=None):
    """
    Simple aux function to help arg sort a graph by distance
    """
    if hub_node is None:
        hub_node = my_graph.find_hub()
    # Get the shortest path
    shortest_path = my_graph.get_shortest_path()
    arg_srt = np.argsort(shortest_path[:, hub_node])
    dist = shortest_path[arg_srt, hub_node]
    dist[dist == np.inf] = -1  # Set the unreachable nodes as -1
    return arg_srt, dist


#########################################################
# Generate new tables
#########################################################


def get_psd_table(table, transient_cutoff=transient_cutoff, normalize=True):
    """
    Get the power spectrum of each column and store in a new matirx.  Take the power spectrum after time 200
    Now average the time sequence data first...
    """
    time, nodes = table.shape
    new_time = time - transient_cutoff
    pow_table = None
    my_window = signal.get_window(window="hamming", Nx=new_time)

    for i in range(nodes):
        # Normalization constant
        # If the normalize argument is set to true, calculate the standar dev
        if normalize:
            norm_const = np.std(table[transient_cutoff:, i])

        # If
        if not normalize or norm_const == 0:
            norm_const = 1

        f, Pxx_den = signal.periodogram(
            table[transient_cutoff:, i] / norm_const, window=my_window
        )
        if pow_table is None:
            pow_table = np.zeros((f.size, nodes))

        # Old normalization code
        # Normalize:
        # if np.std(Pxx_den) != 0:
        # 	pass
        # 	# Pxx_den /= np.std(Pxx_den)
        pow_table[:, i] = Pxx_den
    return f, pow_table


def avg_spectrum_tbl(
    graph,
    node_index,
    period,
    time,
    trials,
    transient_cutoff=transient_cutoff,
    normalize=True,
):
    """
    Take the power spectrum of many initial conditions and average them
    """
    ic = np.copy(graph.get_config())
    tbl = graph.oscillate_update(node_index=node_index, period=period, time=time)
    f, return_table = get_psd_table(
        table=tbl, transient_cutoff=transient_cutoff, normalize=normalize
    )
    return_table /= float(trials)
    for i in range(trials - 1):
        graph.random_config()
        tbl = graph.oscillate_update(node_index=node_index, period=period, time=time)
        f, psd_tbl = get_psd_table(
            table=tbl, transient_cutoff=transient_cutoff, normalize=normalize
        )  # Already normalized
        psd_tbl /= float(trials)
        return_table += psd_tbl
    graph.set_config(np.copy(ic))
    return f, return_table


def avg_spectrum_tbl_ctrl(
    graph, time, trials, transient_cutoff=transient_cutoff, normalize=True
):
    """
    Take the power spectrum of many initial conditions and average them
    """
    ic = np.copy(graph.get_config())
    tbl = graph.update_return_table(time=time)
    f, return_table = get_psd_table(
        table=tbl, transient_cutoff=transient_cutoff, normalize=normalize
    )
    return_table /= float(trials)
    for i in range(trials - 1):
        graph.random_config()
        tbl = graph.update_return_table(time=time)
        f, psd_tbl = get_psd_table(
            table=tbl, transient_cutoff=transient_cutoff, normalize=normalize
        )
        psd_tbl /= float(trials)
        return_table += psd_tbl
    graph.set_config(np.copy(ic))
    return f, return_table


def avg_activity_spectrum(
    graph, hub_index, time, trials, start_period, end_period, step_period
):
    """
    Make a table with the avg activity for each node for a spectrum of frequencies
    """
    test_periods = range(start_period, end_period, step_period)
    results = np.zeros((len(test_periods), graph.nodes.size))
    for index, per in enumerate(test_periods):
        avg_act = np.average(
            a=graph.avg_activity_table_oscil(
                time=time, trials=trials, node=hub_index, period=per
            ),
            axis=0,
        )
        results[index, :] = avg_act
    return results


#########################################################
# Analyze one single node's responses
#########################################################


def single_node_psd_response(
    graph,
    hub,
    start_period,
    end_period,
    out_node,
    step_period=1,
    time=1000,
    trials=20,
    transient_cutoff=transient_cutoff,
    normalize=True,
):
    """
    For analyzing a single node; build a heatmap with the powerspectrum at each frequency
    Plot this in LogLog
    """
    test_periods = range(start_period, end_period, step_period)
    # Get a table
    # freq_ar = np.zeros(len(test_periods))
    for index, per in enumerate(test_periods):
        avg_act = graph.avg_activity_table_oscil(
            time=time, trials=trials, node=hub, period=per
        )
        _f, avgtbl = get_psd_table(
            avg_act, transient_cutoff=transient_cutoff, normalize=normalize
        )
        if index == 0:
            results = np.zeros((_f.size, len(test_periods)))
        spectra = avgtbl[:, out_node]
        hub_freq_index = np.argmax(avgtbl[:, hub])
        # Normalize by dividing by the hub's spectra
        results[:, index] = (spectra + zero_replace) / avgtbl[hub_freq_index, hub]
    # df = pd.DataFrame(results, index = _f)
    return (_f, results)


#########################################################
# Response to a variety of V_0
#########################################################


def freq_range_response(
    graph,
    hub,
    start_period,
    end_period,
    step_period=1,
    time=1000,
    trials=20,
    normalize=True,
):
    """
    Build the single frequency response for different signals
    This uses many initial conditions and averages them

    AVG PSD NOT ACTIVITY
    """
    test_periods = range(start_period, end_period, step_period)
    results = np.zeros((len(test_periods), graph.nodes.size))
    freq_ar = np.zeros(len(test_periods))
    for index, per in enumerate(test_periods):
        _f, avgtbl = avg_spectrum_tbl(
            graph,
            node_index=hub,
            period=per,
            time=time,
            trials=trials,
            normalize=normalize,
        )
        hub_freq_index = np.argmax(avgtbl[:, hub])
        freq_ar[index] = _f[hub_freq_index]  # Save an ar for the frequencies we look at
        # Normalized by dividing by the hub frequency
        results[index, :] = (avgtbl[hub_freq_index, :] + zero_replace) / (
            avgtbl[hub_freq_index, hub]
        )

    df = pd.DataFrame(results, index=test_periods)
    return df


def freq_range_response_ctrl(
    graph,
    start_period,
    end_period,
    step_period=1,
    time=1000,
    trials=20,
    transient_cutoff=transient_cutoff,
    normalize=True,
):
    """
    Get PSD(V_0) for each node at V_0 high frequency when the graph DOES NOT OSCILLATE
    each row represents a different v_0 and each column represents a node

    AVERAGE PSD NOT ACTIVITY
    """
    test_periods = range(start_period, end_period, step_period)
    results = np.zeros((len(test_periods), graph.nodes.size))
    freq_ar = np.zeros(len(test_periods))
    for index, per in enumerate(test_periods):
        _f, avgtbl = avg_spectrum_tbl_ctrl(
            graph=graph, time=time, trials=trials, normalize=normalize
        )
        ctr_seq = Graph.freq_sequence(per, time)

        new_time = time - transient_cutoff
        my_window = signal.get_window(window="hamming", Nx=new_time)

        fctrl, psdctrl = signal.periodogram(
            ctr_seq[transient_cutoff:], window=my_window
        )
        hub_freq_index = np.argmax(psdctrl)
        # Can we normalize this
        results[index, :] = (avgtbl[hub_freq_index, :] + zero_replace) / (
            psdctrl[hub_freq_index]
        )

    df = pd.DataFrame(results, index=test_periods)
    return df


def freq_response_avg_activity(
    graph,
    hub,
    start_period,
    end_period,
    step_period=1,
    time=1000,
    trials=20,
    transient_cutoff=transient_cutoff,
    normalize=True,
):
    """
    Get PSD(V_0) for each node at a number of V_0
    each row represents a different v_0 and each column represents a node
    Uses average activity
    """
    test_periods = range(start_period, end_period, step_period)
    results = np.zeros((len(test_periods), graph.nodes.size))
    freq_ar = np.zeros(len(test_periods))
    for index, per in enumerate(test_periods):
        avg_act = graph.avg_activity_table_oscil(
            time=time, trials=trials, node=hub, period=per
        )
        _f, avgtbl = get_psd_table(
            avg_act, transient_cutoff=transient_cutoff, normalize=normalize
        )
        hub_freq_index = np.argmax(avgtbl[:, hub])
        # Normalized by dividing by the hub frequency
        results[index, :] = (avgtbl[hub_freq_index, :] + zero_replace) / (
            avgtbl[hub_freq_index, hub]
        )
    df = pd.DataFrame(results, index=test_periods)
    return df


def freq_response_avg_activity_ctrl(
    graph,
    start_period,
    end_period,
    step_period=1,
    time=1000,
    trials=20,
    transient_cutoff=transient_cutoff,
    normalize=True,
):
    """
    Get PSD(V_0) for each node at V_0 high frequency when the graph DOES NOT OSCILLATE
    each row represents a different v_0 and each column represents a node
    """
    test_periods = range(start_period, end_period, step_period)
    results = np.zeros((len(test_periods), graph.nodes.size))
    freq_ar = np.zeros(len(test_periods))
    for index, per in enumerate(test_periods):
        avg_act = graph.avg_activity_table(time=time, trials=trials)
        _f, avgtbl = get_psd_table(
            avg_act, transient_cutoff=transient_cutoff, normalize=normalize
        )

        ctr_seq = Graph.freq_sequence(per, time)

        new_time = time - end_period
        my_window = signal.get_window(window="hamming", Nx=new_time)
        fctrl, psdctrl = signal.periodogram(ctr_seq[end_period:], window=my_window)
        hub_freq_index = np.argmax(psdctrl)
        # Can we normalize this
        results[index, :] = (avgtbl[hub_freq_index, :] + zero_replace) / (
            psdctrl[hub_freq_index]
        )

    df = pd.DataFrame(results, index=test_periods)
    return df


#########################################################
# Plot functions
#########################################################


def heatmap_logy(
    freq,
    power,
    xlabel="Node",
    ylabel="Frequency",
    collabel="Power",
    title="Power Spectrum by Node",
    dist_ar=None,
    cm=None,
    low=None,
    high=None,
):

    """
    Plot the heatmap in logy and log color
    Intended to plot the power spectrum by each node
    """

    if freq[0] == 0:
        freq[0] += zero_replace  # Set the smallest frequency

    x_size = power.shape[1]

    # Create grid
    x, y = np.meshgrid(np.arange(x_size), freq)

    if cm is None:
        cm = plt.cm.YlOrRd

    if low == None:
        low = np.min(power)
    if high == None:
        high = max(np.max(power), 1)

    plt.pcolormesh(x, y, power, norm=LogNorm(vmin=low, vmax=high), cmap=cm)

    cbar = plt.colorbar(label=collabel)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ax = plt.gca()
    ax.invert_yaxis()

    plt.yscale("log")
    ax.set_ylim(np.min(freq[1:]), np.max(freq) + 0.02)

    ax.set_xlim(-0.1, x_size + 0.1)

    if dist_ar is not None:
        uniq_dist = set(dist_ar)
        xticks = []
        xtick_labels = []
        for i in uniq_dist:
            xticks.append((np.argmax(dist_ar == i)))
            xtick_labels.append(int(i))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation="vertical")
        ax.set_xlabel("Distance from Hub")

    # if (dist_ar is not None):
    # 	new_labs = [dist_ar[int(i)] if i < dist_ar.size else dist_ar[-1] for i in ax.get_xticks()]
    # 	ax.set_xticklabels(new_labs)
    # 	ax.set_xlabel('Distance from Hub')

    return ax


def heatmap_logy_filters(
    df,
    xlabel="Node",
    ylabel=r"$v_0$",
    collabel="Power($v_0$)",
    title=r"Spectrum response to oscillation frequency $v_0$ at frequency $v_0$",
    dist_ar=None,
    high=None,
    low=None,
    cm=None,
):
    """
    Plot the heatmap in logy and log color
    Used to plot the power of V_0 at V_0;
    We could try taking the Coherence at V_0
    """
    freq = [i**-1 for i in df.index.values]
    y_start = np.min(freq)
    x_size = df.shape[1]
    x, y = np.meshgrid(np.arange(x_size), freq)
    values = df.values
    if high is None:
        high = 1
    if low is None:
        low = 10**-3
    if cm is None:
        cm = plt.cm.YlOrRd
    plt.pcolormesh(x, y, values, norm=LogNorm(vmin=low, vmax=high), cmap=cm, snap=True)
    cbar = plt.colorbar(label=collabel)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.yscale("log")
    ax.set_ylim(np.min(freq), np.max(freq))
    ax.set_xlim(-0.01, df.shape[1])
    if dist_ar is not None:
        new_labs = [
            dist_ar[i] if i < dist_ar.size else dist_ar[-1] for i in ax.get_xticks()
        ]
        ax.set_xticklabels(new_labs)
        ax.set_xlabel("Distance from Hub")
    return ax


def heatmap_loglog(
    freq,
    power,
    per_oscil,
    xlabel="Oscillation Frequency",
    ylabel="PSD",
    collabel="Power",
    low=10**-3,
    high=1,
    cm=None,
    title="Single Node PSD Response",
):
    """
    Plot the heatmap in logy and logx and log color
    Used to plot the frequency response of a single node
    """
    # We need a log scale in the Y;
    freq_oscil = [i**-1 for i in per_oscil]

    period_axis = np.copy(freq)
    period_axis[0] += zero_replace  # Placeholder
    x_size = power.shape[1]
    x, y = np.meshgrid(freq_oscil, period_axis)

    if cm is None:
        cm = plt.cm.YlOrRd
    plt.pcolormesh(x, y, (power), norm=LogNorm(vmin=low, vmax=high), cmap=cm, snap=True)
    cbar = plt.colorbar(label=collabel)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ax = plt.gca()
    ax.invert_yaxis()
    plt.yscale("log")
    ax.set_ylim(np.min(period_axis[1:]), np.max(period_axis))

    ax.set_xscale("log")
    ax.set_xlim(np.min(freq_oscil), np.max(freq_oscil))

    return ax
