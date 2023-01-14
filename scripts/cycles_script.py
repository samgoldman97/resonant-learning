#!/usr/bin/env python


from HomGraph import *
from SFGraph import *
from matplotlib.pyplot import cm
from matplotlib.mlab import frange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
import copy



n = 150
k = 20
r = 1000

f = open("AttractorTimes%d%d.txt" % (n, k), "w")
f.write("Homogeneous Networks: N = %d, k = %d\n" % (n, k))
f.write("Start of Cycle\tEnd of cycle\tLength of Cycle\n")
for q in range(r):
    graph = HomGraph(n, k)
    found = False 
    state_array = {}
    ctr = 0
    state_array[graph.get_config().tostring()] = ctr
    while not found: 
        if ctr % 10000 == 0:
            print ctr
        graph.update()
        ctr += 1
        current_state = graph.get_config().tostring()
        val = state_array.get(current_state)
        if val != None: 
            f.write("%d\t%d\t%d\n" % (val, ctr, (ctr - val)))
            print("%d\t%d\t%d" % (val, ctr, (ctr - val)))
            found = True
        state_array[current_state] = ctr
f.close()
