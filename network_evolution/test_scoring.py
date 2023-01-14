#!/usr/bin/Python
"""
   test_scoring.py
   Sunday, Sept 1 2019
   

   In the past, I was seeing weird results in C when I tried to check our scoring function. It turns out, it is somewhat nonlinear.. The score increases up to some critical point as target and cycle length scale, then it decreases back down to some average around 0.6. This is strange... 
"""

import numpy as np
import matplotlib.pyplot as plt


def lcm(a, b):
    """Calculate the LCM between two numbers"""
    min_num = a if a < b else b
    for i in range(min_num, 0, -1):
        if ((a % i) == 0) and ((b % i) == 0):
            return a * b / i
    return a * b


def score_cycle(target, cycle):
    """Score the cycle in python"""
    # set max
    MAX_SET = 35
    targ_length = len(target)
    cycle_length = len(cycle)
    same_seq = 0
    lcm_val = int(lcm(targ_length, cycle_length))
    max_val = int(cycle_length * MAX_SET)
    iter_val = lcm_val if lcm_val < max_val else max_val
    best_score = 0
    # print("ITER VAL: ", iter_val)
    for q in range(iter_val):
        same_seq = 0
        score = 0
        for j in range(iter_val):
            same_seq += cycle[(q + j) % cycle_length] == target[(j) % targ_length]
        score = same_seq / iter_val
        if score > best_score:
            best_score = score
    return best_score


if __name__ == "__main__":
    # define random seq
    avg_cases = 100
    test_cases = range(0, 200, 5)
    results = []
    for targ_cyc_len in test_cases:
        targ_len = targ_cyc_len
        cyc_len = targ_cyc_len
        ar = np.zeros(avg_cases)
        for i in range(avg_cases):
            rnd_targ = np.random.randint(0, 2, targ_len)
            rnd_cycle = np.random.randint(0, 2, cyc_len)
            ar[int(i)] = score_cycle(rnd_targ, rnd_cycle)
            results.append(ar.mean())
        print(f"Targ len {targ_len}, cycle len {cyc_len}, fitness {ar.mean()}")
    plt.plot(test_cases, results)
    plt.show()
