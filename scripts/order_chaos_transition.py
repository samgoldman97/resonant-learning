"""order_chaos_transition.py 

This script attempts to build order chaos phase transitions

"""

import matplotlib.pyplot as plt
import numpy as np
from resonant_learning.graphs import SFGraph


if __name__ == "__main__":
    n, ic, d, t, trials, networks = 50, 10, 0.05, 10000, 20, 5

    @classmethod
    def avg_hamming_ic(cls, size, lam, config_prob, perturb_frac, time, trials):
        """avg_hamming_ic."""
        graph1 = cls(size, lam, config_prob)
        return super(SFGraph, cls).avg_hamming_ic(
            graph1, config_prob, perturb_frac, time, trials
        )

    lambda_params = np.linspace(1.6, 3.0, 5)
    outputs = []
    for lam in lambda_params:
        print(f"Starting trial for lam {lam}")
        hamming_ic = SFGraph.avg_hamming_network(
            n,
            lam,
            config_prob=0.5,
            perturb_frac=d,
            time=t,
            trials=ic,
            networks=networks,
        )
        print(lam, hamming_ic, np.mean(hamming_ic))
        outputs.append(np.mean(hamming_ic))
    plt.plot(lambda_params, hamming_ic)
    plt.show()
