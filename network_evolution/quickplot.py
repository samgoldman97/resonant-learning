import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys

sns.set(font_scale=4)
sns.set_context("paper", rc={"lines.linewidth": 4})


file_name = sys.argv[1]

with open(file_name, "rb") as fp:
    res = pickle.load(fp)

print("RESULTS: ", res)

plt.plot(np.arange(1, res["scores"].shape[1] + 1), 1 - np.mean(res["scores"], 0))
plt.xlabel("Generations")
plt.ylabel("1 - Fitness")
# plt.ylim([0,0.6])
plt.savefig("/Users/samgoldman/Desktop/temp.png", bbox_inches="tight")
plt.show()
