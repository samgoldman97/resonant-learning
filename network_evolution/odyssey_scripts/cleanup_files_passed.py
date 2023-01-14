#!/usr/bin/python
"""
Sam Goldman
Jul 19, 2018 

cleanup_files.py 

Simple python file to average the results of serial runs from Odyssey

Usage: python3 cleanup_files.py [new_outfile] [file_1] [file_2] [file_3...] ....[file_n] 
python3 clceanup_files.py 3_targets.p *3_targets*
"""


import numpy as np
import pickle
import sys
import os, os.path
import fnmatch


print("Number of args passed:", len(sys.argv))
print("Arg inputs:")
for arg in sys.argv[2:]:
    print(arg)

files = sys.argv[2:]
new_outfile = sys.argv[1]

# The array of all the trials to return...
aggregated_array = None
num_trials = 0
scores_list = []
datum = None

for current_file in files:
    if os.path.isfile(current_file):
        with open(current_file, "rb") as fp:
            datum = pickle.load(fp)
            # In case it gets run with multiple trials per each one..
            ind_trials = datum["scores"].shape[0]
            num_trials += ind_trials
            for j in range(ind_trials):
                scores_list.append(datum["scores"][j])

            os.remove(current_file)
            print("Removing file: ", current_file)
    else:
        print("Missing file: ", current_file)


if datum:
    # Now prepare the output!
    my_out = datum
    my_out_scores = np.array(scores_list)
    my_out["trials"] = num_trials
    my_out["scores"] = my_out_scores
    with open(new_outfile, "wb") as out_file:
        pickle.dump(my_out, out_file)
