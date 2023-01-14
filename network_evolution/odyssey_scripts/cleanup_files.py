#!/usr/bin/python
"""
Sam Goldman
Jul 19, 2018 

cleanup_files.py 

Simple python file to average the results of serial runs from Odyssey

Usage: python3 cleanup_files.py [name_of_file_without_number] [lower bound of range] [upperbound of range, inclusive] 
python3 clceanup_files.py 18-07-19-15-23-14 0 10
"""


import numpy as np
import pickle
import sys
import os, os.path
import fnmatch

if len(sys.argv) != 4:
    raise Exception(
        "Error: See usage of the file.  You must supply 3 command line arguments:\n python3 cleanup_files.py [keyword id start of file] [lower bound of range] [upperbound of range, inclusive]"
    )


outfile_key_word = sys.argv[1]
new_outfile = None
for file in os.listdir("."):
    if fnmatch.fnmatch(file, f"*{outfile_key_word}*"):
        index = file.find(str(outfile_key_word))
        new_outfile = file[index:]
        break

# If we never found this file
if not new_outfile:
    raise Exception(f"Could not find file that has keyword {outfile_key_word}")


lower_bound = int(sys.argv[2])
upper_bound = int(sys.argv[3])

# The array of all the trials to return...
aggregated_array = None
num_trials = 0
scores_list = []
datum = None
for i in range(lower_bound, upper_bound + 1):
    current_file = f"{i}{new_outfile}"
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
