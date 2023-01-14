#!/bin/bash
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-00:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared   # Partition to submit to
#SBATCH --mem=200           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o log_files/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e log_files/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# $sbatch --depend=afterok:[ID of jobs array] odyssey_scripts/consolidate.sh [file_name] [lower bound of range] [upper bound of range]
#Example usage: 
# $sbatch --depend=afterok:1234 odyssey_scripts/consolidate.sh [1234] 1 4

# Load appropriate modules 
module load python/3.6.3-fasrc01

# Launch the script using task id
python3 ~/network_evolution/odyssey_scripts/cleanup_files.py $1 $2 $3
