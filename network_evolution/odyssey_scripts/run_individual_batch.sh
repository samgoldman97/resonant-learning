#!/bin/bash
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-00:5          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=300           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid



module load gcc/7.1.0-fasrc01
module load python/3.6.3-fasrc01

# Build the cython file first
python3 setup.py build_ext --inplace
# Get current date time
DATE=`date '+%y-%m-%d-%H-%M-%S'`

# Launch the script using task id
python3 ~/network_evolution/main.py "testing_$DATE"
