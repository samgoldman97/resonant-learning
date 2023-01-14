#!/bin/bash
#SBATCH -J networks_ev_oscil_start # A single job name for the array
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # All cores on one machine
#SBATCH -p serial_requeue # Partition
#SBATCH --mem 5000 # Memory request (5Gb)
#SBATCH -t 4-00:00 # Maximum execution time (D-HH:MM)
#SBATCH -o log_files/networks_ev_%A_%a.out # Standard output
#SBATCH -e log_files/networks_ev_%A_%a.err # Standard error
#SBATCH --mail-type=END        # Email


#Example usage:
# $sbatch --array=1-3 odyssey_scripts/run_jobs.sh
# then run consolidate.sh through sbash

# Load appropriate modules
module load gcc/7.1.0-fasrc01
module load python/3.6.3-fasrc01

# Build the cython file first
python3 setup.py build_ext --inplace
# Get current date time
DATE=`date '+%y-%m-%d-%H-%M-%S'`


# Launch the script using task id
#python3 ~/network_evolution/main.py "${SLURM_ARRAY_TASK_ID}$DATE"
python3 ~/network_evolution/main.py --timestamp  "${SLURM_ARRAY_TASK_ID}${SLURM_ARRAY_JOB_ID}" --n 500 \
						--time 1000 --pop_size 50 --replicates 3 --network_type 1 \
                        --control_param 1.9 --simulation_type 4 \
                        --target_lengths 2 4 6 --target_node -1 --periods 2 4 6 \
                        --resonance_period -1 --prob_perturb 0 --constant_hub 0 --generations 10 \
                        --switch_time -1 
