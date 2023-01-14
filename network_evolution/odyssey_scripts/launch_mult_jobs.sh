#!/bin/sh


extract_number () {
	echo $4 | awk '{print $NF}'
}


num_in_array=10
for upperbound in `seq 2 2 16`
do
	# Launch one job and clean it up afterward
	jobid=$(sbatch --export=SIMTYPE=8,CONSTHUB=1,LONGPERIOD=$upperbound \
					--array=1-$num_in_array odyssey_scripts/run_jobs_main_fig.sh)
	job_id=$(extract_number $jobid)
	sbatch --depend=afterok:$job_id odyssey_scripts/consolidate.sh $job_id 1 $num_in_array


	jobid=$(sbatch --export=SIMTYPE=8,CONSTHUB=0,LONGPERIOD=$upperbound \
					--array=1-$num_in_array odyssey_scripts/run_jobs_main_fig.sh)
	job_id=$(extract_number $jobid)
	sbatch --depend=afterok:$job_id odyssey_scripts/consolidate.sh $job_id 1 $num_in_array


	jobid=$(sbatch --export=SIMTYPE=4,CONSTHUB=0,LONGPERIOD=$upperbound \
					--array=1-$num_in_array odyssey_scripts/run_jobs_main_fig.sh)
	job_id=$(extract_number $jobid)
	sbatch --depend=afterok:$job_id odyssey_scripts/consolidate.sh $job_id 1 $num_in_array

done
