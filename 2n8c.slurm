#!/bin/bash
#SBATCH --job-name="Processing Twitter Data"
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --time=0-1:00:00

# Check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Load required modules
module purge
module load GCCcore/11.3.0 Python/3.10.4
module load GCC/11.3.0 OpenMPI/4.1.4 mpi4py

# Launch multiple process python code
echo "running with 2 node and 8 cores"
echo "processing file $1"
time srun python process_twitter_data.py $1
