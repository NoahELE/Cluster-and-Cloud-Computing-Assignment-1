#!/bin/bash
#SBATCH --job-name="Processing Twitter Data"
#SBATCH --nodes=1
#SBATCH --ntasks=1
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

# Create and activate virtual environment
python -m venv ~/.venv > /dev/null
source ~/.venv/bin/activate > /dev/null
# Install addtional packages
python -m pip install --upgrade pip > /dev/null
pip install orjson > /dev/null

# Launch multiple process python code
echo "running with 1 node and 1 cores"
echo "processing file $1"
time srun python process_twitter_data.py $1

# Job monitor command to list the resource usage
my-job-stats -a -n -s
