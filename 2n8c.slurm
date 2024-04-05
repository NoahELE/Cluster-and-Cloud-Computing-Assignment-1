#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --time=0-00:60:00

# Load required modules
module purge
module load GCCcore/11.3.0 Python/3.10.4
module load GCC/11.3.0 OpenMPI/4.1.4 mpi4py

# Create and activate virtual environment
python -m venv ~/.venv
source ~/.venv/bin/activate

# Install addtional packages
pip install orjson

# Launch multiple process python code
echo "running with 2 node and 8 cores"
echo "processing file $1"
time srun python process_twitter_data.py $1
