#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=0-12:00:00

# Load required modules
module purge
module load Python/3.10.4
module load mpi4py/3.1.4

# Create and activate virtual environment
virtualenv ~/venvs/venv-3.10.4
source ~/venvs/venv-3.10.4/bin/activate

# Install addtional packages
pip install orjson

# Launch multiple process python code
echo "Searching for happiest hourt"
time srun -n 8 python3 happiest_hour.py
