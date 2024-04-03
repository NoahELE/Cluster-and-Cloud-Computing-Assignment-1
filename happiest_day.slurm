#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=2
#SBATCH --time=0-12:00:00

# Load required modules
module purge
module load GCCcore/11.3.0 Python/3.10.4
module load GCC/11.3.0 OpenMPI/4.1.4 mpi4py

# Create and activate virtual environment
python3 -m venv ~/.venv
source ~/.venv/bin/activate

# Install addtional packages
pip install orjson

# Launch multiple process python code
echo "searching for happiest day..."
time srun -n 8 python3 happiest_day.py
