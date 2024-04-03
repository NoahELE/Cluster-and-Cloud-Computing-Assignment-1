# Cluster and Cloud Computing Assignment 1

## Instructions

Below are the instructions to run the code on SPARTAN.

1. clone the repository to SPARTAN with
   ````
   git clone git@github.com:NoahELE/Cluster-and-Cloud-Computing-Assignment-1.git```
   ````
2. cd into the directory with `cd Cluster-and-Cloud-Computing-Assignment-1`
3. submit the job with `sbatch <job name>.slurm`
   > `1n1c` stands for 1 node and 1 core, `1n8c` stands for 1 node and 8 cores, `2n8c` stands for 2 nodes and 8 core
4. check the job status with `squeue -j <job id>`
5. check the output with `cat <job name>.out`
