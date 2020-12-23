#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=64
#SBATCH --array=64,56,48,40,32,24
#SBATCH --mail-user=jorge.banda@yachaytech.edu.ec
#SBATCH --output=/home/jorge.banda/aco-test/aco_jsp/results/exp1/10ants_%a.out
#SBATCH --mail-type=ALL

srun  --mpi=pmi2 -n $SLURM_ARRAY_TASK_ID python3 main.py --parallel &
srun  --mpi=pmi2 -n $SLURM_ARRAY_TASK_ID python3 main.py --parallel 
srun  --mpi=pmi2 -n $SLURM_ARRAY_TASK_ID python3 main.py --parallel &
srun  --mpi=pmi2 -n $SLURM_ARRAY_TASK_ID python3 main.py --parallel 
srun  --mpi=pmi2 -n $SLURM_ARRAY_TASK_ID python3 main.py --parallel &
srun  --mpi=pmi2 -n $SLURM_ARRAY_TASK_ID python3 main.py --parallel 
wait
