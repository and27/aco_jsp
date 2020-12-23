#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=48
#SBATCH --array=24,16,8,4,2,1
#SBATCH --mail-user=jorge.banda@yachaytech.edu.ec
#SBATCH --output=/home/jorge.banda/aco-test/aco_jsp/results/abz6/100ants_abz6_%a.out
#SBATCH --mail-type=ALL

srun  --mpi=pmi2 -n $SLURM_ARRAY_TASK_ID python3 main.py --parallel &
srun  --mpi=pmi2 -n $SLURM_ARRAY_TASK_ID python3 main.py --parallel 
srun  --mpi=pmi2 -n $SLURM_ARRAY_TASK_ID python3 main.py --parallel &
srun  --mpi=pmi2 -n $SLURM_ARRAY_TASK_ID python3 main.py --parallel 
srun  --mpi=pmi2 -n $SLURM_ARRAY_TASK_ID python3 main.py --parallel &
srun  --mpi=pmi2 -n $SLURM_ARRAY_TASK_ID python3 main.py --parallel 
wait
