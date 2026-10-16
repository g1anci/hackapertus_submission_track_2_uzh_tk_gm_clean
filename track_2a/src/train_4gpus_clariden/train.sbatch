#!/bin/bash
#SBATCH --job-name=apertus-full
#SBATCH --account=<your_cscs_group_name>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.log

ulimit -c 0
srun -ul --environment=apertus \
  papermill -k python3 --log-output syntetic_train_full_4gpu.ipynb out/run_${SLURM_JOB_ID}.ipynb
