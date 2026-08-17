#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --job-name=lifu_gm2_sweep
#SBATCH --output=lifu_gm2_sweep.%j.out
#SBATCH --error=lifu_gm2_sweep.%j.err
#SBATCH --account=              # <-- fill in your TACC allocation

set -euo pipefail

module load python3/3.11

pip install --user numpy scipy

cd "$SLURM_SUBMIT_DIR"

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

python3 run_sweep.py
python3 analyze_results.py

echo "Job finished: $(date)"
