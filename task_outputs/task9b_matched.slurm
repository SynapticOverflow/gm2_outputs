#!/bin/bash
#SBATCH -J gm2_matched
#SBATCH -o /home1/11502/kartheek_nekkanti/gm2_outputs/task_outputs/task9b_matched.%j.out
#SBATCH -e /home1/11502/kartheek_nekkanti/gm2_outputs/task_outputs/task9b_matched.%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A IRI23021

set -euo pipefail

cd /home1/11502/kartheek_nekkanti/gm2_outputs/task_outputs

echo "host=$(hostname)  job=${SLURM_JOB_ID}  start=$(date -u +%FT%TZ)"
echo "model sha256 BEFORE:"
sha256sum ../gm2_analysis/gm2_model_v2.py

python3 task9b_matched_benchmark.py

echo "model sha256 AFTER:"
sha256sum ../gm2_analysis/gm2_model_v2.py
echo "end=$(date -u +%FT%TZ)"
