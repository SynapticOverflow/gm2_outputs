#!/bin/bash
# Usage: sbatch [-p development -t 00:20:00] task10_seed_sensitivity.slurm RUNSET N [OUTDIR]
#   RUNSET in {A,B,C,all}
#SBATCH -J gm2_seedsens
#SBATCH -o /home1/11502/kartheek_nekkanti/gm2_outputs/task_outputs/seed_sensitivity/logs/task10.%x.%j.out
#SBATCH -e /home1/11502/kartheek_nekkanti/gm2_outputs/task_outputs/seed_sensitivity/logs/task10.%x.%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 01:00:00
#SBATCH -A IRI23021

set -euo pipefail
RUNSET=${1:?RUNSET required}
N=${2:-256}
OUTDIR=${3:-N${N}}

cd /home1/11502/kartheek_nekkanti/gm2_outputs/task_outputs/seed_sensitivity
MODEL=../../gm2_analysis/gm2_model_v2.py

echo "host=$(hostname)  job=${SLURM_JOB_ID}  runset=${RUNSET}  N=${N}  outdir=${OUTDIR}  start=$(date -u +%FT%TZ)"
echo "model sha256 BEFORE:"; sha256sum "$MODEL"
sha256sum "$MODEL" | grep -q '^6693427fa909a0a9' || { echo "ABORT: model hash mismatch"; exit 2; }

python3 task10_seed_sensitivity.py "$RUNSET" "$N" "$OUTDIR"

echo "model sha256 AFTER:"; sha256sum "$MODEL"
sha256sum "$MODEL" | grep -q '^6693427fa909a0a9' || { echo "ABORT: model hash changed during run"; exit 3; }
echo "end=$(date -u +%FT%TZ)"
