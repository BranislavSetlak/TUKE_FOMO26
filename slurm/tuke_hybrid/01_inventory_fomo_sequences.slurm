#!/bin/bash
#SBATCH --job-name=fomo-sequence-inventory
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="/mnt/home/brseke961/FOMO26_project/FOMO26_code"
PYTHON="${CODE_ROOT}/fomo_env/bin/python"
SCRIPT="${CODE_ROOT}/asparagus/asparagus/analysis/inventory_fomo_sequences.py"
SPLIT_JSON="/mnt/project/perun2601396/FOMO26_processed/baseline/PT902_FOMO300K_HF/split_99_01_00.json"
OUTPUT_DIR="/mnt/project/perun2601396/FOMO26_results/tuke_hybrid/sequence_inventory_${SLURM_JOB_ID}"

test -x "${PYTHON}"
test -f "${SCRIPT}"
test -f "${SPLIT_JSON}"
mkdir -p "${OUTPUT_DIR}"

echo "START=$(date --iso-8601=seconds)"
echo "NODE=$(hostname)"
echo "SPLIT_JSON=${SPLIT_JSON}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

export PYTHONPATH="/mnt/home/brseke961/FOMO26_project/FOMO26_code/asparagus${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

echo "PYTHONPATH=${PYTHONPATH}"

srun --kill-on-bad-exit=1 \
    "${PYTHON}" "${SCRIPT}" \
    --split-json "${SPLIT_JSON}" \
    --fold 0 \
    --output-dir "${OUTPUT_DIR}" \
    --check-files 200

echo "FINISHED=$(date --iso-8601=seconds)"
find "${OUTPUT_DIR}" -maxdepth 1 -type f -printf '%f\n' | sort
