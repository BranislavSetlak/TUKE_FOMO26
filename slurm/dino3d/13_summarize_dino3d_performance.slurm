#!/bin/bash
#SBATCH --job-name=dino3d-performance
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="/mnt/home/brseke961/FOMO26_project/FOMO26_code"
PYTHON="${CODE_ROOT}/fomo_env/bin/python"
MODEL_ROOT="/mnt/project/perun2601396/FOMO26_models/dino3d_finetuning"
OUTPUT_DIR="/mnt/project/perun2601396/FOMO26_results/dino3d/downstream_performance"

test -x "${PYTHON}"
test -d "${MODEL_ROOT}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_DIR}"
"${PYTHON}" -m asparagus.analysis.summarize_downstream_performance \
  --root "${MODEL_ROOT}" \
  --output-dir "${OUTPUT_DIR}"

find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.csv' -printf '%s bytes  %p\n' | sort
