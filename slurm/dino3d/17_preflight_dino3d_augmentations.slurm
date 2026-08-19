#!/bin/bash
#SBATCH --job-name=dino3d-aug-preflight
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
ENV_FILE="${CODE_ROOT}/asparagus/.env.finetune"
test -f "${ENV_FILE}" || {
  echo "ERROR: ${ENV_FILE} is missing. Submit from the repository root." >&2
  exit 1
}

set -a
source "${ENV_FILE}"
set +a

ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
test -x "${PYTHON}"

export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${CODE_ROOT}/asparagus"
"${PYTHON}" -m asparagus.analysis.preflight_dino3d_augmentations
