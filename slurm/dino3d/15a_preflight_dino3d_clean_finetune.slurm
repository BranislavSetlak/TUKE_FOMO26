#!/bin/bash
#SBATCH --job-name=dino3d-clean-preflight
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
ENV_FILE="${CODE_ROOT}/asparagus/.env.finetune"
test -f "${ENV_FILE}" || {
  echo "ERROR: ${ENV_FILE} is missing. cd to the repository root before sbatch." >&2
  exit 1
}

set -a
source "${ENV_FILE}"
set +a

ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
PRETRAIN_CKPT="${DINO3D_PRETRAIN_CKPT:-${SHARED_ROOT}/FOMO26_checkpoints/dinov3_3d_stage1_71494_last.ckpt}"
SCRIPT="${CODE_ROOT}/asparagus/asparagus/analysis/preflight_dino3d_clean_finetune.py"

test -x "${PYTHON}"
test -s "${PRETRAIN_CKPT}"
test -f "${SCRIPT}"

export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "PREFLIGHT_START time=$(date --iso-8601=seconds) checkpoint=${PRETRAIN_CKPT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
srun "${PYTHON}" "${SCRIPT}" --checkpoint "${PRETRAIN_CKPT}"
echo "PREFLIGHT_FINISHED time=$(date --iso-8601=seconds)"
