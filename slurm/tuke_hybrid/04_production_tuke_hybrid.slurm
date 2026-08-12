#!/bin/bash
#SBATCH --job-name=tuke-hybrid-train
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=30:00:00
#SBATCH --account=perun2601396
#SBATCH --requeue
#SBATCH --signal=USR1@300
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="/mnt/home/brseke961/FOMO26_project/FOMO26_code"
PYTHON="${CODE_ROOT}/fomo_env/bin/python"
ENV_FILE="${CODE_ROOT}/asparagus/.env.pretrain"
CONFIG="projects/tuke_hybrid/pretrain/swinunetr_sequence_production"

test -x "${PYTHON}"
test -f "${ENV_FILE}"

set -a
source "${ENV_FILE}"
set +a

test -f "${ASPARAGUS_DATA}/PT902_FOMO300K_HF/split_99_01_00.json"
mkdir -p "${ASPARAGUS_MODELS}" "${ASPARAGUS_RESULTS}"

export PYTHONPATH="${CODE_ROOT}/asparagus${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${CODE_ROOT}/asparagus"

echo "START=$(date --iso-8601=seconds)"
echo "NODE=$(hostname)"
echo "CONFIG=${CONFIG}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

srun --kill-on-bad-exit=1 \
    "${PYTHON}" -m asparagus.pipeline.run.pretrain \
    --config-name "${CONFIG}"

echo "FINISHED=$(date --iso-8601=seconds)"
