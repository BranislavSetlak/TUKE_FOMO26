#!/bin/bash
#SBATCH --job-name=fomo-preprocess-one
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --account=perun2601396
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

: "${DATASET:?Please submit with --export=ALL,DATASET=CLS002 or similar}"

CODE_ROOT="/mnt/data/home/brseke961/FOMO26_project/FOMO26_code"
ENV_PATH="${CODE_ROOT}/fomo_env"

source "${ENV_PATH}/bin/activate"

set -a
source "${CODE_ROOT}/asparagus/.env.finetune"
set +a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Running on node: $(hostname)"
echo "Started at: $(date)"
echo "Dataset: ${DATASET}"
echo "ASPARAGUS_SOURCE=${ASPARAGUS_SOURCE}"
echo "ASPARAGUS_DATA=${ASPARAGUS_DATA}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"

mkdir -p "${ASPARAGUS_DATA}"

cd "${CODE_ROOT}/asparagus"

asp_process \
  --dataset "${DATASET}" \
  --save_as_tensor \
  --num_workers "${SLURM_CPUS_PER_TASK}"

echo "Finished at: $(date)"
