#!/bin/bash
#SBATCH --job-name=tuke-hybrid-preflight
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="/mnt/home/brseke961/FOMO26_project/FOMO26_code"
SHARED_ROOT="/mnt/project/perun2601396"
PYTHON="${CODE_ROOT}/fomo_env/bin/python"
ENV_FILE="${CODE_ROOT}/asparagus/.env.pretrain"
SCRIPT="${CODE_ROOT}/asparagus/asparagus/analysis/preflight_tuke_hybrid.py"
SPLIT_JSON="${SHARED_ROOT}/FOMO26_processed/baseline/PT902_FOMO300K_HF/split_99_01_00.json"
MAPPING_YAML="${CODE_ROOT}/asparagus/configs/model/tuke_sequence_classes.yaml"
OUTPUT_DIR="${SHARED_ROOT}/FOMO26_results/tuke_hybrid/preflight_${SLURM_JOB_ID}"

test -x "${PYTHON}"
test -f "${ENV_FILE}"
test -f "${SCRIPT}"
test -f "${SPLIT_JSON}"
test -f "${MAPPING_YAML}"
mkdir -p "${OUTPUT_DIR}"

set -a
source "${ENV_FILE}"
set +a

export PYTHONPATH="${CODE_ROOT}/asparagus${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "START=$(date --iso-8601=seconds)"
echo "NODE=$(hostname)"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

srun --kill-on-bad-exit=1 \
    "${PYTHON}" "${SCRIPT}" \
    --split-json "${SPLIT_JSON}" \
    --mapping-yaml "${MAPPING_YAML}" \
    --output-dir "${OUTPUT_DIR}" \
    --fold 0 \
    --expected-train 303144 \
    --expected-validation 3063 \
    --expected-raw-labels 25 \
    --check-files 28 \
    --patch-size 96 96 96 \
    --feature-size 48 \
    --device cuda

cd "${CODE_ROOT}/asparagus"
srun --kill-on-bad-exit=1 \
    "${PYTHON}" -m asparagus.pipeline.run.pretrain \
    --config-name projects/tuke_hybrid/pretrain/swinunetr_sequence_smoke \
    --cfg job --resolve \
    > "${OUTPUT_DIR}/resolved_smoke.yaml"
srun --kill-on-bad-exit=1 \
    "${PYTHON}" -m asparagus.pipeline.run.pretrain \
    --config-name projects/tuke_hybrid/pretrain/swinunetr_sequence_production \
    --cfg job --resolve \
    > "${OUTPUT_DIR}/resolved_production.yaml"

echo "FINISHED=$(date --iso-8601=seconds)"
