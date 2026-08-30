#!/bin/bash
#SBATCH --job-name=fomo26-t1-samples
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${CODE_ROOT}/FOMO26_INFARCT_SUBMISSION}"
SIF_PATH="${SIF_PATH:-${BUNDLE_ROOT}/build/fomo26_task1_tuke_swinunetr.sif}"
ENV_FILE="${CODE_ROOT}/asparagus/.env.finetune"
test -s "${ENV_FILE}"
set -a
source "${ENV_FILE}"
set +a

ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
DATA_ROOT="${INFARCT_DATA_ROOT:-${ASPARAGUS_DATA}/CLS002_FOMO26_Infarct}"
REPORT="${BUNDLE_ROOT}/reports/finetune_sample_check.json"
PER_CLASS="${INFARCT_CHECK_PER_CLASS:-2}"
FAST="${INFARCT_CHECK_FAST:-false}"
EXTRA=()
[[ "${FAST}" == true ]] && EXTRA+=(--fast)

test -x "${PYTHON}"
test -s "${SIF_PATH}"
test -d "${DATA_ROOT}"
export PYTHONUNBUFFERED=1

srun --kill-on-bad-exit=1 "${PYTHON}" "${BUNDLE_ROOT}/tools/check_finetune_cases.py" \
  --sif "${SIF_PATH}" \
  --data-root "${DATA_ROOT}" \
  --report "${REPORT}" \
  --per-class "${PER_CLASS}" \
  "${EXTRA[@]}"
echo "FINETUNE_SAMPLE_CHECK_FINISHED report=${REPORT}"

