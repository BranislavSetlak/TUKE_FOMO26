#!/bin/bash
#SBATCH --job-name=fomo26-t1-export
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${CODE_ROOT}/FOMO26_INFARCT_SUBMISSION}"
ENV_FILE="${CODE_ROOT}/asparagus/.env.finetune"
test -s "${ENV_FILE}"
set -a
source "${ENV_FILE}"
set +a

ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
test -x "${PYTHON}"
test -d "${BUNDLE_ROOT}/container"
export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
: "${CLSREG_EXPERIMENT_ID:?Export CLSREG_EXPERIMENT_ID to the completed cls/reg array job ID}"

VARIANT="${INFARCT_VARIANT:-auto}"
NORMAL_ROOT="${TUKE_SWIN_CLSREG_NORMAL_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_clsreg_normal_cv}/experiment_${CLSREG_EXPERIMENT_ID}"
GIN_ROOT="${TUKE_SWIN_CLSREG_GIN_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_clsreg_gin_cv}/experiment_${CLSREG_EXPERIMENT_ID}"
WEIGHTS_DIR="${BUNDLE_ROOT}/container/weights"
REPORT="${BUNDLE_ROOT}/reports/exported_weights_check.json"
PREPROCESS_REPORT="${BUNDLE_ROOT}/reports/preprocessing_equivalence.json"
DATA_ROOT="${INFARCT_DATA_ROOT:-${ASPARAGUS_DATA}/CLS002_FOMO26_Infarct}"

echo "EXPORT_START experiment=${CLSREG_EXPERIMENT_ID} requested_variant=${VARIANT}"
echo "normal_root=${NORMAL_ROOT}"
echo "gin_root=${GIN_ROOT}"

srun --kill-on-bad-exit=1 "${PYTHON}" "${BUNDLE_ROOT}/tools/export_infarct_ensemble.py" \
  --normal-root "${NORMAL_ROOT}" \
  --gin-root "${GIN_ROOT}" \
  --variant "${VARIANT}" \
  --output-dir "${WEIGHTS_DIR}" \
  --container-code "${BUNDLE_ROOT}/container"

srun --kill-on-bad-exit=1 "${PYTHON}" "${BUNDLE_ROOT}/tools/check_exported_weights.py" \
  --weights-dir "${WEIGHTS_DIR}" \
  --container-code "${BUNDLE_ROOT}/container" \
  --report "${REPORT}"

srun --kill-on-bad-exit=1 "${PYTHON}" "${BUNDLE_ROOT}/tools/check_preprocessing_equivalence.py" \
  --data-root "${DATA_ROOT}" \
  --container-code "${BUNDLE_ROOT}/container" \
  --report "${PREPROCESS_REPORT}" \
  --cases 3

BUNDLE_ROOT="${BUNDLE_ROOT}" REQUIRE_WEIGHTS=true "${BUNDLE_ROOT}/scripts/00_static_preflight.sh"
echo "EXPORT_FINISHED weights=${WEIGHTS_DIR} report=${REPORT} preprocessing_report=${PREPROCESS_REPORT}"
