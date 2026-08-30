#!/bin/bash
#SBATCH --job-name=fomo26-rest-export
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${CODE_ROOT}/FOMO26_REMAINING_SUBMISSIONS}"
test -s "${CODE_ROOT}/asparagus/.env.pretrain"
test -s "${CODE_ROOT}/asparagus/.env.finetune"
SUBMITTED_CODE_ROOT="${CODE_ROOT}"
set -a
source "${CODE_ROOT}/asparagus/.env.pretrain"
set +a
PRETRAIN_MODELS_ROOT="${ASPARAGUS_MODELS}"
set -a
source "${CODE_ROOT}/asparagus/.env.finetune"
set +a
CODE_ROOT="${SUBMITTED_CODE_ROOT}"
ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
test -x "${PYTHON}"
export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

: "${CLSREG_EXPERIMENT_ID:?Set CLSREG_EXPERIMENT_ID to the completed job 14 array ID}"
COMMON_SEG_ID="${SEG_EXPERIMENT_ID:-}"
NORMAL_ID="${SEG_NORMAL_EXPERIMENT_ID:-${COMMON_SEG_ID}}"
GIN_ID="${SEG_GIN_EXPERIMENT_ID:-${COMMON_SEG_ID}}"
CARVEMIX_ID="${SEG_GIN_CARVEMIX_EXPERIMENT_ID:-${COMMON_SEG_ID}}"
if [[ -z "${NORMAL_ID}${GIN_ID}${CARVEMIX_ID}" ]]; then
  echo "ERROR: set SEG_EXPERIMENT_ID, or one or more SEG_*_EXPERIMENT_ID variables" >&2
  exit 1
fi

if [[ -n "${TUKE_HYBRID_PRETRAIN_CKPT:-}" ]]; then
  PRETRAIN_CKPT="${TUKE_HYBRID_PRETRAIN_CKPT}"
else
  PRETRAIN_CKPT="${PRETRAIN_MODELS_ROOT}/PT902_FOMO300K_HF/tuke_hybrid/production/run_id=${TUKE_HYBRID_RUN_ID:-production_8gpu_v1}/checkpoints/last.ckpt"
fi
test -s "${PRETRAIN_CKPT}"

NORMAL_SEG_ROOT="${TUKE_SWIN_NORMAL_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_normal_cv}/experiment_${NORMAL_ID:-missing}"
GIN_SEG_ROOT="${TUKE_SWIN_GIN_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_gin_cv}/experiment_${GIN_ID:-missing}"
CARVEMIX_SEG_ROOT="${TUKE_SWIN_GIN_CARVEMIX_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_gin_carvemix_cv}/experiment_${CARVEMIX_ID:-missing}"
NORMAL_CLSREG_ROOT="${TUKE_SWIN_CLSREG_NORMAL_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_clsreg_normal_cv}/experiment_${CLSREG_EXPERIMENT_ID}"
GIN_CLSREG_ROOT="${TUKE_SWIN_CLSREG_GIN_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_clsreg_gin_cv}/experiment_${CLSREG_EXPERIMENT_ID}"

srun --kill-on-bad-exit=1 "${PYTHON}" "${BUNDLE_ROOT}/tools/export_all_weights.py" \
  --common-code "${BUNDLE_ROOT}/containers/common" \
  --task2-normal-root "${NORMAL_SEG_ROOT}" --task2-gin-root "${GIN_SEG_ROOT}" --task2-gin-carvemix-root "${CARVEMIX_SEG_ROOT}" \
  --task4-normal-root "${NORMAL_SEG_ROOT}" --task4-gin-root "${GIN_SEG_ROOT}" --task4-gin-carvemix-root "${CARVEMIX_SEG_ROOT}" \
  --clsreg-normal-root "${NORMAL_CLSREG_ROOT}" --clsreg-gin-root "${GIN_CLSREG_ROOT}" \
  --task2-variant "${TASK2_VARIANT:-auto}" --task3-variant "${TASK3_VARIANT:-auto}" \
  --task4-variant "${TASK4_VARIANT:-auto}" --task5-variant "${TASK5_VARIANT:-auto}" \
  --pretrain-checkpoint "${PRETRAIN_CKPT}" --output-root "${BUNDLE_ROOT}/containers" \
  --report "${BUNDLE_ROOT}/reports/weight_export.json"

srun --kill-on-bad-exit=1 "${PYTHON}" "${BUNDLE_ROOT}/tools/check_preprocessing_equivalence.py" \
  --data-root "${ASPARAGUS_DATA}" --common-code "${BUNDLE_ROOT}/containers/common" \
  --report "${BUNDLE_ROOT}/reports/preprocessing_equivalence.json" --cases-per-task 2
BUNDLE_ROOT="${BUNDLE_ROOT}" REQUIRE_WEIGHTS=true "${BUNDLE_ROOT}/scripts/00_static_preflight.sh"
echo "ALL_EXPORTS_FINISHED"
