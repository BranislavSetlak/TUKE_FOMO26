#!/bin/bash
#SBATCH --job-name=tuke-swin-clsreg-preflight
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --account=perun2601396
#SBATCH --array=0-2%3
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.err

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
SUBMITTED_CODE_ROOT="${CODE_ROOT}"
PRETRAIN_ENV_FILE="${CODE_ROOT}/asparagus/.env.pretrain"
FINETUNE_ENV_FILE="${CODE_ROOT}/asparagus/.env.finetune"
test -f "${PRETRAIN_ENV_FILE}"
test -f "${FINETUNE_ENV_FILE}"
set -a
source "${PRETRAIN_ENV_FILE}"
set +a
PRETRAIN_MODELS_ROOT="${ASPARAGUS_MODELS}"
set -a
source "${FINETUNE_ENV_FILE}"
set +a
CODE_ROOT="${SUBMITTED_CODE_ROOT}"

ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
test -x "${PYTHON}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

if [[ -n "${TUKE_HYBRID_PRETRAIN_CKPT:-}" ]]; then
  PRETRAIN_CKPT="${TUKE_HYBRID_PRETRAIN_CKPT}"
else
  DEFAULT_CKPT="${PRETRAIN_MODELS_ROOT}/PT902_FOMO300K_HF/tuke_hybrid/production/run_id=${TUKE_HYBRID_RUN_ID:-production_8gpu_v1}/checkpoints/last.ckpt"
  if [[ -s "${DEFAULT_CKPT}" ]]; then
    PRETRAIN_CKPT="${DEFAULT_CKPT}"
  else
    LATEST_LINE="$({
      find "${PRETRAIN_MODELS_ROOT}/PT902_FOMO300K_HF" \
        -type f -path '*/tuke_hybrid/production/run_id=*/checkpoints/last.ckpt' \
        -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | head -n 1)"
    PRETRAIN_CKPT="${LATEST_LINE#* }"
  fi
fi
test -s "${PRETRAIN_CKPT:-}" || {
  echo "ERROR: Set TUKE_HYBRID_PRETRAIN_CKPT to the final checkpoint." >&2
  exit 1
}

case "${SLURM_ARRAY_TASK_ID}" in
  0) DATASET=CLS002_FOMO26_Infarct; TASK_TYPE=classification ;;
  1) DATASET=REGR002_FOMO26_BrainAge; TASK_TYPE=regression ;;
  2) DATASET=CLS003_FOMO26_Polymicrogyria; TASK_TYPE=classification ;;
  *) exit 1 ;;
esac
OUTPUT_ROOT="${TUKE_SWIN_CLSREG_PREFLIGHT_ROOT:-${ASPARAGUS_RESULTS}/tuke_swinunetr_clsreg_preflight}"
OUTPUT="${OUTPUT_ROOT}/job_${SLURM_ARRAY_JOB_ID}/${DATASET}.json"
mkdir -p "$(dirname "${OUTPUT}")"
cd "${CODE_ROOT}/asparagus"
srun --kill-on-bad-exit=1 "${PYTHON}" -m asparagus.analysis.preflight_tuke_swinunetr_clsreg \
  --checkpoint "${PRETRAIN_CKPT}" \
  --task-root "${ASPARAGUS_DATA}/${DATASET}" \
  --task-type "${TASK_TYPE}" \
  --output "${OUTPUT}"
test -s "${OUTPUT}"
