#!/bin/bash
#SBATCH --job-name=tuke-swin-clsreg-analysis
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
ENV_FILE="${CODE_ROOT}/asparagus/.env.finetune"
test -f "${ENV_FILE}"
set -a
source "${ENV_FILE}"
set +a
: "${CLSREG_EXPERIMENT_ID:?Set CLSREG_EXPERIMENT_ID to the 30-element array job ID}"

ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
test -x "${PYTHON}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

NORMAL_ROOT="${TUKE_SWIN_CLSREG_NORMAL_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_clsreg_normal_cv}/experiment_${CLSREG_EXPERIMENT_ID}"
GIN_ROOT="${TUKE_SWIN_CLSREG_GIN_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_clsreg_gin_cv}/experiment_${CLSREG_EXPERIMENT_ID}"
REPORT_ROOT="${TUKE_SWIN_CLSREG_ANALYSIS_ROOT:-${ASPARAGUS_RESULTS}/tuke_swinunetr_clsreg_analysis}"
REPORT="${REPORT_ROOT}/normal_gin_${CLSREG_EXPERIMENT_ID}.txt"
mkdir -p "${REPORT_ROOT}"

ARGS=(
  --normal-root "${NORMAL_ROOT}"
  --gin-root "${GIN_ROOT}"
  --report "${REPORT}"
  --expected-folds 5
)
if [[ "${REQUIRE_COMPLETE:-true}" == true ]]; then
  ARGS+=(--require-complete)
fi
cd "${CODE_ROOT}/asparagus"
srun "${PYTHON}" -m asparagus.analysis.analyze_tuke_swinunetr_clsreg "${ARGS[@]}"
ls -lh "${REPORT}"
echo "TUKE_SWINUNETR_CLSREG_REPORT=${REPORT}"
