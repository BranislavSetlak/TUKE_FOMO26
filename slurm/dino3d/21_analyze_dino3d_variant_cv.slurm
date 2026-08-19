#!/bin/bash
#SBATCH --job-name=dino3d-variant-analysis
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

# Submit from the repository root. The experiment ID is the original training
# array job ID, for example: EXPERIMENT_ID=74929 sbatch slurm/dino3d/16_*.slurm
CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
ENV_FILE="${CODE_ROOT}/asparagus/.env.finetune"
test -f "${ENV_FILE}" || {
  echo "ERROR: ${ENV_FILE} is missing. cd to the repository root before sbatch." >&2
  exit 1
}

set -a
source "${ENV_FILE}"
set +a

: "${EXPERIMENT_ID:?Set EXPERIMENT_ID to the original training array job ID, for example 74929}"
: "${VARIANT:?Set VARIANT to gin, gin_carvemix, or fp_loss}"

ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
TASK_ARGS=()
case "${VARIANT}" in
  gin)
    MODEL_ROOT="${DINO3D_GIN_MODEL_ROOT:-${SHARED_ROOT}/FOMO26_models/dino3d_gin_cv}"
    RESULT_ROOT="${DINO3D_GIN_RESULT_ROOT:-${SHARED_ROOT}/FOMO26_results/dino3d_gin_cv}"
    ;;
  gin_carvemix)
    MODEL_ROOT="${DINO3D_GIN_CARVEMIX_MODEL_ROOT:-${SHARED_ROOT}/FOMO26_models/dino3d_gin_carvemix_cv}"
    RESULT_ROOT="${DINO3D_GIN_CARVEMIX_RESULT_ROOT:-${SHARED_ROOT}/FOMO26_results/dino3d_gin_carvemix_cv}"
    TASK_ARGS=(--tasks SEG009_FOMO26_Meningioma SEG010_FOMO26_TrigeminalNeuralgia)
    ;;
  fp_loss)
    MODEL_ROOT="${DINO3D_FP_LOSS_MODEL_ROOT:-${SHARED_ROOT}/FOMO26_models/dino3d_fp_loss_cv}"
    RESULT_ROOT="${DINO3D_FP_LOSS_RESULT_ROOT:-${SHARED_ROOT}/FOMO26_results/dino3d_fp_loss_cv}"
    TASK_ARGS=(--tasks SEG009_FOMO26_Meningioma SEG010_FOMO26_TrigeminalNeuralgia)
    ;;
  *)
    echo "ERROR: Unknown VARIANT=${VARIANT}; use gin, gin_carvemix, or fp_loss." >&2
    exit 1
    ;;
esac
EXPERIMENT_ROOT="${MODEL_ROOT}/experiment_${EXPERIMENT_ID}"
REPORT="${RESULT_ROOT}/experiment_${EXPERIMENT_ID}/dino3d_${VARIANT}_cv_analysis.txt"
REQUIRE_COMPLETE="${REQUIRE_COMPLETE:-false}"

test -x "${PYTHON}"
test -d "${EXPERIMENT_ROOT}" || {
  echo "ERROR: Experiment directory not found: ${EXPERIMENT_ROOT}" >&2
  exit 1
}

export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

ARGS=(
  --experiment-root "${EXPERIMENT_ROOT}"
  --report "${REPORT}"
  --expected-folds 5
)
ARGS+=("${TASK_ARGS[@]}")
if [[ "${REQUIRE_COMPLETE}" == "true" ]]; then
  ARGS+=(--require-complete)
fi

cd "${CODE_ROOT}/asparagus"
echo "DINO3D_CV_ANALYSIS_START time=$(date --iso-8601=seconds)"
echo "variant=${VARIANT}"
echo "experiment_root=${EXPERIMENT_ROOT}"
echo "report=${REPORT}"
"${PYTHON}" -m asparagus.analysis.analyze_dino3d_clean_cv "${ARGS[@]}"
echo "DINO3D_CV_ANALYSIS_FINISHED time=$(date --iso-8601=seconds)"
ls -lh "${REPORT}"
