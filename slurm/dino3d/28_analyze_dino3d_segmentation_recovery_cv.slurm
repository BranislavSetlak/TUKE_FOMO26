#!/bin/bash
#SBATCH --job-name=dino3d-seg-recovery-analysis
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

: "${EXPERIMENT_ID:?Set EXPERIMENT_ID to the original training array job ID}"
: "${VARIANT:?Set VARIANT to positive_sampling, anatomical_roi, unified_focal, or fn_tversky}"

ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
TASK_ARGS=(--tasks SEG009_FOMO26_Meningioma SEG010_FOMO26_TrigeminalNeuralgia)

case "${VARIANT}" in
  positive_sampling)
    MODEL_ROOT="${DINO3D_POSITIVE_SAMPLING_MODEL_ROOT:-${SHARED_ROOT}/FOMO26_models/dino3d_positive_sampling_cv}"
    RESULT_ROOT="${DINO3D_POSITIVE_SAMPLING_RESULT_ROOT:-${SHARED_ROOT}/FOMO26_results/dino3d_positive_sampling_cv}"
    ;;
  anatomical_roi)
    MODEL_ROOT="${DINO3D_ANATOMICAL_ROI_MODEL_ROOT:-${SHARED_ROOT}/FOMO26_models/dino3d_anatomical_roi_seg010_cv}"
    RESULT_ROOT="${DINO3D_ANATOMICAL_ROI_RESULT_ROOT:-${SHARED_ROOT}/FOMO26_results/dino3d_anatomical_roi_seg010_cv}"
    TASK_ARGS=(--tasks SEG010_FOMO26_TrigeminalNeuralgia)
    ;;
  unified_focal)
    MODEL_ROOT="${DINO3D_UNIFIED_FOCAL_MODEL_ROOT:-${SHARED_ROOT}/FOMO26_models/dino3d_unified_focal_cv}"
    RESULT_ROOT="${DINO3D_UNIFIED_FOCAL_RESULT_ROOT:-${SHARED_ROOT}/FOMO26_results/dino3d_unified_focal_cv}"
    ;;
  fn_tversky)
    MODEL_ROOT="${DINO3D_FN_TVERSKY_MODEL_ROOT:-${SHARED_ROOT}/FOMO26_models/dino3d_fn_tversky_cv}"
    RESULT_ROOT="${DINO3D_FN_TVERSKY_RESULT_ROOT:-${SHARED_ROOT}/FOMO26_results/dino3d_fn_tversky_cv}"
    ;;
  *)
    echo "ERROR: Unknown VARIANT=${VARIANT}." >&2
    exit 1
    ;;
esac

EXPERIMENT_ROOT="${MODEL_ROOT}/experiment_${EXPERIMENT_ID}"
REPORT="${RESULT_ROOT}/experiment_${EXPERIMENT_ID}/dino3d_${VARIANT}_cv_analysis.txt"
REQUIRE_COMPLETE="${REQUIRE_COMPLETE:-false}"
test -x "${PYTHON}"
test -d "${EXPERIMENT_ROOT}"

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
echo "DINO3D_SEGMENTATION_RECOVERY_ANALYSIS_START time=$(date --iso-8601=seconds)"
echo "variant=${VARIANT} experiment_root=${EXPERIMENT_ROOT} report=${REPORT}"
"${PYTHON}" -m asparagus.analysis.analyze_dino3d_clean_cv "${ARGS[@]}"
echo "DINO3D_SEGMENTATION_RECOVERY_ANALYSIS_FINISHED time=$(date --iso-8601=seconds)"
ls -lh "${REPORT}"
