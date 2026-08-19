#!/bin/bash
#SBATCH --job-name=dino3d-epoch-tables
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

# Run from the repository root and supply both original training array IDs:
# GIN_JOB_ID=77093 CARVEMIX_JOB_ID=77094 sbatch slurm/dino3d/22_*.slurm
CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
ENV_FILE="${CODE_ROOT}/asparagus/.env.finetune"
test -f "${ENV_FILE}" || {
  echo "ERROR: ${ENV_FILE} is missing. cd to the repository root before sbatch." >&2
  exit 1
}

set -a
source "${ENV_FILE}"
set +a

: "${GIN_JOB_ID:?Set GIN_JOB_ID to the GIN training array job ID}"
: "${CARVEMIX_JOB_ID:?Set CARVEMIX_JOB_ID to the GIN+CarveMix training array job ID}"

ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
LOG_DIR="${FOMO_JOB_OUTPUTS:-${SHARED_ROOT}/FOMO26_job_outputs}"
OUTPUT_FILE="${OUTPUT_FILE:-${LOG_DIR}/dino3d_epoch_tables_gin${GIN_JOB_ID}_carvemix${CARVEMIX_JOB_ID}.txt}"
REQUIRE_ALL_LOGS="${REQUIRE_ALL_LOGS:-false}"

test -x "${PYTHON}" || {
  echo "ERROR: Python executable not found: ${PYTHON}" >&2
  exit 1
}
test -d "${LOG_DIR}" || {
  echo "ERROR: Job-output directory not found: ${LOG_DIR}" >&2
  exit 1
}

export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

ARGS=(
  --log-dir "${LOG_DIR}"
  --gin-job-id "${GIN_JOB_ID}"
  --carvemix-job-id "${CARVEMIX_JOB_ID}"
  --output "${OUTPUT_FILE}"
)
if [[ "${REQUIRE_ALL_LOGS}" == "true" ]]; then
  ARGS+=(--require-all-logs)
fi

cd "${CODE_ROOT}/asparagus"
echo "DINO3D_EPOCH_TABLES_START time=$(date --iso-8601=seconds)"
echo "gin_job_id=${GIN_JOB_ID}"
echo "carvemix_job_id=${CARVEMIX_JOB_ID}"
echo "output_file=${OUTPUT_FILE}"
"${PYTHON}" -m asparagus.analysis.summarize_dino3d_epoch_logs "${ARGS[@]}"
echo "DINO3D_EPOCH_TABLES_FINISHED time=$(date --iso-8601=seconds)"
ls -lh "${OUTPUT_FILE}"
echo "REPORT_PATH=${OUTPUT_FILE}"
