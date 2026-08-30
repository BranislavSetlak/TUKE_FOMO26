#!/bin/bash
#SBATCH --job-name=fomo26-real-sif
#SBATCH --partition=gpu_short
#SBATCH --account=perun2601396
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-4%3
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/fomo26-real-sif_%A_%a.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/fomo26-real-sif_%A_%a.err

set -euo pipefail

: "${CODE_ROOT:?Export CODE_ROOT before submitting}"

TEST_ROOT="${TEST_ROOT:-${CODE_ROOT}/FOMO26_REAL_SIF_TESTS}"
PYTHON="${CODE_ROOT}/fomo_env/bin/python"
DATA_ROOT="${ASPARAGUS_DATA:-/mnt/project/perun2601396/FOMO26_processed/baseline}"
SOURCE_ROOT="${ASPARAGUS_SOURCE:-/mnt/project/perun2601396/FOMO26_finetune}"
TASK1_SIF="${TASK1_SIF:-${CODE_ROOT}/FOMO26_INFARCT_SUBMISSION/build/fomo26_task1_tuke_swinunetr.sif}"
REMAINING_BUILD_ROOT="${REMAINING_BUILD_ROOT:-${CODE_ROOT}/FOMO26_REMAINING_SUBMISSIONS/build}"
CASES_PER_TASK="${CASES_PER_TASK:-5}"
TEST_SEED="${TEST_SEED:-260821}"
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT:-1800}"
TEST_RUN_ID="${TEST_RUN_ID:-${SLURM_ARRAY_JOB_ID}}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/project/perun2601396/FOMO26_submission_tests/real_sif_${TEST_RUN_ID}}"
TASK_ID="$((SLURM_ARRAY_TASK_ID + 1))"

test -x "${PYTHON}"
test -f "${TEST_ROOT}/tools/run_real_sif_task.py"
mkdir -p "${RESULTS_ROOT}"

EXTRA_ARGS=()
if [[ "${ALLOW_ALL_DATA_FALLBACK:-false}" == "true" ]]; then
    EXTRA_ARGS+=(--allow-all-data-fallback)
fi

srun "${PYTHON}" "${TEST_ROOT}/tools/run_real_sif_task.py" \
    --task "${TASK_ID}" \
    --data-root "${DATA_ROOT}" \
    --source-root "${SOURCE_ROOT}" \
    --task1-sif "${TASK1_SIF}" \
    --remaining-build-root "${REMAINING_BUILD_ROOT}" \
    --results-root "${RESULTS_ROOT}" \
    --cases "${CASES_PER_TASK}" \
    --seed "${TEST_SEED}" \
    --timeout "${PREDICTION_TIMEOUT}" \
    "${EXTRA_ARGS[@]}"

echo "REAL_SIF_SLURM_FINISHED task=${TASK_ID} results=${RESULTS_ROOT}"
