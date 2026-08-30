#!/bin/bash
#SBATCH --job-name=fomo26-real-analysis
#SBATCH --partition=gpu_short
#SBATCH --account=perun2601396
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/fomo26-real-analysis_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/fomo26-real-analysis_%j.err

set -euo pipefail

: "${CODE_ROOT:?Export CODE_ROOT before submitting}"
: "${TEST_RUN_ID:?Export TEST_RUN_ID as the test array job ID}"

TEST_ROOT="${TEST_ROOT:-${CODE_ROOT}/FOMO26_REAL_SIF_TESTS}"
PYTHON="${CODE_ROOT}/fomo_env/bin/python"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/project/perun2601396/FOMO26_submission_tests/real_sif_${TEST_RUN_ID}}"
REPORT="${RESULTS_ROOT}/real_sif_evaluation.txt"

test -x "${PYTHON}"
test -d "${RESULTS_ROOT}"

srun "${PYTHON}" "${TEST_ROOT}/tools/analyze_real_sif_results.py" \
    --results-root "${RESULTS_ROOT}" \
    --output "${REPORT}" \
    --json-output "${RESULTS_ROOT}/real_sif_evaluation.json"

cat "${REPORT}"
