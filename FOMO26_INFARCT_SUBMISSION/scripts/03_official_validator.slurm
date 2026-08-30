#!/bin/bash
#SBATCH --job-name=fomo26-t1-validator
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${CODE_ROOT}/FOMO26_INFARCT_SUBMISSION}"
SIF_PATH="${SIF_PATH:-${BUNDLE_ROOT}/build/fomo26_task1_tuke_swinunetr.sif}"
VALIDATOR_ROOT="${VALIDATOR_ROOT:-${BUNDLE_ROOT}/vendor/container-validator-main}"
ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"

test -x "${PYTHON}"
test -s "${SIF_PATH}"
test -s "${VALIDATOR_ROOT}/container_validator/validate.py"
export PYTHONUNBUFFERED=1

REPORT="${BUNDLE_ROOT}/reports/official_validator.txt"
VALIDATOR_DATA="${BUNDLE_ROOT}/reports/generated_validator_data"
mkdir -p "$(dirname "${REPORT}")"
srun --kill-on-bad-exit=1 "${PYTHON}" "${BUNDLE_ROOT}/tools/prepare_validator_inputs.py" \
  --output-root "${VALIDATOR_DATA}"
cd "${VALIDATOR_ROOT}"
set -o pipefail
srun --kill-on-bad-exit=1 "${PYTHON}" container_validator/validate.py \
  --task task1 --sif "${SIF_PATH}" --manifest "${VALIDATOR_DATA}/manifest.yaml" \
  --timeout 1800 | tee "${REPORT}"
grep -q 'container is ready to submit' "${REPORT}"
echo "OFFICIAL_VALIDATOR_PASS report=${REPORT}"
