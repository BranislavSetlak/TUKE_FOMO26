#!/bin/bash
#SBATCH --job-name=fomo26-rest-data
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${CODE_ROOT}/FOMO26_REMAINING_SUBMISSIONS}"
PYTHON="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}/bin/python"
test -x "${PYTHON}"
mkdir -p "${BUNDLE_ROOT}/reports/generated_validator_data"
srun --kill-on-bad-exit=1 "${PYTHON}" "${BUNDLE_ROOT}/tools/prepare_validator_inputs.py" --output-root "${BUNDLE_ROOT}/reports/generated_validator_data"
