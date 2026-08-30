#!/bin/bash
#SBATCH --job-name=fomo26-rest-real
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${CODE_ROOT}/FOMO26_REMAINING_SUBMISSIONS}"
test -s "${CODE_ROOT}/asparagus/.env.finetune"
set -a
source "${CODE_ROOT}/asparagus/.env.finetune"
set +a
PYTHON="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}/bin/python"
srun --kill-on-bad-exit=1 "${PYTHON}" "${BUNDLE_ROOT}/tools/check_real_finetune_cases.py" \
  --data-root "${ASPARAGUS_DATA}" --build-root "${BUNDLE_ROOT}/build" \
  --report "${BUNDLE_ROOT}/reports/real_finetune_cases.json"
