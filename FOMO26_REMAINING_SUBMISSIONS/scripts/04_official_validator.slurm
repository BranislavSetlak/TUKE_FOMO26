#!/bin/bash
#SBATCH --job-name=fomo26-rest-valid
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=perun2601396
#SBATCH --array=0-4%3
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.err

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${CODE_ROOT}/FOMO26_REMAINING_SUBMISSIONS}"
PYTHON="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}/bin/python"
case "${SLURM_ARRAY_TASK_ID}" in
  0) TASK=task2; SIF=fomo26_task2_meningioma.sif ;;
  1) TASK=task3; SIF=fomo26_task3_brain_age.sif ;;
  2) TASK=task4; SIF=fomo26_task4_trigeminal.sif ;;
  3) TASK=task5; SIF=fomo26_task5_polymicrogyria.sif ;;
  4) TASK=task6_and_7; SIF=fomo26_task6_7_embeddings.sif ;;
  *) exit 2 ;;
esac
test -x "${PYTHON}"
test -s "${BUNDLE_ROOT}/build/${SIF}"
VALIDATOR="${BUNDLE_ROOT}/vendor/container-validator-main"
DATA="${BUNDLE_ROOT}/reports/generated_validator_data"
REPORT="${BUNDLE_ROOT}/reports/official_validator_${TASK}.txt"
cd "${VALIDATOR}"
set -o pipefail
srun --kill-on-bad-exit=1 "${PYTHON}" container_validator/validate.py --task "${TASK}" \
  --sif "${BUNDLE_ROOT}/build/${SIF}" --manifest "${DATA}/manifest.yaml" --timeout 2400 | tee "${REPORT}"
grep -q 'container is ready to submit' "${REPORT}"
echo "OFFICIAL_VALIDATOR_PASS task=${TASK} report=${REPORT}"
