#!/bin/bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(pwd)}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${CODE_ROOT}/FOMO26_INFARCT_SUBMISSION}"
: "${CLSREG_EXPERIMENT_ID:?Export CLSREG_EXPERIMENT_ID first}"
SIF_PATH="${SIF_PATH:-${BUNDLE_ROOT}/build/fomo26_task1_tuke_swinunetr.sif}"
VARIANT="${INFARCT_VARIANT:-auto}"

export_job="$(sbatch --parsable \
  --export="ALL,CODE_ROOT=${CODE_ROOT},BUNDLE_ROOT=${BUNDLE_ROOT},CLSREG_EXPERIMENT_ID=${CLSREG_EXPERIMENT_ID},INFARCT_VARIANT=${VARIANT}" \
  "${BUNDLE_ROOT}/scripts/01_export_infarct_ensemble.slurm")"
export_job="${export_job%%;*}"

build_job="$(sbatch --parsable --dependency="afterok:${export_job}" \
  --export="ALL,CODE_ROOT=${CODE_ROOT},BUNDLE_ROOT=${BUNDLE_ROOT},SIF_PATH=${SIF_PATH}" \
  "${BUNDLE_ROOT}/scripts/02_build_container.slurm")"
build_job="${build_job%%;*}"

validator_job="$(sbatch --parsable --dependency="afterok:${build_job}" \
  --export="ALL,CODE_ROOT=${CODE_ROOT},BUNDLE_ROOT=${BUNDLE_ROOT},SIF_PATH=${SIF_PATH}" \
  "${BUNDLE_ROOT}/scripts/03_official_validator.slurm")"
validator_job="${validator_job%%;*}"

sample_job="$(sbatch --parsable --dependency="afterok:${validator_job}" \
  --export="ALL,CODE_ROOT=${CODE_ROOT},BUNDLE_ROOT=${BUNDLE_ROOT},SIF_PATH=${SIF_PATH}" \
  "${BUNDLE_ROOT}/scripts/04_finetune_sample_check.slurm")"
sample_job="${sample_job%%;*}"

echo "EXPORT_JOB_ID=${export_job}"
echo "BUILD_JOB_ID=${build_job}"
echo "VALIDATOR_JOB_ID=${validator_job}"
echo "SAMPLE_CHECK_JOB_ID=${sample_job}"
echo "SIF_PATH=${SIF_PATH}"

