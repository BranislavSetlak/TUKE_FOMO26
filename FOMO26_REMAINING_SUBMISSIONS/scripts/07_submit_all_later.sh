#!/bin/bash
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-$(pwd)}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${CODE_ROOT}/FOMO26_REMAINING_SUBMISSIONS}"
: "${CLSREG_EXPERIMENT_ID:?Set CLSREG_EXPERIMENT_ID}"
if [[ -z "${SEG_EXPERIMENT_ID:-}${SEG_NORMAL_EXPERIMENT_ID:-}${SEG_GIN_EXPERIMENT_ID:-}${SEG_GIN_CARVEMIX_EXPERIMENT_ID:-}" ]]; then
  echo "Set SEG_EXPERIMENT_ID (combined run) or the per-variant SEG_* IDs." >&2
  exit 1
fi

export_job="$(sbatch --parsable --export=ALL,CODE_ROOT="${CODE_ROOT}",BUNDLE_ROOT="${BUNDLE_ROOT}" "${BUNDLE_ROOT}/scripts/01_export_weights.slurm")"; export_job="${export_job%%;*}"
build_job="$(sbatch --parsable --dependency="afterok:${export_job}" --export=ALL,CODE_ROOT="${CODE_ROOT}",BUNDLE_ROOT="${BUNDLE_ROOT}" "${BUNDLE_ROOT}/scripts/02_build_containers.slurm")"; build_job="${build_job%%;*}"
data_job="$(sbatch --parsable --dependency="afterok:${build_job}" --export=ALL,CODE_ROOT="${CODE_ROOT}",BUNDLE_ROOT="${BUNDLE_ROOT}" "${BUNDLE_ROOT}/scripts/03_prepare_validator_data.slurm")"; data_job="${data_job%%;*}"
validator_job="$(sbatch --parsable --dependency="afterok:${data_job}" --export=ALL,CODE_ROOT="${CODE_ROOT}",BUNDLE_ROOT="${BUNDLE_ROOT}" "${BUNDLE_ROOT}/scripts/04_official_validator.slurm")"; validator_job="${validator_job%%;*}"
contract_job="$(sbatch --parsable --dependency="afterok:${validator_job}" --export=ALL,CODE_ROOT="${CODE_ROOT}",BUNDLE_ROOT="${BUNDLE_ROOT}" "${BUNDLE_ROOT}/scripts/05_contract_checks.slurm")"; contract_job="${contract_job%%;*}"
real_job="$(sbatch --parsable --dependency="afterok:${contract_job}" --export=ALL,CODE_ROOT="${CODE_ROOT}",BUNDLE_ROOT="${BUNDLE_ROOT}" "${BUNDLE_ROOT}/scripts/06_real_finetune_cases.slurm")"; real_job="${real_job%%;*}"
printf 'EXPORT_JOB=%s\nBUILD_ARRAY=%s\nVALIDATOR_DATA_JOB=%s\nVALIDATOR_ARRAY=%s\nCONTRACT_JOB=%s\nREAL_CASE_JOB=%s\n' "${export_job}" "${build_job}" "${data_job}" "${validator_job}" "${contract_job}" "${real_job}"
