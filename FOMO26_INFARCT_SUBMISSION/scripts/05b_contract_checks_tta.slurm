#!/bin/bash
#SBATCH --job-name=fomo26-t1-contract
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
ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
: "${CHECK_FLAIR:?Export CHECK_FLAIR=/absolute/path/flair.nii.gz}"
: "${CHECK_ADC:?Export CHECK_ADC=/absolute/path/adc.nii.gz}"
: "${CHECK_DWI:?Export CHECK_DWI=/absolute/path/dwi_b1000.nii.gz}"

OPTIONAL=()
if [[ -n "${CHECK_SWI:-}" ]]; then
  OPTIONAL=(--swi "${CHECK_SWI}")
elif [[ -n "${CHECK_T2S:-}" ]]; then
  OPTIONAL=(--t2s "${CHECK_T2S}")
else
  echo "ERROR: export CHECK_SWI or CHECK_T2S" >&2
  exit 1
fi

srun --kill-on-bad-exit=1 "${PYTHON}" "${BUNDLE_ROOT}/tools/contract_checks_tta.py" \
  --sif "${SIF_PATH}" \
  --flair "${CHECK_FLAIR}" \
  --adc "${CHECK_ADC}" \
  --dwi "${CHECK_DWI}" \
  "${OPTIONAL[@]}"

