#!/bin/bash
#SBATCH --job-name=fomo26-fivefold
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
ENV_FILE="${CODE_ROOT}/asparagus/.env.finetune"
test -f "${ENV_FILE}" || {
  echo "ERROR: ${ENV_FILE} is missing. cd to the repository root before sbatch." >&2
  exit 1
}

set -a
source "${ENV_FILE}"
set +a

ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
SCRIPT="${CODE_ROOT}/asparagus/asparagus/analysis/create_fomo26_fivefold_splits.py"

test -x "${PYTHON}"
test -f "${SCRIPT}"
test -d "${ASPARAGUS_DATA}"

export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

ARGS=(--data-root "${ASPARAGUS_DATA}" --seed 260826)
if [[ "${OVERWRITE_SPLITS:-false}" == "true" ]]; then
  ARGS+=(--overwrite)
fi

echo "FIVEFOLD_SPLIT_START time=$(date --iso-8601=seconds) data_root=${ASPARAGUS_DATA}"
srun "${PYTHON}" "${SCRIPT}" "${ARGS[@]}"
echo "FIVEFOLD_SPLIT_FINISHED time=$(date --iso-8601=seconds)"
