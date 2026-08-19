#!/bin/bash
#SBATCH --job-name=dino3d-diagnose
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

# Submit this file from the repository root.  Slurm copies job scripts into a
# spool directory, so BASH_SOURCE cannot reliably identify the repository.
CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
SHARED_ROOT="${SHARED_ROOT:-/mnt/project/perun2601396}"
ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
ENV_FILE="${CODE_ROOT}/asparagus/.env.finetune"
REPORT_DIR="${SHARED_ROOT}/FOMO26_results/dino3d/diagnostics_71494"
REPORT="${REPORT_DIR}/dino3d_diagnostic_report.txt"

test -x "${PYTHON}"
test -f "${ENV_FILE}"
test -f "${CODE_ROOT}/asparagus/asparagus/analysis/diagnose_dino3d_transfer.py"

set -a
source "${ENV_FILE}"
set +a

export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"
export SHARED_ROOT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

mkdir -p "${REPORT_DIR}"
cd "${CODE_ROOT}/asparagus"

echo "DINO3D_DIAGNOSTIC_START time=$(date --iso-8601=seconds)"
echo "code_root=${CODE_ROOT}"
echo "shared_root=${SHARED_ROOT}"
echo "data_root=${ASPARAGUS_DATA}"
echo "report=${REPORT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

"${PYTHON}" -m asparagus.analysis.diagnose_dino3d_transfer \
  --code-root "${CODE_ROOT}" \
  --shared-root "${SHARED_ROOT}" \
  --data-root "${ASPARAGUS_DATA}" \
  --feature-samples 24 \
  --output "${REPORT}"

echo "DINO3D_DIAGNOSTIC_FINISHED time=$(date --iso-8601=seconds)"
echo "Paste the following DIAG lines into ChatGPT:"
grep '^DIAG|' "${REPORT}"
