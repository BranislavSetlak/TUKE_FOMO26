#!/bin/bash
#SBATCH --job-name=fomo26-t1-build
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
mkdir -p "$(dirname "${SIF_PATH}")"

BUNDLE_ROOT="${BUNDLE_ROOT}" REQUIRE_WEIGHTS=true "${BUNDLE_ROOT}/scripts/00_static_preflight.sh"
command -v apptainer >/dev/null

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/mnt/project/perun2601396/.apptainer_cache/${USER}}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/mnt/project/perun2601396/.apptainer_tmp/${USER}/${SLURM_JOB_ID}}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

cd "${BUNDLE_ROOT}/container"
echo "BUILD_START output=${SIF_PATH}"
apptainer build --fakeroot --arch amd64 "${SIF_PATH}" Apptainer.def
test -s "${SIF_PATH}"
apptainer inspect "${SIF_PATH}"
apptainer test "${SIF_PATH}"
sha256sum "${SIF_PATH}" | tee "${SIF_PATH}.sha256"
ls -lh "${SIF_PATH}" "${SIF_PATH}.sha256"
echo "BUILD_FINISHED sif=${SIF_PATH}"

