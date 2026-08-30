#!/bin/bash
#SBATCH --job-name=fomo26-rest-build
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --account=perun2601396
#SBATCH --array=0-4%3
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.err

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${CODE_ROOT}/FOMO26_REMAINING_SUBMISSIONS}"
case "${SLURM_ARRAY_TASK_ID}" in
  0) TASK=task2_meningioma; NAME=fomo26_task2_meningioma ;;
  1) TASK=task3_brain_age; NAME=fomo26_task3_brain_age ;;
  2) TASK=task4_trigeminal; NAME=fomo26_task4_trigeminal ;;
  3) TASK=task5_polymicrogyria; NAME=fomo26_task5_polymicrogyria ;;
  4) TASK=task6_7_embeddings; NAME=fomo26_task6_7_embeddings ;;
  *) exit 2 ;;
esac
BUNDLE_ROOT="${BUNDLE_ROOT}" REQUIRE_WEIGHTS=true "${BUNDLE_ROOT}/scripts/00_static_preflight.sh"
command -v apptainer >/dev/null
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/mnt/project/perun2601396/.apptainer_cache/${USER}}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/mnt/project/perun2601396/.apptainer_tmp/${USER}/${SLURM_JOB_ID}}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${BUNDLE_ROOT}/build"
cd "${BUNDLE_ROOT}"
SIF="${BUNDLE_ROOT}/build/${NAME}.sif"
apptainer build --fakeroot --arch amd64 "${SIF}" "containers/${TASK}/Apptainer.def"
test -s "${SIF}"
apptainer test "${SIF}"
sha256sum "${SIF}" > "${SIF}.sha256"
echo "CONTAINER_BUILD_PASS task=${TASK} sif=${SIF}"
