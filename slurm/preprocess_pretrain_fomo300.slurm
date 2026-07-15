#!/bin/bash
#SBATCH --job-name=fomo-preprocess-pt902
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/project/perun2601396/FOMO26_job_outputs/brseke961/%x_%j.out
#SBATCH --error=/project/perun2601396/FOMO26_job_outputs/brseke961/%x_%j.err

set -euo pipefail

CODE_ROOT="/mnt/data/home/brseke961/FOMO26_project/FOMO26_code"
ENV_PATH="${CODE_ROOT}/fomo_env"

source "${ENV_PATH}/bin/activate"

set -a
source "${CODE_ROOT}/asparagus/.env.pretrain"
set +a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Running on node: $(hostname)"
echo "Started at: $(date)"
echo "Python: $(which python)"
python --version

echo "ASPARAGUS_SOURCE=${ASPARAGUS_SOURCE}"
echo "ASPARAGUS_DATA=${ASPARAGUS_DATA}"
echo "Output folder: ${ASPARAGUS_DATA}/PT902_FOMO300K_HF"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"

mkdir -p "${ASPARAGUS_DATA}/PT902_FOMO300K_HF"
mkdir -p "${ASPARAGUS_MODELS}"
mkdir -p "${ASPARAGUS_RESULTS}"
mkdir -p "${ASPARAGUS_RAW_LABELS}"

cd "${CODE_ROOT}/asparagus_preprocessing"

python3 asparagus_preprocessing/datasets_pretraining/PT902_FOMO300K_HF.py \
  --input_dir "${ASPARAGUS_SOURCE}" \
  --output_dir "${ASPARAGUS_DATA}/PT902_FOMO300K_HF" \
  --num_workers "${SLURM_CPUS_PER_TASK}"

echo "Finished preprocessing at: $(date)"

echo
echo "Quick output check:"
ls -ld "${ASPARAGUS_DATA}/PT902_FOMO300K_HF"
find "${ASPARAGUS_DATA}/PT902_FOMO300K_HF" -maxdepth 2 -type f | head -30

deactivate || true
exit 0
