#!/bin/bash
#SBATCH --job-name=fomo-pretrain-split
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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
echo "asp_split: $(which asp_split)"
echo "ASPARAGUS_DATA=${ASPARAGUS_DATA}"

echo
echo "Checking preprocessed FOMO300K folder:"
ls -ld "${ASPARAGUS_DATA}/PT902_FOMO300K_HF"

cd "${CODE_ROOT}/asparagus"

echo
echo "Creating pretraining split:"
asp_split --dataset "PT902_FOMO300K_HF" --vals 99 1 0

echo
echo "Searching for created split/json files:"
find "${ASPARAGUS_DATA}/PT902_FOMO300K_HF" \
  -type f \( -iname "*split*" -o -iname "*.json" \) \
  -print | sort

echo
echo "Finished at: $(date)"
deactivate || true
exit 0
