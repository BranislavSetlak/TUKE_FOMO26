#!/bin/bash
#SBATCH --job-name=fomo-task5-extract
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=perun2601396
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

CODE_ROOT="/mnt/data/home/brseke961/FOMO26_project/FOMO26_code"
ENV_PATH="${CODE_ROOT}/fomo_env"
FINETUNE_DIR="/project/perun2601396/FOMO26_finetune"

source "${ENV_PATH}/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Running on node: $(hostname)"
echo "Started at: $(date)"
echo "Finetune directory: ${FINETUNE_DIR}"

cd "${FINETUNE_DIR}"

echo "Checking required files:"
ls -lh Task_5_extract.py
ls -lh Zhang_Lingfeng_2022_PPMR_Dataset.zip

echo "Running Task 5 extraction..."
python Task_5_extract.py

echo "Task 5 directory after extraction:"
ls -lah "${FINETUNE_DIR}"

echo "Finished at: $(date)"
deactivate || true
exit 0
