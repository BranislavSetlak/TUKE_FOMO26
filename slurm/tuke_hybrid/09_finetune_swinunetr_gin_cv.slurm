#!/bin/bash
#SBATCH --job-name=tuke-swin-gin
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --account=perun2601396
#SBATCH --array=0-9%8
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.err

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
export TUKE_FINETUNE_VARIANT=gin
source "${CODE_ROOT}/slurm/tuke_hybrid/swinunetr_finetune_common.sh"
