#!/bin/bash
#SBATCH --job-name=fomo-splits
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=perun2601396
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

CODE_ROOT="/mnt/data/home/brseke961/FOMO26_project/FOMO26_code"
ENV_PATH="${CODE_ROOT}/fomo_env"

source "${ENV_PATH}/bin/activate"

set -a
source "${CODE_ROOT}/asparagus/.env.finetune"
set +a

cd "${CODE_ROOT}/asparagus"

asp_split --dataset CLS002_FOMO26_Infarct --vals 80 10 10
asp_split --dataset SEG009_FOMO26_Meningioma --vals 80 10 10
asp_split --dataset REGR002_FOMO26_BrainAge --vals 80 10 10
asp_split --dataset SEG010_FOMO26_TrigeminalNeuralgia --vals 80 10 10
asp_split --dataset CLS003_FOMO26_Polymicrogyria --vals 80 10 10
