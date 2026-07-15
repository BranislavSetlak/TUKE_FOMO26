#!/bin/bash
#SBATCH --job-name=fomo-check-env
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
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

echo "=== Python ==="
which python
python --version

echo "=== Asparagus commands ==="
which asp_process
which asp_split
which asp_finetune_cls
which asp_finetune_seg
which asp_finetune_reg

echo "=== Paths ==="
echo "FOMO_ROOT=${FOMO_ROOT}"
echo "ASPARAGUS_SOURCE=${ASPARAGUS_SOURCE}"
echo "ASPARAGUS_DATA=${ASPARAGUS_DATA}"
echo "ASPARAGUS_MODELS=${ASPARAGUS_MODELS}"
echo "ASPARAGUS_RESULTS=${ASPARAGUS_RESULTS}"
echo "ASPARAGUS_RAW_LABELS=${ASPARAGUS_RAW_LABELS}"

echo "=== Raw finetune data ==="
ls -lah "${ASPARAGUS_SOURCE}"
