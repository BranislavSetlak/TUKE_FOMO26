#!/bin/bash
#SBATCH --job-name=fomo-pretrain
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
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
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

echo "Running on node: $(hostname)"
echo "Started at: $(date)"
echo "Python: $(which python)"
python --version

echo
echo "=== GPU ==="
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

echo
echo "=== Paths ==="
echo "ASPARAGUS_DATA=${ASPARAGUS_DATA}"
echo "ASPARAGUS_MODELS=${ASPARAGUS_MODELS}"
echo "ASPARAGUS_RESULTS=${ASPARAGUS_RESULTS}"

echo
echo "=== Checking pretraining dataset ==="
ls -lh "${ASPARAGUS_DATA}/PT902_FOMO300K_HF/paths.json"
ls -lh "${ASPARAGUS_DATA}/PT902_FOMO300K_HF/split_99_01_00.json"

mkdir -p "${ASPARAGUS_MODELS}"
mkdir -p "${ASPARAGUS_RESULTS}"

cd "${CODE_ROOT}/asparagus"

echo
echo "Starting pretraining at: $(date)"

asp_pretrain \
  --config-name projects/datapaper/pretrain/resenc_6m_amaes \
  task=PT902_FOMO300K_HF \
  hardware.num_devices=1 \
  hardware.num_workers="${SLURM_CPUS_PER_TASK}" \
  training.batch_size=4 \
  training.max_samples=1000000 \
  training.accumulate_grad_batches=1 \
  logger.wandb_logging=false

echo
echo "Finished pretraining at: $(date)"

echo
echo "=== Checkpoints ==="
find "${ASPARAGUS_MODELS}" -type f -name "*.ckpt" -print | sort

deactivate || true
exit 0
