#!/bin/bash
#SBATCH --job-name=fomo-test-ft-task1
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/project/perun2601396/FOMO26_job_outputs/brseke961/%x_%j.out
#SBATCH --error=/project/perun2601396/FOMO26_job_outputs/brseke961/%x_%j.err

set -euo pipefail

CODE_ROOT="/mnt/data/home/brseke961/FOMO26_project/FOMO26_code"
ENV_PATH="${CODE_ROOT}/fomo_env"

PRETRAIN_CKPT="/project/perun2601396/FOMO26_checkpoints/pretrain_21746_last.ckpt"

source "${ENV_PATH}/bin/activate"

set -a
source "${CODE_ROOT}/asparagus/.env.finetune"
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
echo "=== Checking checkpoint ==="
ls -lh "${PRETRAIN_CKPT}"

echo
echo "=== Checking Task 1 data ==="
ls -lh "${ASPARAGUS_DATA}/CLS002_FOMO26_Infarct/paths.json"
ls -lh "${ASPARAGUS_DATA}/CLS002_FOMO26_Infarct/split_80_10_10.json"
ls -lh "${ASPARAGUS_DATA}/CLS002_FOMO26_Infarct/TEST_80_10_10.json"

mkdir -p "${ASPARAGUS_MODELS}"
mkdir -p "${ASPARAGUS_RESULTS}"

cd "${CODE_ROOT}/asparagus"

echo
echo "Starting 1-epoch Task 1 finetuning test at: $(date)"

asp_finetune_cls \
  task=CLS002_FOMO26_Infarct \
  +model=resenc_unet_b_clsreg \
  checkpoint_path="${PRETRAIN_CKPT}" \
  data.train_split=split_80_10_10 \
  data.test_split=TEST_80_10_10 \
  data.fold=0 \
  hardware.num_devices=1 \
  hardware.num_workers="${SLURM_CPUS_PER_TASK}" \
  training.batch_size=2 \
  training.epochs=1 \
  logger.wandb_logging=false \
  logger.log_to_stdout=true \
  logger.progress_bar=true

echo
echo "Finished at: $(date)"

echo
echo "=== New finetuning checkpoints ==="
find "${ASPARAGUS_MODELS}" -type f -name "*.ckpt" -printf "%TY-%Tm-%Td %TH:%TM:%TS  %s bytes  %p\n" | sort | tail -20

deactivate || true
exit 0
