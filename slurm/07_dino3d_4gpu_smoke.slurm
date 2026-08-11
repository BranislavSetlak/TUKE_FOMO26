#!/bin/bash
#SBATCH --job-name=dino3d-ddp-smoke
#SBATCH --partition=gpu_short
#SBATCH --account=perun2601396
#SBATCH --qos=perun2601396
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="/mnt/home/brseke961/FOMO26_project/FOMO26_code"
ENV_PATH="${CODE_ROOT}/fomo_env"
PYTHON="${ENV_PATH}/bin/python"

test -x "${PYTHON}"

export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"

hash -r

set -a
source "${CODE_ROOT}/asparagus/.env.pretrain"
set +a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "DDP_SMOKE_START node=$(hostname) time=$(date --iso-8601=seconds)"
echo "python=${PYTHON}"
echo "visible_devices=${CUDA_VISIBLE_DEVICES:-not-set}"

"${PYTHON}" --version
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

cd "${CODE_ROOT}/asparagus"

SMOKE_RUN_ID="${SLURM_JOB_ID}"
TOTAL_WORKERS=48

srun --kill-on-bad-exit=1 --ntasks=4 \
  "${PYTHON}" -m asparagus.pipeline.run.pretrain \
  --config-name projects/datapaper/pretrain/dinov3_3d_stage1 \
  run_id="${SMOKE_RUN_ID}" \
  task=PT902_FOMO300K_HF \
  root=datapaper \
  stem="dinov3_3d_ddp_smoke_${SLURM_JOB_ID}" \
  resume_training=false \
  hardware.num_devices=4 \
  hardware.strategy=ddp_find_unused_parameters_true \
  hardware.num_workers="${TOTAL_WORKERS}" \
  training.batch_size=2 \
  training.accumulate_grad_batches=1 \
  training.max_samples=320 \
  training.steps_per_epoch=20 \
  training.val_steps_per_epoch=0 \
  training.check_val_every_n_epoch=999999 \
  model.ckpt_every_n_epoch=1 \
  logger.wandb_logging=false \
  logger.log_to_stdout=true \
  logger.log_every_n_steps=1

echo "DDP_SMOKE_FINISHED time=$(date --iso-8601=seconds)"

find "${ASPARAGUS_MODELS}" \
  -type f \
  -name '*.ckpt' \
  -path "*dinov3_3d_ddp_smoke_${SLURM_JOB_ID}*" \
  -print
