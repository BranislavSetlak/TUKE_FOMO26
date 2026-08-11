#!/bin/bash
#SBATCH --job-name=dino3d-stage1-4gpu
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=512G
#SBATCH --time=30:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="/mnt/home/brseke961/FOMO26_project/FOMO26_code"
ENV_PATH="${CODE_ROOT}/fomo_env"
PYTHON="${ENV_PATH}/bin/python"

test -x "${PYTHON}"
test -f "${CODE_ROOT}/asparagus/.env.pretrain"

export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"

set -a
source "${CODE_ROOT}/asparagus/.env.pretrain"
set +a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "TRAIN_START time=$(date --iso-8601=seconds)"
echo "node=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "slurm_ntasks=${SLURM_NTASKS}"
echo "slurm_tasks_per_node=${SLURM_TASKS_PER_NODE}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "visible_devices=${CUDA_VISIBLE_DEVICES:-not-set}"

"${PYTHON}" --version
nvidia-smi \
    --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader

# Confirm that the checkpoint fix is installed before starting four GPUs.
CHECKPOINT_SOURCE="${CODE_ROOT}/asparagus/asparagus/pipeline/run/pretrain.py"

grep -q 'save_last=True' "${CHECKPOINT_SOURCE}" || {
    echo "ERROR: save_last=True is missing from pretrain.py" >&2
    exit 1
}

grep -q 'save_on_train_epoch_end=True' "${CHECKPOINT_SOURCE}" || {
    echo "ERROR: save_on_train_epoch_end=True is missing from pretrain.py" >&2
    exit 1
}

grep -q 'save_top_k=-1' "${CHECKPOINT_SOURCE}" || {
    echo "ERROR: save_top_k=-1 is missing from pretrain.py" >&2
    exit 1
}

cd "${CODE_ROOT}/asparagus"

echo "Starting fresh four-GPU DINO training."
echo "Permanent checkpoint interval: 50 pseudo-epochs = 25,000 optimizer steps."
echo "Production run_id=${SLURM_JOB_ID}"

srun --kill-on-bad-exit=1 \
    "${PYTHON}" -m asparagus.pipeline.run.pretrain \
    --config-name projects/datapaper/pretrain/dinov3_3d_stage1 \
    task=PT902_FOMO300K_HF \
    root=datapaper \
    stem=dinov3_3d_stage1_4gpu \
#    run_id="${SLURM_JOB_ID}" \
    resume_training=true \
    training.seed=260826 \
    hardware.num_devices=4 \
    hardware.strategy=ddp_find_unused_parameters_true \
    hardware.num_workers=48 \
    training.batch_size=2 \
    training.accumulate_grad_batches=1 \
    training.max_samples=1000000 \
    training.steps_per_epoch=500 \
    training.val_steps_per_epoch=0 \
    training.check_val_every_n_epoch=999999 \
    model.ckpt_every_n_epoch=50 \
    logger.wandb_logging=false \
    logger.log_to_stdout=true \
    logger.log_every_n_steps=10

echo "TRAIN_FINISHED time=$(date --iso-8601=seconds)"

echo "Saved production checkpoints:"
find "${ASPARAGUS_MODELS}" \
    -type f \
    -path '*stem=dinov3_3d_stage1_4gpu*' \
    -path '*/checkpoints/*.ckpt' \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS  %s bytes  %p\n' \
    | sort
