#!/bin/bash
#SBATCH --job-name=dino3d-stage1-4gpu-resume
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

# This remains the model run ID being resumed.
# It must not be replaced with the new Slurm job ID.
RESUME_RUN_ID=71494

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

echo "RESUME_JOB_START time=$(date --iso-8601=seconds)"
echo "node=$(hostname)"
echo "new_slurm_job_id=${SLURM_JOB_ID}"
echo "resumed_model_run_id=${RESUME_RUN_ID}"
echo "slurm_ntasks=${SLURM_NTASKS}"
echo "slurm_tasks_per_node=${SLURM_TASKS_PER_NODE}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "visible_devices=${CUDA_VISIBLE_DEVICES:-not-set}"

"${PYTHON}" --version

nvidia-smi \
  --query-gpu=index,name,memory.total,memory.free \
  --format=csv,noheader

# Prevent two jobs from writing into run 71494 simultaneously.
if squeue -h -j "${RESUME_RUN_ID}" 2>/dev/null | grep -q .; then
    echo "ERROR: original job ${RESUME_RUN_ID} is still active." >&2
    echo "Do not resume while the original job is running." >&2
    exit 1
fi

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

# Locate exactly the checkpoint belonging to run 71494.
mapfile -t CKPT_MATCHES < <(
    find "${ASPARAGUS_MODELS}" \
      -type f \
      -path '*stem=dinov3_3d_stage1_4gpu*' \
      -path "*/run_id=${RESUME_RUN_ID}/checkpoints/last.ckpt" \
      -print
)

if [[ "${#CKPT_MATCHES[@]}" -ne 1 ]]; then
    echo "ERROR: expected exactly one last.ckpt for run ${RESUME_RUN_ID}." >&2
    echo "Found ${#CKPT_MATCHES[@]} matching files:" >&2
    printf '%s\n' "${CKPT_MATCHES[@]}" >&2
    exit 1
fi

LAST_CKPT="${CKPT_MATCHES[0]}"

test -s "${LAST_CKPT}" || {
    echo "ERROR: checkpoint is missing or empty: ${LAST_CKPT}" >&2
    exit 1
}

echo "CHECKPOINT_PREFLIGHT_OK"
stat --format=$'size=%s bytes\nsaved=%y\npath=%n' "${LAST_CKPT}"

cd "${CODE_ROOT}/asparagus"

echo "Resuming four-GPU DINO training."
echo "Model run ID remains ${RESUME_RUN_ID}."
echo "Training will continue toward the original total of 125,000 steps."

srun --kill-on-bad-exit=1 \
  "${PYTHON}" -m asparagus.pipeline.run.pretrain \
  --config-name projects/datapaper/pretrain/dinov3_3d_stage1 \
  task=PT902_FOMO300K_HF \
  root=datapaper \
  stem=dinov3_3d_stage1_4gpu \
  run_id="${RESUME_RUN_ID}" \
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

echo "RESUME_JOB_FINISHED time=$(date --iso-8601=seconds)"

echo "Checkpoints for run ${RESUME_RUN_ID}:"

find "$(dirname "${LAST_CKPT}")" \
  -maxdepth 1 \
  -type f \
  -name '*.ckpt' \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS  %s bytes  %f\n' \
  | sort
