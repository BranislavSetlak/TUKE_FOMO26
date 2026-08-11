#!/bin/bash
#SBATCH --job-name=dino3d-ft
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --account=perun2601396
#SBATCH --array=1-5%5
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.err

set -euo pipefail

CODE_ROOT="/mnt/home/brseke961/FOMO26_project/FOMO26_code"
ENV_PATH="${CODE_ROOT}/fomo_env"
PYTHON="${ENV_PATH}/bin/python"
SHARED_ROOT="/mnt/project/perun2601396"
PRETRAIN_CKPT="${SHARED_ROOT}/FOMO26_checkpoints/dinov3_3d_stage1_71494_last.ckpt"
FT_MODEL_ROOT="${SHARED_ROOT}/FOMO26_models/dino3d_finetuning"
FT_RESULT_ROOT="${SHARED_ROOT}/FOMO26_results/dino3d/finetuning"

test -x "${PYTHON}"
test -s "${PRETRAIN_CKPT}"
test -f "${CODE_ROOT}/asparagus/.env.finetune"
test -f "${CODE_ROOT}/asparagus/configs/model/dinov3_downstream.yaml"

export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"

set -a
source "${CODE_ROOT}/asparagus/.env.finetune"
set +a

# Repair only the obsolete mount prefixes if .env.finetune predates the
# storage remount. The dataset's remaining directory structure is preserved.
ASPARAGUS_DATA="${ASPARAGUS_DATA/#\/project\//\/mnt\/project\/}"
ASPARAGUS_DATA="${ASPARAGUS_DATA/#\/mnt\/data\/home\//\/mnt\/home\/}"
export ASPARAGUS_DATA

# Keep this experiment separate from the original baseline fine-tuning runs.
export ASPARAGUS_MODELS="${FT_MODEL_ROOT}"
export ASPARAGUS_RESULTS="${FT_RESULT_ROOT}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
FOLD="${FOLD:-0}"
SEED=$((260826 + TASK_INDEX))

case "${TASK_INDEX}" in
  1)
    MODULE="asparagus.pipeline.run.finetune_cls"
    DATASET="CLS002_FOMO26_Infarct"
    EPOCHS=50
    EXTRA_OVERRIDES=(
      "lightning._lightning_module._target_=asparagus.modules.lightning_modules.dino_downstream.DINOClassificationModule"
      "training.target_size=[128,128,128]"
      "training.warmup_epochs=10"
      "training.decoder_warmup_epochs=0"
    )
    ;;
  2)
    MODULE="asparagus.pipeline.run.finetune_seg"
    DATASET="SEG009_FOMO26_Meningioma"
    EPOCHS=150
    EXTRA_OVERRIDES=(
      "training.patch_size=[160,160,160]"
      "training.warmup_epochs=10"
      "training.decoder_warmup_epochs=0"
    )
    ;;
  3)
    MODULE="asparagus.pipeline.run.finetune_reg"
    DATASET="REGR002_FOMO26_BrainAge"
    EPOCHS=50
    EXTRA_OVERRIDES=(
      "training.target_size=[128,128,128]"
      "training.warmup_epochs=10"
    )
    ;;
  4)
    MODULE="asparagus.pipeline.run.finetune_seg"
    DATASET="SEG010_FOMO26_TrigeminalNeuralgia"
    EPOCHS=150
    EXTRA_OVERRIDES=(
      "training.patch_size=[160,160,160]"
      "training.warmup_epochs=10"
      "training.decoder_warmup_epochs=0"
    )
    ;;
  5)
    MODULE="asparagus.pipeline.run.finetune_cls"
    DATASET="CLS003_FOMO26_Polymicrogyria"
    EPOCHS=50
    EXTRA_OVERRIDES=(
      "lightning._lightning_module._target_=asparagus.modules.lightning_modules.dino_downstream.DINOClassificationModule"
      "training.target_size=[128,128,128]"
      "training.warmup_epochs=10"
      "training.decoder_warmup_epochs=0"
    )
    ;;
  *)
    echo "ERROR: Unknown array index ${TASK_INDEX}" >&2
    exit 1
    ;;
esac

for required in paths.json split_80_10_10.json TEST_80_10_10.json; do
  test -s "${ASPARAGUS_DATA}/${DATASET}/${required}" || {
    echo "ERROR: Missing ${ASPARAGUS_DATA}/${DATASET}/${required}" >&2
    exit 1
  }
done

RUN_DIR="${FT_MODEL_ROOT}/${DATASET}/runs/slurm_${SLURM_ARRAY_JOB_ID}_${TASK_INDEX}"
mkdir -p "${RUN_DIR}" "${FT_RESULT_ROOT}"
cd "${CODE_ROOT}/asparagus"

echo "FINETUNE_START time=$(date --iso-8601=seconds)"
echo "task=${DATASET} task_index=${TASK_INDEX} node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "checkpoint=${PRETRAIN_CKPT}"
echo "data_root=${ASPARAGUS_DATA}"
echo "run_dir=${RUN_DIR} epochs=${EPOCHS} seed=${SEED}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

CMD=(
  "${PYTHON}" -m "${MODULE}"
  "task=${DATASET}"
  "+model=dinov3_downstream"
  "checkpoint_path=${PRETRAIN_CKPT}"
  "root=datapaper"
  "stem=dinov3_3d_71494_last"
  "resume_training=false"
  "data.train_split=split_80_10_10"
  "data.test_split=TEST_80_10_10"
  "data.fold=${FOLD}"
  "hardware.num_devices=1"
  "hardware.num_workers=${SLURM_CPUS_PER_TASK}"
  "hardware.precision=bf16-mixed"
  "training.batch_size=2"
  "training.epochs=${EPOCHS}"
  "training.seed=${SEED}"
  "training.accumulate_grad_batches=1"
  "training.load_decoder=false"
  "training.check_val_every_n_epoch=1"
  "model.ckpt_every_n_epoch=25"
  "logger.wandb_logging=false"
  "logger.log_to_stdout=true"
  "logger.progress_bar=true"
  "logger.log_every_n_steps=50"
  "hydra.run.dir=${RUN_DIR}"
  "hydra.job.chdir=false"
)
CMD+=("${EXTRA_OVERRIDES[@]}")

printf 'COMMAND '
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "FINETUNE_FINISHED time=$(date --iso-8601=seconds) task=${DATASET}"
find "${RUN_DIR}" -type f \( -name '*.ckpt' -o -name '*__best.json' \) -printf '%s bytes  %p\n' | sort
