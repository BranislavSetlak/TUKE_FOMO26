#!/bin/bash
#SBATCH --job-name=tuke-swin-overfit
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --account=perun2601396
#SBATCH --array=0-1%2
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
SUBMITTED_CODE_ROOT="${CODE_ROOT}"
PYTHON="${CODE_ROOT}/fomo_env/bin/python"
PRETRAIN_ENV_FILE="${CODE_ROOT}/asparagus/.env.pretrain"
FINETUNE_ENV_FILE="${CODE_ROOT}/asparagus/.env.finetune"

test -d "${CODE_ROOT}/asparagus"
test -x "${PYTHON}"
test -f "${PRETRAIN_ENV_FILE}"
test -f "${FINETUNE_ENV_FILE}"

set -a
source "${PRETRAIN_ENV_FILE}"
set +a
PRETRAIN_MODELS_ROOT="${ASPARAGUS_MODELS}"

set -a
source "${FINETUNE_ENV_FILE}"
set +a
CODE_ROOT="${SUBMITTED_CODE_ROOT}"

export PYTHONPATH="${CODE_ROOT}/asparagus${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=disabled

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    DATASET="SEG009_FOMO26_Meningioma"
    REQUIRED_CLASSES=(1)
    SEED=261000
    ;;
  1)
    DATASET="SEG010_FOMO26_TrigeminalNeuralgia"
    REQUIRED_CLASSES=(1 2)
    SEED=261001
    ;;
  *)
    echo "ERROR: Expected array index 0 or 1." >&2
    exit 1
    ;;
esac

if [[ -n "${TUKE_HYBRID_PRETRAIN_CKPT:-}" ]]; then
  PRETRAIN_CKPT="${TUKE_HYBRID_PRETRAIN_CKPT}"
else
  DEFAULT_CKPT="${PRETRAIN_MODELS_ROOT}/PT902_FOMO300K_HF/tuke_hybrid/production/run_id=${TUKE_HYBRID_RUN_ID:-production_8gpu_v1}/checkpoints/last.ckpt"
  if [[ -f "${DEFAULT_CKPT}" ]]; then
    PRETRAIN_CKPT="${DEFAULT_CKPT}"
  else
    SEARCH_ROOT="${PRETRAIN_MODELS_ROOT}/PT902_FOMO300K_HF"
    LATEST_LINE="$({
      find "${SEARCH_ROOT}" -type f -path '*/tuke_hybrid/production/run_id=*/checkpoints/last.ckpt' \
        -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | head -n 1)"
    PRETRAIN_CKPT="${LATEST_LINE#* }"
  fi
fi
if [[ ! -f "${PRETRAIN_CKPT:-}" ]]; then
  echo "ERROR: Could not find the TUKE hybrid last.ckpt." >&2
  echo "Submit with TUKE_HYBRID_PRETRAIN_CKPT=/absolute/path/to/last.ckpt sbatch $0" >&2
  exit 1
fi

SOURCE_FOLD="${TUKE_OVERFIT_SOURCE_FOLD:-0}"
SOURCE_SPLIT_NAME="${TUKE_OVERFIT_SOURCE_SPLIT:-split_5fold_cv}"
SOURCE_SPLIT="${ASPARAGUS_DATA}/${DATASET}/${SOURCE_SPLIT_NAME}.json"
if [[ ! -f "${SOURCE_SPLIT}" && "${SOURCE_SPLIT_NAME}" == "split_5fold_cv" ]]; then
  SOURCE_SPLIT_NAME="split_80_10_10"
  SOURCE_SPLIT="${ASPARAGUS_DATA}/${DATASET}/${SOURCE_SPLIT_NAME}.json"
fi
DATASET_JSON="${ASPARAGUS_DATA}/${DATASET}/dataset.json"
test -f "${SOURCE_SPLIT}"
test -f "${DATASET_JSON}"

PATCH_SIZE_TEXT="${TUKE_OVERFIT_PATCH_SIZE:-96 96 96}"
read -r PATCH_X PATCH_Y PATCH_Z PATCH_EXTRA <<<"${PATCH_SIZE_TEXT}"
if [[ -n "${PATCH_EXTRA:-}" ]] \
  || [[ ! "${PATCH_X:-}" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "${PATCH_Y:-}" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "${PATCH_Z:-}" =~ ^[1-9][0-9]*$ ]]; then
  echo 'ERROR: TUKE_OVERFIT_PATCH_SIZE must contain three positive integers, for example "96 96 96".' >&2
  exit 1
fi
if (( PATCH_X % 32 != 0 || PATCH_Y % 32 != 0 || PATCH_Z % 32 != 0 )); then
  echo "ERROR: Every SwinUNETR patch dimension must be divisible by 32." >&2
  exit 1
fi
PATCH_HYDRA="[${PATCH_X},${PATCH_Y},${PATCH_Z}]"

EPOCHS="${TUKE_OVERFIT_EPOCHS:-100}"
TRAIN_BATCHES="${TUKE_OVERFIT_TRAIN_BATCHES_PER_EPOCH:-20}"
WARMUP_EPOCHS="${TUKE_OVERFIT_WARMUP_EPOCHS:-5}"
MINIMUM_DICE="${TUKE_OVERFIT_MINIMUM_DICE:-0.90}"
MINIMUM_CLASS_VOXELS="${TUKE_OVERFIT_MINIMUM_CLASS_VOXELS:-8}"
PIPELINE_SMOKE="${TUKE_OVERFIT_PIPELINE_SMOKE:-true}"
for INTEGER_VALUE in "${EPOCHS}" "${TRAIN_BATCHES}" "${WARMUP_EPOCHS}" "${MINIMUM_CLASS_VOXELS}"; do
  if [[ ! "${INTEGER_VALUE}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Epoch, batch, warmup, and voxel settings must be non-negative integers." >&2
    exit 1
  fi
done
if (( EPOCHS < 2 || TRAIN_BATCHES < 1 || WARMUP_EPOCHS >= EPOCHS || MINIMUM_CLASS_VOXELS < 1 )); then
  echo "ERROR: Need epochs>=2, train_batches>=1, 0<=warmup_epochs<epochs, and minimum_class_voxels>=1." >&2
  exit 1
fi
if [[ "${PIPELINE_SMOKE}" != "true" && "${PIPELINE_SMOKE}" != "false" ]]; then
  echo "ERROR: TUKE_OVERFIT_PIPELINE_SMOKE must be true or false." >&2
  exit 1
fi
EXPERIMENT_ID="${TUKE_OVERFIT_EXPERIMENT_ID:-${SLURM_ARRAY_JOB_ID}}"
MODEL_ROOT="${TUKE_OVERFIT_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_one_case_overfit}"
RESULT_ROOT="${TUKE_OVERFIT_RESULT_ROOT:-${ASPARAGUS_RESULTS}/tuke_swinunetr_one_case_overfit}"
RUN_DIR="${MODEL_ROOT}/experiment_${EXPERIMENT_ID}/${DATASET}/source_fold_${SOURCE_FOLD}"
INPUT_DIR="${RUN_DIR}/diagnostic_input"
OVERFIT_SPLIT="${INPUT_DIR}/one_case_split.json"
SAMPLE_FILE="${INPUT_DIR}/one_case_roi.pt"
PREFLIGHT_JSON="${RUN_DIR}/preflight.json"
SOURCE_PRETRAIN_CKPT="${PRETRAIN_CKPT}"
SNAPSHOT_CKPT="${RUN_DIR}/pretrain_weights_snapshot.ckpt"
mkdir -p "${INPUT_DIR}" "${RESULT_ROOT}"

cd "${CODE_ROOT}/asparagus"

PREPARE_ARGS=(
  --split "${SOURCE_SPLIT}"
  --fold "${SOURCE_FOLD}"
  --output-dir "${INPUT_DIR}"
  --output-split "${OVERFIT_SPLIT}"
  --patch-size "${PATCH_X}" "${PATCH_Y}" "${PATCH_Z}"
  --minimum-class-voxels "${MINIMUM_CLASS_VOXELS}"
)
for CLASS_ID in "${REQUIRED_CLASSES[@]}"; do
  PREPARE_ARGS+=(--required-class "${CLASS_ID}")
done

echo "TUKE_SWINUNETR_OVERFIT_START time=$(date --iso-8601=seconds)"
echo "task=${DATASET} source_split=${SOURCE_SPLIT_NAME} source_fold=${SOURCE_FOLD}"
echo "checkpoint_source=${SOURCE_PRETRAIN_CKPT} patch_size=${PATCH_HYDRA} run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

srun --kill-on-bad-exit=1 \
  "${PYTHON}" -m asparagus.analysis.snapshot_lightning_checkpoint \
  --source "${SOURCE_PRETRAIN_CKPT}" \
  --output "${SNAPSHOT_CKPT}"
PRETRAIN_CKPT="${SNAPSHOT_CKPT}"

srun --kill-on-bad-exit=1 \
  "${PYTHON}" -m asparagus.analysis.prepare_tuke_swinunetr_one_case \
  "${PREPARE_ARGS[@]}"

build_finetune_command() {
  local command_run_id="$1"
  local command_run_dir="$2"
  local command_epochs="$3"
  local command_train_batches="$4"
  local command_warmup_epochs="$5"
  FINETUNE_CMD=(
    "${PYTHON}" -m asparagus.pipeline.run.finetune_seg
    "task=${DATASET}"
    "+model=swinunetr_hybrid_seg_b"
    "checkpoint_path='${PRETRAIN_CKPT}'"
    "root=tuke_hybrid"
    "stem=swinunetr_one_case_overfit"
    "run_id=${command_run_id}"
    "resume_training=false"
    "data.train_split=one_case_diagnostic"
    "train_split_path='${OVERFIT_SPLIT}'"
    "data.test_split=TEST_80_10_10"
    "data.fold=0"
    "hardware.num_devices=1"
    "hardware.num_workers=${SLURM_CPUS_PER_TASK}"
    "hardware.precision=bf16-mixed"
    "hardware.compile_mode=null"
    "training.batch_size=1"
    "training.epochs=${command_epochs}"
    "training.train_batches_per_epoch_per_device=${command_train_batches}"
    "training.val_batches_per_epoch_per_device=1"
    "training.seed=${SEED}"
    "training.accumulate_grad_batches=1"
    "training.load_decoder=true"
    "training.repeat_stem_weights=true"
    "training.warmup_epochs=${command_warmup_epochs}"
    "training.decoder_warmup_epochs=0"
    "training.check_val_every_n_epoch=1"
    "training.patch_size=${PATCH_HYDRA}"
    "training.checkpoint_monitor=val/min_foreground_class_dice"
    "training.checkpoint_mode=max"
    "training.run_best_validation_after_fit=true"
    "training.run_test_after_fit=false"
    "transforms.cpu_tr_transforms=CPU_seg_overfit_noaug_transforms"
    "transforms.cpu_val_transforms=CPU_seg_overfit_noaug_transforms"
    "transforms.gpu_tr_transforms=none"
    "model.finetune_lr=3e-4"
    "model.ckpt_every_n_epoch=25"
    "logger.wandb_logging=false"
    "logger.log_to_stdout=true"
    "logger.progress_bar=false"
    "logger.log_every_n_steps=5"
    "hydra.run.dir='${command_run_dir}'"
    "hydra.job.chdir=false"
  )
}

print_command() {
  printf 'RUNNING:'
  printf ' %q' "${FINETUNE_CMD[@]}"
  printf '\n'
}

build_finetune_command \
  "${EXPERIMENT_ID}_${SLURM_ARRAY_TASK_ID}" \
  "${RUN_DIR}" \
  "${EPOCHS}" \
  "${TRAIN_BATCHES}" \
  "${WARMUP_EPOCHS}"
print_command
srun --kill-on-bad-exit=1 "${FINETUNE_CMD[@]}" --cfg job --resolve > "${RUN_DIR}/resolved_config.yaml"

srun --kill-on-bad-exit=1 \
  "${PYTHON}" -m asparagus.analysis.preflight_tuke_swinunetr_overfit \
  --checkpoint "${PRETRAIN_CKPT}" \
  --sample "${SAMPLE_FILE}" \
  --dataset-json "${DATASET_JSON}" \
  --output "${PREFLIGHT_JSON}" \
  --patch-size "${PATCH_X}" "${PATCH_Y}" "${PATCH_Z}" \
  --device cuda

if [[ "${PIPELINE_SMOKE}" == "true" ]]; then
  SMOKE_DIR="${RUN_DIR}/pipeline_smoke"
  mkdir -p "${SMOKE_DIR}"
  echo "TUKE_SWINUNETR_PIPELINE_SMOKE_START time=$(date --iso-8601=seconds) dir=${SMOKE_DIR}"
  build_finetune_command \
    "${EXPERIMENT_ID}_${SLURM_ARRAY_TASK_ID}_smoke" \
    "${SMOKE_DIR}" \
    2 \
    1 \
    1
  print_command
  srun --kill-on-bad-exit=1 "${FINETUNE_CMD[@]}"
  test -s "${SMOKE_DIR}/validation_best_metrics.json"
  test -s "${SMOKE_DIR}/checkpoints/best.ckpt"
  echo "TUKE_SWINUNETR_PIPELINE_SMOKE_OK time=$(date --iso-8601=seconds)"
fi

build_finetune_command \
  "${EXPERIMENT_ID}_${SLURM_ARRAY_TASK_ID}" \
  "${RUN_DIR}" \
  "${EPOCHS}" \
  "${TRAIN_BATCHES}" \
  "${WARMUP_EPOCHS}"
print_command
srun --kill-on-bad-exit=1 "${FINETUNE_CMD[@]}"

METRICS="${RUN_DIR}/validation_best_metrics.json"
test -s "${METRICS}"
CHECK_ARGS=(--metrics "${METRICS}" --minimum-dice "${MINIMUM_DICE}")
for CLASS_ID in "${REQUIRED_CLASSES[@]}"; do
  CHECK_ARGS+=(--required-class "${CLASS_ID}")
done
if [[ "${TUKE_OVERFIT_STRICT:-false}" == "true" ]]; then
  CHECK_ARGS+=(--strict)
fi
srun --kill-on-bad-exit=1 \
  "${PYTHON}" -m asparagus.analysis.check_tuke_swinunetr_overfit \
  "${CHECK_ARGS[@]}"

echo "TUKE_SWINUNETR_OVERFIT_FINISHED time=$(date --iso-8601=seconds) task=${DATASET}"
