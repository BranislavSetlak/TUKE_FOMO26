#!/bin/bash

# Shared implementation for the three controlled segmentation variants.
# The calling Slurm file must set TUKE_FINETUNE_VARIANT.

set -euo pipefail

: "${TUKE_FINETUNE_VARIANT:?Set TUKE_FINETUNE_VARIANT before sourcing this file}"

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
SUBMITTED_CODE_ROOT="${CODE_ROOT}"
PRETRAIN_ENV_FILE="${CODE_ROOT}/asparagus/.env.pretrain"
FINETUNE_ENV_FILE="${CODE_ROOT}/asparagus/.env.finetune"

test -d "${CODE_ROOT}/asparagus"
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

ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
test -x "${PYTHON}"

export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
if [[ ! -s "${PRETRAIN_CKPT:-}" ]]; then
  echo "ERROR: Could not find a non-empty TUKE hybrid checkpoint." >&2
  echo "Set TUKE_HYBRID_PRETRAIN_CKPT=/absolute/path/to/final.ckpt before sbatch." >&2
  exit 1
fi

case "${TUKE_FINETUNE_VARIANT}" in
  normal)
    GPU_TRANSFORMS="GPU_all_train_transforms"
    CARVEMIX_PROBABILITY=0.0
    MODEL_ROOT="${TUKE_SWIN_NORMAL_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_normal_cv}"
    RESULT_ROOT="${TUKE_SWIN_NORMAL_RESULT_ROOT:-${ASPARAGUS_RESULTS}/tuke_swinunetr_normal_cv}"
    ;;
  gin)
    GPU_TRANSFORMS="GPU_all_train_transforms_gin"
    CARVEMIX_PROBABILITY=0.0
    MODEL_ROOT="${TUKE_SWIN_GIN_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_gin_cv}"
    RESULT_ROOT="${TUKE_SWIN_GIN_RESULT_ROOT:-${ASPARAGUS_RESULTS}/tuke_swinunetr_gin_cv}"
    ;;
  gin_carvemix)
    GPU_TRANSFORMS="GPU_all_train_transforms_gin"
    CARVEMIX_PROBABILITY="${TUKE_CARVEMIX_PROBABILITY:-0.5}"
    MODEL_ROOT="${TUKE_SWIN_GIN_CARVEMIX_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_gin_carvemix_cv}"
    RESULT_ROOT="${TUKE_SWIN_GIN_CARVEMIX_RESULT_ROOT:-${ASPARAGUS_RESULTS}/tuke_swinunetr_gin_carvemix_cv}"
    ;;
  *)
    echo "ERROR: Unknown TUKE_FINETUNE_VARIANT=${TUKE_FINETUNE_VARIANT}." >&2
    exit 1
    ;;
esac

RAW_ARRAY_INDEX="${SLURM_ARRAY_TASK_ID}"
ARRAY_INDEX="${TUKE_LOCAL_ARRAY_INDEX:-${RAW_ARRAY_INDEX}}"
FOLD=$((ARRAY_INDEX % 5))
TASK_INDEX=$((ARRAY_INDEX / 5))
case "${TASK_INDEX}" in
  0) DATASET="SEG009_FOMO26_Meningioma" ;;
  1) DATASET="SEG010_FOMO26_TrigeminalNeuralgia" ;;
  *)
    echo "ERROR: Array index must be in 0..9, got ${ARRAY_INDEX}." >&2
    exit 1
    ;;
esac

SEED=$((261500 + TASK_INDEX * 10 + FOLD))
RESUME="${RESUME:-false}"
REQUESTED_EXPERIMENT_ID="${EXPERIMENT_ID:-}"
if [[ "${RESUME}" != "true" && "${RESUME}" != "false" ]]; then
  echo "ERROR: RESUME must be true or false." >&2
  exit 1
fi
if [[ "${RESUME}" == "true" && -z "${REQUESTED_EXPERIMENT_ID}" ]]; then
  echo "ERROR: Resuming requires EXPERIMENT_ID=<original array job ID>." >&2
  exit 1
fi
EXPERIMENT_ID="${REQUESTED_EXPERIMENT_ID:-${SLURM_ARRAY_JOB_ID}}"

for required in paths.json dataset.json split_5fold_cv.json TEST_80_10_10.json; do
  test -s "${ASPARAGUS_DATA}/${DATASET}/${required}" || {
    echo "ERROR: Missing ${ASPARAGUS_DATA}/${DATASET}/${required}" >&2
    exit 1
  }
done
FOLD_COUNT=$({ grep -o '"train"[[:space:]]*:' \
  "${ASPARAGUS_DATA}/${DATASET}/split_5fold_cv.json" || true; } | wc -l)
if (( FOLD_COUNT != 5 )); then
  echo "ERROR: ${DATASET} has ${FOLD_COUNT} folds; exactly five are required." >&2
  exit 1
fi

PATCH_SIZE_TEXT="${TUKE_FINETUNE_PATCH_SIZE:-96 96 96}"
read -r PATCH_X PATCH_Y PATCH_Z PATCH_EXTRA <<<"${PATCH_SIZE_TEXT}"
if [[ -n "${PATCH_EXTRA:-}" ]] \
  || [[ ! "${PATCH_X:-}" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "${PATCH_Y:-}" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "${PATCH_Z:-}" =~ ^[1-9][0-9]*$ ]]; then
  echo 'ERROR: TUKE_FINETUNE_PATCH_SIZE must contain three positive integers.' >&2
  exit 1
fi
if (( PATCH_X % 32 != 0 || PATCH_Y % 32 != 0 || PATCH_Z % 32 != 0 )); then
  echo "ERROR: Every SwinUNETR patch dimension must be divisible by 32." >&2
  exit 1
fi
PATCH_HYDRA="[${PATCH_X},${PATCH_Y},${PATCH_Z}]"

EPOCHS="${TUKE_FINETUNE_EPOCHS:-150}"
TRAIN_BATCHES="${TUKE_FINETUNE_TRAIN_BATCHES_PER_EPOCH:-250}"
WARMUP_EPOCHS="${TUKE_FINETUNE_WARMUP_EPOCHS:-10}"
CHECK_VAL_EVERY="${TUKE_FINETUNE_CHECK_VAL_EVERY:-1}"
CKPT_EVERY="${TUKE_FINETUNE_CKPT_EVERY:-50}"
for integer_value in "${EPOCHS}" "${TRAIN_BATCHES}" "${WARMUP_EPOCHS}" "${CHECK_VAL_EVERY}" "${CKPT_EVERY}"; do
  [[ "${integer_value}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: Epoch and batch controls must be non-negative integers." >&2
    exit 1
  }
done
if (( EPOCHS < 2 || TRAIN_BATCHES < 1 || WARMUP_EPOCHS >= EPOCHS || CHECK_VAL_EVERY < 1 || CKPT_EVERY < 1 )); then
  echo "ERROR: Need epochs>=2, train_batches>=1, warmup<epochs, check_val_every>=1, and ckpt_every>=1." >&2
  exit 1
fi

RUN_DIR="${MODEL_ROOT}/experiment_${EXPERIMENT_ID}/${DATASET}/fold_${FOLD}"
mkdir -p "${RUN_DIR}" "${RESULT_ROOT}"
if [[ "${RESUME}" == "true" && ! -s "${RUN_DIR}/checkpoints/last.ckpt" ]]; then
  echo "ERROR: Resume checkpoint is missing: ${RUN_DIR}/checkpoints/last.ckpt" >&2
  exit 1
fi

cd "${CODE_ROOT}/asparagus"
MARKER="TUKE_SWINUNETR_${TUKE_FINETUNE_VARIANT^^}_FINETUNE"
MARKER="${MARKER//-/_}"
echo "${MARKER}_START time=$(date --iso-8601=seconds)"
echo "variant=${TUKE_FINETUNE_VARIANT} task=${DATASET} fold=${FOLD} array_index=${ARRAY_INDEX} raw_array_index=${RAW_ARRAY_INDEX}"
echo "checkpoint=${PRETRAIN_CKPT} run_dir=${RUN_DIR}"
echo "epochs=${EPOCHS} train_batches=${TRAIN_BATCHES} seed=${SEED} resume=${RESUME}"
echo "patch_size=${PATCH_HYDRA} gpu_transforms=${GPU_TRANSFORMS} carvemix_probability=${CARVEMIX_PROBABILITY}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

CMD=(
  "${PYTHON}" -m asparagus.pipeline.run.finetune_seg
  "task=${DATASET}"
  "+model=swinunetr_hybrid_seg_b"
  "checkpoint_path='${PRETRAIN_CKPT}'"
  "root=tuke_hybrid"
  "stem=swinunetr_${TUKE_FINETUNE_VARIANT}_full"
  "run_id=${EXPERIMENT_ID}_${ARRAY_INDEX}"
  "resume_training=${RESUME}"
  "data.train_split=split_5fold_cv"
  "data.test_split=TEST_80_10_10"
  "data.fold=${FOLD}"
  "hardware.num_devices=1"
  "hardware.num_workers=${SLURM_CPUS_PER_TASK}"
  "hardware.precision=bf16-mixed"
  "hardware.compile_mode=null"
  "training.batch_size=2"
  "training.epochs=${EPOCHS}"
  "training.train_batches_per_epoch_per_device=${TRAIN_BATCHES}"
  "training.val_batches_per_epoch_per_device=1.0"
  "training.val_batch_size=1"
  "training.seed=${SEED}"
  "training.accumulate_grad_batches=1"
  "training.load_decoder=true"
  "training.repeat_stem_weights=true"
  "training.full_validation=true"
  "training.warmup_epochs=${WARMUP_EPOCHS}"
  "training.decoder_warmup_epochs=0"
  "training.check_val_every_n_epoch=${CHECK_VAL_EVERY}"
  "training.patch_size=${PATCH_HYDRA}"
  "training.sliding_window_validation=true"
  "training.inference_overlap=0.5"
  "training.carvemix_probability=${CARVEMIX_PROBABILITY}"
  "training.carvemix_donor_attempts=4"
  "training.checkpoint_monitor=val/min_foreground_class_dice"
  "training.checkpoint_mode=max"
  "training.run_best_validation_after_fit=true"
  "training.run_test_after_fit=false"
  "transforms.cpu_tr_transforms=CPU_seg_train_transforms"
  "transforms.cpu_val_transforms=CPU_seg_test_transforms"
  "transforms.gpu_tr_transforms=${GPU_TRANSFORMS}"
  "model.finetune_lr=3e-4"
  "model.ckpt_every_n_epoch=${CKPT_EVERY}"
  "logger.wandb_logging=false"
  "logger.log_to_stdout=true"
  "logger.progress_bar=false"
  "logger.log_every_n_steps=25"
  "hydra.run.dir='${RUN_DIR}'"
  "hydra.job.chdir=false"
)

printf 'COMMAND'
printf ' %q' "${CMD[@]}"
printf '\n'
srun --kill-on-bad-exit=1 "${CMD[@]}" --cfg job --resolve > "${RUN_DIR}/resolved_config.yaml"
srun --kill-on-bad-exit=1 "${CMD[@]}"

test -s "${RUN_DIR}/validation_best_metrics.json"
test -s "${RUN_DIR}/checkpoints/best.ckpt"
test -s "${RUN_DIR}/checkpoints/last.ckpt"
echo "${MARKER}_FINISHED time=$(date --iso-8601=seconds) task=${DATASET} fold=${FOLD}"
find "${RUN_DIR}" -maxdepth 2 -type f \
  \( -name '*.ckpt' -o -name '*metrics.json' -o -name 'resolved_config.yaml' \) \
  -printf '%s bytes  %p\n' | sort
