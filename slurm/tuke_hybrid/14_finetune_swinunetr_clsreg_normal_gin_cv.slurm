#!/bin/bash
#SBATCH --job-name=tuke-swin-clsreg
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --account=perun2601396
#SBATCH --array=0-29%8
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.err

set -euo pipefail

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
  if [[ -s "${DEFAULT_CKPT}" ]]; then
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
  echo "ERROR: No non-empty TUKE hybrid checkpoint was found." >&2
  echo "Export TUKE_HYBRID_PRETRAIN_CKPT=/absolute/path/to/final.ckpt and resubmit." >&2
  exit 1
fi

RAW_INDEX="${SLURM_ARRAY_TASK_ID}"
VARIANT_INDEX=$((RAW_INDEX / 15))
LOCAL_INDEX=$((RAW_INDEX % 15))
TASK_INDEX=$((LOCAL_INDEX / 5))
FOLD=$((LOCAL_INDEX % 5))

case "${VARIANT_INDEX}" in
  0)
    VARIANT=normal
    GPU_TRANSFORMS=GPU_all_train_transforms
    MODEL_ROOT="${TUKE_SWIN_CLSREG_NORMAL_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_clsreg_normal_cv}"
    RESULT_ROOT="${TUKE_SWIN_CLSREG_NORMAL_RESULT_ROOT:-${ASPARAGUS_RESULTS}/tuke_swinunetr_clsreg_normal_cv}"
    ;;
  1)
    VARIANT=gin
    GPU_TRANSFORMS=GPU_all_train_transforms_gin
    MODEL_ROOT="${TUKE_SWIN_CLSREG_GIN_MODEL_ROOT:-${ASPARAGUS_MODELS}/tuke_swinunetr_clsreg_gin_cv}"
    RESULT_ROOT="${TUKE_SWIN_CLSREG_GIN_RESULT_ROOT:-${ASPARAGUS_RESULTS}/tuke_swinunetr_clsreg_gin_cv}"
    ;;
  *)
    echo "ERROR: Array index must be in 0..29, got ${RAW_INDEX}." >&2
    exit 1
    ;;
esac

case "${TASK_INDEX}" in
  0)
    MODULE=asparagus.pipeline.run.finetune_cls
    DATASET=CLS002_FOMO26_Infarct
    MONITOR=val/loss
    MONITOR_MODE=min
    ;;
  1)
    MODULE=asparagus.pipeline.run.finetune_reg
    DATASET=REGR002_FOMO26_BrainAge
    MONITOR=val/MAE
    MONITOR_MODE=min
    ;;
  2)
    MODULE=asparagus.pipeline.run.finetune_cls
    DATASET=CLS003_FOMO26_Polymicrogyria
    MONITOR=val/loss
    MONITOR_MODE=min
    ;;
  *)
    echo "ERROR: Unknown task index ${TASK_INDEX}." >&2
    exit 1
    ;;
esac

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

EPOCHS="${TUKE_CLSREG_EPOCHS:-50}"
TRAIN_BATCHES="${TUKE_CLSREG_TRAIN_BATCHES_PER_EPOCH:-250}"
WARMUP_EPOCHS="${TUKE_CLSREG_WARMUP_EPOCHS:-10}"
CKPT_EVERY="${TUKE_CLSREG_CKPT_EVERY:-25}"
for VALUE in "${EPOCHS}" "${TRAIN_BATCHES}" "${WARMUP_EPOCHS}" "${CKPT_EVERY}"; do
  [[ "${VALUE}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: Epoch and batch controls must be non-negative integers." >&2
    exit 1
  }
done
if (( EPOCHS < 2 || TRAIN_BATCHES < 1 || WARMUP_EPOCHS >= EPOCHS || CKPT_EVERY < 1 )); then
  echo "ERROR: Need epochs>=2, train_batches>=1, warmup<epochs, ckpt_every>=1." >&2
  exit 1
fi

SEED=$((262000 + TASK_INDEX * 10 + FOLD))
RESUME="${RESUME:-false}"
REQUESTED_EXPERIMENT_ID="${EXPERIMENT_ID:-}"
if [[ "${RESUME}" != true && "${RESUME}" != false ]]; then
  echo "ERROR: RESUME must be true or false." >&2
  exit 1
fi
if [[ "${RESUME}" == true && -z "${REQUESTED_EXPERIMENT_ID}" ]]; then
  echo "ERROR: Resuming requires EXPERIMENT_ID=<original cls/reg array job ID>." >&2
  exit 1
fi
EXPERIMENT_ID="${REQUESTED_EXPERIMENT_ID:-${SLURM_ARRAY_JOB_ID}}"
RUN_DIR="${MODEL_ROOT}/experiment_${EXPERIMENT_ID}/${DATASET}/fold_${FOLD}"
mkdir -p "${RUN_DIR}" "${RESULT_ROOT}"
if [[ "${RESUME}" == true && ! -s "${RUN_DIR}/checkpoints/last.ckpt" ]]; then
  echo "ERROR: Resume checkpoint is missing: ${RUN_DIR}/checkpoints/last.ckpt" >&2
  exit 1
fi

cd "${CODE_ROOT}/asparagus"
echo "TUKE_SWINUNETR_CLSREG_${VARIANT^^}_START time=$(date --iso-8601=seconds)"
echo "variant=${VARIANT} task=${DATASET} fold=${FOLD} raw_array_index=${RAW_INDEX}"
echo "checkpoint=${PRETRAIN_CKPT} run_dir=${RUN_DIR} seed=${SEED} resume=${RESUME}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

CMD=(
  "${PYTHON}" -m "${MODULE}"
  "task=${DATASET}"
  "+model=swinunetr_hybrid_clsreg_b"
  "checkpoint_path='${PRETRAIN_CKPT}'"
  "root=tuke_hybrid"
  "stem=swinunetr_clsreg_${VARIANT}"
  "run_id=${EXPERIMENT_ID}_${RAW_INDEX}"
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
  "training.limit_train_batches=${TRAIN_BATCHES}"
  "training.limit_val_batches=1.0"
  "training.val_batch_size=2"
  "training.target_size=[96,96,96]"
  "training.seed=${SEED}"
  "training.accumulate_grad_batches=1"
  "training.load_decoder=false"
  "training.repeat_stem_weights=true"
  "training.full_validation=true"
  "training.warmup_epochs=${WARMUP_EPOCHS}"
  "training.decoder_warmup_epochs=0"
  "training.check_val_every_n_epoch=1"
  "training.checkpoint_monitor=${MONITOR}"
  "training.checkpoint_mode=${MONITOR_MODE}"
  "transforms.gpu_tr_transforms=${GPU_TRANSFORMS}"
  "model.finetune_lr=1e-4"
  "model.ckpt_every_n_epoch=${CKPT_EVERY}"
  "logger.wandb_logging=false"
  "logger.log_to_stdout=true"
  "logger.progress_bar=true"
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
find "${RUN_DIR}/predictions" -type f -name '*__best.json' -size +0c -print -quit | grep -q .
echo "TUKE_SWINUNETR_CLSREG_${VARIANT^^}_FINISHED time=$(date --iso-8601=seconds) task=${DATASET} fold=${FOLD}"
find "${RUN_DIR}" -maxdepth 3 -type f \
  \( -name '*.ckpt' -o -name '*metrics.json' -o -name '*__best.json' -o -name 'resolved_config.yaml' \) \
  -printf '%s bytes  %p\n' | sort
