#!/bin/bash
#SBATCH --job-name=dino3d-fp-loss
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --account=perun2601396
#SBATCH --array=0-9%5
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.err

set -euo pipefail

# Submit from the repository root. CODE_ROOT may also be exported before
# sbatch, but no user-specific repository path is stored in this script.
CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
ENV_FILE="${CODE_ROOT}/asparagus/.env.finetune"
test -f "${ENV_FILE}" || {
  echo "ERROR: ${ENV_FILE} is missing. cd to the repository root before sbatch." >&2
  exit 1
}

set -a
source "${ENV_FILE}"
set +a

ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
PYTHON="${ENV_PATH}/bin/python"
PRETRAIN_CKPT="${DINO3D_PRETRAIN_CKPT:-${SHARED_ROOT}/FOMO26_checkpoints/dinov3_3d_stage1_71494_last.ckpt}"
FT_MODEL_ROOT="${DINO3D_FP_LOSS_MODEL_ROOT:-${SHARED_ROOT}/FOMO26_models/dino3d_fp_loss_cv}"
FT_RESULT_ROOT="${DINO3D_FP_LOSS_RESULT_ROOT:-${SHARED_ROOT}/FOMO26_results/dino3d_fp_loss_cv}"

test -x "${PYTHON}"
test -s "${PRETRAIN_CKPT}"
test -f "${CODE_ROOT}/asparagus/configs/model/dinov3_downstream.yaml"

export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"
export ASPARAGUS_MODELS="${FT_MODEL_ROOT}"
export ASPARAGUS_RESULTS="${FT_RESULT_ROOT}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID}"
FOLD=$((ARRAY_INDEX % 5))
SEGMENTATION_INDEX=$((ARRAY_INDEX / 5))
RESUME="${RESUME:-false}"
REQUESTED_EXPERIMENT_ID="${EXPERIMENT_ID:-}"

if [[ "${RESUME}" == "true" && -z "${REQUESTED_EXPERIMENT_ID}" ]]; then
  echo "ERROR: To resume, set EXPERIMENT_ID to the original Slurm array job ID." >&2
  exit 1
fi
EXPERIMENT_ID="${REQUESTED_EXPERIMENT_ID:-${SLURM_ARRAY_JOB_ID}}"

case "${SEGMENTATION_INDEX}" in
  0)
    TASK_INDEX=2
    MODULE="asparagus.pipeline.run.finetune_seg"
    DATASET="SEG009_FOMO26_Meningioma"
    ;;
  1)
    TASK_INDEX=4
    MODULE="asparagus.pipeline.run.finetune_seg"
    DATASET="SEG010_FOMO26_TrigeminalNeuralgia"
    ;;
  *)
    echo "ERROR: Unknown segmentation index ${SEGMENTATION_INDEX}" >&2
    exit 1
    ;;
esac

SEED=$((260826 + TASK_INDEX * 10 + FOLD))
EPOCHS=150
EXTRA_OVERRIDES=(
  "training.patch_size=[160,160,160]"
  "training.val_batches_per_epoch_per_device=1.0"
  "training.val_batch_size=1"
  "training.sliding_window_validation=true"
  "training.inference_overlap=0.75"
  "training.checkpoint_monitor=val/foreground_dice"
  "training.checkpoint_mode=max"
  "training.segmentation_loss=asymmetric_tversky_ce"
  "training.tversky_alpha=0.7"
  "training.tversky_beta=0.3"
  "training.tversky_weight=1.0"
  "training.cross_entropy_weight=1.0"
  "transforms.cpu_val_transforms=CPU_seg_test_transforms"
)

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

RUN_DIR="${FT_MODEL_ROOT}/experiment_${EXPERIMENT_ID}/${DATASET}/fold_${FOLD}"
mkdir -p "${RUN_DIR}" "${FT_RESULT_ROOT}"
cd "${CODE_ROOT}/asparagus"

echo "DINO3D_FP_LOSS_FINETUNE_START time=$(date --iso-8601=seconds)"
echo "variant=fp_loss segmentation_loss=asymmetric_tversky_ce alpha=0.7 beta=0.3"
echo "task=${DATASET} task_index=${TASK_INDEX} fold=${FOLD} array_index=${ARRAY_INDEX}"
echo "node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "checkpoint=${PRETRAIN_CKPT}"
echo "run_dir=${RUN_DIR} epochs=${EPOCHS} seed=${SEED} resume=${RESUME}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

CMD=(
  "${PYTHON}" -m "${MODULE}"
  "task=${DATASET}"
  "+model=dinov3_downstream"
  "checkpoint_path=${PRETRAIN_CKPT}"
  "root=datapaper"
  "stem=dinov3_3d_fp_loss_linear"
  "resume_training=${RESUME}"
  "data.train_split=split_5fold_cv"
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
  "training.freeze_backbone=true"
  "training.full_validation=true"
  "training.warmup_epochs=10"
  "training.decoder_warmup_epochs=0"
  "training.check_val_every_n_epoch=1"
  "training.repeat_stem_weights=false"
  "model.ckpt_every_n_epoch=25"
  "logger.wandb_logging=false"
  "logger.log_to_stdout=true"
  "logger.progress_bar=true"
  "logger.log_every_n_steps=25"
  "hydra.run.dir=${RUN_DIR}"
  "hydra.job.chdir=false"
)
CMD+=("${EXTRA_OVERRIDES[@]}")

printf 'COMMAND '
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "DINO3D_FP_LOSS_FINETUNE_FINISHED time=$(date --iso-8601=seconds) task=${DATASET} fold=${FOLD}"
find "${RUN_DIR}" -type f \( -name '*.ckpt' -o -name '*metrics.json' -o -name '*__best.json' \) \
  -printf '%s bytes  %p\n' | sort
