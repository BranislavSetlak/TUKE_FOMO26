#!/bin/bash
#SBATCH --job-name=dino3d-overfit-one
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
source "${CODE_ROOT}/slurm/dino3d/segmentation_recovery_common.sh"
dino3d_recovery_init
dino3d_resolve_experiment_id

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TASK_INDEX=2
    DATASET="SEG009_FOMO26_Meningioma"
    REQUIRED_CLASSES=(1)
    ;;
  1)
    TASK_INDEX=4
    DATASET="SEG010_FOMO26_TrigeminalNeuralgia"
    REQUIRED_CLASSES=(1 2)
    ;;
  *)
    echo "ERROR: Expected array index 0 or 1." >&2
    exit 1
    ;;
esac

SOURCE_FOLD="${SOURCE_FOLD:-0}"
EPOCHS="${OVERFIT_EPOCHS:-100}"
ROI_SIZE_TEXT="${OVERFIT_ROI_SIZE:-160 160 160}"
read -r ROI_X ROI_Y ROI_Z ROI_EXTRA <<<"${ROI_SIZE_TEXT}"
if [[ -n "${ROI_EXTRA:-}" ]] \
  || [[ ! "${ROI_X:-}" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "${ROI_Y:-}" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "${ROI_Z:-}" =~ ^[1-9][0-9]*$ ]]; then
  echo 'ERROR: OVERFIT_ROI_SIZE must contain three positive integers, for example "96 96 96".' >&2
  exit 1
fi
ROI_HYDRA="[${ROI_X},${ROI_Y},${ROI_Z}]"
MIN_ROI_COVERAGE="${OVERFIT_MIN_ROI_COVERAGE:-0.99}"
MINIMUM_DICE="${OVERFIT_MINIMUM_DICE:-0.90}"
MODEL_ROOT="${DINO3D_OVERFIT_MODEL_ROOT:-${SHARED_ROOT}/FOMO26_models/dino3d_one_case_overfit}"
RESULT_ROOT="${DINO3D_OVERFIT_RESULT_ROOT:-${SHARED_ROOT}/FOMO26_results/dino3d_one_case_overfit}"
RUN_DIR="${MODEL_ROOT}/experiment_${EXPERIMENT_ID}/${DATASET}/source_fold_${SOURCE_FOLD}"
mkdir -p "${RUN_DIR}" "${RESULT_ROOT}"

export ASPARAGUS_MODELS="${MODEL_ROOT}"
export ASPARAGUS_RESULTS="${RESULT_ROOT}"

SOURCE_SPLIT="${ASPARAGUS_DATA}/${DATASET}/split_5fold_cv.json"
OVERFIT_SPLIT="${RUN_DIR}/one_case_split.json"
OVERFIT_METADATA="${RUN_DIR}/one_case_metadata.json"
CENTER_FILE="${RUN_DIR}/one_case_center.txt"

cd "${CODE_ROOT}/asparagus"
PREPARE_ARGS=(
  --split "${SOURCE_SPLIT}"
  --fold "${SOURCE_FOLD}"
  --output-split "${OVERFIT_SPLIT}"
  --output-metadata "${OVERFIT_METADATA}"
  --center-output "${CENTER_FILE}"
  --roi-size "${ROI_X}" "${ROI_Y}" "${ROI_Z}"
  --min-roi-coverage "${MIN_ROI_COVERAGE}"
)
for CLASS_ID in "${REQUIRED_CLASSES[@]}"; do
  PREPARE_ARGS+=(--required-class "${CLASS_ID}")
done
"${PYTHON}" -m asparagus.analysis.prepare_one_case_overfit "${PREPARE_ARGS[@]}"
ROI_CENTER="$(<"${CENTER_FILE}")"

echo "DINO3D_ONE_CASE_OVERFIT_START time=$(date --iso-8601=seconds)"
echo "task=${DATASET} source_fold=${SOURCE_FOLD} roi_center=${ROI_CENTER} roi_size=${ROI_HYDRA}"
echo "checkpoint=${PRETRAIN_CKPT} run_dir=${RUN_DIR} epochs=${EPOCHS} resume=${RESUME}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

CMD=(
  "${PYTHON}" -m asparagus.pipeline.run.finetune_seg
  "task=${DATASET}"
  "+model=dinov3_downstream"
  "checkpoint_path=${PRETRAIN_CKPT}"
  "root=datapaper"
  "stem=dinov3_3d_one_case_overfit"
  "resume_training=${RESUME}"
  "data.train_split=one_case_diagnostic"
  "train_split_path=${OVERFIT_SPLIT}"
  "data.test_split=TEST_80_10_10"
  "data.fold=0"
  "hardware.num_devices=1"
  "hardware.num_workers=${SLURM_CPUS_PER_TASK}"
  "hardware.precision=bf16-mixed"
  "training.batch_size=1"
  "training.epochs=${EPOCHS}"
  "training.train_batches_per_epoch_per_device=20"
  "training.seed=$((260900 + TASK_INDEX))"
  "training.accumulate_grad_batches=1"
  "training.load_decoder=false"
  "training.freeze_backbone=false"
  "training.full_validation=true"
  "training.warmup_epochs=5"
  "training.decoder_warmup_epochs=0"
  "training.check_val_every_n_epoch=1"
  "training.repeat_stem_weights=false"
  "training.patch_size=[160,160,160]"
  "training.val_batches_per_epoch_per_device=1.0"
  "training.val_batch_size=1"
  "training.sliding_window_validation=false"
  "training.anatomical_roi_enabled=true"
  "training.anatomical_roi_center=${ROI_CENTER}"
  "training.anatomical_roi_size=${ROI_HYDRA}"
  "training.segmentation_loss=dice_ce"
  "training.checkpoint_monitor=val/foreground_dice"
  "training.checkpoint_mode=max"
  "training.run_best_validation_after_fit=true"
  "training.run_test_after_fit=false"
  "transforms.cpu_tr_transforms=CPU_seg_anatomical_roi_noaug_transforms"
  "transforms.cpu_val_transforms=CPU_seg_test_transforms"
  "transforms.gpu_tr_transforms=none"
  "model.ckpt_every_n_epoch=25"
  "logger.wandb_logging=false"
  "logger.log_to_stdout=true"
  "logger.progress_bar=true"
  "logger.log_every_n_steps=5"
  "hydra.run.dir=${RUN_DIR}"
  "hydra.job.chdir=false"
)
dino3d_print_command_and_run

METRICS="${RUN_DIR}/validation_best_metrics.json"
test -s "${METRICS}"
CHECK_ARGS=(--metrics "${METRICS}" --minimum-dice "${MINIMUM_DICE}")
for CLASS_ID in "${REQUIRED_CLASSES[@]}"; do
  CHECK_ARGS+=(--required-class "${CLASS_ID}")
done
if [[ "${OVERFIT_STRICT:-false}" == "true" ]]; then
  CHECK_ARGS+=(--strict)
fi
"${PYTHON}" -m asparagus.analysis.check_one_case_overfit "${CHECK_ARGS[@]}"
echo "DINO3D_ONE_CASE_OVERFIT_FINISHED time=$(date --iso-8601=seconds) task=${DATASET}"
