#!/bin/bash
#SBATCH --job-name=dino3d-roi-seg010
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --account=perun2601396
#SBATCH --array=0-4%5
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
source "${CODE_ROOT}/slurm/dino3d/segmentation_recovery_common.sh"
dino3d_recovery_init
dino3d_resolve_experiment_id

DATASET="SEG010_FOMO26_TrigeminalNeuralgia"
FOLD="${SLURM_ARRAY_TASK_ID}"
SEED=$((260826 + 40 + FOLD))
ROI_SIZE_TEXT="${ROI_SIZE:-96 96 96}"
read -r ROI_X ROI_Y ROI_Z <<<"${ROI_SIZE_TEXT}"
if [[ -z "${ROI_X:-}" || -z "${ROI_Y:-}" || -z "${ROI_Z:-}" ]]; then
  echo 'ERROR: ROI_SIZE must contain three integers, for example ROI_SIZE="96 96 96".' >&2
  exit 1
fi
ROI_HYDRA="[${ROI_X},${ROI_Y},${ROI_Z}]"
MIN_ROI_COVERAGE="${MIN_ROI_COVERAGE:-0.90}"

dino3d_validate_segmentation_data

MODEL_ROOT="${DINO3D_ANATOMICAL_ROI_MODEL_ROOT:-${SHARED_ROOT}/FOMO26_models/dino3d_anatomical_roi_seg010_cv}"
RESULT_ROOT="${DINO3D_ANATOMICAL_ROI_RESULT_ROOT:-${SHARED_ROOT}/FOMO26_results/dino3d_anatomical_roi_seg010_cv}"
RUN_DIR="${MODEL_ROOT}/experiment_${EXPERIMENT_ID}/${DATASET}/fold_${FOLD}"
mkdir -p "${RUN_DIR}" "${RESULT_ROOT}"
export ASPARAGUS_MODELS="${MODEL_ROOT}"
export ASPARAGUS_RESULTS="${RESULT_ROOT}"

ROI_REPORT="${RUN_DIR}/anatomical_roi_training_report.json"
CENTER_FILE="${RUN_DIR}/anatomical_roi_center.txt"
cd "${CODE_ROOT}/asparagus"
"${PYTHON}" -m asparagus.analysis.estimate_fold_anatomical_roi \
  --split "${ASPARAGUS_DATA}/${DATASET}/split_5fold_cv.json" \
  --fold "${FOLD}" \
  --roi-size "${ROI_X}" "${ROI_Y}" "${ROI_Z}" \
  --min-training-coverage "${MIN_ROI_COVERAGE}" \
  --output "${ROI_REPORT}" \
  --center-output "${CENTER_FILE}"
ROI_CENTER="$(<"${CENTER_FILE}")"

echo "DINO3D_ANATOMICAL_ROI_START time=$(date --iso-8601=seconds)"
echo "task=${DATASET} fold=${FOLD} roi_size=${ROI_HYDRA} network_size=[160,160,160]"
echo "roi_center=${ROI_CENTER} min_training_coverage=${MIN_ROI_COVERAGE}"
echo "checkpoint=${PRETRAIN_CKPT} run_dir=${RUN_DIR} resume=${RESUME}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

CMD=(
  "${PYTHON}" -m asparagus.pipeline.run.finetune_seg
  "task=${DATASET}"
  "+model=dinov3_downstream"
  "checkpoint_path=${PRETRAIN_CKPT}"
  "root=datapaper"
  "stem=dinov3_3d_anatomical_roi_linear"
  "resume_training=${RESUME}"
  "data.train_split=split_5fold_cv"
  "data.test_split=TEST_80_10_10"
  "data.fold=${FOLD}"
  "hardware.num_devices=1"
  "hardware.num_workers=${SLURM_CPUS_PER_TASK}"
  "hardware.precision=bf16-mixed"
  "training.batch_size=2"
  "training.epochs=150"
  "training.seed=${SEED}"
  "training.accumulate_grad_batches=1"
  "training.load_decoder=false"
  "training.freeze_backbone=true"
  "training.full_validation=true"
  "training.warmup_epochs=10"
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
  "training.carvemix_probability=0.0"
  "training.segmentation_loss=dice_ce"
  "training.checkpoint_monitor=val/foreground_dice"
  "training.checkpoint_mode=max"
  "training.run_best_validation_after_fit=true"
  "training.run_test_after_fit=true"
  "transforms.cpu_tr_transforms=CPU_seg_anatomical_roi_train_transforms"
  "transforms.cpu_val_transforms=CPU_seg_test_transforms"
  "transforms.gpu_tr_transforms=GPU_all_train_transforms"
  "model.ckpt_every_n_epoch=25"
  "logger.wandb_logging=false"
  "logger.log_to_stdout=true"
  "logger.progress_bar=true"
  "logger.log_every_n_steps=25"
  "hydra.run.dir=${RUN_DIR}"
  "hydra.job.chdir=false"
)
dino3d_print_command_and_run

echo "DINO3D_ANATOMICAL_ROI_FINISHED time=$(date --iso-8601=seconds) task=${DATASET} fold=${FOLD}"
