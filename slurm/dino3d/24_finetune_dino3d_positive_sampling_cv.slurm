#!/bin/bash
#SBATCH --job-name=dino3d-positive-cv
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --account=perun2601396
#SBATCH --array=0-9%8
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%A_%a.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
source "${CODE_ROOT}/slurm/dino3d/segmentation_recovery_common.sh"
dino3d_recovery_init
dino3d_select_segmentation_task "${SLURM_ARRAY_TASK_ID}"
dino3d_validate_segmentation_data
dino3d_resolve_experiment_id

if [[ "${DATASET}" == "SEG009_FOMO26_Meningioma" ]]; then
  CPU_TRANSFORM="CPU_seg_train_transforms_fg75"
  FOREGROUND_PROBABILITY="0.75"
else
  CPU_TRANSFORM="CPU_seg_train_transforms_fg100"
  FOREGROUND_PROBABILITY="1.00"
fi

MODEL_ROOT="${DINO3D_POSITIVE_SAMPLING_MODEL_ROOT:-${SHARED_ROOT}/FOMO26_models/dino3d_positive_sampling_cv}"
RESULT_ROOT="${DINO3D_POSITIVE_SAMPLING_RESULT_ROOT:-${SHARED_ROOT}/FOMO26_results/dino3d_positive_sampling_cv}"
RUN_DIR="${MODEL_ROOT}/experiment_${EXPERIMENT_ID}/${DATASET}/fold_${FOLD}"
mkdir -p "${RUN_DIR}" "${RESULT_ROOT}"
export ASPARAGUS_MODELS="${MODEL_ROOT}"
export ASPARAGUS_RESULTS="${RESULT_ROOT}"

cd "${CODE_ROOT}/asparagus"
echo "DINO3D_POSITIVE_SAMPLING_START time=$(date --iso-8601=seconds)"
echo "task=${DATASET} fold=${FOLD} foreground_probability=${FOREGROUND_PROBABILITY}"
echo "checkpoint=${PRETRAIN_CKPT} run_dir=${RUN_DIR} resume=${RESUME}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

CMD=(
  "${PYTHON}" -m "${MODULE}"
  "task=${DATASET}"
  "+model=dinov3_downstream"
  "checkpoint_path=${PRETRAIN_CKPT}"
  "root=datapaper"
  "stem=dinov3_3d_positive_sampling_linear"
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
  "training.sliding_window_validation=true"
  "training.inference_overlap=0.75"
  "training.carvemix_probability=0.0"
  "training.segmentation_loss=dice_ce"
  "training.checkpoint_monitor=val/foreground_dice"
  "training.checkpoint_mode=max"
  "training.run_best_validation_after_fit=true"
  "training.run_test_after_fit=true"
  "transforms.cpu_tr_transforms=${CPU_TRANSFORM}"
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

echo "DINO3D_POSITIVE_SAMPLING_FINISHED time=$(date --iso-8601=seconds) task=${DATASET} fold=${FOLD}"
