#!/bin/bash
#SBATCH --job-name=fomo-ft
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/project/perun2601396/FOMO26_job_outputs/brseke961/%x_%j.out
#SBATCH --error=/project/perun2601396/FOMO26_job_outputs/brseke961/%x_%j.err

set -euo pipefail

: "${TASK:?Submit with --export=ALL,TASK=1 or TASK=CLS002 etc.}"

CODE_ROOT="/mnt/data/home/brseke961/FOMO26_project/FOMO26_code"
ENV_PATH="${CODE_ROOT}/fomo_env"

# Clean symlink to the pretrained checkpoint, avoiding '=' characters in the Hydra path.
PRETRAIN_CKPT="/project/perun2601396/FOMO26_checkpoints/pretrain_21746_last.ckpt"

source "${ENV_PATH}/bin/activate"

set -a
source "${CODE_ROOT}/asparagus/.env.finetune"
set +a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

FOLD="${FOLD:-0}"

case "${TASK}" in
  1|CLS002|CLS002_FOMO26_Infarct)
    ASP_CMD="asp_finetune_cls"
    DATASET="CLS002_FOMO26_Infarct"
    MODEL="resenc_unet_b_clsreg"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    EPOCHS="${EPOCHS:-50}"
    EXTRA_OVERRIDES=()
    ;;

  2|SEG009|SEG009_FOMO26_Meningioma)
    ASP_CMD="asp_finetune_seg"
    DATASET="SEG009_FOMO26_Meningioma"
    MODEL="resenc_unet_b"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    EPOCHS="${EPOCHS:-150}"
    EXTRA_OVERRIDES=(
      "training.warmup_epochs=10"
      "training.decoder_warmup_epochs=10"
      "training.patch_size=[160,160,160]"
    )
    ;;

  3|REGR002|REGR002_FOMO26_BrainAge)
    ASP_CMD="asp_finetune_reg"
    DATASET="REGR002_FOMO26_BrainAge"
    MODEL="resenc_unet_b_clsreg"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    EPOCHS="${EPOCHS:-50}"
    EXTRA_OVERRIDES=()
    ;;

  4|SEG010|SEG010_FOMO26_TrigeminalNeuralgia)
    ASP_CMD="asp_finetune_seg"
    DATASET="SEG010_FOMO26_TrigeminalNeuralgia"
    MODEL="resenc_unet_b"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    EPOCHS="${EPOCHS:-150}"
    EXTRA_OVERRIDES=(
      "training.warmup_epochs=10"
      "training.decoder_warmup_epochs=10"
      "training.patch_size=[160,160,160]"
    )
    ;;

  5|CLS003|CLS003_FOMO26_Polymicrogyria)
    ASP_CMD="asp_finetune_cls"
    DATASET="CLS003_FOMO26_Polymicrogyria"
    MODEL="resenc_unet_b_clsreg"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    EPOCHS="${EPOCHS:-50}"
    EXTRA_OVERRIDES=()
    ;;

  *)
    echo "Unknown TASK=${TASK}"
    echo "Use TASK=1,2,3,4,5 or CLS002/SEG009/REGR002/SEG010/CLS003"
    exit 1
    ;;
esac

echo "Running on node: $(hostname)"
echo "Started at: $(date)"
echo "TASK=${TASK}"
echo "DATASET=${DATASET}"
echo "COMMAND=${ASP_CMD}"
echo "MODEL=${MODEL}"
echo "FOLD=${FOLD}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "EPOCHS=${EPOCHS}"
echo "Python: $(which python)"
python --version

echo
echo "=== GPU ==="
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

echo
echo "=== Environment paths ==="
echo "ASPARAGUS_DATA=${ASPARAGUS_DATA}"
echo "ASPARAGUS_MODELS=${ASPARAGUS_MODELS}"
echo "ASPARAGUS_RESULTS=${ASPARAGUS_RESULTS}"

echo
echo "=== Checking pretrained checkpoint ==="
ls -lh "${PRETRAIN_CKPT}"

echo
echo "=== Checking dataset files ==="
ls -lh "${ASPARAGUS_DATA}/${DATASET}/paths.json"
ls -lh "${ASPARAGUS_DATA}/${DATASET}/split_80_10_10.json"
ls -lh "${ASPARAGUS_DATA}/${DATASET}/TEST_80_10_10.json"

mkdir -p "${ASPARAGUS_MODELS}"
mkdir -p "${ASPARAGUS_RESULTS}"

cd "${CODE_ROOT}/asparagus"

RUN_DIR="${ASPARAGUS_MODELS}/${DATASET}/runs/slurm_${SLURM_JOB_ID}"
mkdir -p "$(dirname "${RUN_DIR}")"

echo
echo "Hydra run dir: ${RUN_DIR}"

CMD=(
  "${ASP_CMD}"
  "task=${DATASET}"
  "+model=${MODEL}"
  "checkpoint_path=${PRETRAIN_CKPT}"
  "data.train_split=split_80_10_10"
  "data.test_split=TEST_80_10_10"
  "data.fold=${FOLD}"
  "hardware.num_devices=1"
  "hardware.num_workers=${SLURM_CPUS_PER_TASK}"
  "training.batch_size=${BATCH_SIZE}"
  "training.epochs=${EPOCHS}"
  "logger.wandb_logging=false"
  "logger.log_to_stdout=true"
  "logger.progress_bar=true"
  "hydra.run.dir=${RUN_DIR}"
  "hydra.job.chdir=false"
)

CMD+=( "${EXTRA_OVERRIDES[@]}" )

echo
echo "=== Command ==="
printf '%q ' "${CMD[@]}"
echo

echo
echo "Starting finetuning at: $(date)"
"${CMD[@]}"

echo
echo "Finished at: $(date)"

echo
echo "=== Newest checkpoints ==="
find "${ASPARAGUS_MODELS}" -type f -name "*.ckpt" \
  -printf "%TY-%Tm-%Td %TH:%TM:%TS  %s bytes  %p\n" | sort | tail -30

deactivate || true
exit 0
