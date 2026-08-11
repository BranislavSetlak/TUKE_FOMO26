#!/bin/bash
#SBATCH --job-name=dino3d-pretrain-csv
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="/mnt/home/brseke961/FOMO26_project/FOMO26_code"
PYTHON="${CODE_ROOT}/fomo_env/bin/python"
RUN_ROOT="/mnt/project/perun2601396/FOMO26_models/baseline_pretraining/PT902_FOMO300K_HF/vit_x__3D/script=pretrain/root=datapaper__stem=dinov3_3d_stage1_4gpu/leaf=projects/datapaper/pretrain/dinov3_3d_stage1__clargs=training.seed=260826/split_99_01_00__fold=0/run_id=71494"
JOB_ROOT="/mnt/project/perun2601396/FOMO26_job_outputs"
OUTPUT_DIR="/mnt/project/perun2601396/FOMO26_results/dino3d/pretraining_run_71494"

test -x "${PYTHON}"
test -d "${RUN_ROOT}"
export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"

mapfile -d '' LOG_FILES < <(
  find "${RUN_ROOT}" "${JOB_ROOT}" -type f \
    \( -name 'pretrain.log' -o -name 'dino3d-stage1-4gpu*.out' -o -name 'dino3d-stage1-4gpu*.err' \) \
    -print0 2>/dev/null
)

if (( ${#LOG_FILES[@]} == 0 )); then
  echo "ERROR: No pretraining logs were found." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
"${PYTHON}" -m asparagus.analysis.export_pretrain_history \
  --output-dir "${OUTPUT_DIR}" \
  --steps-per-epoch 500 \
  --global-batch-size 8 \
  --dataset-size 303144 \
  --moving-average 100 \
  "${LOG_FILES[@]}"

echo "Generated files:"
find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.csv' -printf '%s bytes  %p\n' | sort
