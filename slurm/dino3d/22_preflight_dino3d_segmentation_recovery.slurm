#!/bin/bash
#SBATCH --job-name=dino3d-seg-recovery-preflight
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
source "${CODE_ROOT}/slurm/dino3d/segmentation_recovery_common.sh"
dino3d_recovery_init

cd "${CODE_ROOT}/asparagus"
echo "DINO3D_SEGMENTATION_RECOVERY_PREFLIGHT_START time=$(date --iso-8601=seconds)"
"${PYTHON}" -m compileall -q \
  asparagus/analysis/prepare_one_case_overfit.py \
  asparagus/analysis/estimate_fold_anatomical_roi.py \
  asparagus/analysis/check_one_case_overfit.py \
  asparagus/analysis/preflight_dino3d_segmentation_recovery.py \
  asparagus/modules/losses/unified_focal.py \
  asparagus/modules/transforms/anatomical_roi.py \
  asparagus/modules/transforms/presets/train.py \
  asparagus/modules/lightning_modules/segmentation_module.py \
  asparagus/pipeline/run/finetune_seg.py

"${PYTHON}" -m asparagus.analysis.preflight_dino3d_segmentation_recovery \
  --config "${CODE_ROOT}/asparagus/configs/default_finetune_seg.yaml"
echo "DINO3D_SEGMENTATION_RECOVERY_PREFLIGHT_FINISHED time=$(date --iso-8601=seconds)"
