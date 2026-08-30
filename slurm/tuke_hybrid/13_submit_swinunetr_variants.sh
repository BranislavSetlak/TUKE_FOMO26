#!/bin/bash

# Submit the two-task preflight, then one 30-element training array capped at
# eight simultaneous GPUs, then the single-file analyzer.

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(pwd)}"
cd "${CODE_ROOT}"
test -d asparagus
test -d slurm/tuke_hybrid

PREFLIGHT_SUBMISSION=$(sbatch --parsable slurm/tuke_hybrid/07_preflight_swinunetr_finetune.slurm)
PREFLIGHT_JOB_ID="${PREFLIGHT_SUBMISSION%%;*}"
TRAIN_SUBMISSION=$(sbatch --parsable \
  --dependency="afterok:${PREFLIGHT_JOB_ID}" \
  slurm/tuke_hybrid/12_finetune_swinunetr_all_variants_cv.slurm)
TRAIN_JOB_ID="${TRAIN_SUBMISSION%%;*}"
ANALYSIS_SUBMISSION=$(sbatch --parsable \
  --dependency="afterany:${TRAIN_JOB_ID}" \
  --export="ALL,NORMAL_EXPERIMENT_ID=${TRAIN_JOB_ID},GIN_EXPERIMENT_ID=${TRAIN_JOB_ID},GIN_CARVEMIX_EXPERIMENT_ID=${TRAIN_JOB_ID},REQUIRE_COMPLETE=true" \
  slurm/tuke_hybrid/11_analyze_swinunetr_variants.slurm)
ANALYSIS_JOB_ID="${ANALYSIS_SUBMISSION%%;*}"

echo "PREFLIGHT_JOB_ID=${PREFLIGHT_JOB_ID}"
echo "TRAIN_JOB_ID=${TRAIN_JOB_ID}"
echo "ANALYSIS_JOB_ID=${ANALYSIS_JOB_ID}"
echo "Training array mapping: 0-9 normal, 10-19 GIN, 20-29 GIN+CarveMix; at most 8 run concurrently."
