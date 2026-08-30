#!/bin/bash

# Urgent final-day pipeline:
#   1. Fast real-data GPU preflights for cls/reg and repaired CarveMix collation.
#   2. 30 cls/reg jobs: normal + GIN, 3 tasks, 5 folds, max 8 GPUs.
#   3. Fixed 10-job GIN+CarveMix segmentation array after all cls/reg jobs end.
#   4. One-file analyzers for both experiment groups.

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-$(pwd)}"
cd "${CODE_ROOT}"
test -d asparagus
test -d slurm/tuke_hybrid
test -f asparagus/.env.pretrain
test -f asparagus/.env.finetune

# Normal and GIN segmentation results already completed in combined job 77542.
# Override this value if those successful runs used another array job ID.
PRIOR_VARIANT_EXPERIMENT_ID="${PRIOR_VARIANT_EXPERIMENT_ID:-77542}"

CLSREG_PREFLIGHT_SUBMISSION=$(sbatch --parsable \
  slurm/tuke_hybrid/14a_preflight_swinunetr_clsreg.slurm)
CLSREG_PREFLIGHT_JOB_ID="${CLSREG_PREFLIGHT_SUBMISSION%%;*}"
SEG_PREFLIGHT_SUBMISSION=$(sbatch --parsable \
  slurm/tuke_hybrid/07_preflight_swinunetr_finetune.slurm)
SEG_PREFLIGHT_JOB_ID="${SEG_PREFLIGHT_SUBMISSION%%;*}"

CLSREG_SUBMISSION=$(sbatch --parsable \
  --dependency="afterok:${CLSREG_PREFLIGHT_JOB_ID}:${SEG_PREFLIGHT_JOB_ID}" \
  slurm/tuke_hybrid/14_finetune_swinunetr_clsreg_normal_gin_cv.slurm)
CLSREG_JOB_ID="${CLSREG_SUBMISSION%%;*}"

# afterany is intentional: even if one downstream element fails, do not lose
# the remaining submission window for the repaired segmentation experiment.
CARVEMIX_SUBMISSION=$(sbatch --parsable \
  --dependency="afterany:${CLSREG_JOB_ID}" \
  slurm/tuke_hybrid/10_finetune_swinunetr_gin_carvemix_cv.slurm)
CARVEMIX_JOB_ID="${CARVEMIX_SUBMISSION%%;*}"

CLSREG_ANALYSIS_SUBMISSION=$(sbatch --parsable \
  --dependency="afterany:${CLSREG_JOB_ID}" \
  --export="ALL,CLSREG_EXPERIMENT_ID=${CLSREG_JOB_ID},REQUIRE_COMPLETE=true" \
  slurm/tuke_hybrid/15_analyze_swinunetr_clsreg.slurm)
CLSREG_ANALYSIS_JOB_ID="${CLSREG_ANALYSIS_SUBMISSION%%;*}"

SEG_ANALYSIS_SUBMISSION=$(sbatch --parsable \
  --dependency="afterany:${CARVEMIX_JOB_ID}" \
  --export="ALL,NORMAL_EXPERIMENT_ID=${PRIOR_VARIANT_EXPERIMENT_ID},GIN_EXPERIMENT_ID=${PRIOR_VARIANT_EXPERIMENT_ID},GIN_CARVEMIX_EXPERIMENT_ID=${CARVEMIX_JOB_ID},REQUIRE_COMPLETE=true" \
  slurm/tuke_hybrid/11_analyze_swinunetr_variants.slurm)
SEG_ANALYSIS_JOB_ID="${SEG_ANALYSIS_SUBMISSION%%;*}"

RECORD_ROOT="${SHARED_ROOT:-/mnt/project/perun2601396}/FOMO26_job_outputs"
mkdir -p "${RECORD_ROOT}"
RECORD="${RECORD_ROOT}/tuke_last_day_pipeline_${CLSREG_JOB_ID}.txt"
{
  echo "CLSREG_PREFLIGHT_JOB_ID=${CLSREG_PREFLIGHT_JOB_ID}"
  echo "SEG_PREFLIGHT_JOB_ID=${SEG_PREFLIGHT_JOB_ID}"
  echo "CLSREG_JOB_ID=${CLSREG_JOB_ID}"
  echo "CARVEMIX_JOB_ID=${CARVEMIX_JOB_ID}"
  echo "CLSREG_ANALYSIS_JOB_ID=${CLSREG_ANALYSIS_JOB_ID}"
  echo "SEG_ANALYSIS_JOB_ID=${SEG_ANALYSIS_JOB_ID}"
  echo "PRIOR_VARIANT_EXPERIMENT_ID=${PRIOR_VARIANT_EXPERIMENT_ID}"
  echo "CLSREG_MAPPING=0-14 normal; 15-29 GIN; each block is CLS002, REGR002, CLS003 with folds 0-4"
  echo "CARVEMIX_MAPPING=0-4 SEG009; 5-9 SEG010; folds 0-4"
} | tee "${RECORD}"
echo "SUBMISSION_RECORD=${RECORD}"
