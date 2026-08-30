# FOMO26 real-case SIF tests

This bundle tests the final Task 1–5 submission containers on labelled MRI cases from the finetuning dataset. It does not modify or rebuild any SIF.

For every selected case it runs two independent container invocations:

1. the container default, with TTA enabled;
2. the same prediction with `--no-tta`.

It uses `TEST_80_10_10.json` and the corresponding preprocessing metadata to select held-out cases. It repairs the common `/project/...` versus `/mnt/project/...` source-path mismatch. By default it fails instead of silently testing training data.

## Metrics

- Task 1 and Task 5: probability validity, accuracy, sensitivity, specificity, balanced accuracy and AUROC.
- Task 3: MAE, RMSE and signed bias in years.
- Task 2 and Task 4: output shape/affine, valid label values, foreground volume, connected components, per-class Dice, macro foreground Dice, empty-mask/whole-mask warnings and TTA/no-TTA agreement.
- All tasks: prediction success, runtime, stderr/stdout tails and a direct comparison of default TTA with no-TTA.

The resulting numbers are a small operational sanity test. With only five cases per task they are not substitutes for the complete cross-validation results or the official hidden challenge score.

## Install

Extract the bundle at the repository root. You should then have:

```text
FOMO26_code/FOMO26_REAL_SIF_TESTS/
```

No package installation or SIF rebuild is needed. The scripts use the existing `fomo_env`, Apptainer and already-built SIFs.

## Run Tasks 1–5

```bash
cd /mnt/home/brseke961/FOMO26_project/FOMO26_code

export CODE_ROOT="$PWD"
export TEST_ROOT="$CODE_ROOT/FOMO26_REAL_SIF_TESTS"
export CASES_PER_TASK=5

TEST_SUBMISSION=$(sbatch --parsable \
    --export="ALL,CODE_ROOT=$CODE_ROOT,TEST_ROOT=$TEST_ROOT,CASES_PER_TASK=$CASES_PER_TASK" \
    "$TEST_ROOT/slurm/01_test_real_sif_tasks.slurm")

TEST_JOB_ID="${TEST_SUBMISSION%%;*}"

echo "TEST_JOB_ID=$TEST_JOB_ID"
```

The array uses at most three GPUs simultaneously. Array indices map to Tasks 1–5.

Monitor it with:

```bash
squeue -u brseke961
```

After it finishes:

```bash
grep -HnE \
'REAL_SIF_TASK_FINISHED|Traceback|ERROR|error:|CUDA out of memory|CANCELLED|TIME LIMIT' \
/mnt/project/perun2601396/FOMO26_job_outputs/fomo26-real-sif_"${TEST_JOB_ID}"_*.{out,err}
```

All five output files should contain `REAL_SIF_TASK_FINISHED` with all selected cases successful.

## If no held-out source paths can be resolved

The default behavior is to stop, because silently using training cases could make the model look misleadingly good. Inspect the error first. For an operational-only test on any labelled raw finetuning cases, explicitly enable the fallback:

```bash
export ALLOW_ALL_DATA_FALLBACK=true

FALLBACK_SUBMISSION=$(sbatch --parsable \
    --export="ALL,CODE_ROOT=$CODE_ROOT,TEST_ROOT=$TEST_ROOT,CASES_PER_TASK=$CASES_PER_TASK,ALLOW_ALL_DATA_FALLBACK=true" \
    "$TEST_ROOT/slurm/01_test_real_sif_tasks.slurm")

TEST_JOB_ID="${FALLBACK_SUBMISSION%%;*}"
echo "TEST_JOB_ID=$TEST_JOB_ID"
```

The final report will prominently mark these results as not held out.

## Create the single report

Only submit the analysis job after the array is no longer present in `squeue`:

```bash
export TEST_RUN_ID="$TEST_JOB_ID"

ANALYSIS_SUBMISSION=$(sbatch --parsable \
    --export="ALL,CODE_ROOT=$CODE_ROOT,TEST_ROOT=$TEST_ROOT,TEST_RUN_ID=$TEST_RUN_ID" \
    "$TEST_ROOT/slurm/02_analyze_real_sif_tasks.slurm")

ANALYSIS_JOB_ID="${ANALYSIS_SUBMISSION%%;*}"
echo "ANALYSIS_JOB_ID=$ANALYSIS_JOB_ID"
```

When it finishes, display the complete report:

```bash
cat "/mnt/project/perun2601396/FOMO26_submission_tests/real_sif_${TEST_RUN_ID}/real_sif_evaluation.txt"
```

Paste that text into the chat before changing TTA or rebuilding any container.

## Result interpretation

The report gives two separate conclusions:

- `TECHNICAL_STATUS`: whether both TTA and no-TTA produced valid outputs for every selected case.
- `DEFAULT_TTA_RECOMMENDATION`: whether this small labelled sample supports keeping or disabling/reworking TTA.

For segmentation, inspect both Dice and foreground voxel counts. A technically valid all-background NIfTI file is not considered a successful model sanity result when the target contains foreground.
