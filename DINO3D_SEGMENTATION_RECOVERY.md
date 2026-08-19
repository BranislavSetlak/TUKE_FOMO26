# DINO3D segmentation-recovery experiments

This bundle implements recommendations **1, 3, 4, 5, and 6** from the
segmentation-collapse review. It extends the previously applied clean
fine-tuning and GIN/CarveMix patches. It does not change the pretrained DINO
checkpoint or any existing result directory.

## Experiments

1. **One-positive-case overfit diagnostic:** one SEG009 job and one SEG010 job.
   The complete model is unfrozen, all augmentation is disabled, train and
   validation intentionally use the same positive ROI, and validation reports
   foreground Dice at probability thresholds 0.01, 0.05, 0.10, 0.25, and
   0.50. This is a diagnostic, not a publishable result.
2. **Aggressive foreground sampling:** five folds each for SEG009 and SEG010.
   SEG009 uses 75% foreground-guaranteed crops; SEG010 uses 100%. Everything
   else matches the clean frozen-backbone protocol.
3. **Anatomical local stage:** five SEG010 folds. Each fold estimates a fixed
   normalized ROI center using only that fold's training labels. A 96^3 ROI is
   resized to the standard 160^3 network input, predicted locally, resized
   back, and pasted into the full-volume logit canvas. No validation or test
   label is used to choose the ROI.
4. **Asymmetric Unified Focal loss:** five folds each for SEG009 and SEG010,
   using the paper/reference defaults lambda=0.5, delta=0.6, gamma=0.5.
5. **False-negative-oriented Tversky+CE:** five folds each for SEG009 and
   SEG010. In this repository alpha multiplies false positives and beta
   multiplies false negatives, so this experiment uses alpha=0.3, beta=0.7,
   Tversky weight 1.0, and CE weight 0.2.

Except for the one-case diagnostic, every experiment retains the same five
folds, held-out test set, seeds, 150 epochs, batch size, standard augmentation,
frozen backbone, validation protocol, and checkpoint selection as the clean
baseline. This makes each experiment a one-factor ablation.

## Apply the bundle

Unzip it outside the repository, then from the repository root run:

```bash
PATCH_ROOT=/absolute/path/to/FOMO26_DINO3D_SEGMENTATION_RECOVERY_PATCH

cp -a "${PATCH_ROOT}/asparagus/." asparagus/
cp -a "${PATCH_ROOT}/slurm/." slurm/
cp "${PATCH_ROOT}/DINO3D_SEGMENTATION_RECOVERY.md" ./
```

The commands replace only files included in the bundle and add new files. They
do not delete code, checkpoints, or results.

Inspect the result:

```bash
git status --short
git diff --check

for SCRIPT in slurm/dino3d/{22,23,24,25,26,27,28}_*.slurm; do
  bash -n "${SCRIPT}"
done
bash -n slurm/dino3d/segmentation_recovery_common.sh
```

## Optional `.env.finetune` entries

All paths already have defaults under `${SHARED_ROOT}`. Add these only if you
want different locations:

```bash
DINO3D_OVERFIT_MODEL_ROOT="${SHARED_ROOT}/FOMO26_models/dino3d_one_case_overfit"
DINO3D_OVERFIT_RESULT_ROOT="${SHARED_ROOT}/FOMO26_results/dino3d_one_case_overfit"

DINO3D_POSITIVE_SAMPLING_MODEL_ROOT="${SHARED_ROOT}/FOMO26_models/dino3d_positive_sampling_cv"
DINO3D_POSITIVE_SAMPLING_RESULT_ROOT="${SHARED_ROOT}/FOMO26_results/dino3d_positive_sampling_cv"

DINO3D_ANATOMICAL_ROI_MODEL_ROOT="${SHARED_ROOT}/FOMO26_models/dino3d_anatomical_roi_seg010_cv"
DINO3D_ANATOMICAL_ROI_RESULT_ROOT="${SHARED_ROOT}/FOMO26_results/dino3d_anatomical_roi_seg010_cv"

DINO3D_UNIFIED_FOCAL_MODEL_ROOT="${SHARED_ROOT}/FOMO26_models/dino3d_unified_focal_cv"
DINO3D_UNIFIED_FOCAL_RESULT_ROOT="${SHARED_ROOT}/FOMO26_results/dino3d_unified_focal_cv"

DINO3D_FN_TVERSKY_MODEL_ROOT="${SHARED_ROOT}/FOMO26_models/dino3d_fn_tversky_cv"
DINO3D_FN_TVERSKY_RESULT_ROOT="${SHARED_ROOT}/FOMO26_results/dino3d_fn_tversky_cv"
```

Existing `CODE_ROOT`, `SHARED_ROOT`, `FOMO_ENV_PATH`, `ASPARAGUS_DATA`, and
`DINO3D_PRETRAIN_CKPT` continue to be used. No personal repository path is
hard-coded in these scripts.

## Run order

All Python execution occurs inside Slurm.

### 1. Preflight

```bash
cd /absolute/path/to/FOMO26_code
sbatch slurm/dino3d/22_preflight_dino3d_segmentation_recovery.slurm
```

Do not submit training until its output contains:

```text
PREFLIGHT_UNIFIED_FOCAL_OK
PREFLIGHT_FN_TVERSKY_OK
PREFLIGHT_SEGMENTATION_RECOVERY_TRANSFORMS_OK
PREFLIGHT_SEGMENTATION_RECOVERY_CONFIG_OK
DINO3D_SEGMENTATION_RECOVERY_PREFLIGHT_OK
```

### 2. One-case diagnostic

```bash
sbatch slurm/dino3d/23_overfit_one_positive_case.slurm
```

Array element 0 is SEG009 and element 1 is SEG010. The job selects the
positive case with the most stored foreground locations from source fold 0.
Override the source fold with, for example:

```bash
SOURCE_FOLD=2 sbatch slurm/dino3d/23_overfit_one_positive_case.slurm
```

The output ends with either `ONE_CASE_OVERFIT_PASS` or
`ONE_CASE_OVERFIT_WARNING`. A pass requires the best thresholded foreground
Dice to reach at least 0.90. Do not launch all CV arrays if either task cannot
memorize its one positive ROI.

### 3. Screen one fold per experiment

Run one fold per applicable task before spending 35 GPU jobs:

```bash
sbatch --array=0,5%2 slurm/dino3d/24_finetune_dino3d_positive_sampling_cv.slurm
sbatch --array=0 slurm/dino3d/25_finetune_dino3d_anatomical_roi_seg010_cv.slurm
sbatch --array=0,5%2 slurm/dino3d/26_finetune_dino3d_unified_focal_cv.slurm
sbatch --array=0,5%2 slurm/dino3d/27_finetune_dino3d_fn_tversky_cv.slurm
```

For scripts 24, 26, and 27, IDs 0-4 are SEG009 folds 0-4 and IDs 5-9
are SEG010 folds 0-4. Script 25 contains only SEG010 folds 0-4.

### 4. Run full five-fold arrays

After a one-fold experiment shows non-zero foreground recall and meaningful
Dice, submit the complete array:

```bash
sbatch slurm/dino3d/24_finetune_dino3d_positive_sampling_cv.slurm
sbatch slurm/dino3d/25_finetune_dino3d_anatomical_roi_seg010_cv.slurm
sbatch slurm/dino3d/26_finetune_dino3d_unified_focal_cv.slurm
sbatch slurm/dino3d/27_finetune_dino3d_fn_tversky_cv.slurm
```

Scripts 24, 26, and 27 allow up to eight simultaneous one-GPU array elements.
Slurm may place them on one eight-GPU node or across several nodes.

The anatomical ROI job refuses to train when its training-label ROI coverage
is below 90%. If 96^3 is too small, resubmit the same element with a larger ROI:

```bash
ROI_SIZE="128 128 128" sbatch --array=0 \
  slurm/dino3d/25_finetune_dino3d_anatomical_roi_seg010_cv.slurm
```

Do not lower the coverage guard merely to make the job start.

## Monitor collapse and threshold behavior

```bash
grep -hE \
'Current Epoch:|train/positive_patch_fraction:|val/foreground_dice:|val/best_threshold_dice:|val/foreground_dice_t|val/foreground_recall_t|val/pred_foreground_fraction:|val/target_foreground_probability:' \
/mnt/project/perun2601396/FOMO26_job_outputs/dino3d-{overfit-one,positive-cv,roi-seg010,unified-focal,fn-tversky}_*.out \
| tail -n 200
```

For recommendation 3, `train/positive_patch_fraction` should approach 0.75
for SEG009 and 1.0 for SEG010. The probability-threshold metrics distinguish
an uncalibrated low-confidence foreground signal from true representation
collapse.

## Analyze completed CV experiments

Replace each job ID with the ID returned by its corresponding `sbatch` call:

```bash
VARIANT=positive_sampling EXPERIMENT_ID=POSITIVE_JOB_ID REQUIRE_COMPLETE=true \
  sbatch slurm/dino3d/28_analyze_dino3d_segmentation_recovery_cv.slurm

VARIANT=anatomical_roi EXPERIMENT_ID=ROI_JOB_ID REQUIRE_COMPLETE=true \
  sbatch slurm/dino3d/28_analyze_dino3d_segmentation_recovery_cv.slurm

VARIANT=unified_focal EXPERIMENT_ID=UNIFIED_JOB_ID REQUIRE_COMPLETE=true \
  sbatch slurm/dino3d/28_analyze_dino3d_segmentation_recovery_cv.slurm

VARIANT=fn_tversky EXPERIMENT_ID=TVERSKY_JOB_ID REQUIRE_COMPLETE=true \
  sbatch slurm/dino3d/28_analyze_dino3d_segmentation_recovery_cv.slurm
```

Each analysis writes one text report below its variant result root.

## Resume failed elements

Use the original array job ID. For example:

```bash
RESUME=true EXPERIMENT_ID=ORIGINAL_JOB_ID \
  sbatch --array=7 slurm/dino3d/24_finetune_dino3d_positive_sampling_cv.slurm
```

Each CV fold saves `best.ckpt`, `last.ckpt`, and periodic checkpoints every 25
epochs, as in the clean comparison.

## References

- nnU-Net foreground sampling implementation:
  <https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/dataloading/data_loader.py>
- Tiny-target coarse-to-fine segmentation:
  <https://pubmed.ncbi.nlm.nih.gov/31352338/>
- Unified Focal loss paper and reference implementation:
  <https://arxiv.org/abs/2102.04525> and
  <https://github.com/mlyg/unified-focal-loss>
- Tversky loss for 3-D medical segmentation:
  <https://arxiv.org/abs/1706.05721>
