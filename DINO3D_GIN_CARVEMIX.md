# DINO3D GIN and CarveMix fine-tuning experiments

This bundle extends the **corrected clean five-fold DINO3D fine-tuning patch**.
Apply that clean patch first. Do not apply this bundle directly to the original
August 13 repository ZIP, because that ZIP predates the foreground-crop,
checkpoint-loading, full-validation, and segmentation-metric fixes.

## Experimental design

The clean experiment is the unchanged reference. This bundle adds:

1. **GIN:** all five tasks × five folds = **25 new fine-tunings**.
2. **GIN + CarveMix:** the two segmentation tasks × five folds = **10 new
   fine-tunings**.
3. **Optional FP-loss ablation:** the two segmentation tasks × five folds =
   **10 new fine-tunings**. This is separate from the augmentation experiments.

CarveMix requires voxelwise lesion masks. It has no valid target rule for the
two classification tasks or age regression. Therefore a literal second array
of 25 would contain 15 jobs identical to GIN-only and waste GPU time. For a
five-task results table, reuse the three non-segmentation GIN results in the
GIN+CarveMix row. There are 50 task/fold result cells across the two named
experiments, but only **35 distinct, scientifically meaningful trainings**.

The GIN and GIN+CarveMix arrays retain the clean experiment's checkpoint,
folds, test set, seeds, frozen backbone, task budgets, batch size, optimizer,
checkpoint selection, and standard augmentations. The segmentation loss stays
unchanged in both arrays, so augmentation is the only intended change.

## What is implemented

- Paper-style 3-D GIN: four random convolutional layers, kernel sizes 1 or 3,
  Leaky-ReLU between layers, interpolation coefficient sampled from U(0,1),
  and per-sample Frobenius-norm matching.
- GIN is appended after the existing standard GPU intensity transforms and is
  applied only during training.
- Online CarveMix loads a second subject from the same fold's training list,
  independently applies the normal crop/spatial transforms, computes the
  donor lesion's signed Euclidean distance transform, samples the paper's
  adaptive threshold, then pastes image and voxel labels together.
- CarveMix is used with probability 0.5. This gives approximately equal
  original and synthetic samples over training and never touches validation or
  test subjects.
- Multi-class CarveMix defines the ROI from the union of foreground classes
  and pastes the original class labels unchanged.
- Segmentation logs now include predicted foreground fraction, target
  foreground fraction, false-positive fraction, and predicted/target volume
  ratio.
- An optional asymmetric Tversky + cross-entropy loss uses alpha=0.7 for false
  positives and beta=0.3 for false negatives.

SciPy is required for the exact Euclidean distance transform. It is already in
the repository's `requirements.txt`; the preflight will fail clearly if the
active `fomo_env` does not contain it.

## Apply the bundle

Unzip the bundle somewhere outside the repository, then copy its replacement
tree over the repository. From the repository root:

```bash
PATCH_ROOT=/absolute/path/to/FOMO26_DINO3D_GIN_CARVEMIX_PATCH

cp -a "${PATCH_ROOT}/asparagus/." asparagus/
cp -a "${PATCH_ROOT}/slurm/." slurm/
cp "${PATCH_ROOT}/DINO3D_GIN_CARVEMIX.md" ./
```

These commands add new files and replace only the files present in the bundle;
they do not delete other repository files or any checkpoints/results.

Inspect before committing:

```bash
git status --short
git diff --check

bash -n slurm/dino3d/17_preflight_dino3d_augmentations.slurm
bash -n slurm/dino3d/18_finetune_dino3d_gin_cv.slurm
bash -n slurm/dino3d/19_finetune_dino3d_gin_carvemix_cv.slurm
bash -n slurm/dino3d/20_finetune_dino3d_fp_loss_cv.slurm
bash -n slurm/dino3d/21_analyze_dino3d_variant_cv.slurm
```

## Optional `.env.finetune` paths

The scripts have safe defaults under `${SHARED_ROOT}`. You may add these if you
want explicit locations:

```bash
DINO3D_GIN_MODEL_ROOT="${SHARED_ROOT}/FOMO26_models/dino3d_gin_cv"
DINO3D_GIN_RESULT_ROOT="${SHARED_ROOT}/FOMO26_results/dino3d_gin_cv"

DINO3D_GIN_CARVEMIX_MODEL_ROOT="${SHARED_ROOT}/FOMO26_models/dino3d_gin_carvemix_cv"
DINO3D_GIN_CARVEMIX_RESULT_ROOT="${SHARED_ROOT}/FOMO26_results/dino3d_gin_carvemix_cv"

DINO3D_FP_LOSS_MODEL_ROOT="${SHARED_ROOT}/FOMO26_models/dino3d_fp_loss_cv"
DINO3D_FP_LOSS_RESULT_ROOT="${SHARED_ROOT}/FOMO26_results/dino3d_fp_loss_cv"
```

Existing `CODE_ROOT`, `SHARED_ROOT`, `FOMO_ENV_PATH`, `ASPARAGUS_DATA`, and
`DINO3D_PRETRAIN_CKPT` settings continue to be used. No personal code path is
hard-coded in the new Slurm files.

## Run the required preflight

All Python execution remains inside Slurm:

```bash
source asparagus/.env.finetune
cd "${CODE_ROOT}"

sbatch slurm/dino3d/17_preflight_dino3d_augmentations.slurm
```

Do not submit training until its `.out` file contains all four lines:

```text
PREFLIGHT_GIN_OK
PREFLIGHT_CARVEMIX_OK
PREFLIGHT_ASYMMETRIC_TVERSKY_CE_OK
DINO3D_AUGMENTATION_PREFLIGHT_OK
```

## Start the two augmentation experiments

GIN, 25 jobs:

```bash
sbatch slurm/dino3d/18_finetune_dino3d_gin_cv.slurm
```

Array mapping is identical to clean CV: 0-4 Infarct, 5-9 Meningioma,
10-14 BrainAge, 15-19 TrigeminalNeuralgia, and 20-24 Polymicrogyria.

GIN + CarveMix, 10 segmentation jobs:

```bash
sbatch slurm/dino3d/19_finetune_dino3d_gin_carvemix_cv.slurm
```

Array IDs 0-4 are Meningioma folds 0-4; IDs 5-9 are TrigeminalNeuralgia folds
0-4. The two arrays may run concurrently because they write to separate roots.

Record both returned array job IDs. Each job writes `best.ckpt`, `last.ckpt`,
and periodic checkpoints exactly as in the clean experiment.

## Resume one failed element

Use the original experiment/job ID. Examples:

```bash
RESUME=true EXPERIMENT_ID=GIN_ARRAY_JOB_ID \
  sbatch --array=7 slurm/dino3d/18_finetune_dino3d_gin_cv.slurm

RESUME=true EXPERIMENT_ID=CARVEMIX_ARRAY_JOB_ID \
  sbatch --array=7 slurm/dino3d/19_finetune_dino3d_gin_carvemix_cv.slurm
```

## Analyze each experiment

```bash
VARIANT=gin EXPERIMENT_ID=GIN_ARRAY_JOB_ID REQUIRE_COMPLETE=true \
  sbatch slurm/dino3d/21_analyze_dino3d_variant_cv.slurm

VARIANT=gin_carvemix EXPERIMENT_ID=CARVEMIX_ARRAY_JOB_ID REQUIRE_COMPLETE=true \
  sbatch slurm/dino3d/21_analyze_dino3d_variant_cv.slurm
```

The analyzer now accepts a task subset, so the 10-run segmentation experiment
is treated as complete rather than as 15 missing jobs.

## Monitor augmentation and segmentation behavior

```bash
grep -hE \
'Current Epoch:|val/foreground_dice:|val/pred_foreground_fraction:|val/target_foreground_fraction:|val/false_positive_fraction:|val/pred_to_target_volume_ratio:' \
/mnt/project/perun2601396/FOMO26_job_outputs/dino3d-{gin-cv,gin-carvemix}_*.out \
| tail -n 120
```

A predicted/target volume ratio near 1 does not prove spatial correctness, but
ratios in the hundreds or thousands expose over-segmentation immediately.
Always interpret it together with foreground Dice and false-positive fraction.

## False-positive penalty: recommended isolated test

The previous clean result showed two different failures:

- Meningioma: four of five fixed-test folds predicted all background.
- TrigeminalNeuralgia: zero true-positive foreground voxels in every fold, but
  approximately 288,000 to 1,566,466 predicted foreground voxels versus about
  661 true foreground voxels on average.

So simply increasing background pressure can improve the second failure while
making the first worse. The safest first loss ablation is foreground Tversky +
multiclass cross entropy with alpha=0.7 on false positives and beta=0.3 on
false negatives. Run it without GIN or CarveMix:

```bash
sbatch slurm/dino3d/20_finetune_dino3d_fp_loss_cv.slurm
```

Analyze it with `VARIANT=fp_loss`. Select alpha/beta using validation only. If
Meningioma collapses further to background, do not increase alpha; the next
candidate should be alpha=0.6, beta=0.4 rather than a stronger volume penalty.

This loss is a diagnostic, not a complete architectural fix. The DINO dense
head predicts from a frozen 10×10×10 patch-token grid for a 160³ crop (16³
pretraining patches). Both segmentation targets occupy only hundreds of voxels,
and the Trigeminal result has zero spatial overlap. A loss can change the
precision/recall trade-off, but it cannot restore spatial detail absent from
the frozen token grid. If the FP-loss ablation still has near-zero Dice, the
next controlled experiment should unfreeze the last one or two transformer
blocks and/or use a higher-resolution decoder; do not keep increasing the
false-positive coefficient.

Post-processing (validation-tuned probability threshold, connected-component
filtering) may reduce false positives, but it cannot rescue a model with zero
true positives and must not be tuned on the fixed test set.

## Method references

- GIN paper: <https://arxiv.org/abs/2111.12525>
- Official GIN implementation (includes 3-D):
  <https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization>
- CarveMix paper: <https://arxiv.org/abs/2108.06883>
- Official CarveMix implementation: <https://github.com/ZhangxinruBIT/CarveMix>
- Tversky loss paper: <https://arxiv.org/abs/1706.05721>
