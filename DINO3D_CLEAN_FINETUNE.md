# DINO3D clean fine-tuning baseline

This patch creates the first controlled downstream comparison for the trained
DINO3D checkpoint. It deliberately does **not** add GIN, CarveMix, LoRA, or a
new augmentation policy. The existing task preprocessing and training
augmentations remain in place.

## What was wrong with the first run

The checkpoint diagnostic ruled out a broken or collapsed pretrained model:
all teacher-backbone tensors transferred and the extracted features had useful
rank and variance. The downstream protocol nevertheless had four important
problems:

1. The whole ViT was optimized immediately on very small labelled datasets.
2. Classification and regression used only the final CLS token.
3. Segmentation used only the final patch-token layer and a randomly initialized
   transposed-convolution decoder.
4. Validation repeatedly sampled random crops instead of evaluating every
   validation subject; segmentation checkpoints were selected by crop loss.
5. The existing five entries in `split_80_10_10.json` are repeated random
   80/10 holdouts. Their validation sets can overlap, so they are not a true
   five-fold partition.

For multi-modal tasks, the one-channel pretrained stem was also repeated across
modalities. That averages modalities before the transformer and removes their
identity.

## Clean protocol implemented here

- The pretrained teacher backbone is frozen.
- Classification/regression concatenate CLS tokens from the final four blocks
  and the mean final-layer patch token, matching the standard DINOv2 linear
  readout.
- Segmentation concatenates patch tokens from the final four blocks and trains
  a two-layer point-wise dense head, then interpolates the logits.
- Each MRI modality is encoded independently by the shared single-channel
  teacher; a trainable projection mixes modality tokens afterward.
- Validation visits each validation subject exactly once.
- Segmentation validation uses full-volume sliding-window inference and the
  best checkpoint is selected by foreground Dice.
- Regression exposes MAE during validation and selects the best checkpoint by
  validation MAE.
- Classification keeps validation loss as the checkpoint criterion because
  some small folds may not contain both classes; macro AUROC is still logged.
- A deterministic `split_5fold_cv.json` partitions the non-test 90% into five
  disjoint validation folds. Classification folds are stratified. The existing
  held-out 10% test set is left unchanged. This yields approximately 72% train,
  18% validation, and 10% held-out test per fold, so the FOMO baseline must be
  rerun on the same new folds for a strict comparison.
- Five folds are trained for each of the five tasks (25 independent one-GPU jobs).
- Every fold saves `best.ckpt`, `last.ckpt`, and periodic checkpoints every 25
  epochs. It also saves `validation_best_metrics.json` and the existing test
  prediction JSON.

## Required `.env.finetune` values

The Slurm files contain no personal repository path. Submit from the repository
root, or export `CODE_ROOT` before calling `sbatch`.

At minimum, `asparagus/.env.finetune` must define:

```bash
CODE_ROOT="/absolute/path/to/FOMO26_code"
SHARED_ROOT="/mnt/project/perun2601396"

FOMO_ROOT="${CODE_ROOT}"
ASPARAGUS_CONFIGS="${CODE_ROOT}/asparagus/configs"
ASPARAGUS_DATA="${SHARED_ROOT}/FOMO26_processed/baseline"
ASPARAGUS_MODELS="${SHARED_ROOT}/FOMO26_models/${USER}"
ASPARAGUS_RESULTS="${SHARED_ROOT}/FOMO26_results/${USER}"
ASPARAGUS_RAW_LABELS="${SHARED_ROOT}/FOMO26_raw_labels"

# Optional overrides. These defaults are already used by the Slurm scripts.
FOMO_ENV_PATH="${CODE_ROOT}/fomo_env"
DINO3D_PRETRAIN_CKPT="${SHARED_ROOT}/FOMO26_checkpoints/dinov3_3d_stage1_71494_last.ckpt"
DINO3D_CLEAN_MODEL_ROOT="${SHARED_ROOT}/FOMO26_models/dino3d_clean_cv"
DINO3D_CLEAN_RESULT_ROOT="${SHARED_ROOT}/FOMO26_results/dino3d_clean_cv"
```

Do not commit either `.env` file.

## Run it

All Python commands below run inside Slurm jobs.

From the repository root:

```bash
cd /absolute/path/to/FOMO26_code

git diff --check
bash -n slurm/dino3d/15a_preflight_dino3d_clean_finetune.slurm
bash -n slurm/dino3d/15b_create_fomo26_fivefold_splits.slurm
bash -n slurm/dino3d/15_finetune_dino3d_clean_cv.slurm

sbatch slurm/dino3d/15a_preflight_dino3d_clean_finetune.slurm
sbatch slurm/dino3d/15b_create_fomo26_fivefold_splits.slurm
```

Do not submit training until the preflight output ends with:

```text
PREFLIGHT_CLASSIFIER_OK
PREFLIGHT_SEGMENTER_OK
PREFLIGHT_DINO3D_CLEAN_FINETUNE_OK
```

The split job must print `FIVEFOLD_SPLIT_OK` (or
`FIVEFOLD_SPLIT_ALREADY_OK`) for all five tasks. It creates a new split file and
does not alter `split_80_10_10.json` or `TEST_80_10_10.json`. If an existing
`split_5fold_cv.json` has different contents, inspect it before deliberately
replacing it with:

```bash
OVERWRITE_SPLITS=true sbatch slurm/dino3d/15b_create_fomo26_fivefold_splits.slurm
```

Then submit the 25-job cross-validation array:

```bash
sbatch slurm/dino3d/15_finetune_dino3d_clean_cv.slurm
```

The mapping is five consecutive folds per task:

| Array IDs | Task |
|---:|---|
| 0-4 | CLS002 Infarct |
| 5-9 | SEG009 Meningioma |
| 10-14 | REGR002 BrainAge |
| 15-19 | SEG010 TrigeminalNeuralgia |
| 20-24 | CLS003 Polymicrogyria |

The default `%5` limit runs at most five folds concurrently.

## Monitor results

The job output and each run's `.log` contain epoch metrics. Useful commands are:

```bash
grep -hE 'Current Epoch:|val/loss:|val/foreground_dice:|val/MAE:|val/auroc_macro:' \
  /mnt/project/perun2601396/FOMO26_job_outputs/dino3d-clean-cv_*.out | tail -n 100

find "${DINO3D_CLEAN_MODEL_ROOT}" -type f \
  \( -name 'best.ckpt' -o -name 'last.ckpt' -o -name 'periodic-*.ckpt' \
  -o -name 'validation_best_metrics.json' \) -print | sort
```

Classification/regression produce two periodic checkpoints per fold at a
50-epoch budget; segmentation produces six per fold at a 150-epoch budget.
Together with best and last, that is normally four checkpoint files for each
classification/regression fold and eight for each segmentation fold. A
periodic checkpoint can coincide in content with `last.ckpt`, but it is kept as
a separate restart point.

## Resume a failed array element

Use the original array job ID as `EXPERIMENT_ID`. For example, to resume array
element 7 from experiment 81234:

```bash
RESUME=true EXPERIMENT_ID=81234 \
  sbatch --array=7 slurm/dino3d/15_finetune_dino3d_clean_cv.slurm
```

The pipeline passes `ckpt_path="last"` only for a resumed run. A fresh run still
loads the pretrained DINO checkpoint as initialization.

## Rules for the comparison

Use `split_5fold_cv.json`, the same task preprocessing, the same test split, and
the same metric definitions for the FOMO baseline and this DINO run. Report
mean and standard deviation across the five validation folds. Keep the fixed
test set untouched until the protocol and checkpoint rule have been chosen
from validation only.

The next augmentation experiments should start only after this clean result is
recorded. Each later experiment should change one named factor relative to this
protocol.

## Method references

- DINOv2 linear evaluation: <https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/linear.py>
- 3DINO paper and official implementation: <https://www.nature.com/articles/s41746-025-02035-w> and <https://github.com/AICONSlab/3DINO>
- nnU-Net cross-validation workflow: <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md>
