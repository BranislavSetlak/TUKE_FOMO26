# TUKE SwinUNETR controlled segmentation fine-tuning

This repository-relative overlay extends the validated TUKE SwinUNETR one-case overfit
implementation with controlled five-fold segmentation, classification, and regression experiments.

| Variant | SEG009 folds | SEG010 folds | Difference from normal |
| --- | ---: | ---: | --- |
| Normal | 5 | 5 | Standard spatial and MRI intensity augmentation |
| GIN | 5 | 5 | Adds 3-D GIN after the standard GPU transforms |
| GIN + CarveMix | 5 | 5 | Adds GIN plus online lesion-aware CarveMix at probability 0.5 |

The original segmentation comparison is 30 fine-tunings. The final-day launcher adds **30
classification/regression fine-tunings** (normal and GIN across CLS002, REGR002, and CLS003), then
reruns the **10 repaired GIN+CarveMix segmentation folds**. The same task/fold uses the same seed
for normal and GIN.

## Important fixes included before CV

- Validation uses every validation subject with sliding-window inference over the full processed
  volume. It no longer estimates checkpoint quality from random validation crops.
- Best checkpoints maximize `val/min_foreground_class_dice`, protecting the weaker SEG010 class.
- Exact per-case foreground and per-class Dice are accumulated across the complete fold. A class
  absent from both prediction and target is skipped instead of receiving an artificial Dice of 1.
- F1 no longer uses `ignore_index=0`; background voxels now contribute false positives correctly.
- Predicted foreground fraction, target fraction, false-positive fraction, and predicted/target
  volume ratio are logged, so empty-mask and whole-mask collapse remain visible.
- `best.ckpt`, `last.ckpt`, periodic checkpoints, best-checkpoint validation, and restart from
  `last.ckpt` are supported.
- CarveMix now emits an identical nested metadata schema for mixed and unmixed samples. A real
  two-item `default_collate` check catches the exact `KeyError: 'carvemix'` that stopped job 77542.
- The age-regression target is reshaped to `[batch, 1]`, preventing MSE from silently broadcasting
  `[batch,1]` predictions against `[batch]` labels into a wrong batch-by-batch loss matrix.
- Classification/regression use a checkpoint-compatible pooled Swin bottleneck head; only the
  pretrained Swin encoder is transferred, while the new task head is initialized from scratch.

## Install

Download the ZIP to the repository root, where `ls` shows `asparagus`,
`asparagus_preprocessing`, `slurm`, and `README.md`:

```bash
cd /absolute/path/to/FOMO26_code
unzip -o FOMO26_TUKE_SWINUNETR_FINETUNE.zip

git status --short
git diff --check

for SCRIPT in slurm/tuke_hybrid/{07,08,09,10,11,12,14,14a,15}_*.slurm; do
  bash -n "${SCRIPT}"
done
bash -n slurm/tuke_hybrid/swinunetr_finetune_common.sh
bash -n slurm/tuke_hybrid/13_submit_swinunetr_variants.sh
bash -n slurm/tuke_hybrid/16_submit_swinunetr_last_day_pipeline.sh
```

The ZIP overlays only the listed repository files. It does not change either `.env` file, move
the repository, include a virtual environment, or include checkpoints and results.

## Prerequisites

1. Both one-case overfit tasks should have passed the 0.90 minimum-class-Dice gate.
2. `asparagus/.env.pretrain` and `asparagus/.env.finetune` must exist.
3. Both segmentation tasks must contain `dataset.json`, `paths.json`,
   `split_5fold_cv.json`, and `TEST_80_10_10.json` under `${ASPARAGUS_DATA}`.
4. Use the final pretraining checkpoint, not a `last.ckpt` that is still being replaced.
5. The environment must contain SciPy; CarveMix uses its Euclidean distance transform.

Pin the checkpoint explicitly for the final comparison:

```bash
export TUKE_HYBRID_PRETRAIN_CKPT=/absolute/path/to/final.ckpt
```

If this variable is absent, the scripts first try `production_8gpu_v1/checkpoints/last.ckpt`, then the
newest TUKE hybrid production `last.ckpt` below the model root from `.env.pretrain`.

## Urgent final-day submission

Pin the completed pretraining checkpoint and submit one launcher:

```bash
cd /absolute/path/to/FOMO26_code
export TUKE_HYBRID_PRETRAIN_CKPT=/mnt/project/perun2601396/FOMO26_models/baseline_pretraining/PT902_FOMO300K_HF/tuke_hybrid/production/run_id=production_8gpu_v1/checkpoints/last.ckpt
bash slurm/tuke_hybrid/16_submit_swinunetr_last_day_pipeline.sh
```

The launcher submits two short real-data GPU preflights, then:

- cls/reg array `0-14`: normal CLS002, REGR002, CLS003, five folds each;
- cls/reg array `15-29`: GIN CLS002, REGR002, CLS003, five folds each;
- fixed GIN+CarveMix array after the cls/reg array ends: `0-4` SEG009 and `5-9` SEG010;
- one CPU analyzer for cls/reg and one for the segmentation variants.

Every training array is capped at eight simultaneous one-GPU jobs. The CarveMix dependency is
`afterany`, so it still starts if an individual cls/reg element fails. The launcher assumes the
already-successful normal and GIN segmentation results belong to experiment `77542`; override it
when necessary:

```bash
PRIOR_VARIANT_EXPERIMENT_ID=OTHER_JOB_ID \
bash slurm/tuke_hybrid/16_submit_swinunetr_last_day_pipeline.sh
```

The printed job IDs are also saved under:

```text
/mnt/project/perun2601396/FOMO26_job_outputs/tuke_last_day_pipeline_CLSREG_JOB_ID.txt
```

## Original segmentation-only submission

The older launcher submits the two-task GPU preflight, then one combined 30-element training array,
then the one-file analyzer:

```bash
cd /absolute/path/to/FOMO26_code
bash slurm/tuke_hybrid/13_submit_swinunetr_variants.sh
```

The combined array is capped at eight concurrent one-GPU jobs. Its mapping is:

- `0-4`: normal SEG009 folds 0-4
- `5-9`: normal SEG010 folds 0-4
- `10-14`: GIN SEG009 folds 0-4
- `15-19`: GIN SEG010 folds 0-4
- `20-24`: GIN+CarveMix SEG009 folds 0-4
- `25-29`: GIN+CarveMix SEG010 folds 0-4

The launcher prints and records the preflight, training, and analysis job IDs. Training starts
only if both preflight tasks succeed. The analyzer runs even if a training element fails, writes
the partial report, and exits non-zero when the 30-run result is incomplete.

## Manual submission

To run variants separately:

```bash
sbatch slurm/tuke_hybrid/07_preflight_swinunetr_finetune.slurm

sbatch slurm/tuke_hybrid/08_finetune_swinunetr_normal_cv.slurm
sbatch slurm/tuke_hybrid/09_finetune_swinunetr_gin_cv.slurm
sbatch slurm/tuke_hybrid/10_finetune_swinunetr_gin_carvemix_cv.slurm
```

Each individual array contains 10 elements and is capped at eight concurrent jobs. Record the
three returned job IDs, then submit:

```bash
NORMAL_EXPERIMENT_ID=NORMAL_JOB_ID \
GIN_EXPERIMENT_ID=GIN_JOB_ID \
GIN_CARVEMIX_EXPERIMENT_ID=CARVEMIX_JOB_ID \
REQUIRE_COMPLETE=true \
sbatch slurm/tuke_hybrid/11_analyze_swinunetr_variants.slurm
```

## Training controls

All variants default to the validated architecture and these shared settings:

- full-network fine-tuning; encoder and decoder are trainable;
- pretrained encoder and reconstruction decoder loaded;
- 96×96×96 training crops and sliding-window validation windows;
- batch size 2, 250 training batches per epoch, 150 epochs;
- AdamW at `3e-4`, 10 warmup epochs, bf16 mixed precision;
- full-fold validation every epoch, overlap 0.5;
- standard Dice+CE loss;
- no fixed test-set evaluation during model selection.

Optional controls are read from the submission environment:

```bash
TUKE_FINETUNE_EPOCHS=150
TUKE_FINETUNE_TRAIN_BATCHES_PER_EPOCH=250
TUKE_FINETUNE_WARMUP_EPOCHS=10
TUKE_FINETUNE_CHECK_VAL_EVERY=1
TUKE_FINETUNE_CKPT_EVERY=50
TUKE_FINETUNE_PATCH_SIZE="96 96 96"
TUKE_CARVEMIX_PROBABILITY=0.5
```

Every patch dimension must be divisible by 32. Keep these values identical across all three
variants unless the change itself is the experiment.

## Monitor

Replace `TRAIN_JOB_ID` with the combined training-array ID:

```bash
OUTPUT_ROOT=/mnt/project/perun2601396/FOMO26_job_outputs

grep -HnE \
'TUKE_SWINUNETR_.*_(START|FINISHED)|Traceback|Error executing job|CUDA out of memory|CANCELLED|TIME LIMIT' \
"${OUTPUT_ROOT}"/tuke-swin-variants_TRAIN_JOB_ID_*.{out,err}
```

Epoch metrics only:

```bash
grep -hE \
'Current Epoch:|val/(foreground_dice|macro_foreground_class_dice|min_foreground_class_dice|exact_dice_[12]|pred_foreground_fraction|target_foreground_fraction|false_positive_fraction|pred_to_target_volume_ratio|loss):' \
"${OUTPUT_ROOT}"/tuke-swin-variants_TRAIN_JOB_ID_*.out \
| tail -n 240
```

Do not use F1 alone to judge segmentation. The exact Dice values and foreground-volume metrics
are the primary diagnostics.

## Resume a failed element

Use the original combined training job ID and original raw array index. For example, raw element
`17` is GIN, SEG010, fold 2:

```bash
RESUME=true EXPERIMENT_ID=ORIGINAL_TRAIN_JOB_ID \
sbatch --array=17 slurm/tuke_hybrid/12_finetune_swinunetr_all_variants_cv.slurm
```

For an individual variant script, use its original job ID and local array index `0-9`.
The resume path is checked before Python starts, and Lightning continues from that run's
`checkpoints/last.ckpt` including optimizer, scheduler, epoch, and global-step state.

## Outputs and checkpoint count

Variant roots default to:

```text
${ASPARAGUS_MODELS}/tuke_swinunetr_normal_cv
${ASPARAGUS_MODELS}/tuke_swinunetr_gin_cv
${ASPARAGUS_MODELS}/tuke_swinunetr_gin_carvemix_cv
${ASPARAGUS_MODELS}/tuke_swinunetr_clsreg_normal_cv
${ASPARAGUS_MODELS}/tuke_swinunetr_clsreg_gin_cv
```

Each fold writes `resolved_config.yaml`, `validation_best_metrics.json`, `best.ckpt`,
`last.ckpt`, and periodic checkpoints. At 150 epochs with the default 50-epoch interval, that is
up to five checkpoint files per fold: best, last, and periodic snapshots near epochs 50, 100,
and 150.

The analyzer writes one text file below:

```text
${ASPARAGUS_RESULTS}/tuke_swinunetr_variant_analysis/
${ASPARAGUS_RESULTS}/tuke_swinunetr_clsreg_analysis/
```

It contains six fold tables, mean and standard deviation rows, deltas versus normal fine-tuning,
and a complete inventory of best/last/periodic checkpoints and missing outputs.

Only after comparing cross-validation results should one variant be selected for a single final
evaluation on the fixed `TEST_80_10_10` set.
