# FOMO26 remaining submission containers (Tasks 2–7)

This is a build-and-test kit for the five remaining FOMO26 submission files:

| Challenge target | Trained task | Final SIF |
|---|---|---|
| Task 2 | `SEG009_FOMO26_Meningioma` | `fomo26_task2_meningioma.sif` |
| Task 3 | `REGR002_FOMO26_BrainAge` | `fomo26_task3_brain_age.sif` |
| Task 4 | `SEG010_FOMO26_TrigeminalNeuralgia` | `fomo26_task4_trigeminal.sif` |
| Task 5 | `CLS003_FOMO26_Polymicrogyria` | `fomo26_task5_polymicrogyria.sif` |
| Tasks 6 and 7 | final TUKE hybrid pretrained encoder | `fomo26_task6_7_embeddings.sif` |

The ZIP intentionally has no model weights. The export job refuses to proceed
until it finds five non-empty `best.ckpt` files for every downstream task. It
then strict-loads compact inference models before any SIF is built.

## Inference decisions

- Tasks 2–5 select one **complete** validation variant, never a mixture of
  folds from different variants.
- Classification selects the higher mean five-fold validation AUROC; lower
  validation loss is a fallback. Brain-age regression selects lower mean MAE.
  Segmentation selects higher mean `val/min_foreground_class_dice`.
- Final predictions average five folds and eight deterministic 3-D flip TTAs.
- Segmentation averages softmax probabilities from Gaussian sliding-window
  inference (`96³`, overlap `0.5`) before assigning labels.
- Task 2 keeps the largest 26-connected component. This is the conventional
  single-lesion false-positive filter; inspect real-case results before
  submitting if multifocal meningiomas are expected.
- Task 4 does **not** apply a largest-object filter because nerves and vessels
  can be bilateral/disconnected. It removes only components smaller than the
  greater of four voxels or 1% of that class's largest component.
- Task 5 averages positive-class probabilities, not logits or hard labels.
- Tasks 6–7 use the pretrained encoder, not a downstream checkpoint. Mean
  pooled Swin stages 1–4 are concatenated and L2-normalized to a fixed 1,440-D
  float32 vector. Eight flip embeddings are averaged for the final output.

### Important Task 2 channel detail

The challenge requires FLAIR, DWI and one of T2*/SWI. The actual repository
preprocessed and fine-tuned Task 2 with two channels, in this order:

1. FLAIR
2. DWI b1000

The container validates the required T2*/SWI argument but does not inject it
into a two-channel checkpoint. Changing the stem to three channels only at
submission time would be incompatible with training.

## 1. Unpack without overwriting the repository

From the repository root:

```bash
cd /absolute/path/to/FOMO26_code
unzip FOMO26_REMAINING_SUBMISSIONS.zip
```

This creates only `FOMO26_REMAINING_SUBMISSIONS/`.

## 2. Set experiment IDs after training finishes

If segmentation normal, GIN and GIN+CarveMix were all launched by the combined
array, they share one array job ID:

```bash
cd /absolute/path/to/FOMO26_code
export CODE_ROOT="$PWD"
export BUNDLE_ROOT="$CODE_ROOT/FOMO26_REMAINING_SUBMISSIONS"

export SEG_EXPERIMENT_ID=REPLACE_WITH_SEGMENTATION_ARRAY_JOB_ID
export CLSREG_EXPERIMENT_ID=REPLACE_WITH_CLSREG_ARRAY_JOB_ID
export TUKE_HYBRID_PRETRAIN_CKPT=/mnt/project/perun2601396/FOMO26_models/baseline_pretraining/PT902_FOMO300K_HF/tuke_hybrid/production/run_id=production_8gpu_v1/checkpoints/last.ckpt
```

If segmentation variants were separate array submissions, use instead:

```bash
export SEG_NORMAL_EXPERIMENT_ID=JOB_ID_FOR_NORMAL
export SEG_GIN_EXPERIMENT_ID=JOB_ID_FOR_GIN
export SEG_GIN_CARVEMIX_EXPERIMENT_ID=JOB_ID_FOR_GIN_CARVEMIX
```

The exports disappear when the shell closes. Set them again after reconnecting.

Check checkpoint completion before submission:

```bash
find /mnt/project/perun2601396/FOMO26_models -type f \
  -path '*/experiment_*/SEG*_FOMO26_*/fold_*/checkpoints/best.ckpt' \
  -printf '%s %p\n' | sort

find /mnt/project/perun2601396/FOMO26_models -type f \
  -path '*/experiment_*/REGR002_FOMO26_BrainAge/fold_*/checkpoints/best.ckpt' \
  -o -path '*/experiment_*/CLS003_FOMO26_Polymicrogyria/fold_*/checkpoints/best.ckpt'
```

## 3. Submit the full dependency pipeline

```bash
bash "$BUNDLE_ROOT/scripts/07_submit_all_later.sh"
```

The chain is:

1. export and strict-load weights;
2. reproduce training preprocessing on two saved cases per task;
3. build five SIFs in a five-element Slurm array;
4. generate valid local validator inputs;
5. run the supplied official validator on every SIF;
6. enforce exact affine/shape/label/scalar/embedding contracts;
7. run one labeled fine-tune case per downstream task.

All Python is run inside Slurm jobs. The shell helper itself only submits jobs.

Monitor with:

```bash
squeue -u "$USER"
```

## 4. Variant overrides

`auto` is the default. If only one variant is complete, it is selected. To
force a complete variant:

```bash
export TASK2_VARIANT=normal
export TASK3_VARIANT=gin
export TASK4_VARIANT=gin_carvemix
export TASK5_VARIANT=normal
```

Allowed segmentation values are `auto`, `normal`, `gin`, and
`gin_carvemix`. Allowed classification/regression values are `auto`, `normal`,
and `gin`. The exporter fails rather than silently using an incomplete choice.

## 5. Reports that must pass

```bash
cat "$BUNDLE_ROOT/reports/weight_export.json"
cat "$BUNDLE_ROOT/reports/preprocessing_equivalence.json"
cat "$BUNDLE_ROOT/reports/official_validator_task2.txt"
cat "$BUNDLE_ROOT/reports/official_validator_task3.txt"
cat "$BUNDLE_ROOT/reports/official_validator_task4.txt"
cat "$BUNDLE_ROOT/reports/official_validator_task5.txt"
cat "$BUNDLE_ROOT/reports/official_validator_task6_and_7.txt"
cat "$BUNDLE_ROOT/reports/contract_checks.json"
cat "$BUNDLE_ROOT/reports/real_finetune_cases.json"
```

Every official report must contain `container is ready to submit`. The
real-case report is an engineering smoke test, not a valid estimate of model
performance.

The validator files supplied in the source ZIP were Git LFS pointer text rather
than NIfTI bytes. `03_prepare_validator_data.slurm` creates deterministic valid
NIfTIs and passes a custom manifest to the otherwise unchanged official
validator.

## 6. What to upload

Only upload the `.sif` for that challenge task from:

```text
FOMO26_REMAINING_SUBMISSIONS/build/
```

Tasks 6 and 7 use the same `fomo26_task6_7_embeddings.sif`. Do not upload this
ZIP, the reports, weight files, or SHA-256 files as a challenge submission.
On Synapse choose **Submit as a Team**. The challenge allows only three valid
submissions per team, per task, per track.

## 7. If the full helper cannot be queued

Submit stages manually, preserving `afterok` dependencies:

```bash
sbatch --export=ALL,CODE_ROOT="$CODE_ROOT",BUNDLE_ROOT="$BUNDLE_ROOT" \
  "$BUNDLE_ROOT/scripts/01_export_weights.slurm"
```

After it succeeds, run `02_build_containers.slurm`, then
`03_prepare_validator_data.slurm`, `04_official_validator.slurm`,
`05_contract_checks.slurm`, and `06_real_finetune_cases.slurm`.

Do not upload a SIF if export, preprocessing equivalence, build, official
validation, or contract checks failed.
