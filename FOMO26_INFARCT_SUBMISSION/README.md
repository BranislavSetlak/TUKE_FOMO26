# FOMO26 Task 1 — TUKE SwinUNETR submission container

This bundle builds a real FOMO26 Task 1 Apptainer container for infarct
classification. It is wired to the TUKE SwinUNETR classification fine-tuning
jobs in this repository and implements the recommended low-risk inference
scheme:

- select one complete variant (`normal` or `GIN`) using **five-fold validation
  AUROC**; validation loss is the fallback;
- use the best checkpoint from every fold;
- average class-1 probabilities across all five folds;
- use eight deterministic left/right, anterior/posterior and inferior/superior
  flip combinations (TTA);
- write the continuous positive-class probability, not an argmax label.

The ZIP is a build kit, not the final submission. The supplied repository did
not contain the trained fold checkpoints, so no honest `.sif` can be embedded
yet. Once the jobs finish, one command submits export, build and validation as
a dependency chain.

## Important Task 1 detail

The challenge interface requires FLAIR, ADC, DWI and one of T2*/SWI. The
repository's actual `CLS002_FOMO26_Infarct` preprocessor fine-tuned the model
with **three channels only**, in this order:

1. FLAIR
2. ADC
3. DWI b1000

The container therefore accepts and checks the required T2*/SWI path, but does
not feed it to the three-channel checkpoint. Adding a fourth channel only at
submission time would be incompatible with the trained stem and would not be a
valid improvement.

## Contents

- `container/`: Apptainer definition, exact inference model, preprocessing and
  empty weights staging directory.
- `tools/export_infarct_ensemble.py`: chooses the complete normal/GIN variant
  and exports only the Swin encoder plus classifier head.
- `tools/check_exported_weights.py`: strict-loads all five weights and performs
  a synthetic forward pass.
- `tools/check_preprocessing_equivalence.py`: compares three raw cases after
  container preprocessing against the exact saved-tensor validation pipeline.
- `tools/check_finetune_cases.py`: discovers raw source paths through processed
  Task 1 metadata and runs a small balanced sample through the built SIF.
- `tools/contract_checks.py`: checks deterministic output, the one-float output
  contract and the expected failure when T2*/SWI is omitted.
- `vendor/container-validator-main/`: the supplied official FOMO26 validator,
  unchanged.
- `scripts/`: Slurm jobs plus one submission helper.

## 1. Unpack at the repository root

After downloading the ZIP on the cluster:

```bash
cd /absolute/path/to/FOMO26_code
unzip FOMO26_INFARCT_SUBMISSION.zip
```

You should then have:

```text
FOMO26_code/
├── asparagus/
├── fomo_env/
├── slurm/
└── FOMO26_INFARCT_SUBMISSION/
```

No existing repository files are overwritten.

## 2. Identify the completed cls/reg experiment

Use the array job ID from
`14_finetune_swinunetr_clsreg_normal_gin_cv.slurm`. Normal and GIN were tasks
within the same array, so they normally share one experiment ID.

Check that Task 1 has five `best.ckpt` files per intended variant:

```bash
find /mnt/project/perun2601396/FOMO26_models -type f \
  -path '*/experiment_JOB_ID/CLS002_FOMO26_Infarct/fold_*/checkpoints/best.ckpt' \
  -printf '%s %p\n' | sort
```

Replace `JOB_ID` with the real array job ID. A complete normal+GIN run prints
ten paths. Five are enough if you explicitly choose the complete variant.

## 3. Submit the complete build-and-check pipeline

These exports survive only in the current shell, so set them again after
reconnecting:

```bash
cd /absolute/path/to/FOMO26_code

export CODE_ROOT="$PWD"
export BUNDLE_ROOT="$CODE_ROOT/FOMO26_INFARCT_SUBMISSION"
export CLSREG_EXPERIMENT_ID=REPLACE_WITH_ARRAY_JOB_ID
export INFARCT_VARIANT=auto

bash "$BUNDLE_ROOT/scripts/06_submit_pipeline.sh"
```

`INFARCT_VARIANT=auto` uses the only complete variant when just one finished.
When both are complete, it chooses the higher mean five-fold validation AUROC.
You can also force a complete variant explicitly:

```bash
export INFARCT_VARIANT=normal
# or: export INFARCT_VARIANT=gin
```

The helper submits four dependent jobs. The first job performs two checks:

1. export five compact fold checkpoints and strict-load them;
2. prove preprocessing equivalence on three fine-tune cases;
3. build the SIF and create its SHA-256 file;
4. run the official Task 1 container validator;
5. run two label-0 and two label-1 fine-tune cases through the SIF.

Monitor them with:

```bash
squeue -u "$USER"
```

The finished container is:

```text
FOMO26_INFARCT_SUBMISSION/build/fomo26_task1_tuke_swinunetr.sif
```

## 4. Inspect the reports

```bash
cat "$BUNDLE_ROOT/reports/exported_weights_check.json"
cat "$BUNDLE_ROOT/reports/preprocessing_equivalence.json"
cat "$BUNDLE_ROOT/reports/official_validator.txt"
cat "$BUNDLE_ROOT/reports/finetune_sample_check.json"
cat "$BUNDLE_ROOT/build/fomo26_task1_tuke_swinunetr.sif.sha256"
```

Submission readiness requires all of the following:

- five weights strict-loaded successfully;
- three raw cases match the fine-tuning validation tensors within tolerance;
- Apptainer build and `%test` passed;
- official report ends in `container is ready to submit`;
- four real-case probabilities are finite and inside `[0,1]`;
- the `.sif` and `.sha256` files exist and are non-empty.

The four-case check is an engineering sanity test, not a performance estimate.
Its accuracy must not be reported as challenge performance.

The NIfTI files in the supplied validator ZIP are Git LFS pointer text rather
than image bytes. `03_official_validator.slurm` therefore creates two valid,
deterministic local NIfTI subjects (one SWI and one T2*) and passes their
manifest to the otherwise unchanged official validator. This avoids a false
failure caused by trying to load the pointer files.

## 5. Optional manual contract test

Choose one real case and export its paths:

```bash
export CHECK_FLAIR=/absolute/path/to/flair.nii.gz
export CHECK_ADC=/absolute/path/to/adc.nii.gz
export CHECK_DWI=/absolute/path/to/dwi_b1000.nii.gz
export CHECK_SWI=/absolute/path/to/swi.nii.gz

sbatch --export=ALL,CODE_ROOT="$CODE_ROOT",BUNDLE_ROOT="$BUNDLE_ROOT" \
  "$BUNDLE_ROOT/scripts/05_contract_checks.slurm"
```

Use `CHECK_T2S` instead of `CHECK_SWI` for a T2* case. This job runs the same
case twice without TTA and verifies bit-level output stability, exactly one
finite float in `[0,1]`, and correct rejection of a call missing both optional
modalities.

## 6. Manual stages, if the submission helper cannot be queued

```bash
sbatch --export=ALL,CODE_ROOT="$CODE_ROOT",BUNDLE_ROOT="$BUNDLE_ROOT",CLSREG_EXPERIMENT_ID="$CLSREG_EXPERIMENT_ID",INFARCT_VARIANT="$INFARCT_VARIANT" \
  "$BUNDLE_ROOT/scripts/01_export_infarct_ensemble.slurm"
```

After that job succeeds:

```bash
sbatch --export=ALL,CODE_ROOT="$CODE_ROOT",BUNDLE_ROOT="$BUNDLE_ROOT" \
  "$BUNDLE_ROOT/scripts/02_build_container.slurm"
```

Then run `03_official_validator.slurm` and `04_finetune_sample_check.slurm` in
that order. Do not proceed after a failed stage.

## 7. What to upload

Upload only:

```text
fomo26_task1_tuke_swinunetr.sif
```

Do not upload this ZIP, checkpoints, reports or the SHA-256 text file as the
challenge submission. In Synapse, select **Submit as a Team**.

## Inference and preprocessing notes

- NIfTIs are reoriented to canonical RAS.
- ADC and DWI are resampled to the FLAIR grid only if their grid differs.
- Normalization reproduces the common-foreground, per-channel volume-wise
  z-normalization used by `Torch_Normalize`.
- Each channel is symmetrically padded with its minimum intensity if smaller
  than 96 voxels, then center-cropped to `96×96×96`.
- The container requires CUDA and does not contain training data, labels,
  personal paths, API keys or W&B configuration.
