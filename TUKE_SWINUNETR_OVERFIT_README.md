# TUKE SwinUNETR one-case segmentation overfit test

This bundle is an overlay for the `tuke-hybrid-sequence` branch snapshot supplied on
2026-08-20. It runs two independent one-GPU diagnostics in parallel:

- array task `0`: `SEG009_FOMO26_Meningioma`, required foreground class `1`
- array task `1`: `SEG010_FOMO26_TrigeminalNeuralgia`, required foreground classes `1` and `2`

The test is deliberately strict. It selects one positive training case, creates one exact
96×96×96 ROI containing every required class, and uses that identical ROI for training and
validation. There is no augmentation. The ROI is selected from the label, so this workflow
is **diagnostic only** and must never be used to report validation or test performance.

## What the patch changes

The downstream model is the same MONAI SwinUNETR used during hybrid pretraining. The
checkpoint key layout therefore remains `model.swin_unetr.*`:

- the complete Swin encoder is loaded;
- the pretrained reconstruction decoder is loaded;
- the one-channel patch embedding is repeated and divided by the number of input channels
  for the two-modality meningioma task;
- the final output convolution is rejected automatically when its one reconstruction channel
  does not match the downstream task's two or three segmentation classes;
- the sequence-classification head remains present but is not called by segmentation training.

The generic segmentation runner gains configurable checkpoint monitoring, a real `last.ckpt`,
periodic checkpoints, optional best-checkpoint validation, and the option to skip the test set.
The segmentation module also logs foreground Dice and predicted/target foreground fractions,
which distinguish empty-mask and whole-mask collapse from a genuinely improving model.

The runner now omits the TQDM callback when `logger.progress_bar=false`. Its checkpoint loader
also preserves the normal PyTorch return contract, handles an already-identical best checkpoint,
and distinguishes SwinUNETR encoder and decoder keys. These details matter when Lightning reloads
`best.ckpt` after fitting.

## Install the bundle

Download the ZIP to the repository root, where `ls` shows `asparagus`,
`asparagus_preprocessing`, `slurm`, and `README.md`. Then run on the login node:

```bash
cd /absolute/path/to/FOMO26_code
unzip -o FOMO26_TUKE_SWINUNETR_OVERFIT.zip

git status --short
git diff --check
grep -RIn "${USER}" asparagus slurm/tuke_hybrid/06_overfit_swinunetr_segmentation.slurm
```

The ZIP contains repository-relative paths and overlays the changed files in place. It does not
move the repository, delete files, change `.env` files, or include a virtual environment.

## Required environment files

Both files must already exist:

```text
asparagus/.env.pretrain
asparagus/.env.finetune
```

The job reads the pretraining model root from `.env.pretrain`, then uses the data/model/result
paths from `.env.finetune`. `CODE_ROOT` is taken from the submitted environment or, by default,
from `SLURM_SUBMIT_DIR`, so the Slurm file contains no user-specific code-repository path.

## Locate the pretraining checkpoint

The job first uses `TUKE_HYBRID_PRETRAIN_CKPT` when supplied. Otherwise it tries the stable
`production_8gpu_v1/checkpoints/last.ckpt` location and finally selects the newest production
`last.ckpt` below the pretraining model root.

To inspect the candidates yourself:

```bash
set -a
source asparagus/.env.pretrain
set +a

find "${ASPARAGUS_MODELS}/PT902_FOMO300K_HF" \
  -type f -path '*/tuke_hybrid/production/run_id=*/checkpoints/last.ckpt' \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS  %s bytes  %p\n' \
  | sort
```

Using the current `last.ckpt` is valid for a diagnostic even if pretraining is still running,
provided that checkpoint is not being replaced at the exact moment the job starts. For the
final downstream comparison, use the final pretraining checkpoint explicitly.

## Submit

From the repository root:

```bash
sbatch slurm/tuke_hybrid/06_overfit_swinunetr_segmentation.slurm
```

Or pin a checkpoint explicitly:

```bash
TUKE_HYBRID_PRETRAIN_CKPT=/absolute/path/to/last.ckpt \
sbatch slurm/tuke_hybrid/06_overfit_swinunetr_segmentation.slurm
```

The two array tasks can run at the same time. Each requests one GPU, 12 CPUs, 128 GB RAM, and
eight hours. All Python commands execute inside the Slurm allocation.

Before the 100-epoch diagnostic, each task runs a two-epoch, one-batch-per-epoch pipeline smoke
in `pipeline_smoke/`. This exercises the real DataLoader, Dice+CE loss, optimizer, scheduler,
validation metrics, checkpoint callbacks, and best-checkpoint restoration. The full diagnostic
starts only after this smoke succeeds. Keep it enabled for the first successful run.

At job start, the selected pretraining checkpoint is converted to a stable, weights-only snapshot
inside the task directory. The preflight, pipeline smoke, and full diagnostic all load that same
snapshot. This prevents a still-running pretraining job from replacing `last.ckpt` between stages
and removes optimizer state that downstream transfer does not need.

Optional controls:

```bash
TUKE_OVERFIT_EPOCHS=150 \
TUKE_OVERFIT_TRAIN_BATCHES_PER_EPOCH=20 \
TUKE_OVERFIT_WARMUP_EPOCHS=5 \
TUKE_OVERFIT_MINIMUM_DICE=0.90 \
TUKE_OVERFIT_STRICT=true \
sbatch slurm/tuke_hybrid/06_overfit_swinunetr_segmentation.slurm
```

After the complete pipeline has already been proven on the same code revision, the preliminary
smoke can be disabled with `TUKE_OVERFIT_PIPELINE_SMOKE=false`.

`TUKE_OVERFIT_PATCH_SIZE` defaults to `96 96 96`; every dimension must be divisible by 32.
Increase it only if ROI preparation reports too few voxels from a required class and GPU memory
allows it.

## Monitor and interpret

Replace `JOB_ID` below with the returned array job ID:

```bash
OUTPUT_ROOT=/mnt/project/perun2601396/FOMO26_job_outputs

grep -HnE \
  'TUKE_SWINUNETR_(OVERFIT_START|OVERFIT_PREFLIGHT_OK|PIPELINE_SMOKE_OK|OVERFIT_FINISHED)|TUKE_SWINUNETR_ONE_CASE_(PASS|WARNING)|Traceback|Error executing job|CUDA out of memory|CANCELLED|TIME LIMIT' \
  "${OUTPUT_ROOT}"/tuke-swin-overfit_JOB_ID_{0,1}.{out,err}
```

To watch epoch metrics:

```bash
grep -hE \
  'Current Epoch:|train/(loss|dice_[0-9]+|foreground_dice|pred_foreground_fraction):|val/(loss|dice_[0-9]+|foreground_dice|pred_foreground_fraction):' \
  "${OUTPUT_ROOT}"/tuke-swin-overfit_JOB_ID_{0,1}.out \
  | tail -n 160
```

A passing task prints `TUKE_SWINUNETR_ONE_CASE_PASS`. Every required foreground class must
reach Dice ≥ 0.90 at the checkpoint with the highest worst-foreground-class Dice. A warning leaves the job
successful by default so both diagnostics complete; set `TUKE_OVERFIT_STRICT=true` if a failed
gate should return exit code 2.

The foreground-fraction line helps classify a failure:

- ratio near `0`: empty-mask collapse;
- very large ratio: excessive or whole-mask foreground prediction;
- ratio near `1` with low per-class Dice: localization or class-confusion problem;
- high binary foreground Dice but low `val/dice_2`: the model finds foreground but misses the
  second trigeminal-neuralgia class.

Do not launch full SwinUNETR cross-validation unless both tasks can memorize their diagnostic
ROI or there is a documented, task-specific reason for a lower threshold.

## Output and checkpoints

Results are written below the fine-tuning paths from `.env.finetune`:

```text
${ASPARAGUS_MODELS}/tuke_swinunetr_one_case_overfit/experiment_<job-id>/<task>/source_fold_0/
```

Each task writes:

- `diagnostic_input/one_case_roi.pt` and `.pkl`;
- `diagnostic_input/one_case_roi_summary.json`;
- `pretrain_weights_snapshot.ckpt`, the exact weights used by every stage;
- `preflight.json` with checkpoint-transfer and gradient checks;
- `resolved_config.yaml` proving the exact Hydra composition used;
- `pipeline_smoke/` with its two-epoch end-to-end smoke, best checkpoint, last checkpoint, and
  validation metrics;
- `finetune_seg.log` with epoch metrics;
- `validation_best_metrics.json` used by the final gate;
- `checkpoints/best.ckpt`;
- `checkpoints/last.ckpt`;
- four periodic checkpoints at the default 100 epochs (`25`, `50`, `75`, and `100`, subject
  to Lightning's zero-based epoch filename rendering).

Thus the default full run retains up to six checkpoint files per task: best, last, and four
periodic snapshots. The preliminary smoke retains another best and last checkpoint under
`pipeline_smoke/`, for up to eight files per task on the first fully validated run.
