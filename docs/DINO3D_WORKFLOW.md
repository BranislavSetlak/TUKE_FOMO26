# DINO3D post-training and evaluation bundle

This bundle adds downstream heads and four Slurm workflows to the existing
FOMO26 repository. It does not alter the completed pretraining checkpoint.

## What is installed

- A 3D classification/regression wrapper around the pretrained EMA teacher.
- A lightweight 3D segmentation decoder around the same teacher backbone.
- A classification Lightning module that saves class probabilities for AUROC.
- CSV exporters for pretraining history and fine-tuning history.
- A downstream test-performance summarizer.

The checkpoint transfer intentionally matches keys named
`model.teacher_backbone.*`. The student backbone, DINO head, and iBOT head are
not used for downstream inference. The teacher backbone is not frozen: this is
full fine-tuning, as in the original baseline protocol.

## Files

1. `09_install_dino3d_posttraining.slurm` installs and validates the add-on.
2. `10_export_dino3d_pretrain_history.slurm` writes the full SSL metric CSV.
3. `11_finetune_dino3d_all_tasks.slurm` fine-tunes all five tasks as an array.
4. `12_export_dino3d_finetune_history.slurm` writes epoch-level CSV history.
5. `13_summarize_dino3d_performance.slurm` summarizes held-out test results.

All computation, including installation validation and CSV analysis, runs in a
Slurm allocation. The CPU jobs use `cpu_short`; fine-tuning uses one H200 per
array task in `gpu_short`.

## One-time transfer and extraction

Copy the ZIP from Windows/WSL in the same way as the previous bundle:

```bash
rsync -avh --progress -e "ssh -i ~/.ssh/id_ed25519_perun" \
  /mnt/c/Users/brani/dino3d_posttraining_bundle.zip \
  brseke961@login01.perun.tuke.sk:/mnt/home/brseke961/FOMO26_project/FOMO26_code/
```

On `login01`, extract it as file management only:

```bash
cd /mnt/home/brseke961/FOMO26_project/FOMO26_code
unzip -o dino3d_posttraining_bundle.zip
```

## Submit the complete workflow

```bash
cd /mnt/home/brseke961/FOMO26_project/FOMO26_code/dino3d_posttraining_bundle

INSTALL_ID=$(sbatch --parsable slurm/09_install_dino3d_posttraining.slurm)
PRETRAIN_CSV_ID=$(sbatch --parsable --dependency="afterok:${INSTALL_ID}" \
  slurm/10_export_dino3d_pretrain_history.slurm)
FINETUNE_ID=$(sbatch --parsable --dependency="afterok:${INSTALL_ID}" \
  slurm/11_finetune_dino3d_all_tasks.slurm)
FINETUNE_CSV_ID=$(sbatch --parsable --dependency="afterok:${FINETUNE_ID}" \
  slurm/12_export_dino3d_finetune_history.slurm)
PERFORMANCE_ID=$(sbatch --parsable --dependency="afterok:${FINETUNE_ID}" \
  slurm/13_summarize_dino3d_performance.slurm)

echo "install=${INSTALL_ID} pretrain_csv=${PRETRAIN_CSV_ID} finetune=${FINETUNE_ID} finetune_csv=${FINETUNE_CSV_ID} performance=${PERFORMANCE_ID}"
```

`FINETUNE_ID` is the parent ID of five one-GPU array tasks. They are:

1. `CLS002_FOMO26_Infarct` — 50 epochs
2. `SEG009_FOMO26_Meningioma` — 150 epochs
3. `REGR002_FOMO26_BrainAge` — 50 epochs
4. `SEG010_FOMO26_TrigeminalNeuralgia` — 150 epochs
5. `CLS003_FOMO26_Polymicrogyria` — 50 epochs

They use the baseline's `split_80_10_10`, `TEST_80_10_10`, batch size 2,
128-cube classification/regression inputs, and 160-cube segmentation patches.
Validation is run every epoch so `best.ckpt` and the history CSV are more useful.

## Monitor

```bash
squeue -j "${INSTALL_ID},${PRETRAIN_CSV_ID},${FINETUNE_ID},${FINETUNE_CSV_ID},${PERFORMANCE_ID}" \
  -o "%.18i %.28j %.10T %.10M %.30R"

tail -f "/mnt/project/perun2601396/FOMO26_job_outputs/dino3d-ft_${FINETUNE_ID}_1.out"
```

Array output suffixes `_1` through `_5` correspond to the task list above.

## Outputs

Pretraining CSVs:

```text
/mnt/project/perun2601396/FOMO26_results/dino3d/pretraining_run_71494/
  dino3d_pretrain_history.csv
  dino3d_pretrain_summary.csv
```

Fine-tuned checkpoints, logs, and prediction JSON:

```text
/mnt/project/perun2601396/FOMO26_models/dino3d_finetuning/<TASK>/runs/slurm_<ARRAY_ID>_<INDEX>/
```

Fine-tuning CSVs:

```text
/mnt/project/perun2601396/FOMO26_results/dino3d/finetuning_analysis/
  dino3d_finetune_history.csv
  dino3d_finetune_summary.csv
```

Final held-out performance:

```text
/mnt/project/perun2601396/FOMO26_results/dino3d/downstream_performance/
  dino3d_downstream_performance.csv
  dino3d_segmentation_per_label.csv
```

The performance table reports AUROC/average precision and confusion-derived
classification metrics, MAE/MSE/RMSE/R2 for brain-age regression, and
foreground macro segmentation metrics plus a per-label table.

## Run only one task

The array can be restricted at submission time. For example, classification
task 1 only:

```bash
sbatch --array=1 slurm/11_finetune_dino3d_all_tasks.slurm
```

Use `--array=1-3` for one classification, one segmentation, and regression run.
