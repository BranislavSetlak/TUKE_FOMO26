# DINO3D GIN epoch-table report

This bundle creates one text report containing seven tables:

1. `CLS002 - Infarct Classification (GIN)`
2. `SEG009 - Meningioma Segmentation (GIN)`
3. `REGR002 - Brain Age Regression (GIN)`
4. `SEG010 - Trigeminal Neuralgia Segmentation (GIN)`
5. `CLS003 - Polymicrogyria Classification (GIN)`
6. `SEG009 - Meningioma Segmentation (GIN + CarveMix)`
7. `SEG010 - Trigeminal Neuralgia Segmentation (GIN + CarveMix)`

Each epoch cell is the arithmetic mean of that metric across the five folds that
have logged a finite value. `Folds` shows the number of folds represented in an
epoch. The `AVERAGE` row is the mean of the displayed epoch means, ignoring
unfinished (`NA`) epochs.

The script discovers metric columns from the logs, so classification,
regression and segmentation tables automatically contain their respective
training and validation metrics, learning rates and epoch time.

## Install

From the extracted bundle directory, copy the two repository-relative folders
into the repository root:

```bash
CODE_ROOT="/absolute/path/to/FOMO26_code"

cp -r asparagus slurm "${CODE_ROOT}/"
cd "${CODE_ROOT}"
```

This adds two new files and does not replace the training scripts:

```text
asparagus/asparagus/analysis/summarize_dino3d_epoch_logs.py
slurm/dino3d/22_summarize_dino3d_epoch_logs.slurm
```

## Submit

Use the original Slurm array IDs from the successful GIN and GIN+CarveMix
submissions. For example, if the GIN array is `77093` and the GIN+CarveMix array
is `77094`:

```bash
GIN_JOB_ID=77093 CARVEMIX_JOB_ID=77094 \
sbatch slurm/dino3d/22_summarize_dino3d_epoch_logs.slurm
```

Replace `77093` if that is not the successful GIN array ID. The command may be
run while training is active; unfinished epochs appear as `NA`. Rerun the same
command after all jobs finish for the final report.

By default, the report is created at:

```text
/mnt/project/perun2601396/FOMO26_job_outputs/dino3d_epoch_tables_gin<GIN_ID>_carvemix<CARVEMIX_ID>.txt
```

The analysis job's `.out` file ends with `REPORT_PATH=...`, which gives the
exact report location.

To require all 35 expected fold logs before writing the report:

```bash
GIN_JOB_ID=77093 CARVEMIX_JOB_ID=77094 REQUIRE_ALL_LOGS=true \
sbatch slurm/dino3d/22_summarize_dino3d_epoch_logs.slurm
```

To choose another single output file:

```bash
GIN_JOB_ID=77093 CARVEMIX_JOB_ID=77094 \
OUTPUT_FILE=/mnt/project/perun2601396/FOMO26_results/dino3d_epoch_tables.txt \
sbatch slurm/dino3d/22_summarize_dino3d_epoch_logs.slurm
```
