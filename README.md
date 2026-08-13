# TUKE_FOMO26
# Environment setup for Slurm jobs

The project uses separate environment files for pretraining and fine-tuning:

```text
asparagus/.env.pretrain
asparagus/.env.finetune

asparagus/.env.pretrain should contain

CODE_ROOT=""
SHARED_ROOT="/mnt/project/perun2601396"

FOMO_ROOT="${CODE_ROOT}"

ASPARAGUS_SOURCE="${SHARED_ROOT}/FOMO26_data"
ASPARAGUS_CONFIGS="${CODE_ROOT}/asparagus/configs"
ASPARAGUS_DATA="${SHARED_ROOT}/FOMO26_processed/baseline"
ASPARAGUS_MODELS="${SHARED_ROOT}/FOMO26_models/baseline_pretraining"
ASPARAGUS_RESULTS="${SHARED_ROOT}/FOMO26_results/baseline_pretraining"
ASPARAGUS_RAW_LABELS="${SHARED_ROOT}/FOMO26_raw_labels"

asparagus/.env.finetune should contain

CODE_ROOT=""
SHARED_ROOT="/mnt/project/perun2601396"

FOMO_ROOT="${CODE_ROOT}"

ASPARAGUS_SOURCE="${SHARED_ROOT}/FOMO26_finetune"
ASPARAGUS_CONFIGS="${CODE_ROOT}/asparagus/configs"
ASPARAGUS_DATA="${SHARED_ROOT}/FOMO26_processed/baseline"
ASPARAGUS_MODELS="${SHARED_ROOT}/FOMO26_models/${USER}"
ASPARAGUS_RESULTS="${SHARED_ROOT}/FOMO26_results/${USER}"
ASPARAGUS_RAW_LABELS="${SHARED_ROOT}/FOMO26_raw_labels"

where CODE_ROOT should be the absolute path to your code repository, where when you use the ls command you should see the
asparagus  asparagus_preprocessing  README.md  slurm
folders

IF YOU SEE A SLURM SCRIPT THAT HARD CODES THE CODE ROOT EITHER REPLACE IT WITH YOUR PATH OR FIX IT
those exist because of the servers absolute path change and are in need of fixing

ANother thing is the creation
```

# TUKE_FOMO26
This repository contains the TUKE FOMO26 pretraining and fine-tuning code.
# Repository and virtual environment
`CODE_ROOT` must point to the repository root. It is the directory in which the
following entries are visible:
```text
asparagus
asparagus_preprocessing
README.md
slurm
```
The two environment files described below should be stored inside the
`asparagus` directory:
```text
asparagus/.env.pretrain
asparagus/.env.finetune
```
They contain cluster-specific paths and must not be committed to Git.
Pretraining environment
From inside your `CODE_ROOT` create the file:
```bash
nano /asparagus/.env.pretrain
```
Use the following contents:
```bash
export CODE_ROOT=""
export SHARED_ROOT="/mnt/project/perun2601396"

export FOMO_ROOT="${CODE_ROOT}"

export ASPARAGUS_SOURCE="${SHARED_ROOT}/FOMO26_data"
export ASPARAGUS_CONFIGS="${CODE_ROOT}/asparagus/configs"
export ASPARAGUS_DATA="${SHARED_ROOT}/FOMO26_processed/baseline"
export ASPARAGUS_MODELS="${SHARED_ROOT}/FOMO26_models/baseline_pretraining"
export ASPARAGUS_RESULTS="${SHARED_ROOT}/FOMO26_results/baseline_pretraining"
export ASPARAGUS_RAW_LABELS="${SHARED_ROOT}/FOMO26_raw_labels"

# Keep disabled unless a W&B API key has been configured.
export WANDB_MODE="disabled"
```
The pretraining environment is used by the inventory, preflight, smoke,
production, split, preprocessing, and pretraining Slurm jobs.
Fine-tuning environment
From inside your `CODE_ROOT` create the file:
```bash
nano /asparagus/.env.finetune
```
Use the following contents:
```bash
export CODE_ROOT=""
export SHARED_ROOT="/mnt/project/perun2601396"

export FOMO_ROOT="${CODE_ROOT}"

export ASPARAGUS_SOURCE="${SHARED_ROOT}/FOMO26_finetune"
export ASPARAGUS_CONFIGS="${CODE_ROOT}/asparagus/configs"
export ASPARAGUS_DATA="${SHARED_ROOT}/FOMO26_processed/baseline"
export ASPARAGUS_MODELS="${SHARED_ROOT}/FOMO26_models/${USER}"
export ASPARAGUS_RESULTS="${SHARED_ROOT}/FOMO26_results/${USER}"
export ASPARAGUS_RAW_LABELS="${SHARED_ROOT}/FOMO26_raw_labels"

export WANDB_MODE="disabled"
```
The fine-tuning environment is used by the fine-tuning, split, preprocessing,
and environment-check Slurm jobs.

Add these lines to `.gitignore` if they are not already present:
```text
asparagus/.env.pretrain
asparagus/.env.finetune
```
Hard-coded paths in Slurm scripts
The environment files define data, model, result, and label locations. Older
Slurm scripts may still define `CODE_ROOT`, `ENV_PATH`, or checkpoint paths
themselves.

Any script containing the those path will be updated or refactored to use
the env files.
Fine-tuning scripts may also contain a hard-coded `PRETRAIN_CKPT`. Verify that
it points to the pretrained checkpoint that should be used.
# Creating the `fomo_env` virtual environment
Create the environment once from the repository root(CODE_ROOT) with the name `fomo_env` what virtual environment manager gets used is up to the user.
Install dependencies from `requirements.txt`
