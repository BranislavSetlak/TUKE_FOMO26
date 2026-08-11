#!/bin/bash
#SBATCH --job-name=dino3d-path-repair
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=perun2601396
#SBATCH --output=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.out
#SBATCH --error=/mnt/project/perun2601396/FOMO26_job_outputs/%x_%j.err

set -euo pipefail

CODE_ROOT="/mnt/home/brseke961/FOMO26_project/FOMO26_code"
PYTHON="${CODE_ROOT}/fomo_env/bin/python"
DATA_ROOT="/mnt/project/perun2601396/FOMO26_processed/baseline"

test -x "${PYTHON}"
test -d "${DATA_ROOT}"

export PATH="${CODE_ROOT}/fomo_env/bin:${PATH}"
export PYTHONUNBUFFERED=1

TASKS=(
    "CLS002_FOMO26_Infarct"
    "SEG009_FOMO26_Meningioma"
    "REGR002_FOMO26_BrainAge"
    "SEG010_FOMO26_TrigeminalNeuralgia"
    "CLS003_FOMO26_Polymicrogyria"
)

echo "PATH_REPAIR_START time=$(date --iso-8601=seconds)"
echo "data_root=${DATA_ROOT}"

for task in "${TASKS[@]}"; do
    test -d "${DATA_ROOT}/${task}" || {
        echo "ERROR: Missing dataset directory ${DATA_ROOT}/${task}" >&2
        exit 1
    }
done

DATA_ROOT="${DATA_ROOT}" SLURM_JOB_ID="${SLURM_JOB_ID}" "${PYTHON}" - <<'PY'
import json
import os
import shutil
from pathlib import Path

data_root = Path(os.environ["DATA_ROOT"])
job_id = os.environ["SLURM_JOB_ID"]

tasks = [
    "CLS002_FOMO26_Infarct",
    "SEG009_FOMO26_Meningioma",
    "REGR002_FOMO26_BrainAge",
    "SEG010_FOMO26_TrigeminalNeuralgia",
    "CLS003_FOMO26_Polymicrogyria",
]

replacements = [
    ('"/project/perun2601396/', '"/mnt/project/perun2601396/'),
    ('"/mnt/data/home/brseke961/', '"/mnt/home/brseke961/'),
]


def walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_strings(item)


total_replacements = 0
total_pt_paths = 0

for task in tasks:
    task_dir = data_root / task
    json_files = sorted(
        path
        for path in task_dir.glob("*.json")
        if not path.name.startswith("backup_before_downstream_path_fix_")
    )

    if not json_files:
        raise RuntimeError(f"No JSON metadata files found in {task_dir}")

    task_pt_paths = []

    for path in json_files:
        original = path.read_text(encoding="utf-8")
        repaired = original
        replacements_in_file = 0

        for old, new in replacements:
            count = repaired.count(old)
            repaired = repaired.replace(old, new)
            replacements_in_file += count

        if replacements_in_file:
            backup = path.with_name(
                f"backup_before_downstream_path_fix_{job_id}_{path.name}"
            )
            shutil.copy2(path, backup)

            temporary = path.with_suffix(path.suffix + f".tmp_{job_id}")
            temporary.write_text(repaired, encoding="utf-8")

            # Validate the repaired JSON before replacing the original.
            with temporary.open("r", encoding="utf-8") as handle:
                json.load(handle)

            os.replace(temporary, path)

            print(
                f"PATH_METADATA_REPAIRED "
                f"task={task} "
                f"file={path.name} "
                f"replacements={replacements_in_file} "
                f"backup={backup.name}"
            )
        else:
            print(f"PATH_METADATA_CURRENT task={task} file={path.name}")

        with path.open("r", encoding="utf-8") as handle:
            content = json.load(handle)

        for value in walk_strings(content):
            if value.startswith(("/project/", "/mnt/data/home/")):
                raise RuntimeError(
                    f"Obsolete path remains in {path}: {value}"
                )
            if value.endswith(".pt") and value.startswith("/"):
                task_pt_paths.append(value)

        total_replacements += replacements_in_file

    unique_pt_paths = sorted(set(task_pt_paths))
    missing = [path for path in unique_pt_paths[:2000] if not Path(path).is_file()]

    if missing:
        print(f"First missing repaired paths for {task}:")
        for path in missing[:10]:
            print(path)
        raise RuntimeError(
            f"{len(missing)} checked paths are missing for {task}"
        )

    total_pt_paths += len(unique_pt_paths)

    print(
        f"PATH_PREFLIGHT_OK "
        f"task={task} "
        f"referenced_pt_files={len(unique_pt_paths)} "
        f"checked={min(2000, len(unique_pt_paths))}"
    )

print(
    f"DOWNSTREAM_PATH_REPAIR_OK "
    f"replacements={total_replacements} "
    f"referenced_pt_files={total_pt_paths}"
)
PY

echo "PATH_REPAIR_FINISHED time=$(date --iso-8601=seconds)"
