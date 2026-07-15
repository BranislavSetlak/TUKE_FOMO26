#!/bin/bash
#SBATCH --job-name=fomo-test-transform
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --account=perun2601396
#SBATCH --output=/project/perun2601396/FOMO26_job_outputs/brseke961/%x_%j.out
#SBATCH --error=/project/perun2601396/FOMO26_job_outputs/brseke961/%x_%j.err

set -euo pipefail

CODE_ROOT="/mnt/data/home/brseke961/FOMO26_project/FOMO26_code"
ENV_PATH="${CODE_ROOT}/fomo_env"

source "${ENV_PATH}/bin/activate"

cd "${CODE_ROOT}/asparagus"

export HYDRA_FULL_ERROR=1

echo "Running on node: $(hostname)"
echo "Started at: $(date)"
echo "Python: $(which python)"
python --version

python - <<'PY'
print("=== Direct module import ===")
import asparagus.modules.transforms.presets as p
print("presets file:", p.__file__)

for name in [
    "pretrain_CPU_train_transforms",
    "pretrain_CPU_val_transforms",
    "pretrain_GPU_train_transforms",
    "pretrain_GPU_val_transforms",
]:
    print()
    print(name)
    print("exists:", hasattr(p, name))
    obj = getattr(p, name, None)
    print("object:", obj)
    print("type:", type(obj))

print()
print("=== Hydra get_object test ===")
from hydra.utils import get_object

for target in [
    "asparagus.modules.transforms.presets.pretrain_CPU_train_transforms",
    "asparagus.modules.transforms.presets.pretrain_CPU_val_transforms",
    "asparagus.modules.transforms.presets.pretrain_GPU_train_transforms",
    "asparagus.modules.transforms.presets.pretrain_GPU_val_transforms",
]:
    print()
    print("target:", target)
    obj = get_object(target)
    print("loaded:", obj)
    print("type:", type(obj))

print()
print("=== PretrainDataModule import ===")
from asparagus.modules.data_modules.pretraining import PretrainDataModule
print("PretrainDataModule:", PretrainDataModule)
PY

echo "Finished at: $(date)"
deactivate || true
exit 0
