#!/bin/bash
#SBATCH --job-name=fomo-test-ssl-import
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
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
source "${CODE_ROOT}/fomo_env/bin/activate"

cd "${CODE_ROOT}/asparagus"

export HYDRA_FULL_ERROR=1

echo "Running on node: $(hostname)"
echo "Python: $(which python)"
python --version

python - <<'PY'
print("=== Import lightning_modules package ===")
import asparagus.modules.lightning_modules as lm
print("file:", lm.__file__)
print("has SelfSupervisedModule:", hasattr(lm, "SelfSupervisedModule"))

print()
print("=== Hydra get_object test ===")
from hydra.utils import get_object
obj = get_object("asparagus.modules.lightning_modules.SelfSupervisedModule")
print("Loaded:", obj)
print("Type:", type(obj))
PY

echo "Finished at: $(date)"
deactivate || true
exit 0
