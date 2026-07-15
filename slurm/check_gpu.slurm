#!/bin/bash
#SBATCH --job-name=fomo-check-gpu
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --account=perun2601396
#SBATCH --output=/project/perun2601396/FOMO26_job_outputs/brseke961/%x_%j.out
#SBATCH --error=/project/perun2601396/FOMO26_job_outputs/brseke961/%x_%j.err

set -euo pipefail

CODE_ROOT="/mnt/data/home/brseke961/FOMO26_project/FOMO26_code"
ENV_PATH="${CODE_ROOT}/fomo_env"

source "${ENV_PATH}/bin/activate"

echo "Running on node: $(hostname)"
echo "Started at: $(date)"

echo
echo "=== nvidia-smi ==="
nvidia-smi

echo
echo "=== GPU list ==="
nvidia-smi -L

echo
echo "=== Detailed GPU query ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,pci.bus_id --format=csv

echo
echo "=== PyTorch CUDA check ==="
python - <<'PY'
import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"device {i}:", torch.cuda.get_device_name(i))
        props = torch.cuda.get_device_properties(i)
        print(f"  total memory GB: {props.total_memory / 1024**3:.2f}")
        print(f"  compute capability: {props.major}.{props.minor}")
PY

echo
echo "Finished at: $(date)"
deactivate || true
exit 0
