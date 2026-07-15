#!/bin/bash
#SBATCH --job-name=fomo-check-pretrain-light
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --account=perun2601396
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

RAW_DIR="/project/perun2601396/FOMO26_data"
PROCESSED_DIR="/project/perun2601396/FOMO26_processed/baseline/PT902_FOMO300K_HF"

echo "Started at: $(date)"

echo
echo "=== Raw FOMO300K folder exists? ==="
ls -ld "${RAW_DIR}"

echo
echo "=== Top-level raw FOMO300K entries only ==="
find "${RAW_DIR}" -maxdepth 1 -mindepth 1 -printf "%f\n" | sort | head -80

echo
echo "=== Number of top-level raw entries ==="
find "${RAW_DIR}" -maxdepth 1 -mindepth 1 | wc -l

echo
echo "=== Expected processed FOMO300K folder ==="
if [ -d "${PROCESSED_DIR}" ]; then
    echo "Processed folder exists:"
    ls -ld "${PROCESSED_DIR}"

    echo
    echo "Top-level processed entries:"
    find "${PROCESSED_DIR}" -maxdepth 1 -mindepth 1 -printf "%f\n" | sort | head -80

    echo
    echo "Quick check for .pt files near top levels:"
    find "${PROCESSED_DIR}" -maxdepth 3 -type f -name "*.pt" | head -20

    echo
    echo "Quick count up to depth 3 only:"
    find "${PROCESSED_DIR}" -maxdepth 3 -type f -name "*.pt" | wc -l
else
    echo "Processed folder does NOT exist:"
    echo "${PROCESSED_DIR}"
fi

echo
echo "Finished at: $(date)"
exit 0

