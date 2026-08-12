#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_PRODUCTION="${SUBMIT_PRODUCTION:-0}"

preflight_job="$(sbatch --parsable "${SCRIPT_DIR}/02_preflight_tuke_hybrid.slurm")"
smoke_job="$(sbatch --parsable --dependency="afterok:${preflight_job}" "${SCRIPT_DIR}/03_smoke_tuke_hybrid.slurm")"

echo "PREFLIGHT_JOB=${preflight_job}"
echo "SMOKE_JOB=${smoke_job} dependency=afterok:${preflight_job}"

if [[ "${SUBMIT_PRODUCTION}" == "1" ]]; then
    production_job="$(sbatch --parsable --dependency="afterok:${smoke_job}" "${SCRIPT_DIR}/04_production_tuke_hybrid.slurm")"
    echo "PRODUCTION_JOB=${production_job} dependency=afterok:${smoke_job}"
else
    echo "Production was not submitted. After checking the smoke log, run:"
    echo "sbatch --dependency=afterok:${smoke_job} ${SCRIPT_DIR}/04_production_tuke_hybrid.slurm"
fi
