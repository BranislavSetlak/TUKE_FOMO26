#!/bin/bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REQUIRE_WEIGHTS="${REQUIRE_WEIGHTS:-true}"

required=(
  container/Apptainer.def
  container/predict.py
  container/model.py
  container/preprocessing.py
  container/requirements.txt
  tools/export_infarct_ensemble.py
  tools/check_exported_weights.py
  tools/check_finetune_cases.py
  tools/check_preprocessing_equivalence.py
  tools/contract_checks.py
  tools/prepare_validator_inputs.py
)
for relative in "${required[@]}"; do
  test -s "${BUNDLE_ROOT}/${relative}" || {
    echo "ERROR: missing ${BUNDLE_ROOT}/${relative}" >&2
    exit 1
  }
done

while IFS= read -r script; do
  bash -n "${script}"
done < <(find "${BUNDLE_ROOT}/scripts" -maxdepth 1 -type f \( -name '*.sh' -o -name '*.slurm' \) | sort)

if grep -RInE \
  '/mnt/home/brseke961|/mnt/data/home/brseke961|WANDB_API_KEY[[:space:]]*=|PASSWORD[[:space:]]*=|API_KEY[[:space:]]*=' \
  "${BUNDLE_ROOT}" --exclude='*.zip' --exclude='00_static_preflight.sh' --exclude-dir='build' --exclude-dir='reports'; then
  echo "ERROR: hard-coded personal path or possible secret found" >&2
  exit 1
fi

grep -q 'parser.add_argument("--flair", required=True)' "${BUNDLE_ROOT}/container/predict.py"
grep -q 'parser.add_argument("--adc", required=True)' "${BUNDLE_ROOT}/container/predict.py"
grep -q 'parser.add_argument("--dwi", required=True)' "${BUNDLE_ROOT}/container/predict.py"
grep -q 'parser.add_argument("--output", required=True)' "${BUNDLE_ROOT}/container/predict.py"
grep -q 'EXPECTED_FOLDS = 5' "${BUNDLE_ROOT}/container/predict.py"

weight_count="$(find "${BUNDLE_ROOT}/container/weights" -maxdepth 1 -type f -name 'fold_*.pt' | wc -l)"
if [[ "${REQUIRE_WEIGHTS}" == true && "${weight_count}" -ne 5 ]]; then
  echo "ERROR: expected five fold weights, found ${weight_count}" >&2
  echo "Run scripts/01_export_infarct_ensemble.slurm first." >&2
  exit 1
fi

echo "STATIC_PREFLIGHT_PASS bundle=${BUNDLE_ROOT} weights=${weight_count} require_weights=${REQUIRE_WEIGHTS}"
