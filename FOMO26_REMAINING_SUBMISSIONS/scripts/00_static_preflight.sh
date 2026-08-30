#!/bin/bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
for task in task2_meningioma task3_brain_age task4_trigeminal task5_polymicrogyria task6_7_embeddings; do
  test -s "${BUNDLE_ROOT}/containers/${task}/Apptainer.def"
  test -s "${BUNDLE_ROOT}/containers/${task}/predict.py"
done
for file in model.py preprocessing.py inference.py requirements.txt; do
  test -s "${BUNDLE_ROOT}/containers/common/${file}"
done
for file in "${BUNDLE_ROOT}"/scripts/*.sh "${BUNDLE_ROOT}"/scripts/*.slurm; do
  bash -n "${file}"
done
if grep -RInE '/mnt/home/brseke961|/mnt/data/home/brseke961|WANDB_API_KEY=|PASSWORD=' "${BUNDLE_ROOT}" --exclude='00_static_preflight.sh' --exclude-dir='build' --exclude-dir='reports'; then
  echo "ERROR: personal path or secret-like assignment found" >&2
  exit 1
fi
if [[ "${REQUIRE_WEIGHTS:-false}" == true ]]; then
  for task in task2_meningioma task3_brain_age task4_trigeminal task5_polymicrogyria; do
    test "$(find "${BUNDLE_ROOT}/containers/${task}/weights" -maxdepth 1 -type f -name 'fold_*.pt' | wc -l)" -eq 5
  done
  test -s "${BUNDLE_ROOT}/containers/task6_7_embeddings/weights/encoder.pt"
fi
echo "STATIC_PREFLIGHT_PASS bundle=${BUNDLE_ROOT} require_weights=${REQUIRE_WEIGHTS:-false}"
