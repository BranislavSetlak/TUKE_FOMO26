#!/bin/bash

# Shared functions for scripts 22-28. This file is sourced, not submitted.

dino3d_recovery_init() {
  CODE_ROOT="${CODE_ROOT:-${SLURM_SUBMIT_DIR}}"
  ENV_FILE="${CODE_ROOT}/asparagus/.env.finetune"
  test -f "${ENV_FILE}" || {
    echo "ERROR: ${ENV_FILE} is missing. cd to the repository root before sbatch." >&2
    exit 1
  }

  set -a
  source "${ENV_FILE}"
  set +a

  ENV_PATH="${FOMO_ENV_PATH:-${CODE_ROOT}/fomo_env}"
  PYTHON="${ENV_PATH}/bin/python"
  PRETRAIN_CKPT="${DINO3D_PRETRAIN_CKPT:-${SHARED_ROOT}/FOMO26_checkpoints/dinov3_3d_stage1_71494_last.ckpt}"

  test -x "${PYTHON}"
  test -s "${PRETRAIN_CKPT}"
  test -f "${CODE_ROOT}/asparagus/configs/model/dinov3_downstream.yaml"

  export PATH="${ENV_PATH}/bin:${PATH}"
  export PYTHONPATH="${CODE_ROOT}/asparagus:${CODE_ROOT}/asparagus_preprocessing${PYTHONPATH:+:${PYTHONPATH}}"
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export HYDRA_FULL_ERROR=1
  export PYTHONUNBUFFERED=1
  export WANDB_MODE=disabled
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

dino3d_select_segmentation_task() {
  local array_index="$1"
  FOLD=$((array_index % 5))
  SEGMENTATION_INDEX=$((array_index / 5))
  case "${SEGMENTATION_INDEX}" in
    0)
      TASK_INDEX=2
      DATASET="SEG009_FOMO26_Meningioma"
      ;;
    1)
      TASK_INDEX=4
      DATASET="SEG010_FOMO26_TrigeminalNeuralgia"
      ;;
    *)
      echo "ERROR: Unknown segmentation array index ${array_index}" >&2
      exit 1
      ;;
  esac
  MODULE="asparagus.pipeline.run.finetune_seg"
  SEED=$((260826 + TASK_INDEX * 10 + FOLD))
}

dino3d_validate_segmentation_data() {
  local required
  for required in paths.json dataset.json split_5fold_cv.json TEST_80_10_10.json; do
    test -s "${ASPARAGUS_DATA}/${DATASET}/${required}" || {
      echo "ERROR: Missing ${ASPARAGUS_DATA}/${DATASET}/${required}" >&2
      exit 1
    }
  done

  local fold_count
  fold_count=$({ grep -o '"train"[[:space:]]*:' \
    "${ASPARAGUS_DATA}/${DATASET}/split_5fold_cv.json" || true; } | wc -l)
  if (( fold_count != 5 )); then
    echo "ERROR: ${DATASET} has ${fold_count} folds; exactly five are required." >&2
    exit 1
  fi
}

dino3d_resolve_experiment_id() {
  RESUME="${RESUME:-false}"
  REQUESTED_EXPERIMENT_ID="${EXPERIMENT_ID:-}"
  if [[ "${RESUME}" == "true" && -z "${REQUESTED_EXPERIMENT_ID}" ]]; then
    echo "ERROR: To resume, set EXPERIMENT_ID to the original Slurm array job ID." >&2
    exit 1
  fi
  EXPERIMENT_ID="${REQUESTED_EXPERIMENT_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}}"
}

dino3d_print_command_and_run() {
  printf 'COMMAND '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  "${CMD[@]}"
}
