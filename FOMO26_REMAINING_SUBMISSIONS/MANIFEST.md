# Bundle manifest

- `containers/common/`: exact SwinUNETR architecture, preprocessing, ensemble,
  TTA, sliding-window and postprocessing code shared at build time.
- `containers/task*/`: exact challenge CLI plus isolated weight staging area
  and Apptainer definition for each final SIF.
- `tools/export_all_weights.py`: validation-based variant selection, compact
  checkpoint export and strict model loading.
- `tools/check_preprocessing_equivalence.py`: saved-tensor equivalence checks
  for all four downstream tasks.
- `tools/prepare_validator_inputs.py`: valid synthetic inputs for all supplied
  official validator suites.
- `tools/contract_checks.py`: output geometry, label, scalar and embedding
  checks.
- `tools/check_real_finetune_cases.py`: one labeled engineering smoke case for
  each downstream task.
- `scripts/`: Slurm-only Python execution and a dependency submission helper.
- `vendor/container-validator-main/`: supplied validator copied unchanged.

Generated weights, SIFs and reports are intentionally excluded from this ZIP.
