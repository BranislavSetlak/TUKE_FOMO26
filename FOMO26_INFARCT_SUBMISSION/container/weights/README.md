# Exported ensemble weights

Do not place the original pretraining checkpoint here. Run
`scripts/01_export_infarct_ensemble.slurm` after the five-fold infarct
fine-tuning jobs finish. It writes exactly:

- `fold_0.pt`
- `fold_1.pt`
- `fold_2.pt`
- `fold_3.pt`
- `fold_4.pt`

The build refuses to create a submission image unless all five files exist.
