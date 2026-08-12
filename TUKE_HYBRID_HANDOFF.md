# TUKE hybrid implementation handoff

## Verified starting point

- Source archive commit marker: `e709a0c616d80024cfbc86e4609dbbe522e75112`.
- Inventory: 306,207 unique scans; 303,144 train and 3,063 validation.
- Labels: 25 raw sequence labels mapped to 14 classification classes.
- `scan` uses classification target `-100` and remains in reconstruction.
- The archive contained the inventory/parser/mapping work but no SwinUNETR or
  Matej model source. Its existing reconstruction baseline was
  `ResidualEncoderUNet`.

## Implemented experiment

- MONAI 1.6.0 3D SwinUNETR, feature size 48, 96x96x96 crops.
- Masked-region MSE reconstruction as the primary objective.
- Global-average-pooled classifier on the deepest Swin encoder feature map.
- Effective-number class weights computed from the training paths at startup.
- Per-sample weighted sequence cross-entropy, including correct behavior for
  per-GPU batch size one.
- Combined loss:

  `reconstruction_loss + 0.01 * sequence_cross_entropy`

- All-`scan` batches return a differentiable zero classification loss instead
  of the NaN produced by mean cross-entropy when every target is ignored.
- Existing reconstruction-only models remain compatible because all sequence
  arguments default to disabled.

## Reliability changes

- Short Hydra paths avoid the previously observed filename/path-length error.
- Stable environment-overridable run IDs prevent four Slurm ranks racing to
  choose different random IDs.
- `ModelCheckpoint` writes a full `last.ckpt` and one numbered checkpoint.
- Restart passes the newest concrete periodic or signal-time checkpoint path to
  Lightning (`last.ckpt` or `hpc_ckpt_*.ckpt`).
- Production requests Slurm requeue and a `USR1` signal five minutes before the
  allocation limit.

## Cluster launch

Run only the shell submitter on the login node; every Python process is inside
a Slurm allocation:

```bash
cd /mnt/home/brseke961/FOMO26_project/FOMO26_code
bash slurm/tuke_hybrid/05_submit_pipeline.sh
```

This submits preflight and smoke. After the smoke succeeds, use the production
command printed by the submitter. To queue all stages immediately:

```bash
SUBMIT_PRODUCTION=1 bash slurm/tuke_hybrid/05_submit_pipeline.sh
```

Use a new explicit ID only when intentionally starting a separate experiment:

```bash
TUKE_HYBRID_RUN_ID=production_v2 sbatch slurm/tuke_hybrid/04_production_tuke_hybrid.slurm
```

## Verification boundary

Static Python compilation, YAML parsing, shell syntax, archive integrity, and
pure sequence-mapping/weight tests were run in the handoff environment. That
environment did not contain PyTorch, MONAI, Lightning, or the 306,207 cluster
files, so the real tensor/model/DDP checks are intentionally performed by
`02_preflight_tuke_hybrid.slurm` and `03_smoke_tuke_hybrid.slurm` before
production can start through the dependency chain.
