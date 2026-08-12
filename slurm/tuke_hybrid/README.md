# TUKE hybrid sequence pretraining

This pipeline keeps masked reconstruction as the primary objective and adds
class-balanced MRI-sequence classification with
`total_loss = reconstruction_loss + 0.01 * sequence_cross_entropy`.

The uploaded `e709a0c` branch did not contain Matej's original SwinUNETR
source. The implementation here therefore uses MONAI 1.6.0's canonical 3D
SwinUNETR (feature size 48) and attaches the sequence head to its deepest Swin
encoder feature map. It does not claim bit-for-bit equivalence with an
unavailable private implementation.

## Submit safely

From the repository root on the login node:

```bash
bash slurm/tuke_hybrid/05_submit_pipeline.sh
```

This submits the GPU preflight and the four-GPU smoke test with an `afterok`
dependency. It deliberately leaves the expensive production job unsubmitted
until the smoke log has been checked. To submit all three at once:

```bash
SUBMIT_PRODUCTION=1 bash slurm/tuke_hybrid/05_submit_pipeline.sh
```

No Python command is run on the login node.

## Success criteria

The preflight must print `PREFLIGHT_OK=.../preflight_summary.json` after
confirming 303,144 training scans, 3,063 validation scans, 25 raw sequence
labels, readable tensors, 14 non-empty classification classes, and a valid
96-cubed SwinUNETR forward pass.

The smoke test must complete 16 optimizer steps on four GPUs, run validation,
and create `last.ckpt`. The production config uses four GPUs, per-GPU batch size
2, gradient accumulation 4, BF16, and 6,000,000 effective samples.

## Checkpoints and restart

Production writes a full `last.ckpt` every 1,000 training steps while retaining
one numbered checkpoint. The Slurm job requests requeue and sends `USR1` five
minutes before its 30-hour limit so Lightning can checkpoint/requeue. Restart
selects the newest of the periodic `last.ckpt` and Lightning's signal-time
`hpc_ckpt_*.ckpt`, avoiding rollback after a requeue. A manual resubmission of
the same production job finds the same short Hydra run directory and resumes.

All four ranks receive the same fixed run ID, avoiding the race that occurs
when each externally launched DDP process independently searches for a random
unused ID. To start a genuinely separate run, set a new ID at submission time,
for example `TUKE_HYBRID_RUN_ID=production_v2 sbatch .../04_production_tuke_hybrid.slurm`.

The previous Hydra path-length failure is avoided because both TUKE configs use
short stable run directories and do not include CLI overrides in the path.
