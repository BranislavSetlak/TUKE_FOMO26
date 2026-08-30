# TUKE SwinUNETR fine-tuning bundle manifest

Target branch: `tuke-hybrid-sequence`

Source snapshot: `TUKE_FOMO26-tuke-hybrid-sequence(2).zip`, inspected 2026-08-20.

This bundle contains the validated overfit overlay, controlled segmentation variants, normal/GIN
classification and regression, and the repaired final-day GIN+CarveMix rerun.

## Replacement files

- `asparagus/asparagus/modules/networks/swinunetr_hybrid.py`
- `asparagus/asparagus/modules/lightning_modules/base_module.py`
- `asparagus/asparagus/modules/lightning_modules/segmentation_module.py`
- `asparagus/asparagus/modules/lightning_modules/clsreg_module.py`
- `asparagus/asparagus/modules/data_modules/training.py`
- `asparagus/asparagus/modules/datasets/TrainDataset.py`
- `asparagus/asparagus/modules/transforms/presets/train.py`
- `asparagus/asparagus/modules/transforms/presets/__init__.py`
- `asparagus/asparagus/pipeline/run/finetune_seg.py`
- `asparagus/asparagus/pipeline/run/finetune_cls.py`
- `asparagus/asparagus/pipeline/run/finetune_reg.py`
- `asparagus/configs/default_finetune_seg.yaml`
- `asparagus/configs/default_finetune_cls.yaml`
- `asparagus/configs/default_finetune_reg.yaml`
- `asparagus/configs/model/core/swinunetr_hybrid.yaml`

## New files

- `asparagus/asparagus/analysis/prepare_tuke_swinunetr_one_case.py`
- `asparagus/asparagus/analysis/preflight_tuke_swinunetr_overfit.py`
- `asparagus/asparagus/analysis/check_tuke_swinunetr_overfit.py`
- `asparagus/asparagus/analysis/snapshot_lightning_checkpoint.py`
- `asparagus/configs/model/swinunetr_hybrid_seg_b.yaml`
- `asparagus/configs/model/swinunetr_hybrid_clsreg_b.yaml`
- `asparagus/asparagus/analysis/preflight_tuke_swinunetr_finetune.py`
- `asparagus/asparagus/analysis/analyze_tuke_swinunetr_variants.py`
- `asparagus/asparagus/analysis/preflight_tuke_swinunetr_clsreg.py`
- `asparagus/asparagus/analysis/analyze_tuke_swinunetr_clsreg.py`
- `asparagus/asparagus/modules/transforms/gin.py`
- `asparagus/asparagus/modules/transforms/carvemix.py`
- `slurm/tuke_hybrid/06_overfit_swinunetr_segmentation.slurm`
- `slurm/tuke_hybrid/07_preflight_swinunetr_finetune.slurm`
- `slurm/tuke_hybrid/08_finetune_swinunetr_normal_cv.slurm`
- `slurm/tuke_hybrid/09_finetune_swinunetr_gin_cv.slurm`
- `slurm/tuke_hybrid/10_finetune_swinunetr_gin_carvemix_cv.slurm`
- `slurm/tuke_hybrid/11_analyze_swinunetr_variants.slurm`
- `slurm/tuke_hybrid/12_finetune_swinunetr_all_variants_cv.slurm`
- `slurm/tuke_hybrid/13_submit_swinunetr_variants.sh`
- `slurm/tuke_hybrid/14_finetune_swinunetr_clsreg_normal_gin_cv.slurm`
- `slurm/tuke_hybrid/14a_preflight_swinunetr_clsreg.slurm`
- `slurm/tuke_hybrid/15_analyze_swinunetr_clsreg.slurm`
- `slurm/tuke_hybrid/16_submit_swinunetr_last_day_pipeline.sh`
- `slurm/tuke_hybrid/swinunetr_finetune_common.sh`
- `TUKE_SWINUNETR_OVERFIT_README.md`
- `TUKE_SWINUNETR_FINETUNE_README.md`
- `MANIFEST.md`

No `.env`, checkpoint, dataset, metric log, virtual-environment, or user-specific code path is
included.
