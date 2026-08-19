"""CPU preflight for the DINO3D segmentation-recovery experiment bundle."""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from asparagus.modules.losses.asymmetric_tversky import AsymmetricTverskyCrossEntropyLoss
from asparagus.modules.losses.unified_focal import AsymmetricUnifiedFocalLoss
from asparagus.modules.transforms.anatomical_roi import Torch_FixedNormalizedCrop, fixed_roi_slices
from asparagus.modules.transforms.presets.train import (
    CPU_seg_anatomical_roi_noaug_transforms,
    CPU_seg_train_transforms_fg100,
    CPU_seg_train_transforms_fg75,
)


def synthetic_logits_and_target():
    target = torch.zeros((1, 1, 8, 8, 8), dtype=torch.long)
    target[:, :, 3:5, 3:5, 3:5] = 1

    empty = torch.zeros((1, 2, 8, 8, 8), dtype=torch.float32)
    empty[:, 0] = 6.0
    empty.requires_grad_()

    correct = torch.full((1, 2, 8, 8, 8), -6.0, dtype=torch.float32)
    correct[:, 0] = 6.0
    correct[:, 0, 3:5, 3:5, 3:5] = -6.0
    correct[:, 1, 3:5, 3:5, 3:5] = 6.0
    return empty, correct, target


def check_loss(loss, marker: str) -> None:
    empty, correct, target = synthetic_logits_and_target()
    empty_loss = loss(empty, target)
    correct_loss = loss(correct, target)
    if not torch.isfinite(empty_loss) or not torch.isfinite(correct_loss):
        raise RuntimeError(f"{marker} produced a non-finite value")
    if not correct_loss < empty_loss:
        raise RuntimeError(
            f"{marker} should prefer the correct mask: correct={correct_loss} empty={empty_loss}"
        )
    empty_loss.backward()
    if empty.grad is None or not torch.isfinite(empty.grad).all() or empty.grad.abs().sum() == 0:
        raise RuntimeError(f"{marker} did not produce finite non-zero gradients")
    print(f"{marker} correct={correct_loss.item():.6f} empty={empty_loss.item():.6f}")


def check_transforms() -> None:
    fg75 = CPU_seg_train_transforms_fg75([32, 32, 32])
    fg100 = CPU_seg_train_transforms_fg100([32, 32, 32])
    if abs(fg75.transforms[2].p_oversample_foreground - 0.75) > 1e-8:
        raise RuntimeError("75% foreground preset has the wrong sampling probability")
    if abs(fg100.transforms[2].p_oversample_foreground - 1.0) > 1e-8:
        raise RuntimeError("100% foreground preset has the wrong sampling probability")

    image = torch.linspace(-1.0, 1.0, 40**3).reshape(1, 40, 40, 40)
    label = torch.zeros((1, 40, 40, 40), dtype=torch.long)
    label[:, 19:22, 19:22, 19:22] = 1
    crop = Torch_FixedNormalizedCrop([16, 16, 16], [0.5, 0.5, 0.5])
    output = crop({"image": image, "label": label})
    if output["image"].shape != (1, 16, 16, 16) or output["label"].sum() == 0:
        raise RuntimeError("Fixed anatomical ROI crop failed")
    slices = fixed_roi_slices([40, 40, 40], [16, 16, 16], [0.0, 1.0, 0.5])
    if slices[0].start != 0 or slices[1].stop != 40:
        raise RuntimeError("Fixed ROI boundary clamping failed")

    roi_pipeline = CPU_seg_anatomical_roi_noaug_transforms(
        patch_size=[32, 32, 32],
        anatomical_roi_size=[16, 16, 16],
        normalized_center=[0.5, 0.5, 0.5],
    )
    roi_output = roi_pipeline({"image": image, "label": label})
    if (
        roi_output["image"].shape != (1, 32, 32, 32)
        or roi_output["label"].shape != (1, 32, 32, 32)
        or roi_output["label"].sum() == 0
    ):
        raise RuntimeError("ROI crop-and-resize pipeline failed")
    print("PREFLIGHT_SEGMENTATION_RECOVERY_TRANSFORMS_OK")


def check_config(config_path: Path) -> None:
    cfg = OmegaConf.load(config_path)
    required = {
        "unified_focal_weight",
        "unified_focal_delta",
        "unified_focal_gamma",
        "probability_thresholds",
        "anatomical_roi_enabled",
        "anatomical_roi_center",
        "anatomical_roi_size",
        "run_best_validation_after_fit",
        "run_test_after_fit",
    }
    missing = sorted(required - set(cfg.training))
    if missing:
        raise RuntimeError(f"Missing recovery config keys: {missing}")
    print("PREFLIGHT_SEGMENTATION_RECOVERY_CONFIG_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    check_loss(
        AsymmetricUnifiedFocalLoss(weight=0.5, delta=0.6, gamma=0.5),
        "PREFLIGHT_UNIFIED_FOCAL_OK",
    )
    check_loss(
        AsymmetricTverskyCrossEntropyLoss(
            alpha=0.3,
            beta=0.7,
            tversky_weight=1.0,
            cross_entropy_weight=0.2,
        ),
        "PREFLIGHT_FN_TVERSKY_OK",
    )
    check_transforms()
    check_config(args.config)
    print("DINO3D_SEGMENTATION_RECOVERY_PREFLIGHT_OK")


if __name__ == "__main__":
    main()
