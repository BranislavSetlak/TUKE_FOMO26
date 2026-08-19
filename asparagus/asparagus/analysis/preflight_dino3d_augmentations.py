"""Small deterministic checks for the GIN, CarveMix, and FP-loss additions."""

import torch

from asparagus.modules.losses.asymmetric_tversky import AsymmetricTverskyCrossEntropyLoss
from asparagus.modules.transforms.carvemix import Torch_CarveMix
from asparagus.modules.transforms.gin import Torch_GIN
from asparagus.modules.transforms.presets import GPU_all_train_transforms_gin


def check_gin(device):
    torch.manual_seed(260826)
    image = torch.randn(2, 2, 16, 16, 16, device=device)
    label = torch.randint(0, 2, (2, 1, 16, 16, 16), device=device)
    original_image = image.clone()
    original_label = label.clone()
    batch = {"image": image, "label": label, "transforms_applied": {}}
    result = Torch_GIN(probability=1.0)(batch)

    assert result["image"].shape == original_image.shape
    assert torch.isfinite(result["image"]).all()
    assert torch.equal(result["label"], original_label)
    assert not torch.allclose(result["image"], original_image)
    original_norm = torch.linalg.vector_norm(original_image.flatten(start_dim=1), dim=1)
    result_norm = torch.linalg.vector_norm(result["image"].flatten(start_dim=1), dim=1)
    assert torch.allclose(original_norm, result_norm, rtol=2e-4, atol=2e-4)
    print("PREFLIGHT_GIN_OK")


def check_carvemix():
    torch.manual_seed(260826)
    receiver = {
        "image": torch.zeros(2, 24, 24, 24),
        "label": torch.zeros(1, 24, 24, 24),
        "file_path": "receiver.pt",
        "transforms_applied": {},
    }
    donor = {
        "image": torch.ones(2, 24, 24, 24),
        "label": torch.zeros(1, 24, 24, 24),
        "file_path": "donor.pt",
        "transforms_applied": {},
    }
    donor["label"][:, 8:16, 8:16, 8:16] = 1
    result = Torch_CarveMix(probability=1.0)(receiver, donor)

    assert result["image"].shape == donor["image"].shape
    assert result["label"].shape == donor["label"].shape
    assert int((result["label"] > 0).sum()) > 0
    assert torch.all(result["image"][result["label"].expand_as(result["image"]) > 0] == 1)
    assert set(torch.unique(result["label"]).tolist()).issubset({0.0, 1.0})
    assert result["transforms_applied"]["carvemix"]["donor"] == "donor.pt"
    print("PREFLIGHT_CARVEMIX_OK")


def check_fp_loss(device):
    target = torch.zeros(1, 1, 12, 12, 12, device=device)
    target[:, :, 5:7, 5:7, 5:7] = 1
    perfect_logits = torch.full((1, 2, 12, 12, 12), -6.0, device=device)
    perfect_logits[:, 0] = 6.0
    perfect_logits[:, 0, 5:7, 5:7, 5:7] = -6.0
    perfect_logits[:, 1, 5:7, 5:7, 5:7] = 6.0
    all_foreground_logits = torch.full_like(perfect_logits, -6.0)
    all_foreground_logits[:, 1] = 6.0

    fp_weighted = AsymmetricTverskyCrossEntropyLoss(alpha=0.7, beta=0.3)
    fn_weighted = AsymmetricTverskyCrossEntropyLoss(alpha=0.3, beta=0.7)
    perfect_loss = fp_weighted(perfect_logits, target)
    oversegmented_loss = fp_weighted(all_foreground_logits, target)
    less_fp_weighted_loss = fn_weighted(all_foreground_logits, target)
    assert torch.isfinite(perfect_loss)
    assert oversegmented_loss > perfect_loss
    assert oversegmented_loss > less_fp_weighted_loss
    print("PREFLIGHT_ASYMMETRIC_TVERSKY_CE_OK")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this preflight")
    device = torch.device("cuda")
    check_gin(device)
    check_carvemix()
    check_fp_loss(device)
    print("DINO3D_AUGMENTATION_PREFLIGHT_OK")


if __name__ == "__main__":
    main()
