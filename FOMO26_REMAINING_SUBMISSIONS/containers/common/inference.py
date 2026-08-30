"""Shared ensemble, TTA and sliding-window inference helpers."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import torch
from monai.inferers import sliding_window_inference

from model import ClsRegModel, MultiScaleEmbeddingModel, SegmentationModel


def flips(enabled: bool) -> list[tuple[int, ...]]:
    if not enabled:
        return [()]
    axes = (2, 3, 4)
    return [
        tuple(axis for axis, use in zip(axes, bits, strict=True) if use)
        for bits in itertools.product((False, True), repeat=3)
    ]


def exported_folds(weights_dir: Path) -> list[Path]:
    paths = sorted(weights_dir.glob("fold_*.pt"))
    if len(paths) != 5:
        raise RuntimeError(f"Expected five fold weights in {weights_dir}, found {len(paths)}")
    return paths


def _payload(path: Path) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"Unsupported exported checkpoint {path}")
    return payload["state_dict"], dict(payload.get("metadata", {}))


def _device() -> tuple[torch.device, torch.dtype]:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    major, _minor = torch.cuda.get_device_capability(device)
    return device, torch.bfloat16 if major >= 8 else torch.float16


def clsreg_ensemble(image: torch.Tensor, weights_dir: Path, out_channels: int, tta: bool) -> tuple[torch.Tensor, list[dict]]:
    device, dtype = _device()
    model = ClsRegModel(1, out_channels).to(device).eval()
    image = image.to(device)
    rows = []
    outputs = []
    for path in exported_folds(weights_dir):
        state, metadata = _payload(path)
        model.load_state_dict(state, strict=True)
        values = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for axes in flips(tta):
                augmented = torch.flip(image, axes) if axes else image
                values.append(model(augmented).float().cpu())
        fold_output = torch.stack(values).mean(0)
        outputs.append(fold_output)
        rows.append({"fold": metadata.get("fold"), "variant": metadata.get("variant")})
    result = torch.stack(outputs).mean(0)
    if not torch.isfinite(result).all():
        raise RuntimeError("Non-finite ensemble output")
    return result, rows


def classification_probability(image: torch.Tensor, weights_dir: Path, tta: bool) -> tuple[float, list[dict]]:
    """Average positive-class probabilities, never hard labels or raw logits."""
    device, dtype = _device()
    model = ClsRegModel(1, 2).to(device).eval()
    image = image.to(device)
    rows = []
    fold_probabilities = []
    for path in exported_folds(weights_dir):
        state, metadata = _payload(path)
        model.load_state_dict(state, strict=True)
        values = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for axes in flips(tta):
                augmented = torch.flip(image, axes) if axes else image
                values.append(float(torch.softmax(model(augmented).float(), dim=1)[0, 1].cpu()))
        fold_probability = sum(values) / len(values)
        fold_probabilities.append(fold_probability)
        rows.append({"fold": metadata.get("fold"), "variant": metadata.get("variant"), "probability": fold_probability})
    result = sum(fold_probabilities) / len(fold_probabilities)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RuntimeError(f"Invalid ensemble probability {result}")
    return result, rows


def segmentation_ensemble(
    image: torch.Tensor,
    weights_dir: Path,
    in_channels: int,
    out_channels: int,
    roi_size: tuple[int, int, int],
    overlap: float,
    tta: bool,
) -> tuple[torch.Tensor, list[dict]]:
    device, dtype = _device()
    model = SegmentationModel(in_channels, out_channels).to(device).eval()
    total = None
    count = 0
    rows = []
    for path in exported_folds(weights_dir):
        state, metadata = _payload(path)
        model.load_state_dict(state, strict=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for axes in flips(tta):
                augmented = torch.flip(image, axes) if axes else image
                logits = sliding_window_inference(
                    inputs=augmented,
                    roi_size=roi_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                    sw_device=device,
                    device=torch.device("cpu"),
                )
                if axes:
                    logits = torch.flip(logits, axes)
                probability = torch.softmax(logits.float(), dim=1)
                total = probability if total is None else total + probability
                count += 1
        rows.append({"fold": metadata.get("fold"), "variant": metadata.get("variant")})
    if total is None or count == 0:
        raise RuntimeError("No segmentation predictions were produced")
    probabilities = total / count
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Non-finite segmentation probabilities")
    return probabilities, rows


def embedding(image: torch.Tensor, weight_path: Path, tta: bool) -> torch.Tensor:
    device, dtype = _device()
    state, _metadata = _payload(weight_path)
    model = MultiScaleEmbeddingModel().to(device).eval()
    model.load_state_dict(state, strict=True)
    image = image.to(device)
    values = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for axes in flips(tta):
            augmented = torch.flip(image, axes) if axes else image
            values.append(model(augmented).float().cpu())
    result = torch.stack(values).mean(0)
    result = torch.nn.functional.normalize(result, p=2, dim=1, eps=1e-8)
    if result.shape != (1, MultiScaleEmbeddingModel.embedding_dim) or not torch.isfinite(result).all():
        raise RuntimeError(f"Invalid embedding shape {tuple(result.shape)}")
    if not math.isfinite(float(result.norm())):
        raise RuntimeError("Invalid embedding norm")
    return result
