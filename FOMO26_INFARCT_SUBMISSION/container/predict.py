#!/usr/bin/env python3
"""FOMO26 Task 1 infarct probability inference."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

import torch

from model import InfarctSwinClassifier
from preprocessing import load_infarct_tensor


EXPECTED_FOLDS = 5
WEIGHTS_DIR = Path(os.environ.get("FOMO_WEIGHTS_DIR", "/app/weights"))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOMO26 Task 1 infarct classification")
    parser.add_argument("--flair", required=True)
    parser.add_argument("--adc", required=True)
    parser.add_argument("--dwi", required=True)
    parser.add_argument("--t2s")
    parser.add_argument("--swi")
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics", help="Optional JSON diagnostic output")
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable 8-way flip TTA for a faster engineering check",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.t2s and not args.swi:
        raise ValueError("Task 1 requires one of --t2s or --swi")
    for name in ("flair", "adc", "dwi"):
        path = Path(getattr(args, name))
        if not path.is_file():
            raise FileNotFoundError(f"Missing --{name} input: {path}")
    for name in ("t2s", "swi"):
        value = getattr(args, name)
        if value and not Path(value).is_file():
            raise FileNotFoundError(f"Missing --{name} input: {value}")


def _weight_paths() -> list[Path]:
    paths = sorted(WEIGHTS_DIR.glob("fold_*.pt"))
    if len(paths) != EXPECTED_FOLDS:
        raise RuntimeError(
            f"Expected {EXPECTED_FOLDS} exported fold weights in {WEIGHTS_DIR}, found {len(paths)}"
        )
    return paths


def _payload(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"Unsupported exported checkpoint: {path}")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"No state_dict in {path}")
    return state_dict, dict(payload.get("metadata", {}))


def _tta_flips(enabled: bool) -> list[tuple[int, ...]]:
    if not enabled:
        return [()]
    spatial_axes = (2, 3, 4)
    return [
        tuple(axis for axis, selected in zip(spatial_axes, bits, strict=True) if selected)
        for bits in itertools.product((False, True), repeat=3)
    ]


def predict_probability(image: torch.Tensor, tta: bool) -> tuple[float, list[dict]]:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this submission container")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    image = image.to(device, non_blocking=True)
    model = InfarctSwinClassifier().to(device).eval()
    flips = _tta_flips(tta)
    fold_results = []

    major, _minor = torch.cuda.get_device_capability(device)
    amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    for weight_path in _weight_paths():
        state_dict, metadata = _payload(weight_path)
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Weight mismatch in {weight_path}: {incompatible}")
        probabilities = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
            for axes in flips:
                augmented = torch.flip(image, axes) if axes else image
                logits = model(augmented)
                probability = torch.softmax(logits.float(), dim=1)[0, 1]
                probabilities.append(float(probability.cpu()))
        fold_probability = sum(probabilities) / len(probabilities)
        fold_results.append(
            {
                "fold": metadata.get("fold", weight_path.stem),
                "variant": metadata.get("variant", "unknown"),
                "probability": fold_probability,
                "tta_probabilities": probabilities,
            }
        )

    probability = sum(item["probability"] for item in fold_results) / len(fold_results)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError(f"Invalid ensemble probability: {probability}")
    return probability, fold_results


def main() -> int:
    args = arguments()
    _validate_args(args)
    image, preprocessing = load_infarct_tensor(args.flair, args.adc, args.dwi)
    probability, folds = predict_probability(image, tta=not args.no_tta)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{probability:.10f}\n", encoding="utf-8")
    if args.diagnostics:
        diagnostics = Path(args.diagnostics)
        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.write_text(
            json.dumps(
                {
                    "probability": probability,
                    "ensemble_size": len(folds),
                    "tta_count": len(folds[0]["tta_probabilities"]),
                    "preprocessing": preprocessing,
                    "folds": folds,
                    "ignored_optional_modality": "t2s" if args.t2s else "swi",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

