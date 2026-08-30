#!/usr/bin/env python3
"""Compare submission preprocessing against saved fine-tuning validation tensors."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch


TASKS = {
    "SEG009_FOMO26_Meningioma": ("seg", 2),
    "REGR002_FOMO26_BrainAge": ("clsreg", 1),
    "SEG010_FOMO26_TrigeminalNeuralgia": ("seg", 1),
    "CLS003_FOMO26_Polymicrogyria": ("clsreg", 1),
}


def stored_image(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, (list, tuple)):
        return payload[0].float()
    if isinstance(payload, dict) and "image" in payload:
        return payload["image"].float()
    if isinstance(payload, torch.Tensor):
        return payload.float()
    raise ValueError(f"Unsupported tensor payload {type(payload)} in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--common-code", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--cases-per-task", type=int, default=2)
    parser.add_argument("--atol", type=float, default=2e-5)
    args = parser.parse_args()
    sys.path.insert(0, str(args.common_code.resolve()))
    from preprocessing import load_clsreg_tensor, load_segmentation_tensor
    from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop, CPU_seg_test_transforms

    result = {}
    for dataset, (kind, channels) in TASKS.items():
        root = args.data_root / dataset
        transform = CPU_seg_test_transforms([96, 96, 96]) if kind == "seg" else CPU_clsreg_val_test_transforms_crop([96, 96, 96])
        rows = []
        for metadata_path in sorted(root.rglob("*.pkl")):
            tensor_path = metadata_path.with_suffix(".pt")
            if not tensor_path.is_file():
                continue
            try:
                with metadata_path.open("rb") as handle:
                    metadata = pickle.load(handle)
                paths = [str(value) for value in metadata["src_image_paths"][:channels]]
            except Exception:
                continue
            if len(paths) != channels or not all(Path(value).is_file() for value in paths):
                continue
            image = stored_image(tensor_path)
            expected = transform({"image": image.clone(), "transforms_applied": {}, "properties": {}})["image"].unsqueeze(0)
            actual = load_segmentation_tensor(paths, (96, 96, 96))[0] if kind == "seg" else load_clsreg_tensor(paths[0])[0]
            if expected.shape != actual.shape:
                raise RuntimeError(f"Shape mismatch {dataset}: expected={tuple(expected.shape)} actual={tuple(actual.shape)}")
            difference = (expected.float() - actual.float()).abs()
            rows.append({
                "case": str(tensor_path.relative_to(root)),
                "shape": list(actual.shape),
                "max_abs_difference": float(difference.max()),
                "mean_abs_difference": float(difference.mean()),
                "allclose": bool(torch.allclose(expected.float(), actual.float(), rtol=0, atol=args.atol)),
            })
            if len(rows) >= args.cases_per_task:
                break
        if len(rows) < args.cases_per_task:
            raise RuntimeError(f"Only {len(rows)} comparable cases found for {dataset}")
        if not all(row["allclose"] for row in rows):
            raise RuntimeError(f"Preprocessing mismatch for {dataset}: {rows}")
        result[dataset] = rows
    report = {"status": "PASS", "absolute_tolerance": args.atol, "tasks": result}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ALL_PREPROCESSING_EQUIVALENCE_PASS report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
