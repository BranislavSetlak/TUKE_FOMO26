#!/usr/bin/env python3
"""Compare container preprocessing with the fine-tuning validation pipeline."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--container-code", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--atol", type=float, default=2e-5)
    args = parser.parse_args()

    sys.path.insert(0, str(args.container_code.resolve()))
    from preprocessing import load_infarct_tensor
    from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop

    transform = CPU_clsreg_val_test_transforms_crop(target_size=[96, 96, 96])
    rows = []
    for metadata_path in sorted(args.data_root.rglob("*.pkl")):
        tensor_path = metadata_path.with_suffix(".pt")
        if not tensor_path.is_file():
            continue
        try:
            with metadata_path.open("rb") as handle:
                metadata = pickle.load(handle)
        except Exception:
            continue
        source_paths = metadata.get("src_image_paths") if isinstance(metadata, dict) else None
        if not isinstance(source_paths, (list, tuple)) or len(source_paths) < 3:
            continue
        if not all(Path(path).is_file() for path in source_paths[:3]):
            continue

        stored = torch.load(tensor_path, map_location="cpu", weights_only=False)
        stored_image = stored[0].float()
        expected = transform(
            {"image": stored_image.clone(), "transforms_applied": {}, "properties": {}}
        )["image"].unsqueeze(0)
        actual, preprocessing_metadata = load_infarct_tensor(*source_paths[:3])
        difference = (expected.float() - actual.float()).abs()
        row = {
            "case": str(tensor_path.relative_to(args.data_root)),
            "shape": list(actual.shape),
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
            "allclose": bool(torch.allclose(expected.float(), actual.float(), rtol=0.0, atol=args.atol)),
            "preprocessing": preprocessing_metadata,
        }
        rows.append(row)
        if len(rows) >= args.cases:
            break

    if len(rows) < args.cases:
        raise RuntimeError(f"Found only {len(rows)} comparable Task 1 cases; requested {args.cases}")
    passed = all(row["allclose"] for row in rows)
    report = {
        "status": "PASS" if passed else "FAIL",
        "absolute_tolerance": args.atol,
        "cases": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"PREPROCESSING_EQUIVALENCE_{report['status']} cases={len(rows)} "
        f"max_abs={max(row['max_abs_difference'] for row in rows):.9g} report={args.report}"
    )
    if not passed:
        raise SystemExit("Container preprocessing does not reproduce fine-tuning validation preprocessing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
