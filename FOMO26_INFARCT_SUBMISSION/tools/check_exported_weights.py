#!/usr/bin/env python3
"""Strictly load all exported weights and run a small synthetic forward pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--container-code", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.container_code.resolve()))
    from model import InfarctSwinClassifier

    paths = sorted(args.weights_dir.glob("fold_*.pt"))
    if len(paths) != 5:
        raise RuntimeError(f"Expected five weights, found {len(paths)} in {args.weights_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InfarctSwinClassifier().to(device).eval()
    test_input = torch.zeros((1, 3, 96, 96, 96), dtype=torch.float32, device=device)
    test_input[:, :, 24:72, 24:72, 24:72] = 1.0
    rows = []
    with torch.inference_mode():
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["state_dict"], strict=True)
            logits = model(test_input)
            probability = torch.softmax(logits.float(), dim=1)[0, 1]
            if logits.shape != (1, 2) or not torch.isfinite(probability):
                raise RuntimeError(f"Invalid forward result from {path}: shape={tuple(logits.shape)}")
            rows.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "probability": float(probability.cpu()),
                    "metadata": payload.get("metadata", {}),
                }
            )
    report = {
        "status": "PASS",
        "device": str(device),
        "fold_count": len(rows),
        "folds": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"EXPORTED_WEIGHTS_CHECK_PASS folds={len(rows)} report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

