#!/usr/bin/env python3
"""FOMO26 Task 4 trigeminal nerve/vessel multiclass segmentation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from inference import segmentation_ensemble
from postprocessing import task4_mask
from preprocessing import load_segmentation_tensor, remove_padding, save_original_grid_mask


WEIGHTS = Path(os.environ.get("FOMO_WEIGHTS_DIR", "/app/weights"))
ROI_SIZE = (96, 96, 96)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t2", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()
    if not Path(args.t2).is_file():
        raise FileNotFoundError(args.t2)
    image, context = load_segmentation_tensor([args.t2], ROI_SIZE)
    probabilities, folds = segmentation_ensemble(
        image, WEIGHTS, 1, 3, ROI_SIZE, 0.5, not args.no_tta
    )
    unpadded = [remove_padding(probabilities[0, class_id].numpy(), context) for class_id in range(3)]
    mask = task4_mask(np.stack(unpadded), min_voxels=4, min_largest_ratio=0.01)
    save_original_grid_mask(mask, context, args.output)
    if args.diagnostics:
        Path(args.diagnostics).write_text(json.dumps({
            "folds": folds,
            "tta": not args.no_tta,
            "postprocessing": "per_class_remove_lt_max_4_voxels_or_1pct_largest",
            "class_voxels_canonical": {str(c): int((mask == c).sum()) for c in (0, 1, 2)},
            "preprocessing": {key: value for key, value in context.items() if key not in {"original", "canonical"}},
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
