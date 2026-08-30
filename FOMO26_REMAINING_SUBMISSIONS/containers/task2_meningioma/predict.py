#!/usr/bin/env python3
"""FOMO26 Task 2 meningioma segmentation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from inference import segmentation_ensemble
from postprocessing import task2_mask
from preprocessing import load_segmentation_tensor, remove_padding, save_original_grid_mask


WEIGHTS = Path(os.environ.get("FOMO_WEIGHTS_DIR", "/app/weights"))
ROI_SIZE = (96, 96, 96)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flair", required=True)
    parser.add_argument("--dwi", required=True)
    parser.add_argument("--t2s")
    parser.add_argument("--swi")
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()
    if not args.t2s and not args.swi:
        raise ValueError("Task 2 requires one of --t2s or --swi")
    for value in (args.flair, args.dwi, args.t2s or args.swi):
        if not Path(value).is_file():
            raise FileNotFoundError(value)

    image, context = load_segmentation_tensor([args.flair, args.dwi], ROI_SIZE)
    probabilities, folds = segmentation_ensemble(
        image, WEIGHTS, 2, 2, ROI_SIZE, 0.5, not args.no_tta
    )
    probability = remove_padding(probabilities[0, 1].numpy(), context)
    mask = task2_mask(probability, threshold=0.5, keep_components=1)
    save_original_grid_mask(mask, context, args.output)
    if args.diagnostics:
        Path(args.diagnostics).write_text(json.dumps({
            "folds": folds,
            "tta": not args.no_tta,
            "threshold": 0.5,
            "postprocessing": "largest_26_connected_component",
            "ignored_required_modality": "t2s" if args.t2s else "swi",
            "foreground_voxels_canonical": int(mask.sum()),
            "preprocessing": {key: value for key, value in context.items() if key not in {"original", "canonical"}},
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
