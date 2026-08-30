#!/usr/bin/env python3
"""FOMO26 Task 5 polymicrogyria classification probability."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from inference import classification_probability
from preprocessing import load_clsreg_tensor


WEIGHTS = Path(os.environ.get("FOMO_WEIGHTS_DIR", "/app/weights"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()
    if not Path(args.t1).is_file():
        raise FileNotFoundError(args.t1)
    image, _metadata = load_clsreg_tensor(args.t1)
    probability, _folds = classification_probability(image, WEIGHTS, not args.no_tta)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError(f"Invalid probability {probability}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{probability:.10f}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
