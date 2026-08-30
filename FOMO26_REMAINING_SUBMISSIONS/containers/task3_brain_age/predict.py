#!/usr/bin/env python3
"""FOMO26 Task 3 brain-age regression."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from inference import clsreg_ensemble
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
    prediction, _folds = clsreg_ensemble(image, WEIGHTS, 1, not args.no_tta)
    age = float(prediction.reshape(-1)[0])
    if not math.isfinite(age):
        raise RuntimeError(f"Invalid age prediction {age}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{age:.8f}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
