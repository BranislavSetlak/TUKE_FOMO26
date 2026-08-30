#!/usr/bin/env python3
"""FOMO26 Tasks 6 and 7 general MRI embedding inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from inference import embedding
from preprocessing import load_clsreg_tensor


WEIGHTS = Path(os.environ.get("FOMO_WEIGHTS_DIR", "/app/weights"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()
    if not Path(args.input).is_file():
        raise FileNotFoundError(args.input)
    weights = sorted(WEIGHTS.glob("encoder.pt"))
    if len(weights) != 1:
        raise RuntimeError(f"Expected /app/weights/encoder.pt, found {len(weights)}")
    image, _metadata = load_clsreg_tensor(args.input)
    vector = embedding(image, weights[0], not args.no_tta).numpy().reshape(-1).astype(np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, vector, allow_pickle=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
