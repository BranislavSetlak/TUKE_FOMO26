#!/usr/bin/env python3
"""Extra Task 1 contract, determinism, and failure-mode checks."""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from pathlib import Path


def run(command: list[str], expected_success: bool, timeout: int) -> subprocess.CompletedProcess:
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    succeeded = completed.returncode == 0
    if succeeded != expected_success:
        raise RuntimeError(
            f"Unexpected rc={completed.returncode} for {' '.join(command)}\n"
            f"stdout={completed.stdout[-1000:]}\nstderr={completed.stderr[-2000:]}"
        )
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sif", required=True, type=Path)
    parser.add_argument("--flair", required=True, type=Path)
    parser.add_argument("--adc", required=True, type=Path)
    parser.add_argument("--dwi", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--t2s", type=Path)
    group.add_argument("--swi", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    paths = [args.flair, args.adc, args.dwi, args.t2s or args.swi]
    if not all(path and path.is_file() for path in paths):
        raise FileNotFoundError("One or more contract-check inputs do not exist")
    optional_name = "t2s" if args.t2s else "swi"
    optional_path = args.t2s or args.swi
    bind_dirs = sorted({str(path.parent.resolve()) for path in paths if path})

    with tempfile.TemporaryDirectory(prefix="fomo_contract_") as temporary:
        output_dir = Path(temporary)
        base = ["apptainer", "exec", "--nv"]
        for directory in bind_dirs:
            base += ["--bind", f"{directory}:{directory}:ro"]
        base += ["--bind", f"{output_dir.resolve()}:/output:rw", str(args.sif.resolve()), "python", "/app/predict.py"]
        inputs = [
            "--flair", str(args.flair), "--adc", str(args.adc), "--dwi", str(args.dwi),
            f"--{optional_name}", str(optional_path), "--no-tta",
        ]
        values = []
        for index in range(2):
            output = output_dir / f"repeat_{index}.txt"
            run(base + inputs + ["--output", f"/output/{output.name}"], True, args.timeout)
            tokens = output.read_text(encoding="utf-8").strip().split()
            if len(tokens) != 1:
                raise RuntimeError(f"Output is not exactly one token: {tokens}")
            value = float(tokens[0])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError(f"Invalid probability: {value}")
            values.append(value)
        if abs(values[0] - values[1]) > 1e-7:
            raise RuntimeError(f"Non-deterministic repeated inference: {values}")

        missing_optional = output_dir / "missing_optional.txt"
        run(
            base
            + ["--flair", str(args.flair), "--adc", str(args.adc), "--dwi", str(args.dwi), "--output", f"/output/{missing_optional.name}", "--no-tta"],
            False,
            args.timeout,
        )
        if missing_optional.exists():
            raise RuntimeError("Missing-optional negative test unexpectedly wrote output")

    print(f"CONTRACT_CHECKS_PASS deterministic_probability={values[0]:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

