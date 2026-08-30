#!/usr/bin/env python3
"""Run every SIF twice and enforce contracts beyond the official validator."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


def run(sif: Path, args: list[str], root: Path, output: Path, timeout: int) -> subprocess.CompletedProcess:
    command = ["apptainer", "exec", "--nv", "--bind", f"{root.resolve()}:/input:ro", "--bind", f"{output.parent.resolve()}:/output:rw", str(sif.resolve()), "python", "/app/predict.py", *args]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Failed: {' '.join(command)}\nstdout={result.stdout[-2000:]}\nstderr={result.stderr[-4000:]}")
    return result


def nifti_check(output: Path, reference: Path, labels: set[int]) -> dict:
    predicted = nib.load(str(output))
    source = nib.load(str(reference))
    data = np.asarray(predicted.dataobj)
    observed = set(int(v) for v in np.unique(data))
    if predicted.shape != source.shape or not np.allclose(predicted.affine, source.affine, rtol=0, atol=1e-5):
        raise RuntimeError(f"Geometry mismatch output={predicted.shape} input={source.shape}")
    if not observed <= labels:
        raise RuntimeError(f"Invalid labels {observed}")
    return {"shape": list(predicted.shape), "labels": sorted(observed), "foreground_voxels": int((data > 0).sum())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    sifs = {
        "task2": args.build_root / "fomo26_task2_meningioma.sif",
        "task3": args.build_root / "fomo26_task3_brain_age.sif",
        "task4": args.build_root / "fomo26_task4_trigeminal.sif",
        "task5": args.build_root / "fomo26_task5_polymicrogyria.sif",
        "task6": args.build_root / "fomo26_task6_7_embeddings.sif",
    }
    for path in sifs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = {}
    with tempfile.TemporaryDirectory(prefix="fomo26_contract_") as temporary:
        out = Path(temporary)
        t2 = args.input_root / "task2/case_swi"
        task2_out = out / "task2.nii.gz"
        run(sifs["task2"], ["--flair", "/input/flair.nii.gz", "--dwi", "/input/dwi.nii.gz", "--swi", "/input/swi.nii.gz", "--output", f"/output/{task2_out.name}"], t2, task2_out, args.timeout)
        rows["task2"] = nifti_check(task2_out, t2 / "flair.nii.gz", {0, 1})

        t3 = args.input_root / "task3/case_001"
        task3_out = out / "task3.txt"
        run(sifs["task3"], ["--t1", "/input/t1.nii.gz", "--output", f"/output/{task3_out.name}"], t3, task3_out, args.timeout)
        age = float(task3_out.read_text().strip())
        if not math.isfinite(age):
            raise RuntimeError("Non-finite age")
        rows["task3"] = {"prediction": age}

        t4 = args.input_root / "task4/case_001"
        task4_out = out / "task4.nii.gz"
        run(sifs["task4"], ["--t2", "/input/t2.nii.gz", "--output", f"/output/{task4_out.name}"], t4, task4_out, args.timeout)
        rows["task4"] = nifti_check(task4_out, t4 / "t2.nii.gz", {0, 1, 2})

        t5 = args.input_root / "task5/case_001"
        task5_out = out / "task5.txt"
        run(sifs["task5"], ["--t1", "/input/t1.nii.gz", "--output", f"/output/{task5_out.name}"], t5, task5_out, args.timeout)
        probability = float(task5_out.read_text().strip())
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError("Invalid Task 5 probability")
        rows["task5"] = {"probability": probability}

        embeddings = []
        for case in ("case_001", "case_002"):
            source = args.input_root / "task6" / case
            target = out / f"{case}.npy"
            run(sifs["task6"], ["--input", "/input/input.nii.gz", "--output", f"/output/{target.name}"], source, target, args.timeout)
            vector = np.load(target, allow_pickle=False)
            if vector.shape != (1440,) or not np.isfinite(vector).all():
                raise RuntimeError(f"Invalid embedding {case}: {vector.shape}")
            embeddings.append(vector)
        if np.array_equal(embeddings[0], embeddings[1]):
            raise RuntimeError("Different scans produced identical embeddings")
        rows["task6_and_7"] = {"shape": [1440], "cosine": float(np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))}

    report = {"status": "PASS", "checks": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ALL_CONTRACT_CHECKS_PASS report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
