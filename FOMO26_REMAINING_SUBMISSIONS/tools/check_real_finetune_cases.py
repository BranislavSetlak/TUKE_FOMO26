#!/usr/bin/env python3
"""Run one labeled fine-tune case per downstream task through the final SIFs."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


DATASETS = {
    "task2": "SEG009_FOMO26_Meningioma",
    "task3": "REGR002_FOMO26_BrainAge",
    "task4": "SEG010_FOMO26_TrigeminalNeuralgia",
    "task5": "CLS003_FOMO26_Polymicrogyria",
}


def first_case(root: Path) -> dict:
    for metadata_path in sorted(root.rglob("*.pkl")):
        try:
            with metadata_path.open("rb") as handle:
                metadata = pickle.load(handle)
            sources = [Path(value) for value in metadata["src_image_paths"]]
            label = Path(metadata["src_label_path"])
        except Exception:
            continue
        if sources and all(path.is_file() for path in sources) and label.is_file():
            return {"metadata": metadata_path, "sources": sources, "label": label}
    raise RuntimeError(f"No usable metadata case in {root}")


def execute(sif: Path, command: list[str], binds: set[Path], output_dir: Path, timeout: int) -> None:
    full = ["apptainer", "exec", "--nv"]
    for directory in sorted(str(path.resolve()) for path in binds):
        full += ["--bind", f"{directory}:{directory}:ro"]
    full += ["--bind", f"{output_dir.resolve()}:/check:rw", str(sif.resolve()), "python", "/app/predict.py", *command]
    result = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"SIF failed rc={result.returncode}\nstdout={result.stdout[-1500:]}\nstderr={result.stderr[-3000:]}")


def dice(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    p, t = pred == class_id, target == class_id
    denominator = int(p.sum() + t.sum())
    return float(2 * (p & t).sum() / denominator) if denominator else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--full-tta", action="store_true")
    args = parser.parse_args()
    rows = {}
    with tempfile.TemporaryDirectory(prefix="fomo26_real_cases_") as temp:
        out = Path(temp)
        for task, dataset in DATASETS.items():
            case = first_case(args.data_root / dataset)
            sources, label = case["sources"], case["label"]
            common = [] if args.full_tta else ["--no-tta"]
            if task == "task2":
                optional = next((path for name in ("swi.nii.gz", "t2s.nii.gz") if (path := sources[0].parent / name).is_file()), None)
                if optional is None:
                    raise RuntimeError(f"Task 2 case lacks SWI/T2*: {sources[0].parent}")
                target = out / "task2.nii.gz"
                command = ["--flair", str(sources[0]), "--dwi", str(sources[1]), f"--{'swi' if optional.name.startswith('swi') else 't2s'}", str(optional), "--output", "/check/task2.nii.gz", *common]
                execute(args.build_root / "fomo26_task2_meningioma.sif", command, {p.parent for p in [*sources, optional]}, out, args.timeout)
                pred = np.asarray(nib.load(str(target)).dataobj)
                truth = np.asarray(nib.load(str(label)).dataobj)
                rows[task] = {"dice_1": dice(pred, truth, 1), "pred_foreground_fraction": float((pred > 0).mean())}
            elif task == "task3":
                target = out / "task3.txt"
                execute(args.build_root / "fomo26_task3_brain_age.sif", ["--t1", str(sources[0]), "--output", "/check/task3.txt", *common], {sources[0].parent}, out, args.timeout)
                prediction, truth = float(target.read_text().strip()), float(label.read_text().strip())
                if not math.isfinite(prediction): raise RuntimeError("Non-finite age")
                rows[task] = {"prediction": prediction, "target": truth, "absolute_error": abs(prediction - truth)}
            elif task == "task4":
                target = out / "task4.nii.gz"
                execute(args.build_root / "fomo26_task4_trigeminal.sif", ["--t2", str(sources[0]), "--output", "/check/task4.nii.gz", *common], {sources[0].parent}, out, args.timeout)
                pred, truth = np.asarray(nib.load(str(target)).dataobj), np.asarray(nib.load(str(label)).dataobj)
                rows[task] = {"dice_1": dice(pred, truth, 1), "dice_2": dice(pred, truth, 2), "pred_foreground_fraction": float((pred > 0).mean())}
            else:
                target = out / "task5.txt"
                execute(args.build_root / "fomo26_task5_polymicrogyria.sif", ["--t1", str(sources[0]), "--output", "/check/task5.txt", *common], {sources[0].parent}, out, args.timeout)
                probability, truth = float(target.read_text().strip()), int(label.read_text().strip())
                if not 0 <= probability <= 1: raise RuntimeError("Invalid probability")
                rows[task] = {"probability": probability, "target": truth}
    report = {"status": "PASS", "warning": "Engineering smoke sample only; not a performance estimate.", "full_tta": args.full_tta, "tasks": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"REAL_FINETUNE_CASES_PASS report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
