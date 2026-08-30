#!/usr/bin/env python3
"""Run the SIF on a few balanced Task 1 fine-tuning cases."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import subprocess
import tempfile
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sif", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--fast", action="store_true", help="Disable TTA for this engineering check")
    return parser.parse_args()


def _metadata_cases(root: Path) -> list[dict]:
    cases = []
    for metadata_path in sorted(root.rglob("*.pkl")):
        try:
            with metadata_path.open("rb") as handle:
                metadata = pickle.load(handle)
        except Exception:
            continue
        if not isinstance(metadata, dict):
            continue
        paths = metadata.get("src_image_paths")
        label_path = metadata.get("src_label_path")
        if not isinstance(paths, (list, tuple)) or len(paths) < 3 or not label_path:
            continue
        label_file = Path(label_path)
        try:
            label = int(label_file.read_text(encoding="utf-8").strip())
        except Exception:
            continue
        modality_paths = [Path(value) for value in paths[:3]]
        if not all(path.is_file() for path in modality_paths):
            continue
        case_dir = modality_paths[0].parent
        optional = None
        optional_name = None
        for name, candidates in (
            ("swi", ("swi.nii.gz", "SWI.nii.gz")),
            ("t2s", ("t2s.nii.gz", "t2star.nii.gz", "T2S.nii.gz")),
        ):
            match = next((case_dir / candidate for candidate in candidates if (case_dir / candidate).is_file()), None)
            if match is not None:
                optional, optional_name = match, name
                break
        if optional is None:
            continue
        cases.append(
            {
                "id": metadata_path.stem,
                "label": label,
                "flair": modality_paths[0],
                "adc": modality_paths[1],
                "dwi": modality_paths[2],
                optional_name: optional,
                "optional_name": optional_name,
            }
        )
    return cases


def _choose_balanced(cases: list[dict], per_class: int) -> list[dict]:
    selected = []
    for label in (0, 1):
        matches = [case for case in cases if case["label"] == label]
        if len(matches) < per_class:
            raise RuntimeError(f"Found only {len(matches)} usable label={label} cases; need {per_class}")
        selected.extend(matches[:per_class])
    return selected


def _run_case(sif: Path, case: dict, output_dir: Path, timeout: int, fast: bool) -> dict:
    output = output_dir / f"{case['id']}.txt"
    diagnostics = output_dir / f"{case['id']}.json"
    host_paths = [case["flair"], case["adc"], case["dwi"], case[case["optional_name"]]]
    bind_dirs = sorted({str(path.parent.resolve()) for path in host_paths})
    command = ["apptainer", "exec", "--nv"]
    for directory in bind_dirs:
        command += ["--bind", f"{directory}:{directory}:ro"]
    command += ["--bind", f"{output_dir.resolve()}:/check_output:rw", str(sif.resolve()), "python", "/app/predict.py"]
    command += [
        "--flair", str(case["flair"]),
        "--adc", str(case["adc"]),
        "--dwi", str(case["dwi"]),
        f"--{case['optional_name']}", str(case[case["optional_name"]]),
        "--output", f"/check_output/{output.name}",
        "--diagnostics", f"/check_output/{diagnostics.name}",
    ]
    if fast:
        command.append("--no-tta")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Container failed for {case['id']} rc={completed.returncode}\n"
            f"stdout={completed.stdout[-1000:]}\nstderr={completed.stderr[-2000:]}"
        )
    probability = float(output.read_text(encoding="utf-8").strip())
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError(f"Invalid probability for {case['id']}: {probability}")
    return {
        "case_id": case["id"],
        "label": case["label"],
        "probability": probability,
        "prediction": int(probability >= 0.5),
        "optional_modality": case["optional_name"],
        "diagnostics": json.loads(diagnostics.read_text(encoding="utf-8")),
    }


def main() -> int:
    args = arguments()
    if not args.sif.is_file():
        raise FileNotFoundError(args.sif)
    cases = _choose_balanced(_metadata_cases(args.data_root), args.per_class)
    with tempfile.TemporaryDirectory(prefix="fomo26_infarct_check_") as temporary:
        output_dir = Path(temporary)
        rows = [_run_case(args.sif, case, output_dir, args.timeout, args.fast) for case in cases]

    negatives = [row["probability"] for row in rows if row["label"] == 0]
    positives = [row["probability"] for row in rows if row["label"] == 1]
    report = {
        "status": "PASS",
        "warning": "Small engineering sample only; do not report these as challenge performance.",
        "sif": str(args.sif),
        "fast_no_tta": args.fast,
        "n_cases": len(rows),
        "mean_probability_label_0": sum(negatives) / len(negatives),
        "mean_probability_label_1": sum(positives) / len(positives),
        "threshold_accuracy": sum(row["prediction"] == row["label"] for row in rows) / len(rows),
        "cases": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"FINETUNE_CASE_CHECK_PASS cases={len(rows)} "
        f"mean_p0={report['mean_probability_label_0']:.6f} "
        f"mean_p1={report['mean_probability_label_1']:.6f} report={args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

