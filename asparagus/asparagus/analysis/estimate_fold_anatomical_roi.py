"""Estimate a fixed anatomical ROI using only one fold's training labels."""

import argparse
import json
import os
import pickle
import statistics
from pathlib import Path


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def locations_from_metadata(metadata: dict) -> list[list[int]]:
    value = metadata.get("foreground_locations", {})
    if isinstance(value, dict):
        return [list(location) for locations in value.values() for location in locations]
    return [list(location) for location in value]


def centroid(locations: list[list[int]], shape: list[int]) -> list[float]:
    return [
        sum(float(location[axis]) for location in locations)
        / len(locations)
        / max(shape[axis] - 1, 1)
        for axis in range(3)
    ]


def roi_starts(shape: list[int], roi_size: list[int], center: list[float]) -> list[int]:
    starts = []
    for image_dim, roi_dim, center_fraction in zip(shape, roi_size, center):
        if image_dim <= roi_dim:
            starts.append(0)
            continue
        center_index = center_fraction * (image_dim - 1)
        start = int(round(center_index - (roi_dim - 1) / 2.0))
        starts.append(min(max(start, 0), image_dim - roi_dim))
    return starts


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--roi-size", required=True, type=int, nargs=3)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--center-output", required=True, type=Path)
    parser.add_argument("--min-training-coverage", type=float, default=0.90)
    args = parser.parse_args()
    if any(value <= 0 for value in args.roi_size):
        raise ValueError("ROI dimensions must be positive")
    if not 0.0 <= args.min_training_coverage <= 1.0:
        raise ValueError("min-training-coverage must be in [0, 1]")

    folds = read_json(args.split)
    if args.fold < 0 or args.fold >= len(folds):
        raise ValueError(f"Fold {args.fold} is outside the {len(folds)} available folds")

    cases = []
    for file_name in folds[args.fold]["train"]:
        metadata_path = Path(file_name).with_suffix(".pkl")
        metadata = read_pickle(metadata_path)
        locations = locations_from_metadata(metadata)
        if not locations:
            continue
        shape = [int(value) for value in metadata.get("new_size", [])]
        if len(shape) != 3:
            raise ValueError(f"Missing 3-D new_size in {metadata_path}")
        cases.append(
            {
                "file": file_name,
                "shape": shape,
                "locations": locations,
                "centroid": centroid(locations, shape),
            }
        )
    if not cases:
        raise ValueError(f"No positive training labels found in fold {args.fold}")

    center = [statistics.median(case["centroid"][axis] for case in cases) for axis in range(3)]
    center_text = "[" + ",".join(f"{value:.8f}" for value in center) + "]"

    total_locations = 0
    covered_locations = 0
    per_case_coverage = []
    for case in cases:
        starts = roi_starts(case["shape"], args.roi_size, center)
        ends = [
            min(start + roi_dim, image_dim)
            for start, roi_dim, image_dim in zip(starts, args.roi_size, case["shape"])
        ]
        covered = sum(
            all(starts[axis] <= location[axis] < ends[axis] for axis in range(3))
            for location in case["locations"]
        )
        count = len(case["locations"])
        total_locations += count
        covered_locations += covered
        per_case_coverage.append(covered / count)

    global_coverage = covered_locations / total_locations
    report = {
        "source_split": str(args.split),
        "fold": args.fold,
        "training_cases_with_foreground": len(cases),
        "roi_size_before_resize": args.roi_size,
        "normalized_center": center,
        "stored_foreground_locations": total_locations,
        "stored_foreground_locations_in_roi": covered_locations,
        "global_training_foreground_coverage": global_coverage,
        "minimum_case_coverage": min(per_case_coverage),
        "median_case_coverage": statistics.median(per_case_coverage),
        "leakage_guard": "Center and coverage were calculated from the selected fold's training files only.",
    }
    atomic_json(args.output, report)
    args.center_output.parent.mkdir(parents=True, exist_ok=True)
    args.center_output.write_text(center_text + "\n")

    print(
        "ANATOMICAL_ROI_ESTIMATED "
        f"fold={args.fold} center={center_text} roi={args.roi_size} "
        f"training_coverage={global_coverage:.6f} min_case_coverage={min(per_case_coverage):.6f}"
    )
    if global_coverage < args.min_training_coverage:
        raise RuntimeError(
            f"Training foreground coverage {global_coverage:.4f} is below "
            f"the required {args.min_training_coverage:.4f}; increase ROI_SIZE."
        )
    print("ANATOMICAL_ROI_COVERAGE_OK")


if __name__ == "__main__":
    main()
