"""Select one positive training case and create a diagnostic one-case split."""

import argparse
import json
import os
import pickle
from pathlib import Path


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def foreground_locations_by_class(metadata: dict) -> dict[str, list[list[int]]]:
    value = metadata.get("foreground_locations", {})
    if isinstance(value, dict):
        return {
            str(class_id): [list(location) for location in locations]
            for class_id, locations in value.items()
            if len(locations) > 0
        }
    locations = [list(location) for location in value]
    return {"1": locations} if locations else {}


def flatten_locations(locations_by_class: dict[str, list[list[int]]]) -> list[list[int]]:
    return [
        location
        for locations in locations_by_class.values()
        for location in locations
    ]


def normalized_centroid(locations: list[list[int]], shape: list[int]) -> list[float]:
    if not locations:
        raise ValueError("Cannot calculate a foreground centroid without foreground locations")
    return [
        sum(float(location[axis]) for location in locations)
        / len(locations)
        / max(int(shape[axis]) - 1, 1)
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


def locations_in_roi(
    locations: list[list[int]],
    starts: list[int],
    ends: list[int],
) -> int:
    return sum(
        all(starts[axis] <= location[axis] < ends[axis] for axis in range(3))
        for location in locations
    )


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
    parser.add_argument("--output-split", required=True, type=Path)
    parser.add_argument("--output-metadata", required=True, type=Path)
    parser.add_argument("--center-output", required=True, type=Path)
    parser.add_argument("--required-class", action="append", default=[], type=int)
    parser.add_argument("--roi-size", required=True, nargs=3, type=int)
    parser.add_argument("--min-roi-coverage", default=0.99, type=float)
    args = parser.parse_args()
    if any(value <= 0 for value in args.roi_size):
        raise ValueError("ROI dimensions must be positive")
    if not 0.0 <= args.min_roi_coverage <= 1.0:
        raise ValueError("min-roi-coverage must be in [0, 1]")
    required_classes = [str(class_id) for class_id in args.required_class]

    folds = read_json(args.split)
    if args.fold < 0 or args.fold >= len(folds):
        raise ValueError(f"Fold {args.fold} is outside the {len(folds)} available folds")

    candidates = []
    for file_name in folds[args.fold]["train"]:
        tensor_path = Path(file_name)
        metadata_path = tensor_path.with_suffix(".pkl")
        metadata = read_pickle(metadata_path)
        locations_by_class = foreground_locations_by_class(metadata)
        if any(not locations_by_class.get(class_id) for class_id in required_classes):
            continue
        locations = flatten_locations(locations_by_class)
        if not locations:
            continue
        shape = [int(value) for value in metadata.get("new_size", [])]
        if len(shape) != 3:
            raise ValueError(f"Missing 3-D new_size in {metadata_path}")
        candidates.append(
            (
                len(locations),
                str(tensor_path),
                shape,
                locations,
                locations_by_class,
            )
        )

    if not candidates:
        required_text = ",".join(required_classes) if required_classes else "any foreground"
        raise ValueError(
            f"No training case containing {required_text} found in fold {args.fold}"
        )
    count, selected_file, shape, locations, locations_by_class = max(
        candidates,
        key=lambda value: (value[0], value[1]),
    )
    center = normalized_centroid(locations, shape)
    center_text = "[" + ",".join(f"{value:.8f}" for value in center) + "]"

    starts = roi_starts(shape, args.roi_size, center)
    ends = [
        min(start + roi_dim, image_dim)
        for start, roi_dim, image_dim in zip(starts, args.roi_size, shape)
    ]
    coverage_by_class = {
        class_id: locations_in_roi(class_locations, starts, ends) / len(class_locations)
        for class_id, class_locations in locations_by_class.items()
    }
    required_coverage = {
        class_id: coverage_by_class[class_id]
        for class_id in required_classes
    }
    if not required_coverage:
        required_coverage = coverage_by_class
    minimum_coverage = min(required_coverage.values())

    atomic_json(args.output_split, [{"train": [selected_file], "val": [selected_file]}])
    atomic_json(
        args.output_metadata,
        {
            "source_split": str(args.split),
            "source_fold": args.fold,
            "selected_file": selected_file,
            "stored_foreground_locations": count,
            "stored_foreground_locations_by_class": {
                class_id: len(class_locations)
                for class_id, class_locations in locations_by_class.items()
            },
            "processed_shape": shape,
            "normalized_foreground_centroid": center,
            "roi_size_before_resize": args.roi_size,
            "roi_starts": starts,
            "roi_ends": ends,
            "stored_foreground_coverage_by_class": coverage_by_class,
            "minimum_required_class_coverage": minimum_coverage,
            "warning": "Diagnostic only: train and validation intentionally contain the same case.",
        },
    )
    args.center_output.parent.mkdir(parents=True, exist_ok=True)
    args.center_output.write_text(center_text + "\n")

    print(
        "ONE_CASE_OVERFIT_SPLIT_OK "
        f"source_fold={args.fold} selected={selected_file} locations={count} "
        f"center={center_text} roi={args.roi_size} coverage={coverage_by_class}"
    )
    if minimum_coverage < args.min_roi_coverage:
        raise RuntimeError(
            f"Minimum required-class ROI coverage {minimum_coverage:.4f} is below "
            f"the required {args.min_roi_coverage:.4f}; increase OVERFIT_ROI_SIZE."
        )
    print("ONE_CASE_OVERFIT_ROI_COVERAGE_OK")


if __name__ == "__main__":
    main()
