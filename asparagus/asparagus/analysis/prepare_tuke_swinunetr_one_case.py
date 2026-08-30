"""Create one deterministic positive ROI for a SwinUNETR memorization test.

This is deliberately label-derived and must never be used as a validation or
test preprocessing strategy.  Its only purpose is to prove that the complete
checkpoint-to-segmentation training path can memorize one labelled example.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def foreground_locations_by_class(metadata: dict) -> dict[str, list[list[int]]]:
    value = metadata.get("foreground_locations", {})
    if isinstance(value, dict):
        return {
            str(class_id): [list(map(int, location)) for location in locations]
            for class_id, locations in value.items()
            if len(locations) > 0
        }
    locations = [list(map(int, location)) for location in value]
    return {"1": locations} if locations else {}


def clamped_start(center: torch.Tensor, shape: tuple[int, int, int], patch: tuple[int, int, int]):
    starts = []
    for axis in range(3):
        maximum = max(shape[axis] - patch[axis], 0)
        proposed = int(round(float(center[axis]) - (patch[axis] - 1) / 2.0))
        starts.append(min(max(proposed, 0), maximum))
    return tuple(starts)


def candidate_starts(
    coordinates: dict[int, torch.Tensor],
    shape: tuple[int, int, int],
    patch: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    centers = []
    class_centers = []
    for class_coordinates in coordinates.values():
        class_coordinates = class_coordinates.float()
        class_centers.append(class_coordinates.mean(dim=0))
        centers.append(class_coordinates.mean(dim=0))
        centers.append((class_coordinates.amin(dim=0) + class_coordinates.amax(dim=0)) / 2.0)
        stride = max(class_coordinates.shape[0] // 64, 1)
        centers.extend(class_coordinates[::stride][:64])
    centers.append(torch.stack(class_centers).mean(dim=0))
    all_coordinates = torch.cat(list(coordinates.values()), dim=0).float()
    centers.append(all_coordinates.mean(dim=0))
    centers.append((all_coordinates.amin(dim=0) + all_coordinates.amax(dim=0)) / 2.0)
    return sorted({clamped_start(center, shape, patch) for center in centers})


def score_crop(
    coordinates: dict[int, torch.Tensor],
    starts: tuple[int, int, int],
    patch: tuple[int, int, int],
):
    counts = {}
    coverage = {}
    for class_id, class_coordinates in coordinates.items():
        inside = torch.ones(class_coordinates.shape[0], dtype=torch.bool)
        for axis in range(3):
            inside &= class_coordinates[:, axis] >= starts[axis]
            inside &= class_coordinates[:, axis] < starts[axis] + patch[axis]
        counts[class_id] = int(inside.sum())
        coverage[class_id] = counts[class_id] / max(int(class_coordinates.shape[0]), 1)
    objective = (
        int(all(value > 0 for value in counts.values())),
        min(counts.values()),
        min(coverage.values()),
        sum(counts.values()),
    )
    return objective, counts, coverage


def pad_to_patch(data: torch.Tensor, patch: tuple[int, int, int]):
    spatial_shape = tuple(int(value) for value in data.shape[-3:])
    before = [max((patch[axis] - spatial_shape[axis]) // 2, 0) for axis in range(3)]
    after = [max(patch[axis] - spatial_shape[axis] - before[axis], 0) for axis in range(3)]
    if any(before) or any(after):
        data = F.pad(
            data,
            (before[2], after[2], before[1], after[1], before[0], after[0]),
        )
    return data, before


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-split", required=True, type=Path)
    parser.add_argument("--required-class", action="append", required=True, type=int)
    parser.add_argument("--patch-size", nargs=3, default=(96, 96, 96), type=int)
    parser.add_argument("--maximum-candidates", default=24, type=int)
    parser.add_argument("--minimum-class-voxels", default=8, type=int)
    args = parser.parse_args()

    patch = tuple(int(value) for value in args.patch_size)
    required_classes = tuple(sorted(set(args.required_class)))
    if any(value <= 0 for value in patch):
        raise ValueError(f"Patch dimensions must be positive, got {patch}")
    if any(class_id <= 0 for class_id in required_classes):
        raise ValueError("Required classes must be positive foreground IDs")

    folds = read_json(args.split)
    if args.fold < 0 or args.fold >= len(folds):
        raise ValueError(f"Fold {args.fold} is outside the {len(folds)} available folds")

    metadata_candidates = []
    for file_name in folds[args.fold]["train"]:
        tensor_path = Path(file_name)
        metadata_path = tensor_path.with_suffix(".pkl")
        if not tensor_path.is_file() or not metadata_path.is_file():
            continue
        locations = foreground_locations_by_class(read_pickle(metadata_path))
        required_counts = [len(locations.get(str(class_id), [])) for class_id in required_classes]
        if any(count == 0 for count in required_counts):
            continue
        metadata_candidates.append((min(required_counts), sum(required_counts), str(tensor_path)))

    if not metadata_candidates:
        raise RuntimeError(
            f"No fold-{args.fold} training case contains every required class {required_classes}"
        )
    metadata_candidates.sort(reverse=True)

    best = None
    load_errors = []
    for _, _, file_name in metadata_candidates[: args.maximum_candidates]:
        try:
            data = torch.load(file_name, map_location="cpu", weights_only=False)
        except Exception as error:
            load_errors.append(f"{file_name}: {error}")
            continue
        if not isinstance(data, torch.Tensor) or data.ndim != 4 or data.shape[0] < 2:
            load_errors.append(f"{file_name}: expected [C+1,X,Y,Z] tensor, got {type(data)} {getattr(data, 'shape', None)}")
            continue

        data, pad_before = pad_to_patch(data, patch)
        label = data[-1].long()
        coordinates = {
            class_id: torch.nonzero(label == class_id, as_tuple=False)
            for class_id in required_classes
        }
        if any(value.numel() == 0 for value in coordinates.values()):
            continue
        shape = tuple(int(value) for value in label.shape)
        for starts in candidate_starts(coordinates, shape, patch):
            objective, counts, coverage = score_crop(coordinates, starts, patch)
            candidate = (objective, file_name, data, pad_before, starts, counts, coverage)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        detail = "\n".join(load_errors[:10])
        raise RuntimeError(f"Could not prepare a valid positive ROI.\n{detail}")

    objective, selected_file, data, pad_before, starts, counts, coverage = best
    if objective[0] != 1 or min(counts.values()) < args.minimum_class_voxels:
        raise RuntimeError(
            f"Best ROI has required-class counts {counts}; need at least "
            f"{args.minimum_class_voxels} voxels for every class. Increase TUKE_OVERFIT_PATCH_SIZE."
        )

    slices = tuple(slice(starts[axis], starts[axis] + patch[axis]) for axis in range(3))
    roi = data[(slice(None), *slices)].contiguous()
    if tuple(roi.shape[-3:]) != patch:
        raise AssertionError(f"Prepared ROI shape {tuple(roi.shape)} does not match {patch}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "one_case_roi.pt"
    temporary_sample = sample_path.with_suffix(".pt.tmp")
    torch.save(roi, temporary_sample)
    os.replace(temporary_sample, sample_path)

    roi_label = roi[-1].long()
    locations_by_class = {}
    all_class_counts = {}
    for class_id in sorted(int(value) for value in torch.unique(roi_label).tolist() if int(value) > 0):
        class_coordinates = torch.nonzero(roi_label == class_id, as_tuple=False)
        all_class_counts[str(class_id)] = int(class_coordinates.shape[0])
        locations_by_class[str(class_id)] = class_coordinates.tolist()

    source_metadata = read_pickle(Path(selected_file).with_suffix(".pkl"))
    source_metadata.update(
        {
            "new_size": list(patch),
            "foreground_locations": locations_by_class,
            "diagnostic_source_file": selected_file,
            "diagnostic_pad_before": pad_before,
            "diagnostic_crop_starts_after_padding": list(starts),
            "diagnostic_label_derived": True,
        }
    )
    metadata_path = sample_path.with_suffix(".pkl")
    temporary_metadata = metadata_path.with_suffix(".pkl.tmp")
    with temporary_metadata.open("wb") as handle:
        pickle.dump(source_metadata, handle)
    os.replace(temporary_metadata, metadata_path)

    atomic_json(args.output_split, [{"train": [str(sample_path)], "val": [str(sample_path)]}])
    summary_path = args.output_dir / "one_case_roi_summary.json"
    atomic_json(
        summary_path,
        {
            "warning": "Diagnostic only: label-derived ROI and identical train/validation case.",
            "source_split": str(args.split),
            "source_fold": args.fold,
            "source_file": selected_file,
            "sample_file": str(sample_path),
            "patch_size": list(patch),
            "pad_before": pad_before,
            "crop_starts_after_padding": list(starts),
            "required_classes": list(required_classes),
            "required_class_voxels_in_roi": {str(key): value for key, value in counts.items()},
            "required_class_source_coverage": {str(key): value for key, value in coverage.items()},
            "all_foreground_class_voxels_in_roi": all_class_counts,
        },
    )
    print(
        "TUKE_SWINUNETR_ONE_CASE_READY "
        f"source={selected_file} sample={sample_path} patch={patch} "
        f"counts={counts} coverage={coverage}"
    )
    print(f"TUKE_SWINUNETR_ONE_CASE_SUMMARY={summary_path}")


if __name__ == "__main__":
    main()
