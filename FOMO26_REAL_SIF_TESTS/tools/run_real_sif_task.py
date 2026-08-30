#!/usr/bin/env python3
"""Run one FOMO26 submission SIF on real labelled finetuning cases.

This script is intended to run inside a Slurm allocation.  It discovers cases
from TEST_80_10_10.json, repairs stale source paths using the current raw-data
root, runs both default TTA and --no-tta through direct ``apptainer exec --nv``,
and records task-specific metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage


TASKS: dict[int, dict[str, Any]] = {
    1: {
        "dataset": "CLS002_FOMO26_Infarct",
        "kind": "classification",
        "anchor": "flair.nii.gz",
        "label_name": "label.txt",
        "raw_marker": "Task_1",
        "output_ext": ".txt",
    },
    2: {
        "dataset": "SEG009_FOMO26_Meningioma",
        "kind": "segmentation",
        "anchor": "flair.nii.gz",
        "label_name": "seg.nii.gz",
        "raw_marker": "Task_2",
        "output_ext": ".nii.gz",
        "allowed_labels": [0, 1],
    },
    3: {
        "dataset": "REGR002_FOMO26_BrainAge",
        "kind": "regression",
        "anchor": "t1w.nii.gz",
        "label_name": "labels.txt",
        "raw_marker": "Task_3",
        "output_ext": ".txt",
    },
    4: {
        "dataset": "SEG010_FOMO26_TrigeminalNeuralgia",
        "kind": "segmentation",
        "anchor": "t2w.nii.gz",
        "label_name": "seg.nii.gz",
        "raw_marker": "Task_4",
        "output_ext": ".nii.gz",
        "allowed_labels": [0, 1, 2],
    },
    5: {
        "dataset": "CLS003_FOMO26_Polymicrogyria",
        "kind": "classification",
        "anchor": "t1.nii.gz",
        "label_name": "labels.txt",
        "raw_marker": "Task_5",
        "output_ext": ".txt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=TASKS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--task1-sif", type=Path, required=True)
    parser.add_argument("--remaining-build-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260821)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--apptainer", default="apptainer")
    parser.add_argument(
        "--allow-all-data-fallback",
        action="store_true",
        help="Allow raw-data fallback if no held-out split case can be resolved.",
    )
    return parser.parse_args()


def flatten_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(flatten_strings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(flatten_strings(item))
    return found


def first_existing(candidates: list[Path]) -> Path | None:
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_processed(path_text: str, task_dir: Path, dataset: str) -> Path | None:
    original = Path(path_text)
    candidates = [original]
    normalized = str(original).replace("/project/", "/mnt/project/")
    candidates.append(Path(normalized))
    parts = list(original.parts)
    if dataset in parts:
        idx = parts.index(dataset)
        candidates.append(task_dir.joinpath(*parts[idx + 1 :]))
    candidates.append(task_dir / original.name)
    resolved = first_existing(candidates)
    if resolved is not None:
        return resolved
    matches = list(task_dir.rglob(original.name))
    if len(matches) == 1:
        return matches[0].resolve()
    return None


def resolve_source(path_text: str, source_root: Path, raw_marker: str) -> Path | None:
    original = Path(path_text)
    text = str(original)
    candidates = [original, Path(text.replace("/project/", "/mnt/project/"))]

    marker = "FOMO26_finetune/"
    if marker in text:
        candidates.append(source_root / text.split(marker, 1)[1])

    parts = list(original.parts)
    if raw_marker in parts:
        idx = parts.index(raw_marker)
        candidates.append(source_root.joinpath(*parts[idx:]))

    return first_existing(candidates)


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected dictionary metadata in {path}, got {type(value)}")
    return value


def inferred_label_path(anchor: Path, label_name: str) -> Path:
    text = str(anchor)
    if "/preprocessed/" in text:
        text = text.replace("/preprocessed/", "/labels/")
    return Path(text).with_name(label_name)


def sibling(case_dir: Path, names: list[str]) -> Path | None:
    return first_existing([case_dir / name for name in names])


def modalities_for_task(task: int, metadata_paths: list[Path]) -> dict[str, Path] | None:
    by_name = {path.name.lower(): path for path in metadata_paths}
    if not metadata_paths:
        return None
    case_dir = metadata_paths[0].parent

    if task == 1:
        flair = by_name.get("flair.nii.gz") or sibling(case_dir, ["flair.nii.gz"])
        adc = by_name.get("adc.nii.gz") or sibling(case_dir, ["adc.nii.gz"])
        dwi = by_name.get("dwi_b1000.nii.gz") or sibling(
            case_dir, ["dwi_b1000.nii.gz", "dwi.nii.gz"]
        )
        optional = sibling(case_dir, ["swi.nii.gz", "t2s.nii.gz"])
        if not all([flair, adc, dwi, optional]):
            return None
        optional_key = "swi" if optional.name.lower() == "swi.nii.gz" else "t2s"
        return {"flair": flair, "adc": adc, "dwi": dwi, optional_key: optional}

    if task == 2:
        flair = by_name.get("flair.nii.gz") or sibling(case_dir, ["flair.nii.gz"])
        dwi = by_name.get("dwi_b1000.nii.gz") or sibling(
            case_dir, ["dwi_b1000.nii.gz", "dwi.nii.gz"]
        )
        optional = sibling(case_dir, ["swi.nii.gz", "t2s.nii.gz"])
        if not all([flair, dwi, optional]):
            return None
        optional_key = "swi" if optional.name.lower() == "swi.nii.gz" else "t2s"
        return {"flair": flair, "dwi": dwi, optional_key: optional}

    if task == 3:
        image = by_name.get("t1w.nii.gz") or sibling(case_dir, ["t1w.nii.gz"])
        return {"t1": image} if image else None

    if task == 4:
        image = by_name.get("t2w.nii.gz") or sibling(case_dir, ["t2w.nii.gz"])
        return {"t2": image} if image else None

    image = by_name.get("t1.nii.gz") or sibling(case_dir, ["t1.nii.gz"])
    return {"input": image} if image else None


def read_target(kind: str, label_path: Path) -> tuple[float | int | None, dict[str, Any]]:
    if kind == "segmentation":
        image = nib.load(str(label_path))
        data = np.asarray(image.get_fdata())
        rounded = np.rint(data).astype(np.int16)
        labels, counts = np.unique(rounded, return_counts=True)
        class_voxels = {str(int(k)): int(v) for k, v in zip(labels, counts)}
        foreground_voxels = int(np.count_nonzero(rounded > 0))
        return None, {
            "foreground_voxels": foreground_voxels,
            "class_voxels": class_voxels,
            "shape": list(rounded.shape),
        }
    text = label_path.read_text(encoding="utf-8").strip().split()[0]
    value = float(text)
    if kind == "classification":
        return int(round(value)), {}
    return float(value), {}


def record_from_processed(
    task: int,
    processed: Path,
    source_root: Path,
    sample_origin: str,
) -> dict[str, Any] | None:
    spec = TASKS[task]
    metadata_path = processed.with_suffix(".pkl")
    if not metadata_path.is_file():
        return None
    try:
        metadata = load_pickle(metadata_path)
    except Exception:
        return None

    src_values = metadata.get("src_image_paths", [])
    if isinstance(src_values, str):
        src_values = [src_values]
    resolved_images: list[Path] = []
    for value in src_values:
        resolved = resolve_source(str(value), source_root, spec["raw_marker"])
        if resolved is None:
            return None
        resolved_images.append(resolved)
    modalities = modalities_for_task(task, resolved_images)
    if modalities is None:
        return None

    label_value = metadata.get("src_label_path")
    label_path = (
        resolve_source(str(label_value), source_root, spec["raw_marker"])
        if label_value
        else None
    )
    anchor = next(iter(modalities.values()))
    if label_path is None:
        candidate = inferred_label_path(anchor, spec["label_name"])
        label_path = candidate if candidate.is_file() else None
    if label_path is None:
        return None

    try:
        target, target_meta = read_target(spec["kind"], label_path)
    except Exception:
        return None

    relative = str(processed)
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:8]
    case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", processed.stem)[-48:] + "_" + digest
    return {
        "case_id": case_id,
        "processed_path": str(processed),
        "sample_origin": sample_origin,
        "modalities": {key: str(value) for key, value in modalities.items()},
        "label_path": str(label_path),
        "target": target,
        "target_meta": target_meta,
    }


def raw_fallback_records(task: int, source_root: Path) -> list[dict[str, Any]]:
    spec = TASKS[task]
    search_root = source_root / spec["raw_marker"]
    if not search_root.exists():
        search_root = source_root
    records: list[dict[str, Any]] = []
    for anchor in search_root.rglob(spec["anchor"]):
        if "labels" in anchor.parts:
            continue
        modalities = modalities_for_task(task, [anchor.resolve()])
        if modalities is None:
            continue
        label_path = inferred_label_path(anchor, spec["label_name"])
        if not label_path.is_file():
            continue
        try:
            target, target_meta = read_target(spec["kind"], label_path)
        except Exception:
            continue
        rel = str(anchor.relative_to(source_root))
        digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
        records.append(
            {
                "case_id": re.sub(r"[^A-Za-z0-9_.-]+", "_", anchor.parent.name)[-48:]
                + "_"
                + digest,
                "processed_path": None,
                "sample_origin": "RAW_ALL_FALLBACK_NOT_HELD_OUT",
                "modalities": {key: str(value) for key, value in modalities.items()},
                "label_path": str(label_path),
                "target": target,
                "target_meta": target_meta,
            }
        )
    return records


def discover_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = TASKS[args.task]
    task_dir = args.data_root / spec["dataset"]
    split_path = task_dir / "TEST_80_10_10.json"
    diagnostics: dict[str, Any] = {
        "split_path": str(split_path),
        "split_exists": split_path.is_file(),
        "split_entries": 0,
        "resolved_processed": 0,
        "resolved_real_cases": 0,
        "fallback_used": False,
    }
    records: list[dict[str, Any]] = []
    if split_path.is_file():
        with split_path.open("r", encoding="utf-8") as handle:
            entries = flatten_strings(json.load(handle))
        diagnostics["split_entries"] = len(entries)
        seen: set[str] = set()
        for entry in entries:
            processed = resolve_processed(entry, task_dir, spec["dataset"])
            if processed is None or str(processed) in seen:
                continue
            seen.add(str(processed))
            diagnostics["resolved_processed"] += 1
            record = record_from_processed(
                args.task, processed, args.source_root, "HELD_OUT_TEST_SPLIT"
            )
            if record is not None:
                records.append(record)

    diagnostics["resolved_real_cases"] = len(records)
    if not records and args.allow_all_data_fallback:
        records = raw_fallback_records(args.task, args.source_root)
        diagnostics["fallback_used"] = True
        diagnostics["resolved_real_cases"] = len(records)
    return records, diagnostics


def select_cases(
    records: list[dict[str, Any]], kind: str, count: int, seed: int
) -> list[dict[str, Any]]:
    if count <= 0 or not records:
        return []
    rng = random.Random(seed)
    if kind == "classification":
        groups: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            groups.setdefault(int(record["target"]), []).append(record)
        for values in groups.values():
            rng.shuffle(values)
        selected: list[dict[str, Any]] = []
        labels = sorted(groups)
        while len(selected) < count and any(groups.values()):
            for label in labels:
                if groups[label] and len(selected) < count:
                    selected.append(groups[label].pop())
        return selected

    if kind == "regression":
        ordered = sorted(records, key=lambda row: float(row["target"]))
    else:
        positives = [
            row for row in records if row["target_meta"].get("foreground_voxels", 0) > 0
        ]
        ordered = sorted(
            positives or records,
            key=lambda row: row["target_meta"].get("foreground_voxels", 0),
        )
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    return [ordered[index] for index in indices]


def sif_for_task(args: argparse.Namespace) -> Path:
    if args.task == 1:
        return args.task1_sif
    names = {
        2: "fomo26_task2_meningioma.sif",
        3: "fomo26_task3_brain_age.sif",
        4: "fomo26_task4_trigeminal.sif",
        5: "fomo26_task5_polymicrogyria.sif",
    }
    return args.remaining_build_root / names[args.task]


def cli_arguments(task: int, paths: dict[str, str], output: str) -> list[str]:
    args: list[str] = []
    ordered = {
        1: ["flair", "adc", "dwi", "swi", "t2s"],
        2: ["flair", "dwi", "swi", "t2s"],
        3: ["t1"],
        4: ["t2"],
        5: ["input"],
    }[task]
    for key in ordered:
        if key in paths:
            args.extend([f"--{key}", paths[key]])
    args.extend(["--output", output])
    return args


def run_prediction(
    args: argparse.Namespace,
    sif: Path,
    record: dict[str, Any],
    mode: str,
    predictions_dir: Path,
) -> dict[str, Any]:
    spec = TASKS[args.task]
    result: dict[str, Any] = {"mode": mode, "success": False}
    with tempfile.TemporaryDirectory(prefix=f"fomo26_task{args.task}_") as tmp:
        output_dir = Path(tmp)
        output_name = "prediction" + spec["output_ext"]
        host_output = output_dir / output_name
        command = [args.apptainer, "exec", "--nv"]
        container_paths: dict[str, str] = {}
        for index, (key, value) in enumerate(record["modalities"].items()):
            host_path = Path(value)
            mount = f"/case_{index}"
            command.extend(["--bind", f"{host_path.parent}:{mount}:ro"])
            container_paths[key] = f"{mount}/{host_path.name}"
        command.extend(["--bind", f"{output_dir}:/output:rw"])
        command.extend([str(sif), "python", "/app/predict.py"])
        command.extend(cli_arguments(args.task, container_paths, f"/output/{output_name}"))
        if mode == "no_tta":
            command.append("--no-tta")

        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            result.update(
                {
                    "returncode": completed.returncode,
                    "runtime_seconds": time.monotonic() - started,
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-4000:],
                    "command": command,
                }
            )
        except subprocess.TimeoutExpired as exc:
            result.update(
                {
                    "returncode": None,
                    "runtime_seconds": time.monotonic() - started,
                    "error": f"Timed out after {args.timeout}s",
                    "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                    "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                    "command": command,
                }
            )
            return result

        if result["returncode"] != 0:
            result["error"] = f"predict.py exited with rc={result['returncode']}"
            return result
        if not host_output.is_file() or host_output.stat().st_size == 0:
            result["error"] = "Prediction output is missing or empty"
            return result

        predictions_dir.mkdir(parents=True, exist_ok=True)
        saved_output = predictions_dir / (
            f"{record['case_id']}__{mode}" + spec["output_ext"]
        )
        shutil.copy2(host_output, saved_output)
        result["output_path"] = str(saved_output)
        result["success"] = True
        return result


def parse_text_prediction(path: Path) -> float:
    value = float(path.read_text(encoding="utf-8").strip().split()[0])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite prediction in {path}: {value}")
    return value


def dice_score(prediction: np.ndarray, target: np.ndarray, label: int) -> float:
    pred = prediction == label
    true = target == label
    denominator = int(pred.sum()) + int(true.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, true).sum() / denominator)


def component_stats(mask: np.ndarray) -> dict[str, Any]:
    labelled, count = ndimage.label(mask)
    if count == 0:
        return {"count": 0, "largest_voxels": 0, "largest_fraction": 0.0}
    sizes = np.bincount(labelled.ravel())[1:]
    largest = int(sizes.max()) if sizes.size else 0
    total = int(mask.sum())
    return {
        "count": int(count),
        "largest_voxels": largest,
        "largest_fraction": float(largest / total) if total else 0.0,
    }


def analyze_segmentation(
    output_path: Path,
    label_path: Path,
    reference_path: Path,
    allowed_labels: list[int],
) -> dict[str, Any]:
    pred_img = nib.load(str(output_path))
    target_img = nib.load(str(label_path))
    reference_img = nib.load(str(reference_path))
    pred_raw = np.asarray(pred_img.get_fdata())
    target_raw = np.asarray(target_img.get_fdata())
    pred = np.rint(pred_raw).astype(np.int16)
    target = np.rint(target_raw).astype(np.int16)
    pred_labels = sorted(int(v) for v in np.unique(pred))
    integer_output = bool(np.allclose(pred_raw, pred, atol=1e-4))
    shape_matches_reference = tuple(pred.shape) == tuple(reference_img.shape)
    affine_matches_reference = bool(
        np.allclose(pred_img.affine, reference_img.affine, rtol=1e-4, atol=1e-3)
    )
    shape_matches_label = pred.shape == target.shape
    affine_matches_label = bool(
        np.allclose(pred_img.affine, target_img.affine, rtol=1e-4, atol=1e-3)
    )
    allowed = set(allowed_labels)
    valid_labels = set(pred_labels).issubset(allowed)
    metrics: dict[str, Any] = {
        "shape": list(pred.shape),
        "predicted_labels": pred_labels,
        "integer_output": integer_output,
        "valid_labels": valid_labels,
        "shape_matches_reference": shape_matches_reference,
        "affine_matches_reference": affine_matches_reference,
        "shape_matches_label": shape_matches_label,
        "affine_matches_label": affine_matches_label,
        "pred_foreground_voxels": int(np.count_nonzero(pred > 0)),
        "pred_foreground_fraction": float(np.mean(pred > 0)),
        "target_foreground_voxels": int(np.count_nonzero(target > 0)),
        "target_foreground_fraction": float(np.mean(target > 0)),
    }
    if shape_matches_label:
        foreground_labels = [label for label in allowed_labels if label != 0]
        per_class = {
            str(label): dice_score(pred, target, label) for label in foreground_labels
        }
        metrics["dice_per_foreground_class"] = per_class
        metrics["foreground_macro_dice"] = float(np.mean(list(per_class.values())))
    else:
        metrics["dice_per_foreground_class"] = {}
        metrics["foreground_macro_dice"] = None
    metrics["components"] = {
        str(label): component_stats(pred == label)
        for label in allowed_labels
        if label != 0
    }
    warnings: list[str] = []
    if metrics["target_foreground_voxels"] > 0 and metrics["pred_foreground_voxels"] == 0:
        warnings.append("EMPTY_PREDICTION_ON_POSITIVE_CASE")
    if metrics["pred_foreground_fraction"] > 0.25:
        warnings.append("PREDICTION_COVERS_MORE_THAN_25_PERCENT_OF_IMAGE")
    if not shape_matches_reference or not affine_matches_reference:
        warnings.append("OUTPUT_GEOMETRY_MISMATCH")
    if not integer_output or not valid_labels:
        warnings.append("INVALID_SEGMENTATION_LABEL_VALUES")
    metrics["warnings"] = warnings
    return metrics


def analyze_case_outputs(
    task: int, record: dict[str, Any], runs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    spec = TASKS[task]
    successful = all(run.get("success") for run in runs.values())
    analysis: dict[str, Any] = {"both_modes_succeeded": successful}
    if not successful:
        return analysis

    if spec["kind"] in {"classification", "regression"}:
        values = {
            mode: parse_text_prediction(Path(run["output_path"]))
            for mode, run in runs.items()
        }
        analysis["predictions"] = values
        analysis["tta_minus_no_tta"] = values["tta"] - values["no_tta"]
        if spec["kind"] == "classification":
            target = int(record["target"])
            analysis["predicted_classes"] = {
                mode: int(value >= 0.5) for mode, value in values.items()
            }
            analysis["correct"] = {
                mode: int(value >= 0.5) == target for mode, value in values.items()
            }
            analysis["probabilities_valid"] = {
                mode: 0.0 <= value <= 1.0 for mode, value in values.items()
            }
        else:
            target = float(record["target"])
            analysis["absolute_errors"] = {
                mode: abs(value - target) for mode, value in values.items()
            }
        return analysis

    reference_key = "flair" if task == 2 else "t2"
    segmentations = {
        mode: analyze_segmentation(
            Path(run["output_path"]),
            Path(record["label_path"]),
            Path(record["modalities"][reference_key]),
            spec["allowed_labels"],
        )
        for mode, run in runs.items()
    }
    analysis["segmentations"] = segmentations
    pred_no = np.rint(
        nib.load(runs["no_tta"]["output_path"]).get_fdata()
    ).astype(np.int16)
    pred_tta = np.rint(nib.load(runs["tta"]["output_path"]).get_fdata()).astype(
        np.int16
    )
    if pred_no.shape == pred_tta.shape:
        foreground_labels = [label for label in spec["allowed_labels"] if label != 0]
        agreement = [dice_score(pred_tta, pred_no, label) for label in foreground_labels]
        analysis["tta_no_tta_foreground_agreement"] = float(np.mean(agreement))
    else:
        analysis["tta_no_tta_foreground_agreement"] = None
    return analysis


def main() -> int:
    args = parse_args()
    spec = TASKS[args.task]
    if args.cases < 1:
        raise ValueError("--cases must be at least 1")
    for path in [args.data_root, args.source_root]:
        if not path.exists():
            raise FileNotFoundError(path)
    sif = sif_for_task(args)
    if not sif.is_file():
        raise FileNotFoundError(sif)

    records, diagnostics = discover_records(args)
    selected = select_cases(records, spec["kind"], args.cases, args.seed + args.task)
    if not selected:
        raise RuntimeError(
            "No real labelled cases could be resolved. "
            "Inspect discovery diagnostics and set --allow-all-data-fallback only if needed."
        )

    task_root = args.results_root / f"task{args.task}"
    predictions_dir = task_root / "predictions"
    task_root.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {
        "task": args.task,
        "dataset": spec["dataset"],
        "kind": spec["kind"],
        "sif": str(sif),
        "cases_requested": args.cases,
        "cases_selected": len(selected),
        "discovery": diagnostics,
        "cases": [],
    }

    print(
        f"REAL_SIF_TEST_START task={args.task} dataset={spec['dataset']} "
        f"selected={len(selected)} fallback={diagnostics['fallback_used']}"
    )
    for index, record in enumerate(selected, start=1):
        print(
            f"CASE_START task={args.task} case={index}/{len(selected)} "
            f"id={record['case_id']} origin={record['sample_origin']}"
        )
        runs = {
            mode: run_prediction(args, sif, record, mode, predictions_dir)
            for mode in ["no_tta", "tta"]
        }
        try:
            analysis = analyze_case_outputs(args.task, record, runs)
        except Exception as exc:
            analysis = {"both_modes_succeeded": False, "analysis_error": repr(exc)}
        case_output = dict(record)
        case_output["runs"] = runs
        case_output["analysis"] = analysis
        output["cases"].append(case_output)
        print(
            f"CASE_FINISH task={args.task} id={record['case_id']} "
            f"no_tta={runs['no_tta'].get('success')} tta={runs['tta'].get('success')}"
        )

    output_path = task_root / f"task{args.task}_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    successes = sum(
        1 for case in output["cases"] if case["analysis"].get("both_modes_succeeded")
    )
    print(
        f"REAL_SIF_TASK_FINISHED task={args.task} successful_cases={successes}/"
        f"{len(selected)} result={output_path}"
    )
    return 0 if successes == len(selected) else 2


if __name__ == "__main__":
    raise SystemExit(main())
