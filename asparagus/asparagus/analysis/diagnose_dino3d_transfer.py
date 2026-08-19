"""Diagnose the completed FOMO26 DINO3D pretraining and downstream transfer.

The script is intentionally read-only.  It inspects the production SSL
checkpoint, the five fine-tuning runs, their prediction JSON files, and a
small deterministic feature sample.  Every paste-worthy line begins with
``DIAG|`` and is also written to a standalone report file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


RUN_ID = 71494
EXPECTED_GLOBAL_STEPS = 125_000
PRETRAIN_TEACHER_PREFIX = "model.teacher_backbone."
PRETRAIN_STUDENT_PREFIX = "model.student_backbone."

TASKS = {
    "CLS002_FOMO26_Infarct": {"kind": "classification", "channels": 3, "classes": 2},
    "SEG009_FOMO26_Meningioma": {"kind": "segmentation", "channels": 2, "classes": 2},
    "REGR002_FOMO26_BrainAge": {"kind": "regression", "channels": 1, "classes": 1},
    "SEG010_FOMO26_TrigeminalNeuralgia": {"kind": "segmentation", "channels": 1, "classes": 3},
    "CLS003_FOMO26_Polymicrogyria": {"kind": "classification", "channels": 1, "classes": 2},
}


def clean(value) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


class Reporter:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")

    def emit(self, section: str, **values) -> None:
        fields = ["DIAG", clean(section)]
        fields.extend(f"{clean(key)}={clean(value)}" for key, value in values.items())
        line = "|".join(fields)
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def canonical_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    canonical = {}
    for key, value in state_dict.items():
        key = key.replace("model._orig_mod.", "model.")
        canonical[key] = value
    return canonical


def checkpoint_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "state_dict" in checkpoint:
        return canonical_state_dict(checkpoint["state_dict"])
    if "network_weights" in checkpoint:
        return canonical_state_dict(checkpoint["network_weights"])
    raise ValueError("Checkpoint has neither state_dict nor network_weights")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def relative_l2_and_cosine(pairs: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float | int]:
    diff_sq = 0.0
    ref_sq = 0.0
    other_sq = 0.0
    dot = 0.0
    tensors = 0
    exact_tensors = 0
    elements = 0
    for reference, other in pairs:
        if reference.shape != other.shape:
            continue
        reference = reference.detach().float().reshape(-1)
        other = other.detach().float().reshape(-1)
        difference = other - reference
        diff_sq += float(torch.dot(difference, difference))
        ref_sq += float(torch.dot(reference, reference))
        other_sq += float(torch.dot(other, other))
        dot += float(torch.dot(reference, other))
        tensors += 1
        elements += reference.numel()
        if torch.equal(reference, other):
            exact_tensors += 1
    return {
        "tensors": tensors,
        "elements": elements,
        "exact_tensors": exact_tensors,
        "relative_l2": math.sqrt(diff_sq / max(ref_sq, 1e-30)),
        "cosine": dot / max(math.sqrt(ref_sq * other_sq), 1e-30),
    }


def tensor_health(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, int | float]:
    tensors = 0
    elements = 0
    nonfinite = 0
    zeros = 0
    squared_norm = 0.0
    for key, tensor in state_dict.items():
        if not key.startswith(prefix) or not torch.is_tensor(tensor):
            continue
        values = tensor.detach()
        tensors += 1
        elements += values.numel()
        if values.is_floating_point():
            finite = torch.isfinite(values)
            nonfinite += int((~finite).sum())
            safe = torch.where(finite, values, torch.zeros_like(values)).float()
        else:
            safe = values.float()
        zeros += int((safe == 0).sum())
        squared_norm += float(torch.sum(safe * safe))
    return {
        "tensors": tensors,
        "elements": elements,
        "nonfinite": nonfinite,
        "zero_fraction": zeros / max(elements, 1),
        "l2_norm": math.sqrt(squared_norm),
    }


def paired_prefixes(
    state_dict: dict[str, torch.Tensor], first_prefix: str, second_prefix: str
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pairs = []
    for key, first in state_dict.items():
        if not key.startswith(first_prefix):
            continue
        second_key = second_prefix + key[len(first_prefix) :]
        second = state_dict.get(second_key)
        if torch.is_tensor(second) and first.shape == second.shape:
            pairs.append((first, second))
    return pairs


def find_pretrain_checkpoint(args) -> Path:
    explicit = Path(args.pretrain_checkpoint)
    if explicit.is_file():
        return explicit
    candidates = sorted(
        Path(args.pretrain_model_root).rglob(f"run_id={RUN_ID}/checkpoints/last.ckpt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint at {explicit} and no run_id={RUN_ID}/checkpoints/last.ckpt under "
            f"{args.pretrain_model_root}"
        )
    return candidates[-1]


def model_for_task(spec: dict):
    from asparagus.modules.networks.dino_downstream import (
        DINO3DClassifierRegressor,
        DINO3DSegmenter,
    )

    if spec["kind"] == "segmentation":
        return DINO3DSegmenter(spec["channels"], spec["classes"])
    return DINO3DClassifierRegressor(spec["channels"], spec["classes"])


def transfer_backbone(
    source_state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    source_prefix: str = PRETRAIN_TEACHER_PREFIX,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int | list[str]]]:
    target_state = model.state_dict()
    mapped = {}
    missing = []
    shape_mismatch = []
    adapted = []
    backbone_elements = 0
    loaded_elements = 0

    for target_key, target_value in target_state.items():
        if not target_key.startswith("teacher_backbone."):
            continue
        backbone_elements += target_value.numel()
        source_key = source_prefix + target_key[len("teacher_backbone.") :]
        source_value = source_state.get(source_key)
        if source_value is None:
            missing.append(target_key)
            continue
        if source_value.shape == target_value.shape:
            mapped[target_key] = source_value
            loaded_elements += target_value.numel()
            continue
        is_stem = target_key.endswith("patch_embedding.patch_embeddings.weight")
        same_nonchannel_shape = (
            source_value.ndim == target_value.ndim == 5
            and source_value.shape[0] == target_value.shape[0]
            and source_value.shape[1] == 1
            and source_value.shape[2:] == target_value.shape[2:]
        )
        if is_stem and same_nonchannel_shape and target_value.shape[1] > 1:
            mapped[target_key] = source_value.repeat(1, target_value.shape[1], 1, 1, 1) / target_value.shape[1]
            loaded_elements += target_value.numel()
            adapted.append(target_key)
        else:
            shape_mismatch.append(
                f"{target_key}:{tuple(source_value.shape)}->{tuple(target_value.shape)}"
            )

    incompatible = model.load_state_dict(mapped, strict=False)
    report = {
        "mapped_tensors": len(mapped),
        "backbone_tensors": sum(key.startswith("teacher_backbone.") for key in target_state),
        "loaded_elements": loaded_elements,
        "backbone_elements": backbone_elements,
        "element_coverage": loaded_elements / max(backbone_elements, 1),
        "missing": missing,
        "shape_mismatch": shape_mismatch,
        "adapted": adapted,
        "load_missing_count": len(incompatible.missing_keys),
        "load_unexpected_count": len(incompatible.unexpected_keys),
    }
    return mapped, report


def locate_task_run(root: Path, task: str) -> dict[str, object]:
    task_root = root / task
    best_checkpoints = sorted(task_root.rglob("checkpoints/best.ckpt"), key=lambda path: path.stat().st_mtime)
    completed = []
    for checkpoint in best_checkpoints:
        run_dir = checkpoint.parent.parent
        predictions = sorted(run_dir.glob("predictions/*__best.json"))
        if predictions:
            completed.append((checkpoint, predictions[-1], run_dir))
    if completed:
        checkpoint, prediction, run_dir = completed[-1]
    elif best_checkpoints:
        checkpoint = best_checkpoints[-1]
        run_dir = checkpoint.parent.parent
        prediction = None
    else:
        return {"candidate_count": 0, "completed_count": 0}
    return {
        "candidate_count": len(best_checkpoints),
        "completed_count": len(completed),
        "checkpoint": checkpoint,
        "prediction": prediction,
        "run_dir": run_dir,
    }


def parse_transfer_log(run_dir: Path) -> tuple[str, str]:
    pattern = re.compile(r"Succesfully transferred weights for\s+(\d+)/(\d+)\s+layers")
    found = []
    for path in sorted(run_dir.rglob("*.log")):
        try:
            for match in pattern.finditer(path.read_text(encoding="utf-8", errors="replace")):
                found.append((match.group(1), match.group(2), path))
        except OSError:
            continue
    if not found:
        return "not_found", "not_found"
    successful, total, _ = found[-1]
    return successful, total


def counts_text(values: Iterable) -> str:
    counts = Counter(values)
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts, key=str))


def binary_auc(labels: list[int], scores: list[float]) -> float:
    n_pos = sum(label == 1 for label in labels)
    n_neg = sum(label == 0 for label in labels)
    if not n_pos or not n_neg:
        return float("nan")
    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    xt = xt - xt.mean()
    yt = yt - yt.mean()
    denominator = torch.linalg.vector_norm(xt) * torch.linalg.vector_norm(yt)
    return float(torch.dot(xt, yt) / denominator) if denominator > 0 else float("nan")


def summarize_predictions(task: str, spec: dict, path: Path, reporter: Reporter) -> dict[str, float | bool]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    kind = spec["kind"]
    flags = {}
    if kind == "classification":
        records = [
            value
            for key, value in data.items()
            if key != "metrics" and isinstance(value, dict) and "label" in value
        ]
        labels = [int(record["label"]) for record in records]
        predictions = [int(record["prediction"]) for record in records]
        probabilities = [
            float(record["probabilities"][1])
            for record in records
            if isinstance(record.get("probabilities"), list) and len(record["probabilities"]) >= 2
        ]
        accuracy = sum(p == y for p, y in zip(predictions, labels)) / max(len(labels), 1)
        probability_std = float(torch.tensor(probabilities).std(unbiased=False)) if probabilities else float("nan")
        auc = binary_auc(labels, probabilities) if len(probabilities) == len(labels) else float("nan")
        constant = len(set(predictions)) <= 1
        flags["constant_output"] = constant
        reporter.emit(
            "PREDICTIONS",
            task=task,
            kind=kind,
            n=len(labels),
            labels=counts_text(labels),
            predictions=counts_text(predictions),
            accuracy=f"{accuracy:.6g}",
            auc=f"{auc:.6g}",
            probability_std=f"{probability_std:.6g}",
            constant_output=constant,
        )
    elif kind == "regression":
        records = [
            value
            for key, value in data.items()
            if key != "metrics" and isinstance(value, dict) and "label" in value
        ]
        labels = [float(record["label"]) for record in records]
        predictions = [float(record["prediction"]) for record in records]
        errors = [prediction - label for prediction, label in zip(predictions, labels)]
        label_std = float(torch.tensor(labels).std(unbiased=False)) if labels else float("nan")
        prediction_std = float(torch.tensor(predictions).std(unbiased=False)) if predictions else float("nan")
        mae = sum(abs(error) for error in errors) / max(len(errors), 1)
        collapsed = prediction_std < 0.1 * max(label_std, 1e-12)
        flags["constant_output"] = collapsed
        reporter.emit(
            "PREDICTIONS",
            task=task,
            kind=kind,
            n=len(labels),
            mae=f"{mae:.6g}",
            bias=f"{sum(errors) / max(len(errors), 1):.6g}",
            label_mean=f"{sum(labels) / max(len(labels), 1):.6g}",
            label_std=f"{label_std:.6g}",
            prediction_mean=f"{sum(predictions) / max(len(predictions), 1):.6g}",
            prediction_std=f"{prediction_std:.6g}",
            pearson=f"{pearson(labels, predictions):.6g}",
            near_constant_output=collapsed,
        )
    else:
        mean = data.get("mean", {})
        records = [value for key, value in data.items() if key != "mean" and isinstance(value, dict)]
        foreground = [str(index) for index in range(1, int(spec["classes"]))]
        for label in foreground:
            dice_value = finite_float(mean.get(label, {}).get("dice"))
            gt_positive_cases = 0
            zero_prediction_cases = 0
            total_gt = 0.0
            total_pred = 0.0
            per_case_dice = []
            for record in records:
                metrics = record.get(label, {})
                gt = finite_float(metrics.get("total_pos_gt")) or 0.0
                pred = finite_float(metrics.get("total_pos_pred")) or 0.0
                case_dice = finite_float(metrics.get("dice"))
                total_gt += gt
                total_pred += pred
                if gt > 0:
                    gt_positive_cases += 1
                    if pred == 0:
                        zero_prediction_cases += 1
                if case_dice is not None:
                    per_case_dice.append(case_dice)
            zero_fraction = zero_prediction_cases / max(gt_positive_cases, 1)
            flags[f"zero_fraction_label_{label}"] = zero_fraction
            reporter.emit(
                "PREDICTIONS",
                task=task,
                kind=kind,
                label=label,
                n=len(records),
                mean_dice="nan" if dice_value is None else f"{dice_value:.6g}",
                recomputed_case_dice=(
                    "nan" if not per_case_dice else f"{sum(per_case_dice) / len(per_case_dice):.6g}"
                ),
                gt_positive_cases=gt_positive_cases,
                zero_prediction_cases=zero_prediction_cases,
                zero_prediction_fraction=f"{zero_fraction:.6g}",
                predicted_to_gt_voxel_ratio=f"{total_pred / max(total_gt, 1e-12):.6g}",
            )
    return flags


def feature_health(features: torch.Tensor) -> dict[str, float | int]:
    features = features.detach().float().cpu()
    n_samples, n_features = features.shape
    centered = features - features.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    singular_values = singular_values[singular_values > 1e-12]
    if singular_values.numel():
        probabilities = singular_values / singular_values.sum()
        effective_rank = float(torch.exp(-(probabilities * probabilities.log()).sum()))
    else:
        effective_rank = 0.0
    normalized = F.normalize(features, dim=1)
    similarity = normalized @ normalized.T
    off_diagonal = similarity[~torch.eye(n_samples, dtype=torch.bool)] if n_samples > 1 else torch.tensor([])
    std = features.std(dim=0, unbiased=False)
    return {
        "samples": n_samples,
        "dimensions": n_features,
        "effective_rank": effective_rank,
        "normalized_effective_rank": effective_rank / max(min(n_samples - 1, n_features), 1),
        "mean_dimension_std": float(std.mean()),
        "near_zero_dimension_fraction": float((std < 1e-6).float().mean()),
        "mean_offdiag_cosine": float(off_diagonal.mean()) if off_diagonal.numel() else float("nan"),
        "max_offdiag_cosine": float(off_diagonal.max()) if off_diagonal.numel() else float("nan"),
    }


def split_train_paths(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    fold = payload
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        fold = payload[0]
    elif isinstance(payload, dict) and "0" in payload and isinstance(payload["0"], dict):
        fold = payload["0"]
    if isinstance(fold, dict):
        for key in ("train", "training"):
            if isinstance(fold.get(key), list):
                return fold[key]
    if isinstance(payload, list) and (not payload or isinstance(payload[0], str)):
        return payload
    raise ValueError(f"Could not find a training path list in {path}")


def evenly_spaced(items: list[str], count: int) -> list[str]:
    if len(items) <= count:
        return items
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


@torch.no_grad()
def run_feature_probe(
    source_state: dict[str, torch.Tensor], data_root: Path, sample_count: int, reporter: Reporter
) -> dict[str, dict[str, float | int]]:
    from asparagus.functional.loading import load_image_file
    from asparagus.modules.networks.dino_downstream import DINO3DClassifierRegressor
    from asparagus.modules.transforms.presets.train import CPU_clsreg_val_test_transforms_crop

    split_path = data_root / "PT902_FOMO300K_HF" / "split_99_01_00.json"
    all_paths = split_train_paths(split_path)
    selected = evenly_spaced(all_paths, sample_count)
    transform = CPU_clsreg_val_test_transforms_crop(target_size=[96, 96, 96])
    volumes = []
    failures = []
    for path in selected:
        try:
            image = load_image_file(path).float()
            transformed = transform({"image": image, "transforms_applied": {}})["image"]
            if transformed.ndim != 4 or transformed.shape[0] != 1:
                raise ValueError(f"unexpected transformed shape {tuple(transformed.shape)}")
            volumes.append(transformed)
        except Exception as error:  # diagnostics should continue after a bad sample
            failures.append(f"{Path(path).name}:{type(error).__name__}")
    if len(volumes) < 4:
        raise RuntimeError(f"Only {len(volumes)} feature-probe samples loaded; failures={failures[:5]}")

    teacher_model = DINO3DClassifierRegressor(1, 1)
    _, teacher_transfer = transfer_backbone(source_state, teacher_model, PRETRAIN_TEACHER_PREFIX)
    student_model = DINO3DClassifierRegressor(1, 1)
    _, student_transfer = transfer_backbone(source_state, student_model, PRETRAIN_STUDENT_PREFIX)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher_model.teacher_backbone.eval().to(device)
    student = student_model.teacher_backbone.eval().to(device)
    cls_teacher = []
    cls_student = []
    patch_mean_teacher = []
    spatial_patch_std = []

    for start in range(0, len(volumes), 2):
        batch = torch.stack(volumes[start : start + 2]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            teacher_tokens = teacher(batch, mask=None)
            student_tokens = student(batch, mask=None)
        cls_teacher.append(teacher_tokens[:, 0].float().cpu())
        cls_student.append(student_tokens[:, 0].float().cpu())
        patches = teacher_tokens[:, 1:].float().cpu()
        patch_mean_teacher.append(patches.mean(dim=1))
        spatial_patch_std.extend(patches.std(dim=1, unbiased=False).mean(dim=1).tolist())

    cls_teacher = torch.cat(cls_teacher)
    cls_student = torch.cat(cls_student)
    patch_mean_teacher = torch.cat(patch_mean_teacher)
    concatenated = torch.cat([cls_teacher, patch_mean_teacher], dim=1)
    teacher_student_cosine = float(
        F.cosine_similarity(cls_teacher, cls_student, dim=1).mean()
    )

    health = {
        "teacher_cls": feature_health(cls_teacher),
        "teacher_mean_patch": feature_health(patch_mean_teacher),
        "teacher_cls_plus_mean_patch": feature_health(concatenated),
    }
    for representation, values in health.items():
        reporter.emit(
            "FEATURE_HEALTH",
            representation=representation,
            **{key: f"{value:.6g}" if isinstance(value, float) else value for key, value in values.items()},
        )
    reporter.emit(
        "FEATURE_HEALTH",
        representation="patch_spatial_variation",
        samples=len(spatial_patch_std),
        mean_patch_dimension_std=f"{sum(spatial_patch_std) / len(spatial_patch_std):.6g}",
        teacher_student_cls_cosine=f"{teacher_student_cosine:.6g}",
        load_failures=len(failures),
        teacher_transfer_coverage=f"{teacher_transfer['element_coverage']:.6g}",
        student_transfer_coverage=f"{student_transfer['element_coverage']:.6g}",
    )
    del teacher, student, teacher_model, student_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return health


def summarize_csv_histories(result_root: Path, reporter: Reporter) -> None:
    pretrain_files = sorted(result_root.rglob("dino3d_pretrain_history.csv"))
    reporter.emit("HISTORY_FILES", kind="pretrain", count=len(pretrain_files))
    if pretrain_files:
        path = pretrain_files[-1]
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            metrics = [
                key
                for key in rows[0]
                if any(token in key for token in ("total_loss", "dino_loss", "ibot_loss", "koleo_loss", "mask_ratio"))
            ]
            for metric in metrics:
                values = [finite_float(row.get(metric)) for row in rows]
                values = [value for value in values if value is not None]
                if values:
                    reporter.emit(
                        "PRETRAIN_HISTORY",
                        metric=metric,
                        points=len(values),
                        first=f"{values[0]:.6g}",
                        last=f"{values[-1]:.6g}",
                        minimum=f"{min(values):.6g}",
                        maximum=f"{max(values):.6g}",
                        source=path,
                    )

    finetune_files = sorted(result_root.rglob("dino3d_finetune_history.csv"))
    reporter.emit("HISTORY_FILES", kind="finetune", count=len(finetune_files))
    if finetune_files:
        path = finetune_files[-1]
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for task in TASKS:
            task_rows = [row for row in rows if row.get("task") == task]
            task_rows.sort(key=lambda row: int(float(row.get("epoch", 0))))
            if not task_rows:
                continue
            for metric in ("train_loss", "val_loss"):
                values = [finite_float(row.get(metric)) for row in task_rows]
                values = [value for value in values if value is not None]
                if values:
                    reporter.emit(
                        "FINETUNE_HISTORY",
                        task=task,
                        metric=metric,
                        points=len(values),
                        first=f"{values[0]:.6g}",
                        last=f"{values[-1]:.6g}",
                        best=f"{min(values):.6g}",
                        source=path,
                    )


def main() -> int:
    code_root_default = Path(__file__).resolve().parents[3]
    shared_default = Path(os.environ.get("SHARED_ROOT", "/mnt/project/perun2601396"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", default=str(code_root_default))
    parser.add_argument("--shared-root", default=str(shared_default))
    parser.add_argument(
        "--pretrain-checkpoint",
        default=str(shared_default / "FOMO26_checkpoints" / f"dinov3_3d_stage1_{RUN_ID}_last.ckpt"),
    )
    parser.add_argument(
        "--pretrain-model-root",
        default=str(shared_default / "FOMO26_models" / "baseline_pretraining"),
    )
    parser.add_argument(
        "--finetune-root",
        default=str(shared_default / "FOMO26_models" / "dino3d_finetuning"),
    )
    parser.add_argument(
        "--result-root",
        default=str(shared_default / "FOMO26_results" / "dino3d"),
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get(
            "ASPARAGUS_DATA", str(shared_default / "FOMO26_processed" / "baseline")
        ),
    )
    parser.add_argument("--feature-samples", type=int, default=24)
    parser.add_argument(
        "--output",
        default=str(
            shared_default
            / "FOMO26_results"
            / "dino3d"
            / f"diagnostics_{RUN_ID}"
            / "dino3d_diagnostic_report.txt"
        ),
    )
    args = parser.parse_args()

    reporter = Reporter(Path(args.output))
    findings: list[tuple[str, str, str]] = []
    task_flags: dict[str, dict] = {}
    try:
        reporter.emit(
            "START",
            run_id=RUN_ID,
            python=sys.version.split()[0],
            torch=torch.__version__,
            cuda=torch.cuda.is_available(),
            device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            code_root=args.code_root,
            data_root=args.data_root,
        )

        pretrain_path = find_pretrain_checkpoint(args)
        reporter.emit(
            "PRETRAIN_CHECKPOINT",
            path=pretrain_path,
            size_bytes=pretrain_path.stat().st_size,
            sha256=sha256_file(pretrain_path),
        )
        pretrain_checkpoint = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        source_state = checkpoint_state(pretrain_checkpoint)
        global_step = int(pretrain_checkpoint.get("global_step", -1))
        epoch = int(pretrain_checkpoint.get("epoch", -1))
        reporter.emit(
            "PRETRAIN_META",
            global_step=global_step,
            expected_global_steps=EXPECTED_GLOBAL_STEPS,
            completion_fraction=f"{global_step / EXPECTED_GLOBAL_STEPS:.6g}",
            epoch=epoch,
            state_tensors=len(source_state),
        )
        if global_step < int(0.95 * EXPECTED_GLOBAL_STEPS):
            findings.append(("HIGH", "PRETRAIN_INCOMPLETE", f"global_step={global_step}"))

        teacher_health = tensor_health(source_state, PRETRAIN_TEACHER_PREFIX)
        student_health = tensor_health(source_state, PRETRAIN_STUDENT_PREFIX)
        reporter.emit("WEIGHT_HEALTH", component="teacher_backbone", **teacher_health)
        reporter.emit("WEIGHT_HEALTH", component="student_backbone", **student_health)
        if int(teacher_health["nonfinite"]) or int(student_health["nonfinite"]):
            findings.append(("HIGH", "NONFINITE_BACKBONE_WEIGHTS", "checkpoint contains NaN/Inf"))

        teacher_student = relative_l2_and_cosine(
            paired_prefixes(source_state, PRETRAIN_TEACHER_PREFIX, PRETRAIN_STUDENT_PREFIX)
        )
        reporter.emit("TEACHER_STUDENT", **teacher_student)
        if teacher_student["tensors"] and teacher_student["exact_tensors"] == teacher_student["tensors"]:
            findings.append(("HIGH", "TEACHER_STUDENT_IDENTICAL", "all matching tensors are exactly equal"))

        reporter.emit(
            "ARCHITECTURE",
            backbone="MONAI_ViT_3D",
            layers=8,
            hidden_size=384,
            patch_size="16x16x16",
            pretrain_crop="96x96x96",
            clsreg_features="final_CLS_only",
            segmentation_features="final_layer_patch_tokens_only",
            segmentation_decoder="four_transposed_convolutions",
        )

        finetune_root = Path(args.finetune_root)
        for task, spec in TASKS.items():
            run = locate_task_run(finetune_root, task)
            reporter.emit(
                "TASK_ARTIFACTS",
                task=task,
                checkpoint_candidates=run.get("candidate_count", 0),
                completed_runs=run.get("completed_count", 0),
                selected_run=run.get("run_dir", "not_found"),
            )
            if "checkpoint" not in run:
                findings.append(("HIGH", "MISSING_FINETUNE_CHECKPOINT", task))
                continue

            model = model_for_task(spec)
            mapped, transfer = transfer_backbone(source_state, model)
            reporter.emit(
                "TRANSFER_COVERAGE",
                task=task,
                mapped_tensors=transfer["mapped_tensors"],
                backbone_tensors=transfer["backbone_tensors"],
                element_coverage=f"{transfer['element_coverage']:.9g}",
                adapted_stem=len(transfer["adapted"]),
                missing=len(transfer["missing"]),
                shape_mismatch=len(transfer["shape_mismatch"]),
            )
            if float(transfer["element_coverage"]) < 0.99:
                findings.append(
                    ("HIGH", "LOW_TRANSFER_COVERAGE", f"{task}:{transfer['element_coverage']:.6g}")
                )

            successful_log, total_log = parse_transfer_log(run["run_dir"])
            reporter.emit(
                "TRANSFER_LOG",
                task=task,
                successful_layers=successful_log,
                total_layers=total_log,
            )

            finetune_checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
            finetune_state = checkpoint_state(finetune_checkpoint)
            drift_pairs = []
            for target_key, initial_value in mapped.items():
                final_value = finetune_state.get("model." + target_key)
                if torch.is_tensor(final_value) and final_value.shape == initial_value.shape:
                    drift_pairs.append((initial_value, final_value))
            drift = relative_l2_and_cosine(drift_pairs)
            decoder_health = tensor_health(finetune_state, "model.decoder.")
            reporter.emit(
                "FINETUNE_CHECKPOINT",
                task=task,
                path=run["checkpoint"],
                epoch=finetune_checkpoint.get("epoch", "missing"),
                global_step=finetune_checkpoint.get("global_step", "missing"),
                backbone_relative_l2=f"{drift['relative_l2']:.6g}",
                backbone_cosine=f"{drift['cosine']:.6g}",
                unchanged_backbone_tensors=f"{drift['exact_tensors']}/{drift['tensors']}",
                decoder_nonfinite=decoder_health["nonfinite"],
                decoder_l2=f"{decoder_health['l2_norm']:.6g}",
            )
            if drift["tensors"] and drift["exact_tensors"] == drift["tensors"]:
                findings.append(("HIGH", "BACKBONE_NEVER_UPDATED", task))
            if int(decoder_health["nonfinite"]):
                findings.append(("HIGH", "NONFINITE_DOWNSTREAM_HEAD", task))

            if run.get("prediction"):
                task_flags[task] = summarize_predictions(
                    task, spec, run["prediction"], reporter
                )
            else:
                findings.append(("HIGH", "MISSING_PREDICTIONS", task))
            del model, finetune_checkpoint, finetune_state

        summarize_csv_histories(Path(args.result_root), reporter)

        try:
            feature_results = run_feature_probe(
                source_state,
                Path(args.data_root),
                max(4, int(args.feature_samples)),
                reporter,
            )
            cls_health = feature_results["teacher_cls"]
            patch_health = feature_results["teacher_mean_patch"]
            if float(cls_health["effective_rank"]) < 2.0 or float(cls_health["mean_offdiag_cosine"]) > 0.995:
                findings.append(
                    (
                        "HIGH",
                        "CLS_REPRESENTATION_COLLAPSE",
                        f"rank={cls_health['effective_rank']:.4g},cos={cls_health['mean_offdiag_cosine']:.4g}",
                    )
                )
            elif float(cls_health["normalized_effective_rank"]) < 0.2:
                findings.append(
                    (
                        "MEDIUM",
                        "LOW_CLS_EFFECTIVE_RANK",
                        f"normalized_rank={cls_health['normalized_effective_rank']:.4g}",
                    )
                )
            if float(patch_health["effective_rank"]) > 1.5 * max(float(cls_health["effective_rank"]), 1e-9):
                findings.append(
                    (
                        "MEDIUM",
                        "PATCH_FEATURES_HEALTHIER_THAN_CLS",
                        "CLS-only downstream pooling may discard useful features",
                    )
                )
        except Exception as error:
            reporter.emit(
                "FEATURE_PROBE_ERROR",
                error_type=type(error).__name__,
                message=error,
                traceback=traceback.format_exc(limit=3),
            )
            findings.append(("MEDIUM", "FEATURE_PROBE_FAILED", clean(error)))

        for task, flags in task_flags.items():
            if flags.get("constant_output"):
                findings.append(("HIGH", "CONSTANT_DOWNSTREAM_OUTPUT", task))
            for key, value in flags.items():
                if key.startswith("zero_fraction_label_") and float(value) >= 0.9:
                    findings.append(("HIGH", "SEGMENTATION_ALL_BACKGROUND", f"{task}:{key}={value:.3g}"))

        findings.append(
            (
                "MEDIUM",
                "DOWNSTREAM_HEAD_MISMATCH_RISK",
                "evaluation uses final CLS/final patch layer rather than multi-layer or adapter features",
            )
        )
        if not findings:
            findings.append(("INFO", "NO_AUTOMATIC_FAILURE_FOUND", "manual history review required"))
        for severity, code, evidence in findings:
            reporter.emit("FINDING", severity=severity, code=code, evidence=evidence)
        reporter.emit("END", status="complete", report=reporter.path, findings=len(findings))
        return 0
    except Exception as error:
        reporter.emit(
            "FATAL",
            error_type=type(error).__name__,
            message=error,
            traceback=traceback.format_exc(limit=8),
        )
        reporter.emit("END", status="failed", report=reporter.path)
        return 1
    finally:
        reporter.close()


if __name__ == "__main__":
    raise SystemExit(main())
