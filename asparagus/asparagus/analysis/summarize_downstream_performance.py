"""Summarize completed downstream test JSON files using challenge-relevant metrics."""

import argparse
import csv
import json
import math
from pathlib import Path


TASK_TYPES = {
    "CLS002_FOMO26_Infarct": "classification",
    "SEG009_FOMO26_Meningioma": "segmentation",
    "REGR002_FOMO26_BrainAge": "regression",
    "SEG010_FOMO26_TrigeminalNeuralgia": "segmentation",
    "CLS003_FOMO26_Polymicrogyria": "classification",
}


def task_from_path(path):
    text = str(path)
    return next((task for task in TASK_TYPES if task in text), "unknown")


def safe_div(a, b):
    return a / b if b else float("nan")


def binary_auc(labels, scores):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = average_rank
        i = j
    rank_sum_pos = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels, scores):
    positives = sum(labels)
    if not positives:
        return float("nan")
    ordered = sorted(zip(scores, labels), reverse=True)
    true_positives = 0
    total = 0.0
    for rank, (_, label) in enumerate(ordered, 1):
        if label == 1:
            true_positives += 1
            total += true_positives / rank
    return total / positives


def summarize_classification(data):
    records = [value for key, value in data.items() if key != "metrics" and isinstance(value, dict) and "label" in value]
    if not records:
        raise ValueError("No classification predictions")
    labels = [int(record["label"]) for record in records]
    predictions = [int(record["prediction"]) for record in records]
    has_probabilities = all("probabilities" in record and len(record["probabilities"]) >= 2 for record in records)
    scores = [float(record["probabilities"][1]) for record in records] if has_probabilities else []
    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(predictions, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "n_test": len(records),
        "accuracy": safe_div(tp + tn, len(records)),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "roc_auc": binary_auc(labels, scores) if scores else float("nan"),
        "average_precision": average_precision(labels, scores) if scores else float("nan"),
        "probabilities_available": has_probabilities,
    }


def summarize_regression(data):
    records = [value for key, value in data.items() if key != "metrics" and isinstance(value, dict) and "label" in value]
    if not records:
        raise ValueError("No regression predictions")
    labels = [float(record["label"]) for record in records]
    predictions = [float(record["prediction"]) for record in records]
    errors = [pred - label for pred, label in zip(predictions, labels)]
    mse = sum(error * error for error in errors) / len(errors)
    mean_label = sum(labels) / len(labels)
    ss_total = sum((label - mean_label) ** 2 for label in labels)
    ss_residual = sum(error * error for error in errors)
    return {
        "n_test": len(records),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": 1.0 - ss_residual / ss_total if ss_total else float("nan"),
    }


def summarize_segmentation(data):
    mean = data.get("mean")
    if not isinstance(mean, dict):
        raise ValueError("Segmentation JSON has no mean section")
    labels = sorted(mean, key=lambda value: int(value) if str(value).isdigit() else str(value))
    foreground = [label for label in labels if str(label) != "0"] or labels
    metric_names = sorted({name for label in foreground for name in mean[label]})
    summary = {"n_test": max(0, len(data) - 1), "foreground_labels": ";".join(map(str, foreground))}
    for name in metric_names:
        values = [float(mean[label][name]) for label in foreground if name in mean[label] and math.isfinite(float(mean[label][name]))]
        summary[f"foreground_macro_{name}"] = sum(values) / len(values) if values else float("nan")
    return summary, mean


def write_csv(path, rows):
    fields = ["task", "task_type", "run", "predictions_json"]
    fields += sorted({key for row in rows for key in row if key not in fields})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="DINO fine-tuning model root")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    json_files = sorted(root.rglob("predictions/*__best.json"))
    if not json_files:
        raise SystemExit(f"No completed best-checkpoint prediction JSON files found under {root}")
    rows = []
    per_label_rows = []
    for path in json_files:
        task = task_from_path(path)
        task_type = TASK_TYPES.get(task, "unknown")
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        base = {
            "task": task,
            "task_type": task_type,
            "run": next((part for part in path.parts if part.startswith("slurm_") or part.startswith("run_")), path.parent.parent.name),
            "predictions_json": str(path),
        }
        if task_type == "classification":
            base.update(summarize_classification(data))
        elif task_type == "regression":
            base.update(summarize_regression(data))
        elif task_type == "segmentation":
            summary, labels = summarize_segmentation(data)
            base.update(summary)
            for label, metrics in labels.items():
                per_label_rows.append({**base, "label": label, **metrics})
        else:
            continue
        rows.append(base)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "dino3d_downstream_performance.csv", rows)
    if per_label_rows:
        write_csv(output_dir / "dino3d_segmentation_per_label.csv", per_label_rows)
    print(f"DOWNSTREAM_PERFORMANCE_OK completed_tasks={len(rows)}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
