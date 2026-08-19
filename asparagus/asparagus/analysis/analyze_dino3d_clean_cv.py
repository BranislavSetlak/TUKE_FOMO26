"""Analyze the controlled five-fold DINO3D downstream experiment.

The command deliberately writes one result artifact: a plain-text report with
machine-readable TSV sections.  It can analyze a partially completed Slurm
array, while ``--require-complete`` makes missing fold outputs an error.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score


TASKS = {
    "CLS002_FOMO26_Infarct": {
        "type": "classification",
        "validation_primary": "val/auroc_macro",
        "test_primary": "roc_auc",
    },
    "SEG009_FOMO26_Meningioma": {
        "type": "segmentation",
        "validation_primary": "val/foreground_dice",
        "test_primary": "foreground_macro_dice",
    },
    "REGR002_FOMO26_BrainAge": {
        "type": "regression",
        "validation_primary": "val/MAE",
        "test_primary": "mae",
    },
    "SEG010_FOMO26_TrigeminalNeuralgia": {
        "type": "segmentation",
        "validation_primary": "val/foreground_dice",
        "test_primary": "foreground_macro_dice",
    },
    "CLS003_FOMO26_Polymicrogyria": {
        "type": "classification",
        "validation_primary": "val/auroc_macro",
        "test_primary": "roc_auc",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        required=True,
        help="Directory containing one task directory per dataset, for example experiment_74929",
    )
    parser.add_argument("--report", required=True, help="Single text report to create")
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASKS),
        help="Optional task subset; omit to analyze all five downstream tasks",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero unless all expected best-checkpoint results exist",
    )
    return parser.parse_args()


def finite_number(value):
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def fmt(value):
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if value is None:
        return "NA"
    if isinstance(value, Path):
        return str(value)
    if finite_number(value):
        number = float(value)
        if number.is_integer() and abs(number) < 1e12:
            return str(int(number))
        return f"{number:.8g}"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return str(value).replace("\t", " ").replace("\n", " ")


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_numbers(value, prefix=""):
    """Flatten numeric JSON values while retaining the original metric names."""
    flattened = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_numbers(child, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            name = f"{prefix}[{index}]"
            flattened.update(flatten_numbers(child, name))
    elif finite_number(value):
        flattened[prefix] = float(value)
    return flattened


def prediction_records(data):
    return {
        str(key): value
        for key, value in data.items()
        if key not in {"metrics", "mean"}
        and isinstance(value, dict)
        and "label" in value
        and "prediction" in value
    }


def classification_metrics(labels, predictions, scores):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    scores = np.asarray(scores, dtype=float) if scores is not None else None
    if labels.size == 0:
        raise ValueError("No classification predictions")

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * precision * sensitivity, precision + sensitivity)
    both_classes = len(set(labels.tolist())) == 2

    result = {
        "n_test": int(labels.size),
        "n_negative": int(np.sum(labels == 0)),
        "n_positive": int(np.sum(labels == 1)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "precision": precision,
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "label_positive_fraction": float(np.mean(labels == 1)),
        "predicted_positive_fraction": float(np.mean(predictions == 1)),
        "constant_predictions": len(set(predictions.tolist())) == 1,
    }
    if scores is not None and scores.size == labels.size:
        result.update(
            {
                "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else float("nan"),
                "average_precision": (
                    float(average_precision_score(labels, scores))
                    if np.any(labels == 1)
                    else float("nan")
                ),
                "brier_score": float(np.mean((scores - labels) ** 2)),
                "positive_probability_mean": float(np.mean(scores)),
                "positive_probability_std": float(np.std(scores)),
                "near_constant_probabilities": float(np.std(scores)) < 1e-4,
            }
        )
    else:
        result.update(
            {
                "roc_auc": float("nan"),
                "average_precision": float("nan"),
                "brier_score": float("nan"),
                "positive_probability_mean": float("nan"),
                "positive_probability_std": float("nan"),
                "near_constant_probabilities": True,
            }
        )
    return result


def summarize_classification(data):
    records = prediction_records(data)
    labels = [int(record["label"]) for record in records.values()]
    predictions = [int(record["prediction"]) for record in records.values()]
    probabilities_available = all(
        isinstance(record.get("probabilities"), list) and len(record["probabilities"]) >= 2
        for record in records.values()
    )
    scores = (
        [float(record["probabilities"][1]) for record in records.values()]
        if probabilities_available
        else None
    )
    return classification_metrics(labels, predictions, scores), records


def regression_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if labels.size == 0:
        raise ValueError("No regression predictions")
    errors = predictions - labels
    ss_total = float(np.sum((labels - np.mean(labels)) ** 2))
    ss_residual = float(np.sum(errors**2))
    label_std = float(np.std(labels))
    prediction_std = float(np.std(predictions))
    pearson = (
        float(np.corrcoef(labels, predictions)[0, 1])
        if labels.size > 1 and label_std > 0 and prediction_std > 0
        else float("nan")
    )
    mse = float(np.mean(errors**2))
    return {
        "n_test": int(labels.size),
        "mae": float(np.mean(np.abs(errors))),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": 1 - ss_residual / ss_total if ss_total else float("nan"),
        "pearson_r": pearson,
        "mean_error_bias": float(np.mean(errors)),
        "label_mean": float(np.mean(labels)),
        "label_std": label_std,
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": prediction_std,
        "prediction_min": float(np.min(predictions)),
        "prediction_max": float(np.max(predictions)),
        "collapsed_predictions": prediction_std < max(1e-6, 0.1 * label_std),
    }


def summarize_regression(data):
    records = prediction_records(data)
    labels = [float(record["label"]) for record in records.values()]
    predictions = [float(record["prediction"]) for record in records.values()]
    return regression_metrics(labels, predictions), records


def metric_case_insensitive(metrics, wanted):
    for key, value in metrics.items():
        if key.lower() == wanted.lower() and finite_number(value):
            return float(value)
    return float("nan")


def summarize_segmentation(data):
    mean = data.get("mean")
    if not isinstance(mean, dict) or not mean:
        raise ValueError("Segmentation prediction JSON has no mean section")
    labels = sorted(mean, key=lambda item: int(item) if str(item).isdigit() else str(item))
    foreground = [label for label in labels if str(label) != "0"] or labels
    metric_names = sorted({name for label in foreground for name in mean[label]})
    result = {
        "n_test": max(0, len(data) - 1),
        "foreground_labels": ";".join(map(str, foreground)),
    }
    for name in metric_names:
        values = [
            float(mean[label][name])
            for label in foreground
            if name in mean[label] and finite_number(mean[label][name])
        ]
        result[f"foreground_macro_{name}"] = (
            float(np.mean(values)) if values else float("nan")
        )
    predicted_foreground = [
        metric_case_insensitive(mean[label], "total_pos_pred") for label in foreground
    ]
    foreground_dice = [metric_case_insensitive(mean[label], "dice") for label in foreground]
    result["all_background_prediction"] = bool(
        predicted_foreground
        and all(math.isfinite(value) and value <= 0 for value in predicted_foreground)
    )
    result["all_foreground_dice_zero"] = bool(
        foreground_dice
        and all(math.isfinite(value) and value <= 0 for value in foreground_dice)
    )
    return result


def summary_statistics(values):
    array = np.asarray([float(value) for value in values if finite_number(value)], dtype=float)
    count = int(array.size)
    if count == 0:
        return None
    mean = float(np.mean(array))
    sample_sd = float(np.std(array, ddof=1)) if count > 1 else float("nan")
    if count > 1:
        half_width = float(student_t.ppf(0.975, count - 1) * sample_sd / math.sqrt(count))
        ci_low, ci_high = mean - half_width, mean + half_width
    else:
        ci_low = ci_high = float("nan")
    return {
        "n": count,
        "mean": mean,
        "sample_sd": sample_sd,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_rows(rows, tasks):
    summaries = []
    for task, task_spec in tasks.items():
        task_rows = [row for row in rows if row["task"] == task]
        metric_names = sorted(
            {
                metric
                for row in task_rows
                for metric, value in row["metrics"].items()
                if finite_number(value)
            }
        )
        for metric in metric_names:
            stats = summary_statistics(row["metrics"].get(metric) for row in task_rows)
            if stats:
                summaries.append(
                    {
                        "task": task,
                        "task_type": task_spec["type"],
                        "metric": metric,
                        **stats,
                    }
                )
    return summaries


def find_checkpoint(run_dir, name):
    candidates = sorted(run_dir.rglob(name)) if run_dir.exists() else []
    return candidates[0] if candidates else None


def inspect_runs(experiment_root, expected_folds, tasks):
    statuses = []
    validation_rows = []
    test_rows = []
    payloads = {task: [] for task in tasks}
    warnings = []

    for task, task_spec in tasks.items():
        for fold in range(expected_folds):
            run_dir = experiment_root / task / f"fold_{fold}"
            best = find_checkpoint(run_dir, "best.ckpt")
            last = find_checkpoint(run_dir, "last.ckpt")
            periodic = sorted(run_dir.rglob("periodic-*.ckpt")) if run_dir.exists() else []
            validation_path = run_dir / "validation_best_metrics.json"
            prediction_paths = sorted(run_dir.glob("predictions/*__best.json")) if run_dir.exists() else []
            prediction_path = prediction_paths[0] if len(prediction_paths) == 1 else None
            complete = bool(
                best
                and best.stat().st_size > 0
                and validation_path.is_file()
                and validation_path.stat().st_size > 0
                and prediction_path
                and prediction_path.stat().st_size > 0
            )
            statuses.append(
                {
                    "task": task,
                    "task_type": task_spec["type"],
                    "fold": fold,
                    "run_dir_exists": run_dir.is_dir(),
                    "best_ckpt": str(best) if best else "MISSING",
                    "last_ckpt": str(last) if last else "MISSING",
                    "periodic_ckpts": len(periodic),
                    "validation_json": str(validation_path) if validation_path.is_file() else "MISSING",
                    "prediction_json": str(prediction_path) if prediction_path else "MISSING",
                    "prediction_json_count": len(prediction_paths),
                    "complete": complete,
                }
            )
            if len(prediction_paths) > 1:
                warnings.append(
                    f"{task} fold {fold}: found {len(prediction_paths)} best prediction JSON files; expected one."
                )

            if validation_path.is_file():
                try:
                    metrics = flatten_numbers(load_json(validation_path))
                    validation_rows.append(
                        {"task": task, "task_type": task_spec["type"], "fold": fold, "metrics": metrics}
                    )
                except Exception as error:
                    warnings.append(f"{task} fold {fold}: could not read validation metrics: {error}")

            if prediction_path:
                try:
                    data = load_json(prediction_path)
                    if task_spec["type"] == "classification":
                        metrics, records = summarize_classification(data)
                        payloads[task].append((fold, records))
                    elif task_spec["type"] == "regression":
                        metrics, records = summarize_regression(data)
                        payloads[task].append((fold, records))
                    else:
                        metrics = summarize_segmentation(data)
                    test_rows.append(
                        {"task": task, "task_type": task_spec["type"], "fold": fold, "metrics": metrics}
                    )
                except Exception as error:
                    warnings.append(f"{task} fold {fold}: could not analyze test predictions: {error}")
    return statuses, validation_rows, test_rows, payloads, warnings


def build_ensembles(payloads, warnings, tasks):
    rows = []
    for task, task_spec in tasks.items():
        task_type = task_spec["type"]
        fold_payloads = sorted(payloads[task])
        if task_type == "segmentation":
            rows.append(
                {
                    "task": task,
                    "task_type": task_type,
                    "n_models": 0,
                    "method": "unavailable: saved JSON contains metrics, not voxel probabilities",
                    "metrics": {},
                }
            )
            continue
        if not fold_payloads:
            continue

        reference_keys = set(fold_payloads[0][1])
        if any(set(records) != reference_keys for _, records in fold_payloads[1:]):
            warnings.append(
                f"{task}: fold test sample keys differ, so no prediction ensemble was computed."
            )
            continue
        keys = sorted(reference_keys)
        if not keys:
            continue
        for key in keys:
            labels = [records[key]["label"] for _, records in fold_payloads]
            if any(label != labels[0] for label in labels[1:]):
                warnings.append(f"{task}: labels disagree between folds for {key}; no ensemble computed.")
                keys = []
                break
        if not keys:
            continue

        if task_type == "classification":
            probabilities_available = all(
                isinstance(records[key].get("probabilities"), list)
                and len(records[key]["probabilities"]) >= 2
                for _, records in fold_payloads
                for key in keys
            )
            if not probabilities_available:
                warnings.append(f"{task}: probabilities are missing, so no classification ensemble was computed.")
                continue
            labels = [int(fold_payloads[0][1][key]["label"]) for key in keys]
            scores = [
                float(np.mean([records[key]["probabilities"][1] for _, records in fold_payloads]))
                for key in keys
            ]
            predictions = [int(score >= 0.5) for score in scores]
            metrics = classification_metrics(labels, predictions, scores)
            method = "mean positive-class probability; threshold=0.5"
        else:
            labels = [float(fold_payloads[0][1][key]["label"]) for key in keys]
            predictions = [
                float(np.mean([records[key]["prediction"] for _, records in fold_payloads]))
                for key in keys
            ]
            metrics = regression_metrics(labels, predictions)
            method = "mean prediction"
        rows.append(
            {
                "task": task,
                "task_type": task_type,
                "n_models": len(fold_payloads),
                "method": method,
                "metrics": metrics,
            }
        )
    return rows


def append_tsv(lines, title, fields, rows):
    lines.extend(["", f"[{title}]", "\t".join(fields)])
    for row in rows:
        lines.append("\t".join(fmt(row.get(field)) for field in fields))


def long_metric_rows(rows):
    output = []
    for row in rows:
        for metric, value in sorted(row["metrics"].items()):
            output.append(
                {
                    "task": row["task"],
                    "task_type": row["task_type"],
                    "fold": row.get("fold"),
                    "metric": metric,
                    "value": value,
                }
            )
    return output


def lookup_summary(summaries, task, metric):
    return next(
        (row for row in summaries if row["task"] == task and row["metric"] == metric),
        None,
    )


def flag_warnings(test_rows, warnings):
    flag_names = {
        "constant_predictions": "constant predicted class",
        "near_constant_probabilities": "near-constant class probabilities",
        "collapsed_predictions": "collapsed regression predictions",
        "all_background_prediction": "all-background segmentation prediction",
        "all_foreground_dice_zero": "zero Dice for every foreground class",
    }
    for row in test_rows:
        for metric, description in flag_names.items():
            if row["metrics"].get(metric) is True:
                warnings.append(f"{row['task']} fold {row['fold']}: {description}.")


def create_report(
    experiment_root,
    tasks,
    statuses,
    validation_rows,
    validation_summary,
    test_rows,
    test_summary,
    ensemble_rows,
    warnings,
):
    completed = sum(status["complete"] for status in statuses)
    lines = [
        "DINO3D FIVE-FOLD FINE-TUNING ANALYSIS",
        f"generated_utc\t{datetime.now(timezone.utc).isoformat()}",
        f"experiment_root\t{experiment_root}",
        f"completed_runs\t{completed}/{len(statuses)}",
        "",
        "INTERPRETATION",
        "- Validation summaries are mean, sample SD, and two-sided 95% Student-t intervals across folds.",
        "- Every fold is evaluated on the same fixed held-out test set. Test fold SD/CI therefore describe",
        "  variability among the five fold-trained models; they are not patient-level confidence intervals.",
        "- Classification and regression ensembles average predictions for identical fixed-test samples.",
        "- Segmentation cannot be ensembled from the saved metric JSON because voxel probabilities were not saved.",
        "- Select checkpoints and experimental choices using validation only; inspect the fixed test set afterward.",
    ]

    primary_rows = []
    for task, task_spec in tasks.items():
        val = lookup_summary(validation_summary, task, task_spec["validation_primary"])
        test = lookup_summary(test_summary, task, task_spec["test_primary"])
        ensemble = next((row for row in ensemble_rows if row["task"] == task), None)
        primary_rows.append(
            {
                "task": task,
                "task_type": task_spec["type"],
                "validation_metric": task_spec["validation_primary"],
                "validation_n": val["n"] if val else None,
                "validation_mean": val["mean"] if val else None,
                "validation_sd": val["sample_sd"] if val else None,
                "validation_ci95": (
                    f"[{fmt(val['ci95_low'])}, {fmt(val['ci95_high'])}]" if val else "NA"
                ),
                "test_metric": task_spec["test_primary"],
                "test_models": test["n"] if test else None,
                "test_mean": test["mean"] if test else None,
                "test_sd": test["sample_sd"] if test else None,
                "ensemble": (
                    ensemble["metrics"].get(task_spec["test_primary"])
                    if ensemble and ensemble["metrics"]
                    else None
                ),
            }
        )
    append_tsv(
        lines,
        "PRIMARY_RESULTS",
        [
            "task",
            "task_type",
            "validation_metric",
            "validation_n",
            "validation_mean",
            "validation_sd",
            "validation_ci95",
            "test_metric",
            "test_models",
            "test_mean",
            "test_sd",
            "ensemble",
        ],
        primary_rows,
    )
    append_tsv(
        lines,
        "RUN_STATUS",
        [
            "task",
            "task_type",
            "fold",
            "complete",
            "run_dir_exists",
            "best_ckpt",
            "last_ckpt",
            "periodic_ckpts",
            "validation_json",
            "prediction_json",
            "prediction_json_count",
        ],
        statuses,
    )
    append_tsv(
        lines,
        "VALIDATION_PER_FOLD",
        ["task", "task_type", "fold", "metric", "value"],
        long_metric_rows(validation_rows),
    )
    append_tsv(
        lines,
        "VALIDATION_CV_SUMMARY",
        ["task", "task_type", "metric", "n", "mean", "sample_sd", "ci95_low", "ci95_high", "median", "min", "max"],
        validation_summary,
    )
    append_tsv(
        lines,
        "FIXED_TEST_PER_FOLD",
        ["task", "task_type", "fold", "metric", "value"],
        long_metric_rows(test_rows),
    )
    append_tsv(
        lines,
        "FIXED_TEST_MODEL_VARIABILITY_SUMMARY",
        ["task", "task_type", "metric", "n", "mean", "sample_sd", "ci95_low", "ci95_high", "median", "min", "max"],
        test_summary,
    )
    ensemble_long = []
    for row in ensemble_rows:
        if row["metrics"]:
            for metric, value in sorted(row["metrics"].items()):
                ensemble_long.append({**row, "metric": metric, "value": value})
        else:
            ensemble_long.append({**row, "metric": "NA", "value": "NA"})
    append_tsv(
        lines,
        "FIXED_TEST_ENSEMBLE",
        ["task", "task_type", "n_models", "method", "metric", "value"],
        ensemble_long,
    )
    lines.extend(["", "[WARNINGS]"])
    if warnings:
        lines.extend(f"- {warning}" for warning in sorted(set(warnings)))
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "[STATUS]",
            f"DINO3D_CV_ANALYSIS_{'OK' if completed == len(statuses) else 'PARTIAL'} completed={completed}/{len(statuses)}",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    tasks = {name: TASKS[name] for name in (args.tasks or TASKS)}
    experiment_root = Path(args.experiment_root).resolve()
    report_path = Path(args.report).resolve()
    if not experiment_root.is_dir():
        raise SystemExit(f"Experiment root does not exist: {experiment_root}")
    if args.expected_folds < 1:
        raise SystemExit("--expected-folds must be positive")

    statuses, validation_rows, test_rows, payloads, warnings = inspect_runs(
        experiment_root, args.expected_folds, tasks
    )
    flag_warnings(test_rows, warnings)
    validation_summary = aggregate_rows(validation_rows, tasks)
    test_summary = aggregate_rows(test_rows, tasks)
    ensemble_rows = build_ensembles(payloads, warnings, tasks)
    report = create_report(
        experiment_root,
        tasks,
        statuses,
        validation_rows,
        validation_summary,
        test_rows,
        test_summary,
        ensemble_rows,
        warnings,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    completed = sum(status["complete"] for status in statuses)
    print(f"DINO3D_CV_ANALYSIS_{'OK' if completed == len(statuses) else 'PARTIAL'} completed={completed}/{len(statuses)}")
    print(f"report={report_path}")
    print("primary_results:")
    for task, task_spec in tasks.items():
        summary = lookup_summary(test_summary, task, task_spec["test_primary"])
        value = f"mean={fmt(summary['mean'])} sd={fmt(summary['sample_sd'])}" if summary else "not_available"
        print(f"  {task} fixed_test_{task_spec['test_primary']} {value}")

    if args.require_complete and completed != len(statuses):
        raise SystemExit(
            f"Incomplete experiment: {completed}/{len(statuses)} runs have best checkpoint, validation metrics, and test predictions"
        )


if __name__ == "__main__":
    main()
