"""Write one text report for normal and GIN SwinUNETR cls/reg CV."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


TASKS = {
    "CLS002_FOMO26_Infarct": "classification",
    "REGR002_FOMO26_BrainAge": "regression",
    "CLS003_FOMO26_Polymicrogyria": "classification",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-root", required=True)
    parser.add_argument("--gin-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prediction_file(fold_root):
    matches = sorted((fold_root / "predictions").glob("*__best.json"))
    return matches[0] if matches else None


def records(data):
    return [
        value
        for key, value in data.items()
        if key != "metrics" and isinstance(value, dict) and "prediction" in value and "label" in value
    ]


def classification_metrics(items):
    labels = np.asarray([int(item["label"]) for item in items])
    predictions = np.asarray([int(item["prediction"]) for item in items])
    scores = np.asarray([float(item["probabilities"][1]) for item in items])
    result = {
        "n_test": float(labels.size),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "predicted_positive_fraction": float(np.mean(predictions == 1)),
    }
    result["roc_auc"] = float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")
    return result


def regression_metrics(items):
    labels = np.asarray([float(item["label"]) for item in items])
    predictions = np.asarray([float(item["prediction"]) for item in items])
    errors = predictions - labels
    mse = float(np.mean(errors**2))
    total = float(np.sum((labels - labels.mean()) ** 2))
    return {
        "n_test": float(labels.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": math.sqrt(mse),
        "r2": 1.0 - float(np.sum(errors**2)) / total if total else float("nan"),
        "pearson_r": float(np.corrcoef(labels, predictions)[0, 1])
        if labels.size > 1 and labels.std() > 0 and predictions.std() > 0
        else float("nan"),
        "prediction_mean": float(predictions.mean()),
        "prediction_std": float(predictions.std()),
    }


def format_number(value):
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if not math.isfinite(float(value)):
        return "NaN"
    return f"{float(value):.8g}"


def main():
    args = arguments()
    roots = {"normal": Path(args.normal_root), "gin": Path(args.gin_root)}
    rows = []
    missing = []
    for variant, root in roots.items():
        for task, task_type in TASKS.items():
            for fold in range(args.expected_folds):
                fold_root = root / task / f"fold_{fold}"
                prediction = prediction_file(fold_root)
                validation = fold_root / "validation_best_metrics.json"
                if prediction is None or not validation.is_file():
                    missing.append(f"{variant}\t{task}\tfold_{fold}\t{fold_root}")
                    continue
                prediction_data = load_json(prediction)
                item_rows = records(prediction_data)
                metrics = (
                    classification_metrics(item_rows)
                    if task_type == "classification"
                    else regression_metrics(item_rows)
                )
                validation_data = load_json(validation)
                for key, value in validation_data.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics[f"validation:{key}"] = float(value)
                rows.append({"variant": variant, "task": task, "fold": fold, **metrics})

    metric_names = sorted({key for row in rows for key in row if key not in {"variant", "task", "fold"}})
    lines = [
        "TUKE_SWINUNETR_CLASSIFICATION_REGRESSION_CV",
        f"normal_root\t{roots['normal']}",
        f"gin_root\t{roots['gin']}",
        "",
        "PER_FOLD",
        "variant\ttask\tfold\t" + "\t".join(metric_names),
    ]
    for row in rows:
        lines.append(
            f"{row['variant']}\t{row['task']}\t{row['fold']}\t"
            + "\t".join(format_number(row.get(metric)) for metric in metric_names)
        )

    lines.extend(["", "MEAN_AND_SAMPLE_SD", "variant\ttask\tmetric\tn\tmean\tsample_sd"])
    for variant in roots:
        for task in TASKS:
            task_rows = [row for row in rows if row["variant"] == variant and row["task"] == task]
            for metric in metric_names:
                values = np.asarray(
                    [float(row[metric]) for row in task_rows if metric in row and math.isfinite(float(row[metric]))]
                )
                if not values.size:
                    continue
                sample_sd = float(values.std(ddof=1)) if values.size > 1 else float("nan")
                lines.append(
                    f"{variant}\t{task}\t{metric}\t{values.size}\t"
                    f"{format_number(values.mean())}\t{format_number(sample_sd)}"
                )

    lines.extend(["", "MISSING", "variant\ttask\tfold\tpath"])
    lines.extend(missing or ["none"])
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"TUKE_SWINUNETR_CLSREG_ANALYSIS_WRITTEN report={output} rows={len(rows)} missing={len(missing)}")
    if args.require_complete and missing:
        raise SystemExit(f"Incomplete experiment: {len(missing)} fold outputs are missing")


if __name__ == "__main__":
    main()
