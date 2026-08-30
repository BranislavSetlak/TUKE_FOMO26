"""Create one text report for three TUKE SwinUNETR five-fold variants."""

import argparse
import json
import math
import statistics
from pathlib import Path


TASKS = (
    "SEG009_FOMO26_Meningioma",
    "SEG010_FOMO26_TrigeminalNeuralgia",
)
METRICS = (
    ("val/foreground_dice", "FgDice"),
    ("val/macro_foreground_class_dice", "MacroCls"),
    ("val/min_foreground_class_dice", "MinCls"),
    ("val/exact_dice_1", "ExactC1"),
    ("val/exact_dice_2", "ExactC2"),
    ("val/F1_1", "F1-C1"),
    ("val/F1_2", "F1-C2"),
    ("val/pred_foreground_fraction", "PredFrac"),
    ("val/target_foreground_fraction", "TgtFrac"),
    ("val/false_positive_fraction", "FPFrac"),
    ("val/pred_to_target_volume_ratio", "VolRatio"),
    ("val/loss", "Loss"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-root", required=True)
    parser.add_argument("--gin-root", required=True)
    parser.add_argument("--gin-carvemix-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def flatten_numbers(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numbers(child, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(flatten_numbers(child, f"{prefix}[{index}]"))
    elif finite_number(value):
        result[prefix] = float(value)
    return result


def load_metrics(path):
    with path.open("r", encoding="utf-8") as handle:
        return flatten_numbers(json.load(handle))


def checkpoint(run_dir, name):
    candidates = sorted(run_dir.rglob(name)) if run_dir.is_dir() else []
    return candidates[0] if candidates else None


def inspect_variant(name, root, expected_folds):
    rows = {task: [] for task in TASKS}
    status = []
    for task in TASKS:
        for fold in range(expected_folds):
            run_dir = root / task / f"fold_{fold}"
            metrics_path = run_dir / "validation_best_metrics.json"
            best = checkpoint(run_dir, "best.ckpt")
            last = checkpoint(run_dir, "last.ckpt")
            periodic = sorted(run_dir.rglob("periodic-*.ckpt")) if run_dir.is_dir() else []
            complete = bool(
                metrics_path.is_file()
                and metrics_path.stat().st_size > 0
                and best
                and best.stat().st_size > 0
                and last
                and last.stat().st_size > 0
            )
            metrics = {}
            error = ""
            if metrics_path.is_file():
                try:
                    metrics = load_metrics(metrics_path)
                except Exception as exception:
                    error = str(exception)
            rows[task].append({"fold": fold, "metrics": metrics, "complete": complete})
            status.append(
                {
                    "variant": name,
                    "task": task,
                    "fold": fold,
                    "complete": complete,
                    "best": str(best) if best else "MISSING",
                    "last": str(last) if last else "MISSING",
                    "periodic": len(periodic),
                    "metrics": str(metrics_path) if metrics_path.is_file() else "MISSING",
                    "error": error,
                }
            )
    return rows, status


def fmt(value):
    return "NA" if value is None or not finite_number(value) else f"{float(value):.6f}"


def values_for(rows, metric):
    return [row["metrics"].get(metric) for row in rows if finite_number(row["metrics"].get(metric))]


def summary(values, kind):
    if not values:
        return None
    if kind == "mean":
        return statistics.fmean(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def make_table(rows):
    header = ["Fold"] + [label for _, label in METRICS]
    body = []
    for row in rows:
        body.append([str(row["fold"])] + [fmt(row["metrics"].get(metric)) for metric, _ in METRICS])
    body.append(["MEAN"] + [fmt(summary(values_for(rows, metric), "mean")) for metric, _ in METRICS])
    body.append(["SD"] + [fmt(summary(values_for(rows, metric), "sd")) for metric, _ in METRICS])

    widths = [max(len(header[index]), *(len(row[index]) for row in body)) for index in range(len(header))]
    rule = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [rule]
    lines.append("|" + "|".join(f" {value:<{widths[index]}} " for index, value in enumerate(header)) + "|")
    lines.append(rule)
    for row in body:
        lines.append("|" + "|".join(f" {value:<{widths[index]}} " for index, value in enumerate(row)) + "|")
    lines.append(rule)
    return lines


def main():
    args = parse_args()
    variants = {
        "normal": Path(args.normal_root),
        "gin": Path(args.gin_root),
        "gin_carvemix": Path(args.gin_carvemix_root),
    }
    all_rows = {}
    statuses = []
    for name, root in variants.items():
        rows, status = inspect_variant(name, root, args.expected_folds)
        all_rows[name] = rows
        statuses.extend(status)

    lines = [
        "TUKE SWINUNETR SEGMENTATION VARIANT REPORT",
        "",
        "All values come from each fold's best validation checkpoint.",
        "Best checkpoint monitor: val/min_foreground_class_dice.",
        "",
    ]
    for variant in variants:
        for task in TASKS:
            readable = "Meningioma Segmentation" if task.startswith("SEG009") else "Trigeminal Neuralgia Segmentation"
            lines.extend([f"{variant.upper()} - {task} - {readable}", ""])
            lines.extend(make_table(all_rows[variant][task]))
            lines.append("")

    lines.extend(["PRIMARY COMPARISON", "", "Variant\tTask\tN folds\tMean MinCls\tDelta vs normal"])
    normal_means = {}
    for task in TASKS:
        normal_means[task] = summary(values_for(all_rows["normal"][task], "val/min_foreground_class_dice"), "mean")
    for variant in variants:
        for task in TASKS:
            values = values_for(all_rows[variant][task], "val/min_foreground_class_dice")
            mean = summary(values, "mean")
            baseline = normal_means[task]
            delta = mean - baseline if mean is not None and baseline is not None else None
            lines.append(f"{variant}\t{task}\t{len(values)}\t{fmt(mean)}\t{fmt(delta)}")

    lines.extend(["", "RUN COMPLETENESS", "", "Variant\tTask\tFold\tComplete\tPeriodic\tBest\tLast\tMetrics\tError"])
    for status in statuses:
        lines.append(
            "\t".join(
                [
                    status["variant"],
                    status["task"],
                    str(status["fold"]),
                    str(status["complete"]).lower(),
                    str(status["periodic"]),
                    status["best"],
                    status["last"],
                    status["metrics"],
                    status["error"].replace("\t", " ").replace("\n", " "),
                ]
            )
        )

    missing = [status for status in statuses if not status["complete"]]
    lines.extend(["", f"COMPLETE_RUNS={len(statuses) - len(missing)}/{len(statuses)}"])
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"TUKE_SWINUNETR_VARIANT_ANALYSIS_FINISHED report={report}")
    if args.require_complete and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
