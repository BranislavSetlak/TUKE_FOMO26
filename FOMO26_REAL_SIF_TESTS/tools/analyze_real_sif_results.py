#!/usr/bin/env python3
"""Create one readable report from Tasks 1-5 real-SIF test results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def binary_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def classification_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    labels = [int(case["target"]) for case in cases]
    for mode in ["no_tta", "tta"]:
        scores = [float(case["analysis"]["predictions"][mode]) for case in cases]
        predictions = [int(score >= 0.5) for score in scores]
        correct = [pred == label for pred, label in zip(predictions, labels)]
        sensitivity_values = [pred for pred, label in zip(predictions, labels) if label == 1]
        specificity_values = [1 - pred for pred, label in zip(predictions, labels) if label == 0]
        sensitivity = mean([float(value) for value in sensitivity_values])
        specificity = mean([float(value) for value in specificity_values])
        summary[mode] = {
            "n": len(labels),
            "accuracy": mean([float(value) for value in correct]),
            "auc": binary_auc(labels, scores),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": (
                (sensitivity + specificity) / 2.0
                if sensitivity is not None and specificity is not None
                else None
            ),
            "mean_probability": mean(scores),
        }
    return summary


def regression_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    targets = [float(case["target"]) for case in cases]
    for mode in ["no_tta", "tta"]:
        predictions = [float(case["analysis"]["predictions"][mode]) for case in cases]
        errors = [pred - target for pred, target in zip(predictions, targets)]
        summary[mode] = {
            "n": len(targets),
            "mae": mean([abs(value) for value in errors]),
            "rmse": math.sqrt(mean([value * value for value in errors]) or 0.0),
            "bias": mean(errors),
            "mean_prediction": mean(predictions),
            "mean_target": mean(targets),
        }
    return summary


def segmentation_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in ["no_tta", "tta"]:
        rows = [case["analysis"]["segmentations"][mode] for case in cases]
        dice = [
            float(row["foreground_macro_dice"])
            for row in rows
            if row.get("foreground_macro_dice") is not None
        ]
        summary[mode] = {
            "n": len(rows),
            "mean_foreground_macro_dice": mean(dice),
            "minimum_foreground_macro_dice": min(dice) if dice else None,
            "nonempty_predictions": sum(row["pred_foreground_voxels"] > 0 for row in rows),
            "empty_on_positive": sum(
                "EMPTY_PREDICTION_ON_POSITIVE_CASE" in row.get("warnings", [])
                for row in rows
            ),
            "geometry_failures": sum(
                not row["shape_matches_reference"] or not row["affine_matches_reference"]
                for row in rows
            ),
            "invalid_label_failures": sum(
                not row["integer_output"] or not row["valid_labels"] for row in rows
            ),
            "mean_pred_foreground_fraction": mean(
                [float(row["pred_foreground_fraction"]) for row in rows]
            ),
            "mean_target_foreground_fraction": mean(
                [float(row["target_foreground_fraction"]) for row in rows]
            ),
        }
    agreements = [
        float(case["analysis"]["tta_no_tta_foreground_agreement"])
        for case in cases
        if case["analysis"].get("tta_no_tta_foreground_agreement") is not None
    ]
    summary["mean_tta_no_tta_foreground_agreement"] = mean(agreements)
    return summary


def tta_recommendation(kind: str, summary: dict[str, Any]) -> tuple[str, str]:
    no_tta = summary["no_tta"]
    tta = summary["tta"]
    if kind == "segmentation":
        no_metric = no_tta["mean_foreground_macro_dice"]
        tta_metric = tta["mean_foreground_macro_dice"]
        if no_metric is None or tta_metric is None:
            return "INCONCLUSIVE", "Dice could not be calculated."
        delta = tta_metric - no_metric
        if delta < -0.02:
            return "DISABLE_OR_REWORK_TTA", f"Mean Dice changed by {delta:.4f}."
        if delta > 0.02:
            return "KEEP_TTA", f"Mean Dice changed by +{delta:.4f}."
        return "NO_CLEAR_DIFFERENCE", f"Mean Dice changed by {delta:+.4f}."
    if kind == "regression":
        delta = tta["mae"] - no_tta["mae"]
        if delta > 0.5:
            return "DISABLE_OR_REWORK_TTA", f"MAE worsened by {delta:.3f} years."
        if delta < -0.5:
            return "KEEP_TTA", f"MAE improved by {-delta:.3f} years."
        return "NO_CLEAR_DIFFERENCE", f"MAE changed by {delta:+.3f} years."
    no_metric = no_tta["balanced_accuracy"]
    tta_metric = tta["balanced_accuracy"]
    if no_metric is None or tta_metric is None:
        return "INCONCLUSIVE", "Both classes were not represented."
    delta = tta_metric - no_metric
    if delta < -0.05:
        return "DISABLE_OR_REWORK_TTA", f"Balanced accuracy changed by {delta:.3f}."
    if delta > 0.05:
        return "KEEP_TTA", f"Balanced accuracy changed by +{delta:.3f}."
    return "NO_CLEAR_DIFFERENCE", f"Balanced accuracy changed by {delta:+.3f}."


def case_table(task: int, kind: str, cases: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    if kind == "classification":
        lines.append("case | label | no-TTA prob | TTA prob | no-TTA correct | TTA correct")
        lines.append("-" * 88)
        for case in cases:
            analysis = case["analysis"]
            lines.append(
                f"{case['case_id']} | {case['target']} | "
                f"{fmt(analysis['predictions']['no_tta'])} | {fmt(analysis['predictions']['tta'])} | "
                f"{fmt(analysis['correct']['no_tta'])} | {fmt(analysis['correct']['tta'])}"
            )
    elif kind == "regression":
        lines.append("case | age | no-TTA prediction | TTA prediction | no-TTA AE | TTA AE")
        lines.append("-" * 88)
        for case in cases:
            analysis = case["analysis"]
            lines.append(
                f"{case['case_id']} | {fmt(case['target'], 2)} | "
                f"{fmt(analysis['predictions']['no_tta'], 2)} | "
                f"{fmt(analysis['predictions']['tta'], 2)} | "
                f"{fmt(analysis['absolute_errors']['no_tta'], 2)} | "
                f"{fmt(analysis['absolute_errors']['tta'], 2)}"
            )
    else:
        lines.append(
            "case | target voxels | no-TTA voxels | TTA voxels | no-TTA Dice | TTA Dice | TTA/no-TTA agreement"
        )
        lines.append("-" * 120)
        for case in cases:
            analysis = case["analysis"]
            no_tta = analysis["segmentations"]["no_tta"]
            tta = analysis["segmentations"]["tta"]
            lines.append(
                f"{case['case_id']} | {no_tta['target_foreground_voxels']} | "
                f"{no_tta['pred_foreground_voxels']} | {tta['pred_foreground_voxels']} | "
                f"{fmt(no_tta['foreground_macro_dice'])} | "
                f"{fmt(tta['foreground_macro_dice'])} | "
                f"{fmt(analysis.get('tta_no_tta_foreground_agreement'))}"
            )
    return lines


def main() -> int:
    args = parse_args()
    reports: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    for task in range(1, 6):
        path = args.results_root / f"task{task}" / f"task{task}_results.json"
        if not path.is_file():
            missing.append(task)
            continue
        with path.open("r", encoding="utf-8") as handle:
            reports[task] = json.load(handle)

    lines = [
        "FOMO26 REAL FINETUNING-CASE SIF EVALUATION",
        f"results_root={args.results_root}",
        "",
        "This is a small operational sanity test, not the official challenge score.",
        "TTA means the container default; no-TTA uses the --no-tta flag.",
        "",
    ]
    summary_json: dict[str, Any] = {"results_root": str(args.results_root), "tasks": {}}
    names = {
        1: "Infarct classification",
        2: "Meningioma segmentation",
        3: "Brain-age regression",
        4: "Trigeminal neuralgia segmentation",
        5: "Polymicrogyria classification",
    }
    for task in range(1, 6):
        lines.extend(["=" * 120, f"TASK {task}: {names[task]}", "=" * 120])
        if task not in reports:
            lines.extend(["RESULT: MISSING", ""])
            continue
        report = reports[task]
        valid_cases = [
            case
            for case in report["cases"]
            if case.get("analysis", {}).get("both_modes_succeeded")
        ]
        fallback = bool(report["discovery"].get("fallback_used"))
        lines.append(
            f"cases={len(valid_cases)}/{report['cases_selected']} "
            f"held_out={not fallback} fallback_used={fallback}"
        )
        if fallback:
            lines.append(
                "WARNING: cases came from the full raw finetuning dataset, so metrics are not an unbiased test estimate."
            )
        failed_cases = report["cases_selected"] - len(valid_cases)
        if not valid_cases:
            lines.extend(["RESULT: NO SUCCESSFUL CASES", ""])
            summary_json["tasks"][str(task)] = {"status": "FAILED", "failed_cases": failed_cases}
            continue
        lines.extend(case_table(task, report["kind"], valid_cases))
        if report["kind"] == "classification":
            metrics = classification_summary(valid_cases)
            lines.extend(
                [
                    "",
                    "Aggregate metrics:",
                    f"no-TTA: accuracy={fmt(metrics['no_tta']['accuracy'])} AUC={fmt(metrics['no_tta']['auc'])} "
                    f"balanced_accuracy={fmt(metrics['no_tta']['balanced_accuracy'])}",
                    f"TTA:    accuracy={fmt(metrics['tta']['accuracy'])} AUC={fmt(metrics['tta']['auc'])} "
                    f"balanced_accuracy={fmt(metrics['tta']['balanced_accuracy'])}",
                ]
            )
        elif report["kind"] == "regression":
            metrics = regression_summary(valid_cases)
            lines.extend(
                [
                    "",
                    "Aggregate metrics:",
                    f"no-TTA: MAE={fmt(metrics['no_tta']['mae'], 3)} RMSE={fmt(metrics['no_tta']['rmse'], 3)} "
                    f"bias={fmt(metrics['no_tta']['bias'], 3)}",
                    f"TTA:    MAE={fmt(metrics['tta']['mae'], 3)} RMSE={fmt(metrics['tta']['rmse'], 3)} "
                    f"bias={fmt(metrics['tta']['bias'], 3)}",
                ]
            )
        else:
            metrics = segmentation_summary(valid_cases)
            lines.extend(
                [
                    "",
                    "Aggregate metrics:",
                    f"no-TTA: mean_Dice={fmt(metrics['no_tta']['mean_foreground_macro_dice'])} "
                    f"minimum_Dice={fmt(metrics['no_tta']['minimum_foreground_macro_dice'])} "
                    f"nonempty={metrics['no_tta']['nonempty_predictions']}/{metrics['no_tta']['n']} "
                    f"empty_on_positive={metrics['no_tta']['empty_on_positive']} "
                    f"geometry_failures={metrics['no_tta']['geometry_failures']}",
                    f"TTA:    mean_Dice={fmt(metrics['tta']['mean_foreground_macro_dice'])} "
                    f"minimum_Dice={fmt(metrics['tta']['minimum_foreground_macro_dice'])} "
                    f"nonempty={metrics['tta']['nonempty_predictions']}/{metrics['tta']['n']} "
                    f"empty_on_positive={metrics['tta']['empty_on_positive']} "
                    f"geometry_failures={metrics['tta']['geometry_failures']}",
                    f"mean TTA/no-TTA foreground agreement="
                    f"{fmt(metrics['mean_tta_no_tta_foreground_agreement'])}",
                ]
            )
        recommendation, reason = tta_recommendation(report["kind"], metrics)
        technical_status = "PASS" if failed_cases == 0 else "PARTIAL_FAILURE"
        lines.extend(
            [
                f"TECHNICAL_STATUS={technical_status}",
                f"DEFAULT_TTA_RECOMMENDATION={recommendation}",
                f"TTA_REASON={reason}",
                "",
            ]
        )
        summary_json["tasks"][str(task)] = {
            "name": names[task],
            "kind": report["kind"],
            "held_out": not fallback,
            "successful_cases": len(valid_cases),
            "failed_cases": failed_cases,
            "metrics": metrics,
            "tta_recommendation": recommendation,
            "tta_reason": reason,
        }

    lines.extend(
        [
            "=" * 120,
            f"MISSING_TASK_RESULTS={','.join(map(str, missing)) if missing else 'none'}",
            "Paste this complete report into the chat for interpretation before changing or rebuilding any SIF.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_output = args.json_output or args.output.with_suffix(".json")
    json_output.write_text(json.dumps(summary_json, indent=2, sort_keys=True), encoding="utf-8")
    print(f"REAL_SIF_ANALYSIS_FINISHED report={args.output} json={json_output}")
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
