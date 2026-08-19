"""Summarize the one-case memorization diagnostic."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--minimum-dice", type=float, default=0.90)
    parser.add_argument("--required-class", action="append", default=[], type=int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    with args.metrics.open() as handle:
        metrics = json.load(handle)
    interesting = {
        key: float(value)
        for key, value in metrics.items()
        if key == "val/foreground_dice"
        or key == "val/best_threshold_dice"
        or key.startswith("val/dice_")
        or key.startswith("val/foreground_dice_t")
        or key.startswith("val/foreground_recall_t")
        or key in {
            "val/pred_foreground_fraction",
            "val/target_foreground_fraction",
            "val/target_foreground_probability",
            "val/max_foreground_probability",
        }
    }
    print("ONE_CASE_OVERFIT_METRICS")
    for key in sorted(interesting):
        print(f"{key}={interesting[key]:.10g}")

    required_keys = [f"val/dice_{class_id}" for class_id in args.required_class]
    if required_keys:
        missing = [key for key in required_keys if key not in interesting]
        class_scores = {key: interesting.get(key, 0.0) for key in required_keys}
        for key, value in class_scores.items():
            print(f"ONE_CASE_OVERFIT_CLASS key={key} dice={value:.6f}")
        score = min(class_scores.values())
        score_name = "minimum_required_class_dice"
        passed = not missing and all(
            value >= args.minimum_dice for value in class_scores.values()
        )
    else:
        missing = []
        score = interesting.get("val/foreground_dice", 0.0)
        score_name = "foreground_dice"
        passed = score >= args.minimum_dice

    if passed:
        print(
            f"ONE_CASE_OVERFIT_PASS {score_name}={score:.6f} "
            f"minimum={args.minimum_dice:.6f}"
        )
        return

    if missing:
        print(f"ONE_CASE_OVERFIT_MISSING_METRICS keys={','.join(missing)}")
    print(
        f"ONE_CASE_OVERFIT_WARNING {score_name}={score:.6f} "
        f"minimum={args.minimum_dice:.6f}"
    )
    print(
        "The complete train-to-mask path could not memorize one positive ROI. "
        "Do not launch full CV until labels, transforms, resolution, and gradients are inspected."
    )
    if args.strict:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
