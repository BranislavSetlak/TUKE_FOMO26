"""Gate and summarize the TUKE SwinUNETR one-case memorization result."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--required-class", action="append", required=True, type=int)
    parser.add_argument("--minimum-dice", default=0.90, type=float)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    with args.metrics.open() as handle:
        metrics = json.load(handle)
    numeric_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
    interesting = {
        key: value
        for key, value in numeric_metrics.items()
        if key.startswith("val/")
    }

    print("TUKE_SWINUNETR_ONE_CASE_METRICS")
    for key in sorted(interesting):
        print(f"{key}={interesting[key]:.10g}")

    required_keys = [f"val/dice_{class_id}" for class_id in args.required_class]
    missing = [key for key in required_keys if key not in numeric_metrics]
    class_scores = {key: numeric_metrics.get(key, 0.0) for key in required_keys}
    for key, score in class_scores.items():
        print(f"TUKE_SWINUNETR_ONE_CASE_CLASS key={key} dice={score:.6f}")

    score = min(class_scores.values()) if class_scores else 0.0
    passed = not missing and all(value >= args.minimum_dice for value in class_scores.values())
    predicted_fraction = numeric_metrics.get("val/pred_foreground_fraction")
    target_fraction = numeric_metrics.get("val/target_foreground_fraction")
    if predicted_fraction is not None and target_fraction is not None:
        ratio = predicted_fraction / target_fraction if target_fraction > 0 else float("inf")
        print(
            "TUKE_SWINUNETR_ONE_CASE_FOREGROUND "
            f"predicted={predicted_fraction:.8g} target={target_fraction:.8g} ratio={ratio:.6g}"
        )

    if passed:
        print(
            "TUKE_SWINUNETR_ONE_CASE_PASS "
            f"minimum_required_class_dice={score:.6f} minimum={args.minimum_dice:.6f}"
        )
        return

    if missing:
        print(f"TUKE_SWINUNETR_ONE_CASE_MISSING_METRICS keys={','.join(missing)}")
    print(
        "TUKE_SWINUNETR_ONE_CASE_WARNING "
        f"minimum_required_class_dice={score:.6f} minimum={args.minimum_dice:.6f}"
    )
    print(
        "Do not launch SwinUNETR segmentation cross-validation yet. Inspect the prepared ROI, "
        "checkpoint transfer counts, gradients, loss curve, and foreground-fraction diagnostics."
    )
    if args.strict:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
