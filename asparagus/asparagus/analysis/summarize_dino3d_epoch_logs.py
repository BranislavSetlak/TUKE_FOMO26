"""Create one text report containing seven fold-averaged epoch tables.

The input files are the stdout files produced by the DINO3D GIN and
GIN+CarveMix Slurm arrays. Each task has five folds. For every epoch and metric,
the table reports the arithmetic mean across the folds that logged a finite
value. The final AVERAGE row is the arithmetic mean of the displayed epoch
means, ignoring unavailable epochs.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:inf|nan)"
LOG_VALUE_RE = re.compile(
    rf"^\d{{4}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}\s+"
    rf"(?P<name>[^:]+):\s*(?P<value>{NUMBER})\s*$",
    flags=re.IGNORECASE,
)
PROGRESS_EPOCH_RE = re.compile(r"^Epoch\s+(?P<epoch>\d+):")


@dataclass(frozen=True)
class TableSpec:
    title: str
    variant: str
    array_indices: tuple[int, ...]
    expected_epochs: int


TABLES = (
    TableSpec(
        "CLS002 - Infarct Classification (GIN)",
        "gin",
        tuple(range(0, 5)),
        50,
    ),
    TableSpec(
        "SEG009 - Meningioma Segmentation (GIN)",
        "gin",
        tuple(range(5, 10)),
        150,
    ),
    TableSpec(
        "REGR002 - Brain Age Regression (GIN)",
        "gin",
        tuple(range(10, 15)),
        50,
    ),
    TableSpec(
        "SEG010 - Trigeminal Neuralgia Segmentation (GIN)",
        "gin",
        tuple(range(15, 20)),
        150,
    ),
    TableSpec(
        "CLS003 - Polymicrogyria Classification (GIN)",
        "gin",
        tuple(range(20, 25)),
        50,
    ),
    TableSpec(
        "SEG009 - Meningioma Segmentation (GIN + CarveMix)",
        "gin_carvemix",
        tuple(range(0, 5)),
        150,
    ),
    TableSpec(
        "SEG010 - Trigeminal Neuralgia Segmentation (GIN + CarveMix)",
        "gin_carvemix",
        tuple(range(5, 10)),
        150,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/mnt/project/perun2601396/FOMO26_job_outputs"),
        help="Directory containing the Slurm .out files",
    )
    parser.add_argument("--gin-job-id", required=True, help="GIN array job ID")
    parser.add_argument(
        "--carvemix-job-id",
        required=True,
        help="GIN+CarveMix array job ID",
    )
    parser.add_argument("--output", type=Path, required=True, help="Single report to create")
    parser.add_argument(
        "--require-all-logs",
        action="store_true",
        help="Fail if any of the expected 35 array-element logs is missing",
    )
    return parser.parse_args()


def log_path(log_dir: Path, variant: str, job_id: str, array_index: int) -> Path:
    if variant == "gin":
        prefix = "dino3d-gin-cv"
    elif variant == "gin_carvemix":
        prefix = "dino3d-gin-carvemix"
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return log_dir / f"{prefix}_{job_id}_{array_index}.out"


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    """Return epoch -> metric -> last finite or non-finite logged value."""
    epochs: dict[int, dict[str, float]] = defaultdict(dict)
    current_epoch: int | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip().replace("\r", "")
            progress_match = PROGRESS_EPOCH_RE.match(line)
            if progress_match:
                current_epoch = int(progress_match.group("epoch"))

            value_match = LOG_VALUE_RE.match(line)
            if not value_match:
                continue

            name = value_match.group("name").strip()
            value = float(value_match.group("value"))
            if name == "Current Epoch":
                if math.isfinite(value):
                    current_epoch = int(value)
                continue
            if current_epoch is None or current_epoch < 0:
                continue
            epochs[current_epoch][name] = value

    return dict(epochs)


def metric_sort_key(name: str) -> tuple[int, str]:
    if name == "Epoch Time":
        group = 0
    elif name.startswith("lr-"):
        group = 1
    elif name == "train/loss":
        group = 2
    elif name.startswith("train/"):
        group = 3
    elif name == "val/loss":
        group = 4
    elif name.startswith("val/"):
        group = 5
    else:
        group = 6
    return group, name.lower()


def format_number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    if value == 0:
        return "0"
    absolute = abs(value)
    if absolute >= 1e5 or absolute < 1e-4:
        return f"{value:.6e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def aggregate_table(
    spec: TableSpec,
    fold_logs: list[dict[int, dict[str, float]]],
) -> tuple[list[str], list[dict[str, float]], list[int]]:
    metric_names = sorted(
        {
            metric
            for fold_log in fold_logs
            for epoch_metrics in fold_log.values()
            for metric in epoch_metrics
        },
        key=metric_sort_key,
    )

    epoch_rows: list[dict[str, float]] = []
    fold_counts: list[int] = []
    for epoch in range(spec.expected_epochs):
        row: dict[str, float] = {}
        fold_counts.append(sum(epoch in fold_log for fold_log in fold_logs))
        for metric in metric_names:
            values = [
                fold_log[epoch][metric]
                for fold_log in fold_logs
                if epoch in fold_log
                and metric in fold_log[epoch]
                and math.isfinite(fold_log[epoch][metric])
            ]
            if values:
                row[metric] = fmean(values)
        epoch_rows.append(row)

    return metric_names, epoch_rows, fold_counts


def average_row(metric_names: list[str], epoch_rows: list[dict[str, float]]) -> dict[str, float]:
    result = {}
    for metric in metric_names:
        values = [row[metric] for row in epoch_rows if metric in row and math.isfinite(row[metric])]
        if values:
            result[metric] = fmean(values)
    return result


def write_table(
    handle,
    spec: TableSpec,
    paths: list[Path],
    fold_logs: list[dict[int, dict[str, float]]],
) -> None:
    metric_names, epoch_rows, fold_counts = aggregate_table(spec, fold_logs)
    observed_epochs = sum(count > 0 for count in fold_counts)
    completed_epochs = sum(count == len(spec.array_indices) for count in fold_counts)

    handle.write("=" * 120 + "\n")
    handle.write(spec.title + "\n")
    handle.write("=" * 120 + "\n")
    handle.write(
        f"Fold logs found: {len(paths)}/{len(spec.array_indices)} | "
        f"epochs with any fold: {observed_epochs}/{spec.expected_epochs} | "
        f"epochs with all five folds: {completed_epochs}/{spec.expected_epochs}\n"
    )
    handle.write("Per-epoch metric cells are arithmetic means across available folds.\n")
    handle.write("AVERAGE is the arithmetic mean of the displayed finite epoch means.\n\n")

    columns = ["Epoch", "Folds"] + metric_names
    handle.write("\t".join(columns) + "\n")
    for epoch, (row, fold_count) in enumerate(zip(epoch_rows, fold_counts)):
        values = [str(epoch), str(fold_count)]
        values.extend(format_number(row.get(metric)) for metric in metric_names)
        handle.write("\t".join(values) + "\n")

    summary = average_row(metric_names, epoch_rows)
    values = ["AVERAGE", "NA"]
    values.extend(format_number(summary.get(metric)) for metric in metric_names)
    handle.write("\t".join(values) + "\n\n")


def main() -> None:
    args = parse_args()
    if not args.log_dir.is_dir():
        raise SystemExit(f"Log directory does not exist: {args.log_dir}")

    job_ids = {
        "gin": str(args.gin_job_id),
        "gin_carvemix": str(args.carvemix_job_id),
    }
    table_inputs: list[tuple[TableSpec, list[Path], list[dict[int, dict[str, float]]]]] = []
    missing_paths: list[Path] = []

    for spec in TABLES:
        paths = []
        parsed = []
        for array_index in spec.array_indices:
            path = log_path(args.log_dir, spec.variant, job_ids[spec.variant], array_index)
            if not path.is_file():
                missing_paths.append(path)
                continue
            paths.append(path)
            parsed.append(parse_log(path))
        table_inputs.append((spec, paths, parsed))

    if args.require_all_logs and missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise SystemExit(f"Missing {len(missing_paths)} expected log files:\n{missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("DINO3D GIN AND GIN+CARVEMIX PER-EPOCH METRIC TABLES\n")
        handle.write(f"GIN array job ID: {job_ids['gin']}\n")
        handle.write(f"GIN+CarveMix array job ID: {job_ids['gin_carvemix']}\n")
        handle.write(f"Log directory: {args.log_dir}\n")
        handle.write(f"Missing expected logs: {len(missing_paths)}\n")
        if missing_paths:
            for path in missing_paths:
                handle.write(f"  MISSING: {path.name}\n")
        handle.write("\n")

        for spec, paths, parsed in table_inputs:
            write_table(handle, spec, paths, parsed)

    print(f"DINO3D_EPOCH_TABLES_OK report={args.output}")


if __name__ == "__main__":
    main()
