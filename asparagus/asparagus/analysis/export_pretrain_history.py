"""Export the rank-zero SSL_METRICS stream to graph-ready CSV files."""

import argparse
import ast
import csv
import math
import re
from collections import deque
from pathlib import Path


METRIC_RE = re.compile(r"SSL_METRICS\s+step=(\d+)\s+(\{.*\})")


def parse_inputs(paths):
    rows = {}
    sources = {}
    for path_text in paths:
        path = Path(path_text)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                match = METRIC_RE.search(line)
                if not match:
                    continue
                step = int(match.group(1))
                try:
                    values = ast.literal_eval(match.group(2))
                except (SyntaxError, ValueError) as exc:
                    raise RuntimeError(f"Cannot parse {path}:{line_number}: {exc}") from exc
                numeric = {}
                for key, value in values.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        numeric[str(key)] = float(value)
                # Repeated rank-zero lines or resumed logs are expected. The
                # last occurrence is retained and reported in the source field.
                rows[step] = numeric
                sources[step] = str(path)
    return rows, sources


def add_moving_averages(rows, metric_names, window):
    buffers = {name: deque(maxlen=window) for name in metric_names}
    sums = {name: 0.0 for name in metric_names}
    for row in rows:
        for name in metric_names:
            buffer = buffers[name]
            if len(buffer) == window:
                sums[name] -= buffer[0]
            value = row.get(name)
            if value is not None:
                buffer.append(value)
                sums[name] += value
            row[f"{name}_ma{window}"] = sums[name] / len(buffer) if buffer else ""


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Pretrain log and/or Slurm stdout files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--dataset-size", type=int, default=303144)
    parser.add_argument("--moving-average", type=int, default=100)
    args = parser.parse_args()

    parsed, sources = parse_inputs(args.inputs)
    if not parsed:
        raise SystemExit("No SSL_METRICS lines were found in the supplied files.")

    metric_names = sorted({key for values in parsed.values() for key in values})
    rows = []
    for step in sorted(parsed):
        samples_seen = step * args.global_batch_size
        row = {
            "step": step,
            "epoch": 0 if step <= 0 else (step - 1) // args.steps_per_epoch,
            "step_in_epoch": 0 if step <= 0 else ((step - 1) % args.steps_per_epoch) + 1,
            "samples_seen": samples_seen,
            "dataset_equivalent_passes": samples_seen / args.dataset_size,
            "source_file": sources[step],
        }
        row.update(parsed[step])
        rows.append(row)

    add_moving_averages(rows, metric_names, args.moving_average)
    fixed = ["step", "epoch", "step_in_epoch", "samples_seen", "dataset_equivalent_passes"]
    moving = [f"{name}_ma{args.moving_average}" for name in metric_names]
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "dino3d_pretrain_history.csv", rows, fixed + metric_names + moving + ["source_file"])

    summary_rows = []
    for name in metric_names:
        values = [(row["step"], row[name]) for row in rows if name in row]
        if not values:
            continue
        minimum = min(values, key=lambda item: item[1])
        maximum = max(values, key=lambda item: item[1])
        tail = values[-min(args.moving_average, len(values)) :]
        summary_rows.append(
            {
                "metric": name,
                "first": values[0][1],
                "last": values[-1][1],
                "minimum": minimum[1],
                "minimum_step": minimum[0],
                "maximum": maximum[1],
                "maximum_step": maximum[0],
                f"last_{len(tail)}_mean": sum(value for _, value in tail) / len(tail),
                "observations": len(values),
            }
        )
    tail_name = f"last_{min(args.moving_average, len(rows))}_mean"
    write_csv(
        output_dir / "dino3d_pretrain_summary.csv",
        summary_rows,
        ["metric", "first", "last", "minimum", "minimum_step", "maximum", "maximum_step", tail_name, "observations"],
    )
    print(f"PRETRAIN_HISTORY_OK rows={len(rows)} first_step={rows[0]['step']} last_step={rows[-1]['step']}")
    print(output_dir / "dino3d_pretrain_history.csv")


if __name__ == "__main__":
    main()
