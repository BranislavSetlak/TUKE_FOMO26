"""Convert Asparagus plaintext fine-tuning logs to wide epoch CSV files."""

import argparse
import ast
import csv
import math
import re
from pathlib import Path


LINE_RE = re.compile(r"^(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\s+([^:]+):\s*(.*)$")
TASKS = (
    "CLS002_FOMO26_Infarct",
    "SEG009_FOMO26_Meningioma",
    "REGR002_FOMO26_BrainAge",
    "SEG010_FOMO26_TrigeminalNeuralgia",
    "CLS003_FOMO26_Polymicrogyria",
)


def task_from_path(path):
    text = str(path)
    return next((task for task in TASKS if task in text), "unknown")


def scalar_or_items(text):
    text = text.strip()
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            return {"": float(text)}
        except ValueError:
            return {}
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return {"": float(value)}
    if isinstance(value, (list, tuple)):
        return {f"_class_{i}": float(item) for i, item in enumerate(value) if isinstance(item, (int, float))}
    return {}


def clean_name(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()


def parse_log(path):
    task = task_from_path(path)
    run = next((part for part in path.parts if part.startswith("slurm_") or part.startswith("run_")), path.parent.name)
    rows = {}
    current_epoch = None
    lrs = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LINE_RE.search(line.rstrip())
            if not match:
                continue
            timestamp, raw_name, raw_value = match.groups()
            name = raw_name.strip()
            if name == "Current Epoch":
                try:
                    current_epoch = int(float(raw_value.strip()))
                except ValueError:
                    current_epoch = None
                if current_epoch is not None:
                    rows.setdefault(current_epoch, {"task": task, "run": run, "epoch": current_epoch, "timestamp": timestamp})
                    lrs.setdefault(current_epoch, [])
                continue
            if current_epoch is None:
                continue
            row = rows[current_epoch]
            if name == "Epoch Time":
                row["epoch_time_seconds"] = float(raw_value)
                continue
            if name.startswith("lr-"):
                if not name.endswith("-momentum"):
                    try:
                        lrs[current_epoch].append(float(raw_value))
                    except ValueError:
                        pass
                continue
            if not (name.startswith("train") or name.startswith("val")):
                continue
            base = clean_name(name)
            for suffix, value in scalar_or_items(raw_value).items():
                row[base + suffix] = value

    for epoch, values in lrs.items():
        if values:
            rows[epoch]["learning_rate_min"] = min(values)
            rows[epoch]["learning_rate_mean"] = sum(values) / len(values)
            rows[epoch]["learning_rate_max"] = max(values)
    for row in rows.values():
        row["source_log"] = str(path)
    return list(rows.values())


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="DINO fine-tuning model root")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    logs = sorted(path for path in root.rglob("*.log") if "finetune" in path.name)
    if not logs:
        raise SystemExit(f"No fine-tuning .log files found under {root}")
    rows = []
    for log in logs:
        rows.extend(parse_log(log))
    rows.sort(key=lambda row: (row["task"], row["run"], row["epoch"]))
    if not rows:
        raise SystemExit("Fine-tuning logs were found, but no epoch records could be parsed.")

    fixed = ["task", "run", "epoch", "timestamp", "epoch_time_seconds", "learning_rate_min", "learning_rate_mean", "learning_rate_max"]
    dynamic = sorted({key for row in rows for key in row if key not in fixed and key != "source_log"})
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "dino3d_finetune_history.csv", rows, fixed + dynamic + ["source_log"])

    summaries = []
    for task_run in sorted({(row["task"], row["run"]) for row in rows}):
        subset = [row for row in rows if (row["task"], row["run"]) == task_run]
        val_rows = [row for row in subset if "val_loss" in row]
        best = min(val_rows, key=lambda row: row["val_loss"]) if val_rows else None
        summaries.append(
            {
                "task": task_run[0],
                "run": task_run[1],
                "epochs_logged": len(subset),
                "first_epoch": subset[0]["epoch"],
                "last_epoch": subset[-1]["epoch"],
                "total_logged_time_seconds": sum(row.get("epoch_time_seconds", 0.0) for row in subset),
                "best_val_loss": best.get("val_loss", "") if best else "",
                "best_val_loss_epoch": best.get("epoch", "") if best else "",
                "final_train_loss": subset[-1].get("train_loss", ""),
            }
        )
    write_csv(
        output_dir / "dino3d_finetune_summary.csv",
        summaries,
        ["task", "run", "epochs_logged", "first_epoch", "last_epoch", "total_logged_time_seconds", "best_val_loss", "best_val_loss_epoch", "final_train_loss"],
    )
    print(f"FINETUNE_HISTORY_OK logs={len(logs)} rows={len(rows)}")
    print(output_dir / "dino3d_finetune_history.csv")


if __name__ == "__main__":
    main()
