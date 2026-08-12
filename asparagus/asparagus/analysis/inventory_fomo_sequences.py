#!/usr/bin/env python3
"""Inventory FOMO26 sequence labels without loading any image tensors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from asparagus.functional.sequence_labels import (
    path_datatype,
    sequence_and_variant,
)


SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--check-files",
        type=int,
        default=200,
        help="Check existence of this many paths; use 0 to skip.",
    )
    return parser.parse_args()


def contains_split_keys(value: Any) -> bool:
    return isinstance(value, dict) and any(
        str(key).lower() in SPLIT_ALIASES for key in value
    )


def select_fold(document: Any, fold: int) -> tuple[Any, str]:
    if (
        isinstance(document, list)
        and document
        and all(contains_split_keys(item) for item in document)
    ):
        if not 0 <= fold < len(document):
            raise IndexError(f"Fold {fold} is outside 0..{len(document) - 1}")
        return document[fold], f"root list index {fold} of {len(document)} folds"

    if isinstance(document, dict):
        folds = document.get("folds")
        if isinstance(folds, list) and folds and all(
            contains_split_keys(item) for item in folds
        ):
            if not 0 <= fold < len(folds):
                raise IndexError(f"Fold {fold} is outside 0..{len(folds) - 1}")
            return folds[fold], f"document['folds'][{fold}] of {len(folds)} folds"

        for key in (str(fold), f"fold_{fold}", f"fold{fold}"):
            if key in document and contains_split_keys(document[key]):
                return document[key], f"document[{key!r}]"

    return document, "document already represents one fold"


def iter_pt_paths(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.lower().endswith(".pt"):
            yield candidate
        return

    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str) and key.strip().lower().endswith(".pt"):
                yield key.strip()
            yield from iter_pt_paths(nested)
        return

    if isinstance(value, (list, tuple)):
        for nested in value:
            yield from iter_pt_paths(nested)


def extract_split_paths(fold_data: Any) -> dict[str, set[str]]:
    if not isinstance(fold_data, dict):
        raise TypeError(
            "Selected fold is not a dictionary with train/validation/test keys: "
            f"got {type(fold_data).__name__}"
        )

    result: dict[str, set[str]] = defaultdict(set)
    for key, value in fold_data.items():
        split = SPLIT_ALIASES.get(str(key).lower())
        if split is not None:
            result[split].update(iter_pt_paths(value))

    if not result:
        raise KeyError(
            "No train/validation/test keys found. Top-level keys were: "
            + ", ".join(map(str, fold_data.keys()))
        )
    return dict(result)


def session_key(path: str) -> str:
    parts = Path(path).parts
    dataset = next((p for p in parts if re.match(r"(?i)^PT\d+", p)), "")
    subject = next((p for p in parts if re.match(r"(?i)^sub[-_]", p)), "")
    session = next((p for p in parts if re.match(r"(?i)^ses[-_]", p)), "")
    if subject and session:
        return "/".join(part for part in (dataset, subject, session) if part)
    return str(Path(path).parent)


def current_path(path: str) -> str:
    legacy = "/project/perun2601396/"
    current = "/mnt/project/perun2601396/"
    return current + path[len(legacy) :] if path.startswith(legacy) else path


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def session_statistics(paths: set[str]) -> tuple[dict[str, int], Counter[tuple[str, ...]]]:
    sessions: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        sessions[session_key(path)].append(path)

    combinations: Counter[tuple[str, ...]] = Counter()
    single_scan = 0
    multi_scan = 0
    multi_sequence = 0
    for session_paths in sessions.values():
        sequences = tuple(sorted({sequence_and_variant(p)[0] for p in session_paths}))
        combinations[sequences] += 1
        if len(session_paths) == 1:
            single_scan += 1
        else:
            multi_scan += 1
        if len(sequences) >= 2:
            multi_sequence += 1

    stats = {
        "scans": len(paths),
        "sessions": len(sessions),
        "single_scan_sessions": single_scan,
        "multi_scan_sessions": multi_scan,
        "multi_sequence_sessions": multi_sequence,
    }
    return stats, combinations


def main() -> None:
    args = parse_args()
    if not args.split_json.is_file():
        raise FileNotFoundError(args.split_json)
    if args.fold < 0:
        raise ValueError("--fold must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.split_json.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    fold_data, fold_description = select_fold(document, args.fold)
    split_paths = extract_split_paths(fold_data)
    all_paths = set().union(*split_paths.values())
    if not all_paths:
        raise RuntimeError("The selected fold contains zero .pt paths")

    split_order = ["train", "validation", "test"]
    counts_by_split: dict[str, Counter[str]] = {}
    variants: Counter[tuple[str, str]] = Counter()
    variant_example: dict[tuple[str, str], str] = {}
    datatype_counts: Counter[str] = Counter()

    for split, paths in split_paths.items():
        counts_by_split[split] = Counter(sequence_and_variant(p)[0] for p in paths)

    all_counts: Counter[str] = Counter()
    for path in all_paths:
        sequence, variant = sequence_and_variant(path)
        all_counts[sequence] += 1
        variants[(sequence, variant)] += 1
        variant_example.setdefault((sequence, variant), path)
        datatype_counts[path_datatype(path)] += 1

    sequence_rows = []
    for sequence, count in all_counts.most_common():
        row: dict[str, Any] = {
            "sequence": sequence,
            "all_count": count,
            "all_percent": round(100.0 * count / len(all_paths), 6),
        }
        for split in split_order:
            row[f"{split}_count"] = counts_by_split.get(split, Counter()).get(sequence, 0)
        sequence_rows.append(row)

    write_csv(
        args.output_dir / "sequence_counts.csv",
        [
            "sequence",
            "all_count",
            "all_percent",
            "train_count",
            "validation_count",
            "test_count",
        ],
        sequence_rows,
    )

    variant_rows = [
        {
            "sequence": sequence,
            "filename_variant": variant,
            "count": count,
            "example_path": variant_example[(sequence, variant)],
        }
        for (sequence, variant), count in variants.most_common()
    ]
    write_csv(
        args.output_dir / "sequence_variants.csv",
        ["sequence", "filename_variant", "count", "example_path"],
        variant_rows,
    )

    mapping_rows = [
        {
            "sequence": row["sequence"],
            "train_count": row["train_count"],
            "class_id": "",
            "class_name": "",
            "include": "",
        }
        for row in sequence_rows
    ]
    write_csv(
        args.output_dir / "classification_mapping_template.csv",
        ["sequence", "train_count", "class_id", "class_name", "include"],
        mapping_rows,
    )

    session_rows = []
    combination_rows = []
    for split, paths in list(split_paths.items()) + [("all", all_paths)]:
        stats, combinations = session_statistics(paths)
        session_rows.append({"split": split, **stats})
        for combination, count in combinations.most_common():
            combination_rows.append(
                {
                    "split": split,
                    "sequence_combination": "+".join(combination),
                    "sessions": count,
                }
            )

    write_csv(
        args.output_dir / "session_summary.csv",
        [
            "split",
            "scans",
            "sessions",
            "single_scan_sessions",
            "multi_scan_sessions",
            "multi_sequence_sessions",
        ],
        session_rows,
    )
    write_csv(
        args.output_dir / "session_sequence_combinations.csv",
        ["split", "sequence_combination", "sessions"],
        combination_rows,
    )

    write_csv(
        args.output_dir / "datatype_counts.csv",
        ["datatype", "count"],
        (
            {"datatype": datatype, "count": count}
            for datatype, count in datatype_counts.most_common()
        ),
    )

    checked = 0
    missing: list[str] = []
    if args.check_files > 0:
        for path in sorted(all_paths)[: args.check_files]:
            checked += 1
            resolved = current_path(path)
            if not os.path.isfile(resolved):
                missing.append(resolved)

    summary = {
        "split_json": str(args.split_json),
        "fold": args.fold,
        "fold_selection": fold_description,
        "split_scan_counts": {key: len(value) for key, value in split_paths.items()},
        "unique_scans_across_selected_fold": len(all_paths),
        "number_of_sequence_labels": len(all_counts),
        "files_checked": checked,
        "missing_in_checked_sample": missing,
    }
    with (args.output_dir / "inventory_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"FOLD_SELECTION={fold_description}")
    print("SPLIT_COUNTS=" + json.dumps(summary["split_scan_counts"], sort_keys=True))
    print(f"UNIQUE_SCANS={len(all_paths)}")
    print(f"SEQUENCE_LABELS={len(all_counts)}")
    print(f"CHECKED_FILES={checked} MISSING={len(missing)}")
    print("\nsequence                         count     percent")
    print("-------------------------------------------------")
    for row in sequence_rows:
        print(f"{row['sequence'][:30]:30s} {row['all_count']:9d} {row['all_percent']:10.4f}%")
    print(f"\nOUTPUT_DIR={args.output_dir}")


if __name__ == "__main__":
    main()
