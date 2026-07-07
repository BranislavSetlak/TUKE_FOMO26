#!/usr/bin/env python3
"""
Rewrite JSON paths to use a new base directory.

This script reads a JSON file containing a list of file paths, identifies
directory prefixes preceding any 'PTNNN' folder (where NNN is a number),
and replaces those prefixes with a user-specified base path.

Features:
- Handles different prefixes line-to-line.
- Prompts user confirmation for each unique prefix.
- Displays progress using tqdm.
- Creates a backup file before overwriting the input file.

Usage:
    ./rewrite_paths.py input.json /new/base/path
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

PT_RE = re.compile(r"^PT\d+")


def load_paths(path):
    """Load a list of file paths from a JSON file and validate the structure."""
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        print("Input JSON must be a list of strings.", file=sys.stderr)
        sys.exit(1)
    return data


def find_prefix(p):
    """Extract the prefix before the PTNNN directory in a given path."""
    parts = Path(p).parts
    for i, comp in enumerate(parts):
        if PT_RE.match(comp):
            return str(Path(*parts[:i]))
    return None


def unique_prefixes(paths):
    """Return a sorted set of unique prefixes found across all paths."""
    return sorted({pref for p in paths if (pref := find_prefix(p)) is not None})


def confirm_prefixes(prefixes, new_base):
    """Prompt the user to confirm each unique prefix replacement."""
    confirmed = set()
    for pref in prefixes:
        while True:
            ans = input(f"Replace '{pref}' -> '{new_base}'? [y/N]: ").strip().lower()
            if ans in ("y", "yes"):
                confirmed.add(pref)
                break
            if ans in ("n", "no", ""):
                break
    return confirmed


def first_pt_index(parts):
    """Return the index of the PTNNN directory in a path's parts."""
    for i, comp in enumerate(parts):
        if PT_RE.match(comp):
            return i
    return None


def rewrite_paths(paths, new_base, confirmed):
    """Replace confirmed prefixes with new_base, showing progress."""
    out, replaced, skipped_no_pt, skipped_unconfirmed = [], 0, 0, 0
    for p in tqdm(paths, desc="Rewriting paths"):
        pref = find_prefix(p)
        if pref is None:
            out.append(p)
            skipped_no_pt += 1
            continue
        if pref not in confirmed:
            out.append(p)
            skipped_unconfirmed += 1
            continue
        parts = Path(p).parts
        i_pt = first_pt_index(parts)
        tail = Path(*parts[i_pt:])
        out.append(str(Path(new_base) / tail))
        replaced += 1
    return out, replaced, skipped_no_pt, skipped_unconfirmed


def backup_file(path):
    """Create a backup copy of the input JSON file."""
    p = Path(path)
    backup_path = p.with_name(p.stem + "_backup.json")
    shutil.copy2(p, backup_path)
    print(f"Backup saved to {backup_path}", file=sys.stderr)
    return backup_path


def save_paths(path, data):
    """Write updated paths back to the same JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="Replace path prefixes preceding PTNNN directories.")
    ap.add_argument("input_json", help="Path to input JSON file containing list of paths.")
    ap.add_argument("new_base", help="New base path to replace prefixes with.")
    return ap.parse_args()


def main():
    args = parse_args()
    paths = load_paths(args.input_json)
    prefixes = unique_prefixes(paths)
    confirmed = confirm_prefixes(prefixes, args.new_base)
    backup_file(args.input_json)
    out, rep, sk_np, sk_uc = rewrite_paths(paths, args.new_base, confirmed)
    save_paths(args.input_json, out)
    print(
        f"Replaced: {rep} | Skipped (no PT*): {sk_np} | Skipped (unconfirmed prefix): {sk_uc}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
