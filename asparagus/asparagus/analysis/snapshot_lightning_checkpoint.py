"""Create a stable, weights-only snapshot of a Lightning checkpoint."""

import argparse
import os
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempts", default=3, type=int)
    args = parser.parse_args()

    if args.attempts < 1:
        raise ValueError("--attempts must be at least one")
    if not args.source.is_file():
        raise FileNotFoundError(args.source)

    checkpoint = None
    source_stat = None
    last_error = None
    for attempt in range(1, args.attempts + 1):
        before = args.source.stat()
        try:
            candidate = torch.load(args.source, map_location="cpu", weights_only=False)
        except Exception as error:
            last_error = error
            candidate = None
        after = args.source.stat()
        stable = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
        if candidate is not None and stable:
            checkpoint = candidate
            source_stat = after
            break
        if attempt < args.attempts:
            print(
                "TUKE_SWINUNETR_CHECKPOINT_SNAPSHOT_RETRY "
                f"attempt={attempt} stable={stable} error={last_error}"
            )
            time.sleep(5)

    if checkpoint is None or source_stat is None:
        raise RuntimeError(
            f"Could not read a stable checkpoint after {args.attempts} attempts: {last_error}"
        )
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"{args.source} has no Lightning state_dict")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"{args.source} contains an empty or invalid state_dict")

    payload = {
        "state_dict": state_dict,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "snapshot_source": str(args.source),
        "snapshot_source_size": source_stat.st_size,
        "snapshot_source_mtime_ns": source_stat.st_mtime_ns,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)

    print(
        "TUKE_SWINUNETR_CHECKPOINT_SNAPSHOT_OK "
        f"source={args.source} output={args.output} tensors={len(state_dict)} "
        f"epoch={payload['epoch']} global_step={payload['global_step']}"
    )


if __name__ == "__main__":
    main()
