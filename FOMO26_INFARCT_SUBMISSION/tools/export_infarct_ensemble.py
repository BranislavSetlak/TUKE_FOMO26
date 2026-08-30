#!/usr/bin/env python3
"""Select a complete CV variant and export five compact inference weights."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch


TASK = "CLS002_FOMO26_Infarct"
EXPECTED_FOLDS = 5


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-root", required=True, type=Path)
    parser.add_argument("--gin-root", required=True, type=Path)
    parser.add_argument("--variant", choices=("auto", "normal", "gin"), default="auto")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--container-code", required=True, type=Path)
    return parser.parse_args()


def _fold_root(root: Path, fold: int) -> Path:
    return root / TASK / f"fold_{fold}"


def _metric_dict(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected metric structure in {path}")
    return {
        str(key): float(value)
        for key, value in data.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _variant_summary(root: Path) -> dict:
    folds = []
    missing = []
    for fold in range(EXPECTED_FOLDS):
        directory = _fold_root(root, fold)
        checkpoint = directory / "checkpoints" / "best.ckpt"
        metrics_path = directory / "validation_best_metrics.json"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not metrics_path.is_file():
            missing.append(fold)
            continue
        metrics = _metric_dict(metrics_path)
        folds.append(
            {
                "fold": fold,
                "checkpoint": checkpoint,
                "metrics_path": metrics_path,
                "metrics": metrics,
            }
        )

    aurocs = []
    losses = []
    for item in folds:
        metrics = item["metrics"]
        for key in ("val/auroc_macro", "val/auroc_1", "val/auroc"):
            if key in metrics and math.isfinite(metrics[key]):
                aurocs.append(metrics[key])
                break
        if "val/loss" in metrics and math.isfinite(metrics["val/loss"]):
            losses.append(metrics["val/loss"])
    return {
        "root": root,
        "folds": folds,
        "missing": missing,
        "mean_auroc": sum(aurocs) / len(aurocs) if len(aurocs) == EXPECTED_FOLDS else None,
        "mean_loss": sum(losses) / len(losses) if len(losses) == EXPECTED_FOLDS else None,
    }


def _select(requested: str, summaries: dict[str, dict]) -> str:
    complete = [name for name, summary in summaries.items() if not summary["missing"]]
    if requested != "auto":
        if requested not in complete:
            raise RuntimeError(
                f"Requested {requested!r} is incomplete; missing folds={summaries[requested]['missing']}"
            )
        return requested
    if not complete:
        details = ", ".join(f"{name}: missing {value['missing']}" for name, value in summaries.items())
        raise RuntimeError(f"Neither normal nor GIN has five complete folds ({details})")
    if len(complete) == 1:
        return complete[0]

    normal, gin = summaries["normal"], summaries["gin"]
    if normal["mean_auroc"] is not None and gin["mean_auroc"] is not None:
        return "gin" if gin["mean_auroc"] > normal["mean_auroc"] else "normal"
    if normal["mean_loss"] is not None and gin["mean_loss"] is not None:
        return "gin" if gin["mean_loss"] < normal["mean_loss"] else "normal"
    raise RuntimeError("Both variants are complete but comparable AUROC/loss metrics are missing")


def _source_state_dict(checkpoint: Path) -> tuple[dict, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a mapping: {checkpoint}")
    state_dict = payload.get("state_dict", payload.get("network_weights"))
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"No state_dict/network_weights in {checkpoint}")
    metadata = {
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
    }
    return state_dict, metadata


def _compact_state_dict(source: dict, checkpoint: Path) -> dict[str, torch.Tensor]:
    compact = {}
    for raw_key, value in source.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        key = key.replace("model._orig_mod.", "model.", 1)
        encoder_prefix = "model.swin_unetr.swinViT."
        head_prefix = "model.downstream_head."
        if key.startswith(encoder_prefix):
            compact["encoder." + key[len(encoder_prefix) :]] = value.detach().cpu()
        elif key.startswith(head_prefix):
            compact["head." + key[len(head_prefix) :]] = value.detach().cpu()
    if not any(key.startswith("encoder.") for key in compact):
        raise RuntimeError(f"No Swin encoder tensors found in {checkpoint}")
    if not any(key.startswith("head.") for key in compact):
        raise RuntimeError(f"No downstream head tensors found in {checkpoint}")
    return compact


def main() -> int:
    args = arguments()
    summaries = {
        "normal": _variant_summary(args.normal_root),
        "gin": _variant_summary(args.gin_root),
    }
    selected = _select(args.variant, summaries)
    selected_summary = summaries[selected]

    sys.path.insert(0, str(args.container_code.resolve()))
    from model import InfarctSwinClassifier

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    manifest = {
        "format_version": 1,
        "task": TASK,
        "selected_variant": selected,
        "selection_rule": "higher mean five-fold validation AUROC; lower loss fallback",
        "variant_scores": {
            name: {"mean_auroc": item["mean_auroc"], "mean_loss": item["mean_loss"]}
            for name, item in summaries.items()
        },
        "folds": [],
    }
    try:
        for item in selected_summary["folds"]:
            fold = int(item["fold"])
            source, checkpoint_metadata = _source_state_dict(item["checkpoint"])
            compact = _compact_state_dict(source, item["checkpoint"])
            model = InfarctSwinClassifier()
            incompatible = model.load_state_dict(compact, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"Strict export validation failed for fold {fold}: {incompatible}")

            fold_metadata = {
                "fold": fold,
                "variant": selected,
                "epoch": checkpoint_metadata["epoch"],
                "global_step": checkpoint_metadata["global_step"],
                "validation_metrics": item["metrics"],
                "tensor_count": len(compact),
                "parameter_count": int(sum(value.numel() for value in compact.values())),
            }
            destination = staging / f"fold_{fold}.pt"
            torch.save(
                {
                    "format_version": 1,
                    "architecture": "tuke_swinunetr_clsreg_b_3ch_2class",
                    "state_dict": compact,
                    "metadata": fold_metadata,
                },
                destination,
            )
            manifest["folds"].append({**fold_metadata, "file": destination.name})
            print(
                f"EXPORTED fold={fold} variant={selected} tensors={len(compact)} "
                f"bytes={destination.stat().st_size}"
            )

        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for stale in output_dir.glob("fold_*.pt"):
            stale.unlink()
        for source in staging.iterdir():
            shutil.move(str(source), output_dir / source.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(
        f"INFARCT_ENSEMBLE_EXPORT_OK variant={selected} folds={EXPECTED_FOLDS} "
        f"output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

