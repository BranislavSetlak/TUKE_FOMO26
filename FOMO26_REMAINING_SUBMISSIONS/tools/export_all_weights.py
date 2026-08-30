#!/usr/bin/env python3
"""Select complete CV variants and export compact weights for Tasks 2--7."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch


SPECS = {
    "task2": {"dataset": "SEG009_FOMO26_Meningioma", "kind": "seg", "in": 2, "out": 2},
    "task3": {"dataset": "REGR002_FOMO26_BrainAge", "kind": "reg", "in": 1, "out": 1},
    "task4": {"dataset": "SEG010_FOMO26_TrigeminalNeuralgia", "kind": "seg", "in": 1, "out": 3},
    "task5": {"dataset": "CLS003_FOMO26_Polymicrogyria", "kind": "cls", "in": 1, "out": 2},
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-code", required=True, type=Path)
    parser.add_argument("--task2-normal-root", type=Path)
    parser.add_argument("--task2-gin-root", type=Path)
    parser.add_argument("--task2-gin-carvemix-root", type=Path)
    parser.add_argument("--task4-normal-root", type=Path)
    parser.add_argument("--task4-gin-root", type=Path)
    parser.add_argument("--task4-gin-carvemix-root", type=Path)
    parser.add_argument("--clsreg-normal-root", required=True, type=Path)
    parser.add_argument("--clsreg-gin-root", required=True, type=Path)
    parser.add_argument("--task2-variant", default="auto", choices=("auto", "normal", "gin", "gin_carvemix"))
    parser.add_argument("--task3-variant", default="auto", choices=("auto", "normal", "gin"))
    parser.add_argument("--task4-variant", default="auto", choices=("auto", "normal", "gin", "gin_carvemix"))
    parser.add_argument("--task5-variant", default="auto", choices=("auto", "normal", "gin"))
    parser.add_argument("--pretrain-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected metrics structure in {path}")
    return {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def summary(root: Path | None, spec: dict) -> dict:
    if root is None:
        return {"root": None, "folds": [], "missing": list(range(5)), "score": None, "score_name": None}
    folds = []
    missing = []
    for fold in range(5):
        directory = root / spec["dataset"] / f"fold_{fold}"
        checkpoint = directory / "checkpoints" / "best.ckpt"
        metric_path = directory / "validation_best_metrics.json"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not metric_path.is_file():
            missing.append(fold)
            continue
        folds.append({"fold": fold, "checkpoint": checkpoint, "metrics": metrics(metric_path)})
    score_keys = {
        "seg": ("val/min_foreground_class_dice", "val/macro_foreground_class_dice", "val/foreground_dice"),
        "cls": ("val/auroc_macro", "val/auroc_1", "val/auroc"),
        "reg": ("val/MAE", "val/loss"),
    }[spec["kind"]]
    score = None
    score_name = None
    if not missing:
        for key in score_keys:
            values = [item["metrics"].get(key) for item in folds]
            if all(value is not None and math.isfinite(value) for value in values):
                score = sum(values) / len(values)
                score_name = key
                break
    return {"root": str(root), "folds": folds, "missing": missing, "score": score, "score_name": score_name}


def select(requested: str, candidates: dict[str, dict], kind: str) -> str:
    complete = [name for name, value in candidates.items() if not value["missing"]]
    if requested != "auto":
        if requested not in complete:
            raise RuntimeError(f"Requested {requested} is incomplete: {candidates.get(requested, {}).get('missing')}")
        return requested
    if not complete:
        raise RuntimeError("No candidate variant has five complete folds")
    if len(complete) == 1:
        return complete[0]
    scored = [name for name in complete if candidates[name]["score"] is not None]
    if not scored:
        raise RuntimeError(f"Complete variants {complete} have no comparable validation metric; choose explicitly")
    minimize = kind == "reg" or candidates[scored[0]]["score_name"] == "val/loss"
    return min(scored, key=lambda name: candidates[name]["score"]) if minimize else max(scored, key=lambda name: candidates[name]["score"])


def source_state(path: Path) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    state = payload.get("state_dict", payload.get("network_weights"))
    if not isinstance(state, dict) or not state:
        raise ValueError(f"No weights in {path}")
    return state, {"epoch": payload.get("epoch"), "global_step": payload.get("global_step")}


def normalized_key(raw: str) -> str:
    key = raw[len("module."):] if raw.startswith("module.") else raw
    return key.replace("model._orig_mod.", "model.", 1)


def compact_downstream(source: dict, kind: str) -> dict[str, torch.Tensor]:
    result = {}
    for raw, value in source.items():
        key = normalized_key(str(raw))
        if kind == "seg" and key.startswith("model.swin_unetr."):
            result["net." + key[len("model.swin_unetr."):]] = value.detach().cpu()
        elif kind in {"cls", "reg"} and key.startswith("model.swin_unetr.swinViT."):
            result["encoder." + key[len("model.swin_unetr.swinViT."):]] = value.detach().cpu()
        elif kind in {"cls", "reg"} and key.startswith("model.downstream_head."):
            result["head." + key[len("model.downstream_head."):]] = value.detach().cpu()
    return result


def write_task(task: str, spec: dict, candidates: dict, requested: str, output: Path, model_module) -> dict:
    variant = select(requested, candidates, spec["kind"])
    chosen = candidates[variant]
    staging = output.parent / f".{output.name}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    fold_rows = []
    try:
        for item in chosen["folds"]:
            source, checkpoint_info = source_state(item["checkpoint"])
            compact = compact_downstream(source, spec["kind"])
            model = model_module.SegmentationModel(spec["in"], spec["out"]) if spec["kind"] == "seg" else model_module.ClsRegModel(spec["in"], spec["out"])
            model.load_state_dict(compact, strict=True)
            metadata = {
                "task": task,
                "dataset": spec["dataset"],
                "kind": spec["kind"],
                "fold": item["fold"],
                "variant": variant,
                "epoch": checkpoint_info["epoch"],
                "global_step": checkpoint_info["global_step"],
                "validation_metrics": item["metrics"],
                "roi_size": [96, 96, 96],
                "overlap": 0.5,
            }
            destination = staging / f"fold_{item['fold']}.pt"
            torch.save({"format_version": 1, "state_dict": compact, "metadata": metadata}, destination)
            fold_rows.append({**metadata, "file": destination.name, "bytes": destination.stat().st_size})
        manifest = {"format_version": 1, "task": task, "selected_variant": variant, "candidates": {
            name: {"root": value["root"], "missing": value["missing"], "score": value["score"], "score_name": value["score_name"]}
            for name, value in candidates.items()
        }, "folds": fold_rows}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output.mkdir(parents=True, exist_ok=True)
        for stale in output.glob("fold_*.pt"):
            stale.unlink()
        (output / "manifest.json").unlink(missing_ok=True)
        for source in staging.iterdir():
            shutil.move(str(source), output / source.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest


def export_embedding(checkpoint: Path, output: Path, model_module) -> dict:
    source, info = source_state(checkpoint)
    compact = {}
    for raw, value in source.items():
        key = normalized_key(str(raw))
        prefix = "model.swin_unetr.swinViT."
        if key.startswith(prefix):
            compact["encoder." + key[len(prefix):]] = value.detach().cpu()
    model = model_module.MultiScaleEmbeddingModel()
    model.load_state_dict(compact, strict=True)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "encoder.pt"
    metadata = {"task": "task6_and_7", "epoch": info["epoch"], "global_step": info["global_step"], "embedding_dim": model.embedding_dim, "pooling": "mean pooled Swin stages 1-4, concatenated and L2 normalized"}
    torch.save({"format_version": 1, "state_dict": compact, "metadata": metadata}, destination)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**metadata, "file": destination.name, "bytes": destination.stat().st_size}


def main() -> int:
    args = arguments()
    if not args.pretrain_checkpoint.is_file():
        raise FileNotFoundError(args.pretrain_checkpoint)
    sys.path.insert(0, str(args.common_code.resolve()))
    import model as model_module

    roots = {
        "task2": {"normal": args.task2_normal_root, "gin": args.task2_gin_root, "gin_carvemix": args.task2_gin_carvemix_root},
        "task3": {"normal": args.clsreg_normal_root, "gin": args.clsreg_gin_root},
        "task4": {"normal": args.task4_normal_root, "gin": args.task4_gin_root, "gin_carvemix": args.task4_gin_carvemix_root},
        "task5": {"normal": args.clsreg_normal_root, "gin": args.clsreg_gin_root},
    }
    report = {"tasks": {}}
    for task, spec in SPECS.items():
        candidates = {name: summary(root, spec) for name, root in roots[task].items()}
        output_name = {"task2": "task2_meningioma", "task3": "task3_brain_age", "task4": "task4_trigeminal", "task5": "task5_polymicrogyria"}[task]
        report["tasks"][task] = write_task(task, spec, candidates, getattr(args, f"{task}_variant"), args.output_root / output_name / "weights", model_module)
        print(f"EXPORTED {task} variant={report['tasks'][task]['selected_variant']}")
    report["tasks"]["task6_and_7"] = export_embedding(args.pretrain_checkpoint, args.output_root / "task6_7_embeddings" / "weights", model_module)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ALL_WEIGHT_EXPORTS_OK report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
