"""GPU preflight for the TUKE SwinUNETR one-case segmentation diagnostic."""

import argparse
import json
import math
import os
from pathlib import Path

import torch

from asparagus.modules.lightning_modules.segmentation_module import SegmentationModule
from asparagus.modules.networks.swinunetr_hybrid import SwinUNETRSegmentation
from asparagus.modules.transforms.presets import CPU_seg_overfit_noaug_transforms


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError(f"{path} has no Lightning state_dict")
    state_dict = checkpoint["state_dict"]
    if any("_orig_mod." in key for key in state_dict):
        state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
    if not any(key.startswith("model.swin_unetr.swinViT.") for key in state_dict):
        raise ValueError("Checkpoint has no model.swin_unetr.swinViT.* encoder weights")
    return state_dict


def gradient_norm(module: torch.nn.Module, name_fragment: str) -> float:
    squared = 0.0
    for name, parameter in module.named_parameters():
        if name_fragment in name and parameter.grad is not None:
            squared += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return squared**0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--dataset-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--patch-size", nargs=3, default=(96, 96, 96), type=int)
    parser.add_argument("--feature-size", default=48, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.sample.is_file():
        raise FileNotFoundError(args.sample)
    with args.dataset_json.open() as handle:
        dataset_json = json.load(handle)
    metadata = dataset_json["metadata"]
    input_channels = int(metadata["n_modalities"])
    output_channels = int(metadata["n_classes"])
    patch_size = tuple(int(value) for value in args.patch_size)

    state_dict = load_checkpoint(args.checkpoint)
    stem_keys = (
        "model.swin_unetr.swinViT.patch_embed.proj.weight",
        "model.swin_unetr.encoder1.layer.conv1.conv.weight",
        "model.swin_unetr.encoder1.layer.conv3.conv.weight",
    )
    missing_source_stems = [key for key in stem_keys if key not in state_dict]
    if missing_source_stems:
        raise RuntimeError(f"Missing raw-input checkpoint weights: {missing_source_stems}")
    original_source_stems = {key: state_dict[key].clone() for key in stem_keys}
    model = SwinUNETRSegmentation(
        input_channels=input_channels,
        output_channels=output_channels,
        num_sequence_classes=14,
        feature_size=args.feature_size,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        use_checkpoint=True,
    )
    module = SegmentationModule(
        model=model,
        weights=state_dict,
        optimizer="AdamW",
        learning_rate=3e-4,
        warmup_epochs=5,
        decoder_warmup_epochs=0,
        load_decoder=True,
        repeat_stem_weights=True,
        inference_patch_size=list(patch_size),
    )

    loaded_state = module.state_dict()
    exact_loaded = [
        key
        for key, value in state_dict.items()
        if key in loaded_state
        and tuple(value.shape) == tuple(loaded_state[key].shape)
        and torch.equal(value.cpu(), loaded_state[key].cpu())
    ]
    encoder_loaded = [key for key in exact_loaded if key.startswith("model.swin_unetr.swinViT.")]
    decoder_loaded = [
        key
        for key in exact_loaded
        if key.startswith("model.swin_unetr.encoder") or key.startswith("model.swin_unetr.decoder")
    ]
    if not encoder_loaded:
        raise RuntimeError("No exact Swin encoder tensor was transferred")
    if not decoder_loaded:
        raise RuntimeError("No exact SwinUNETR decoder tensor was transferred")

    stem_transfer = {}
    for stem_key, original_source_stem in original_source_stems.items():
        if stem_key not in loaded_state:
            raise RuntimeError(f"Missing downstream raw-input weight {stem_key}")
        source_stem = original_source_stem.cpu()
        target_stem = loaded_state[stem_key].cpu()
        if input_channels == source_stem.shape[1]:
            stem_ok = torch.equal(source_stem, target_stem)
        elif source_stem.shape[1] == 1 and input_channels > 1:
            repeats = [1] * source_stem.ndim
            repeats[1] = input_channels
            expected = source_stem.repeat(*repeats) / input_channels
            stem_ok = torch.allclose(expected, target_stem)
        else:
            stem_ok = False
        stem_transfer[stem_key] = stem_ok
        if not stem_ok:
            raise RuntimeError(
                f"Raw-input transfer failed for {stem_key}: source={tuple(source_stem.shape)} "
                f"target={tuple(target_stem.shape)}"
            )

    sample = torch.load(args.sample, map_location="cpu", weights_only=False)
    if not isinstance(sample, torch.Tensor) or sample.ndim != 4:
        raise ValueError(f"Expected [C+1,X,Y,Z] sample tensor, got {type(sample)}")
    transformed = CPU_seg_overfit_noaug_transforms(patch_size=patch_size)(
        {
            "image": sample[:-1].float(),
            "label": sample[-1:].float(),
            "transforms_applied": {},
        }
    )
    image = transformed["image"].unsqueeze(0).to(args.device)
    label = transformed["label"].unsqueeze(0).long().to(args.device)
    if tuple(image.shape) != (1, input_channels, *patch_size):
        raise RuntimeError(f"Unexpected image shape {tuple(image.shape)}")
    if tuple(label.shape) != (1, 1, *patch_size):
        raise RuntimeError(f"Unexpected label shape {tuple(label.shape)}")
    present_classes = sorted(int(value) for value in torch.unique(label).cpu().tolist())
    if max(present_classes) >= output_channels:
        raise RuntimeError(
            f"Label classes {present_classes} exceed model output channels {output_channels}"
        )

    module = module.to(args.device).train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = module.model(image)
        loss = module.train_loss(logits, label)
    if tuple(logits.shape) != (1, output_channels, *patch_size):
        raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Preflight logits contain non-finite values")
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite preflight loss {float(loss.detach().cpu())}")
    loss.backward()
    encoder_gradient = gradient_norm(module, "swin_unetr.swinViT")
    decoder_gradient = gradient_norm(module, "swin_unetr.decoder")
    output_gradient = gradient_norm(module, "swin_unetr.out")
    gradients = (encoder_gradient, decoder_gradient, output_gradient)
    if any(not math.isfinite(value) or value <= 0.0 for value in gradients):
        raise RuntimeError(
            f"Missing gradients: encoder={encoder_gradient} decoder={decoder_gradient} "
            f"output={output_gradient}"
        )

    result = {
        "checkpoint": str(args.checkpoint),
        "sample": str(args.sample),
        "input_channels": input_channels,
        "output_channels": output_channels,
        "present_label_classes": present_classes,
        "image_shape": list(image.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "exact_loaded_tensor_count": len(exact_loaded),
        "exact_loaded_encoder_tensor_count": len(encoder_loaded),
        "exact_loaded_decoder_tensor_count": len(decoder_loaded),
        "stem_transfer_ok": stem_transfer,
        "encoder_gradient_norm": encoder_gradient,
        "decoder_gradient_norm": decoder_gradient,
        "output_gradient_norm": output_gradient,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    atomic_json(args.output, result)
    print(
        "TUKE_SWINUNETR_OVERFIT_PREFLIGHT_OK "
        f"loaded={len(exact_loaded)} encoder={len(encoder_loaded)} "
        f"decoder={len(decoder_loaded)} loss={result['loss']:.6f} "
        f"encoder_grad={encoder_gradient:.6g} decoder_grad={decoder_gradient:.6g} "
        f"output_grad={output_gradient:.6g}"
    )
    print(f"TUKE_SWINUNETR_OVERFIT_PREFLIGHT={args.output}")


if __name__ == "__main__":
    main()
