"""GPU preflight for the clean frozen-feature DINO3D downstream protocol."""

import argparse

import torch

from asparagus.modules.networks.dino_downstream import DINO3DClassifierRegressor, DINO3DSegmenter


def checkpoint_state(path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint state_dict is not a mapping")
    return state


def load_teacher_backbone(model: torch.nn.Module, source_state: dict[str, torch.Tensor]) -> None:
    normalized = {}
    for key, value in source_state.items():
        key = key.replace("model._orig_mod.", "").replace("model.", "")
        if key.startswith("teacher_backbone."):
            normalized[key] = value

    target = model.state_dict()
    expected = {key for key in target if key.startswith("teacher_backbone.")}
    compatible = {
        key: value
        for key, value in normalized.items()
        if key in target and target[key].shape == value.shape
    }
    missing = sorted(expected - compatible.keys())
    if missing:
        raise RuntimeError(
            f"Only {len(compatible)}/{len(expected)} teacher-backbone tensors are compatible; "
            f"first missing keys: {missing[:5]}"
        )
    model.load_state_dict(compatible, strict=False)


def assert_frozen_backbone(model: torch.nn.Module) -> None:
    model.freeze_backbone()
    if any(parameter.requires_grad for parameter in model.teacher_backbone.parameters()):
        raise RuntimeError("Backbone still has trainable parameters")
    if not any(parameter.requires_grad for parameter in model.decoder.parameters()):
        raise RuntimeError("Decoder has no trainable parameters")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This preflight must run in a GPU Slurm allocation")

    source_state = checkpoint_state(args.checkpoint)
    device = torch.device("cuda")

    classifier = DINO3DClassifierRegressor(input_channels=4, output_channels=2).to(device)
    load_teacher_backbone(classifier, source_state)
    assert_frozen_backbone(classifier)
    classifier.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        classifier_input = torch.randn(1, 4, 96, 96, 96, device=device)
        classifier_features = classifier.forward_features(classifier_input)
        classifier_output = classifier(classifier_input)
    if classifier_features.shape != (1, 384 * 5):
        raise RuntimeError(f"Unexpected classification feature shape {classifier_features.shape}")
    if classifier_output.shape != (1, 2):
        raise RuntimeError(f"Unexpected classification output shape {classifier_output.shape}")
    print(
        "PREFLIGHT_CLASSIFIER_OK "
        f"features={tuple(classifier_features.shape)} output={tuple(classifier_output.shape)}"
    )

    del classifier, classifier_input, classifier_features, classifier_output
    torch.cuda.empty_cache()

    segmenter = DINO3DSegmenter(input_channels=4, output_channels=3).to(device)
    load_teacher_backbone(segmenter, source_state)
    assert_frozen_backbone(segmenter)
    segmenter.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        segmentation_input = torch.randn(1, 4, 96, 96, 96, device=device)
        segmentation_features = segmenter.forward_features(segmentation_input)
        segmentation_output = segmenter(segmentation_input)
    if segmentation_features.shape != (1, 384 * 4, 6, 6, 6):
        raise RuntimeError(f"Unexpected segmentation feature shape {segmentation_features.shape}")
    if segmentation_output.shape != (1, 3, 96, 96, 96):
        raise RuntimeError(f"Unexpected segmentation output shape {segmentation_output.shape}")
    print(
        "PREFLIGHT_SEGMENTER_OK "
        f"features={tuple(segmentation_features.shape)} output={tuple(segmentation_output.shape)}"
    )
    print("PREFLIGHT_DINO3D_CLEAN_FINETUNE_OK")


if __name__ == "__main__":
    main()
