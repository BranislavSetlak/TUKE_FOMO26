"""Online lesion-aware CarveMix for already cropped 3-D training samples."""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


class Torch_CarveMix:
    """Paste a signed-distance ROI from a donor image and its voxel label.

    The ROI is computed from the union of all non-background donor classes.
    The donor label itself is pasted unchanged, so this also gives a practical
    multi-class extension of the binary method described in the paper.
    """

    def __init__(self, probability: float = 0.5):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        self.probability = float(probability)

    @staticmethod
    def _carved_roi(label: torch.Tensor) -> tuple[torch.Tensor | None, float | None]:
        if label.ndim != 4 or label.shape[0] != 1:
            raise ValueError(f"CarveMix expects label [1,D,H,W], got {tuple(label.shape)}")

        lesion = label[0].detach().cpu().numpy() > 0
        if not np.any(lesion):
            return None, None

        outside_distance = distance_transform_edt(~lesion)
        inside_distance = distance_transform_edt(lesion)
        signed_distance = outside_distance
        signed_distance[lesion] = -inside_distance[lesion]
        lesion_radius = float(-signed_distance.min())
        if lesion_radius <= 0:
            return None, None

        if float(torch.rand(())) < 0.5:
            threshold = float(torch.empty(()).uniform_(-0.5 * lesion_radius, 0.0))
        else:
            threshold = float(torch.empty(()).uniform_(0.0, lesion_radius))
        roi = torch.from_numpy(signed_distance <= threshold)
        return roi, threshold

    def __call__(self, receiver: dict, donor: dict, force: bool = False) -> dict:
        transforms_applied = receiver.setdefault("transforms_applied", {})
        transforms_applied["carvemix"] = {
            "applied": False,
            "donor": "",
            "threshold": 0.0,
            "roi_voxels": 0,
        }
        if not force and (self.probability == 0.0 or float(torch.rand(())) >= self.probability):
            return receiver

        receiver_image, donor_image = receiver["image"], donor["image"]
        receiver_label, donor_label = receiver["label"], donor["label"]
        if receiver_image.shape != donor_image.shape or receiver_label.shape != donor_label.shape:
            raise ValueError(
                "CarveMix requires equal post-transform shapes, got "
                f"images {tuple(receiver_image.shape)} and {tuple(donor_image.shape)}, "
                f"labels {tuple(receiver_label.shape)} and {tuple(donor_label.shape)}"
            )

        roi, threshold = self._carved_roi(donor_label)
        if roi is None or not torch.any(roi):
            return receiver
        roi = roi.to(device=receiver_image.device)

        receiver["image"] = torch.where(roi.unsqueeze(0), donor_image, receiver_image)
        receiver["label"] = torch.where(roi.unsqueeze(0), donor_label, receiver_label)
        transforms_applied["carvemix"] = {
            "applied": True,
            "donor": donor.get("file_path", "unknown"),
            "threshold": threshold,
            "roi_voxels": int(roi.sum()),
        }
        return receiver
