"""Conservative connected-component postprocessing for segmentation tasks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def components(binary: np.ndarray) -> list[tuple[int, int]]:
    labeled, count = ndimage.label(binary, structure=ndimage.generate_binary_structure(3, 3))
    if count == 0:
        return []
    sizes = np.bincount(labeled.ravel())[1:]
    return sorted(((index + 1, int(size)) for index, size in enumerate(sizes)), key=lambda x: x[1], reverse=True)


def task2_mask(probability: np.ndarray, threshold: float = 0.5, keep_components: int = 1) -> np.ndarray:
    raw = probability >= threshold
    labeled, _count = ndimage.label(raw, structure=ndimage.generate_binary_structure(3, 3))
    ranked = components(raw)
    keep = {label for label, _size in ranked[:keep_components]}
    if not keep:
        return np.zeros(raw.shape, dtype=np.uint8)
    return np.isin(labeled, list(keep)).astype(np.uint8)


def task4_mask(probabilities: np.ndarray, min_voxels: int = 4, min_largest_ratio: float = 0.01) -> np.ndarray:
    raw = np.argmax(probabilities, axis=0).astype(np.uint8)
    output = np.zeros(raw.shape, dtype=np.uint8)
    for class_id in (1, 2):
        binary = raw == class_id
        labeled, _count = ndimage.label(binary, structure=ndimage.generate_binary_structure(3, 3))
        ranked = components(binary)
        if not ranked:
            continue
        cutoff = max(int(min_voxels), int(np.ceil(ranked[0][1] * min_largest_ratio)))
        keep = [label for label, size in ranked if size >= cutoff]
        output[np.isin(labeled, keep)] = class_id
    return output
