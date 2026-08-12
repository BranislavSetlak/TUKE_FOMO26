"""Utilities for extracting and weighting MRI sequence labels from FOMO paths."""

from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping


CANONICAL_CASE = {
    "dwi": "dwi",
    "adc": "ADC",
    "t1": "T1",
    "t1w": "T1w",
    "t1c": "T1c",
    "t2": "T2",
    "t2w": "T2w",
    "t2star": "T2star",
    "t2s": "T2star",
    "flair": "FLAIR",
    "gre": "gre",
    "mp2rage": "MP2RAGE",
    "pd": "PD",
    "pdw": "PDw",
    "asl": "asl",
    "unit1": "UNIT1",
    "swi": "swi",
    "angio": "angio",
    "m0scan": "m0scan",
    "cbf": "cbf",
    "mese": "MESE",
    "scan": "scan",
    "unknown": "unknown",
}


def path_datatype(path: str) -> str:
    """Return the BIDS-style datatype directory, when available."""

    for part in reversed(Path(path).parts[:-1]):
        lowered = part.lower()
        if lowered in {"anat", "dwi", "perf"}:
            return lowered
    return "unspecified"


def sequence_and_variant(path: str) -> tuple[str, str]:
    """Extract canonical sequence and original filename variant."""

    stem = Path(path).stem
    if stem.lower().endswith(".nii"):
        stem = stem[:-4]

    tokens = [token for token in stem.split("_") if token]
    lowered = [token.lower() for token in tokens]
    datatype = path_datatype(path)

    if "adc" in lowered:
        index = lowered.index("adc")
        return "ADC", "_".join(tokens[index:])

    if "dwi" in lowered:
        index = lowered.index("dwi")
        return "dwi", "_".join(tokens[index:])

    if datatype == "dwi":
        return "dwi", tokens[-1] if tokens else stem

    suffix = tokens[-1] if tokens else stem
    return CANONICAL_CASE.get(suffix.lower(), suffix), suffix


def sequence_class_id(
    path: str,
    raw_to_class: Mapping[str, int],
    ignored_sequences: Mapping[str, int],
    other_class_id: int,
) -> int:
    """Map a preprocessed scan path to its sequence-class target.

    Explicitly ignored labels (currently ``scan``) keep their reconstruction
    target but receive the configured classification ``ignore_index``. Any
    previously unseen suffix is assigned to the catch-all ``other`` class.
    """

    raw_sequence, _ = sequence_and_variant(path)
    if raw_sequence in ignored_sequences:
        return int(ignored_sequences[raw_sequence])
    return int(raw_to_class.get(raw_sequence, other_class_id))


def sequence_class_counts(
    paths: Iterable[str],
    raw_to_class: Mapping[str, int],
    ignored_sequences: Mapping[str, int],
    other_class_id: int,
    num_classes: int,
) -> list[int]:
    """Count non-ignored sequence targets and validate the configured IDs."""

    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    if not 0 <= other_class_id < num_classes:
        raise ValueError(f"other_class_id={other_class_id} is outside 0..{num_classes - 1}")

    configured_ids = {int(value) for value in raw_to_class.values()}
    invalid_ids = sorted(value for value in configured_ids if not 0 <= value < num_classes)
    if invalid_ids:
        raise ValueError(f"raw_to_class contains invalid class IDs: {invalid_ids}")

    counts: Counter[int] = Counter()
    ignored_ids = {int(value) for value in ignored_sequences.values()}
    for path in paths:
        class_id = sequence_class_id(
            path,
            raw_to_class=raw_to_class,
            ignored_sequences=ignored_sequences,
            other_class_id=other_class_id,
        )
        if class_id not in ignored_ids:
            counts[class_id] += 1

    return [counts[class_id] for class_id in range(num_classes)]


def effective_number_class_weights(counts: Iterable[int], beta: float = 0.9999) -> list[float]:
    """Return class-balanced weights based on the effective number of samples.

    The weights follow Cui et al.'s ``(1-beta)/(1-beta**n)`` formulation and
    are normalized to mean one. Requiring every configured class to occur in
    the training split turns stale mappings into an early, actionable error.
    """

    counts = [int(count) for count in counts]
    if not counts:
        raise ValueError("counts must not be empty")
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    missing = [index for index, count in enumerate(counts) if count <= 0]
    if missing:
        raise ValueError(f"The training split has no samples for sequence classes: {missing}")

    if beta == 0:
        return [1.0] * len(counts)

    weights = [(1.0 - beta) / (1.0 - beta**count) for count in counts]
    scale = len(weights) / sum(weights)
    return [weight * scale for weight in weights]
