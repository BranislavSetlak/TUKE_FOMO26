"""Utilities for extracting canonical MRI sequence labels from FOMO paths."""

from pathlib import Path


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
