"""
Split loader for court keypoint detection.

data_train.json (6630 records) → train split as-is.
data_val.json   (2211 records) → deterministically split 50/50 by sorted MD5
                                  hash of record id: first half → val,
                                  second half → test, minus known-bad GT.
"""

import hashlib
import json
from pathlib import Path


_DEFAULT_DATA_DIR = Path(__file__).parent

_BAD_TEST_RECORD_IDS = {
    # Semantically wrong GT: annotations mark a small net/service-box region
    # instead of the full-court keypoint layout.
    "zKIU4fWsRTM_1500",
    "PuAPCalPLM4_1700",
    "-5zNAhwRoPE_200",
    "UJHVcyTNo-k_2150",
    # Additional bad annotations.
    "1ueaSm-2-lo_1650",
    "oTsZKnpPiRw_800",
}


def _load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_records(split: str, data_dir: str | Path | None = None) -> list[dict]:
    """Return records for the requested split.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"val"``, or ``"test"``.
    data_dir : str or Path, optional
        Directory containing ``data_train.json`` and ``data_val.json``.
        Defaults to the ``data/`` folder next to this file.

    Returns
    -------
    list[dict]
        Each dict has keys ``"id"``, ``"metric"``, ``"kps"``.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")

    d = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR

    if split == "train":
        return _load_json(d / "data_train.json")

    # val + test come from data_val.json, deterministically halved by md5 hash
    records = _load_json(d / "data_val.json")

    def _sort_key(r: dict) -> str:
        return hashlib.md5(str(r["id"]).encode()).hexdigest()

    sorted_records = sorted(records, key=_sort_key)
    mid = len(sorted_records) // 2  # 1105 val / 1106 test before exclusions

    if split == "val":
        return sorted_records[:mid]
    else:  # "test"
        return [
            r for r in sorted_records[mid:]
            if str(r["id"]) not in _BAD_TEST_RECORD_IDS
        ]
