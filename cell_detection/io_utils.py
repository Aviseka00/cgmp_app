"""Load images and Cellpose-style *_seg.npy masks; deduplicate identical images."""

from __future__ import annotations

import glob
import hashlib
import os
from collections import defaultdict

import numpy as np


def _file_md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_instance_masks(seg_path: str) -> np.ndarray:
    """Return (H, W) uint16 instance mask (0 = background)."""
    arr = np.load(seg_path, allow_pickle=True)
    if arr.dtype == object:
        data = arr.item()
        if not isinstance(data, dict) or "masks" not in data:
            raise ValueError(f"Unexpected seg format in {seg_path}")
        return np.asarray(data["masks"], dtype=np.uint16)
    return np.asarray(arr, dtype=np.uint16)


def instance_to_binary(instance: np.ndarray) -> np.ndarray:
    """Float32 {0,1} positive pixel mask."""
    return (instance > 0).astype(np.float32)


def positive_pixel_count(instance: np.ndarray) -> int:
    return int(np.count_nonzero(instance > 0))


def discover_pairs(data_dir: str) -> list[tuple[str, str]]:
    """Return list of (image_path, seg_npy_path)."""
    data_dir = os.path.abspath(data_dir)
    pairs: list[tuple[str, str]] = []
    for jpg in sorted(glob.glob(os.path.join(data_dir, "*.jpg"))):
        stem, _ = os.path.splitext(jpg)
        seg = stem + "_seg.npy"
        if os.path.isfile(seg):
            pairs.append((jpg, seg))
    return pairs


def dedupe_pairs_by_image_hash(
    pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """If two JPGs are byte-identical, keep the pair with more annotated positive pixels."""
    groups: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for img, seg in pairs:
        digest = _file_md5(img)
        inst = load_instance_masks(seg)
        groups[digest].append((img, seg, positive_pixel_count(inst)))
    out: list[tuple[str, str]] = []
    for items in groups.values():
        if len(items) == 1:
            out.append((items[0][0], items[0][1]))
        else:
            best = max(items, key=lambda t: t[2])
            out.append((best[0], best[1]))
    out.sort(key=lambda x: x[0])
    return out
