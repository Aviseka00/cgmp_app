from __future__ import annotations

import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from io_utils import instance_to_binary, load_instance_masks


def _read_image_bgr(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return im


def build_train_augmentation(patch: int) -> A.Compose:
    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.85, 1.15),
                rotate=(-20, 20),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=0.4,
            ),
            A.GaussNoise(std_range=(0.02, 0.06), mean_range=(0.0, 0.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.25),
            A.ElasticTransform(
                alpha=80,
                sigma=8,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.2,
            ),
        ],
        additional_targets={"mask": "mask"},
    )


class PositiveCellCropDataset(Dataset):
    """Random crops biased toward regions containing positive pixels."""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        patch: int = 512,
        crops_per_image: int = 40,
        pos_fraction: float = 0.72,
        seed: int = 42,
    ) -> None:
        self.pairs = pairs
        self.patch = patch
        self.crops_per_image = crops_per_image
        self.pos_fraction = pos_fraction
        self.rng = random.Random(seed)
        self.aug = build_train_augmentation(patch)

    def __len__(self) -> int:
        return len(self.pairs) * self.crops_per_image

    def _sample_crop_xy(
        self, h: int, w: int, mask: np.ndarray, want_positive: bool
    ) -> tuple[int, int]:
        ph, pw = self.patch, self.patch
        if h < ph or w < pw:
            raise ValueError(f"Image {h}x{w} smaller than patch {ph}x{pw}")
        if want_positive:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                return self.rng.randint(0, h - ph), self.rng.randint(0, w - pw)
            idx = self.rng.randrange(len(xs))
            cy, cx = int(ys[idx]), int(xs[idx])
            y0 = min(max(cy - ph // 2, 0), h - ph)
            x0 = min(max(cx - pw // 2, 0), w - pw)
            return y0, x0
        return self.rng.randint(0, h - ph), self.rng.randint(0, w - pw)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        img_idx = index // self.crops_per_image
        img_path, seg_path = self.pairs[img_idx]
        bgr = _read_image_bgr(img_path)
        inst = load_instance_masks(seg_path)
        if bgr.shape[:2] != inst.shape[:2]:
            raise ValueError(
                f"Shape mismatch {img_path}: image {bgr.shape[:2]} vs mask {inst.shape[:2]}"
            )
        bin_mask = (inst > 0).astype(np.uint8)
        h, w = bgr.shape[:2]
        want_pos = self.rng.random() < self.pos_fraction
        y0, x0 = self._sample_crop_xy(h, w, inst, want_pos)
        ph, pw = self.patch, self.patch
        crop_bgr = bgr[y0 : y0 + ph, x0 : x0 + pw]
        crop_m = bin_mask[y0 : y0 + ph, x0 : x0 + pw]
        augmented = self.aug(image=crop_bgr, mask=crop_m)
        image = augmented["image"]
        mask = augmented["mask"]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = mask.astype(np.float32)
        chw = np.transpose(image, (2, 0, 1))
        return {
            "image": torch.from_numpy(chw),
            "mask": torch.from_numpy(mask[None, ...]),
        }


class FullImageTensorDataset(Dataset):
    """Whole images resized for validation (memory-friendly)."""

    def __init__(self, pairs: list[tuple[str, str]], long_side: int = 1024) -> None:
        self.pairs = pairs
        self.long_side = long_side

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        img_path, seg_path = self.pairs[idx]
        bgr = _read_image_bgr(img_path)
        inst = load_instance_masks(seg_path)
        bin_mask = instance_to_binary(inst)
        h, w = bgr.shape[:2]
        scale = self.long_side / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        nh = max(32, (nh // 32) * 32)
        nw = max(32, (nw // 32) * 32)
        bgr_r = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        m_r = cv2.resize(bin_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        return {
            "image": torch.from_numpy(chw),
            "mask": torch.from_numpy(m_r.astype(np.float32)[None, ...]),
            "path": img_path,
            "orig_hw": torch.tensor([h, w], dtype=torch.int32),
        }
