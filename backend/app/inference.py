import os
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from .config import settings


CELL_DETECTION_DIR = Path(settings.cell_detection_dir)
if str(CELL_DETECTION_DIR) not in sys.path:
    sys.path.append(str(CELL_DETECTION_DIR))

from modeling import build_model  # type: ignore  # noqa: E402
from predict import predict_full_image  # type: ignore  # noqa: E402


@lru_cache(maxsize=1)
def get_model_bundle() -> tuple[torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(settings.best_model_path, map_location=device, weights_only=False)
    encoder = ckpt.get("encoder", "mit_b0")
    model = build_model(encoder=encoder).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, device


def _count_instances(binary_mask: np.ndarray) -> int:
    n_labels, _ = cv2.connectedComponents((binary_mask > 0).astype(np.uint8))
    return max(0, n_labels - 1)


def _apply_speckle_morph_open(binary_255: np.ndarray, ksize: int) -> np.ndarray:
    """Remove tiny isolated positives before area filtering (opening shrinks salt-and-pepper hits)."""
    ksize = int(ksize)
    if ksize < 3:
        return binary_255
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    u8 = (binary_255 > 0).astype(np.uint8)
    opened = cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel)
    return (opened * 255).astype(np.uint8)


def _filter_components_by_area(
    binary: np.ndarray,
    min_area: int,
    min_short_side: int = 0,
) -> tuple[np.ndarray, list, list, int, int]:
    """
    Split thresholded mask into large (kept) vs small (noise) components.
    Also rejects components whose bbox short side is below min_short_side (when > 0).
    Returns filtered mask (uint8 0/255), accepted contours, rejected contours,
    noise_instance count, noise_pixel count.
    """
    binary_u8 = (binary > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    h, w = binary_u8.shape
    filtered = np.zeros((h, w), dtype=np.uint8)
    accepted: list = []
    rejected: list = []
    noise_instances = 0
    noise_pixels = 0
    mss = max(0, int(min_short_side))
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        short = min(bw, bh)
        too_small = area < min_area or (mss > 0 and short < mss)
        if too_small:
            noise_instances += 1
            noise_pixels += area
            comp = (labels == i).astype(np.uint8) * 255
            ct, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rejected.extend(ct)
            continue
        filtered[labels == i] = 1
        comp = (labels == i).astype(np.uint8) * 255
        ct, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        accepted.extend(ct)
    return (filtered * 255).astype(np.uint8), accepted, rejected, noise_instances, noise_pixels


def _annotate_overlay_with_counts(
    overlay: np.ndarray,
    *,
    cell_count: int,
    positive_pixels: int,
    noise_instances: int,
    min_area: int,
    min_short_side: int,
    morph_k: int,
) -> None:
    """Draw count summary on the overlay (in-place) for saved heatmap PNGs."""
    h, w = overlay.shape[:2]
    base = float(max(480, min(h, w)))
    font_scale = float(np.clip(0.42 * (base / 1000.0), 0.32, 1.05))
    thickness = max(1, int(round(font_scale * 2)))
    line_height = int(26 * font_scale + 8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    rules = f"area≥{min_area}px²"
    if min_short_side > 0:
        rules += f"  short≥{min_short_side}px"
    if morph_k >= 3:
        rules += f"  open{morph_k}×{morph_k}"

    lines = [
        f"Objects counted: {cell_count}",
        f"Positive pixels: {positive_pixels:,}",
        f"Filtered (noise): {noise_instances}",
        rules,
    ]
    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    max_w = max(s[0] for s in sizes) + 24
    total_h = line_height * len(lines) + 20

    x0, y0 = 10, 10
    cv2.rectangle(
        overlay,
        (x0, y0),
        (min(w - 2, x0 + max_w), min(h - 2, y0 + total_h)),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.rectangle(
        overlay,
        (x0, y0),
        (min(w - 2, x0 + max_w), min(h - 2, y0 + total_h)),
        (255, 255, 255),
        thickness=1,
    )

    y = y0 + int(22 * font_scale) + 6
    for i, line in enumerate(lines):
        cv2.putText(
            overlay,
            line,
            (x0 + 12, y + i * line_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )


def _load_image_bgr(input_path: str) -> np.ndarray:
    """
    Load a BGR uint8 image for OpenCV / model input.
    cv2.imread alone often returns None for CMYK JPEGs, some progressive JPEGs, or odd TIFF/WebP cases;
    imdecode + Pillow cover most uploads that browsers accept.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Image file not found: {input_path}")
    size = os.path.getsize(input_path)
    if size == 0:
        raise ValueError(f"Image file is empty: {input_path}")

    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is not None:
        return bgr

    raw = np.fromfile(input_path, dtype=np.uint8)
    if raw.size > 0:
        decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if decoded is not None:
            return decoded

    try:
        with Image.open(input_path) as im:
            im = ImageOps.exif_transpose(im)
            rgb = np.asarray(im.convert("RGB"))
    except Exception as e:
        raise ValueError(
            f"Could not read image: {input_path}. "
            f"OpenCV could not decode the file; Pillow error: {e}. "
            "Export as RGB JPEG or PNG if this keeps happening."
        ) from e

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def analyze_image(input_path: str, out_mask_path: str, out_heatmap_path: str) -> dict:
    model, device = get_model_bundle()
    bgr = _load_image_bgr(input_path)
    prob = predict_full_image(
        model,
        bgr,
        device,
        tile=settings.model_tile,
        stride=settings.model_stride,
    )
    binary = (prob >= settings.model_threshold).astype(np.uint8) * 255
    morph_k = int(settings.speckle_morph_open_kernel)
    binary = _apply_speckle_morph_open(binary, morph_k)

    min_a = max(0, int(settings.min_detection_area_pixels))
    min_ss = max(0, int(settings.min_detection_short_side_pixels))
    filtered_mask, acc_contours, rej_contours, noise_instances, noise_pixels = _filter_components_by_area(
        binary, min_a, min_short_side=min_ss
    )

    heat = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    # High-contrast overlay: vivid TURBO heatmap + boundaries (cyan = kept, yellow = area-filtered noise).
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(bgr, 0.42, heatmap, 0.58, 0)
    # BGR: yellow (0,255,255) = speckle / below min area; cyan (255,255,0) = counted positives.
    if rej_contours:
        cv2.drawContours(overlay, rej_contours, -1, (0, 255, 255), 2)
    if acc_contours:
        cv2.drawContours(overlay, acc_contours, -1, (255, 255, 0), 2)

    cell_count = _count_instances(filtered_mask)
    positive_pixels = int(np.count_nonzero(filtered_mask))
    _annotate_overlay_with_counts(
        overlay,
        cell_count=cell_count,
        positive_pixels=positive_pixels,
        noise_instances=noise_instances,
        min_area=min_a,
        min_short_side=min_ss,
        morph_k=morph_k,
    )

    os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_heatmap_path), exist_ok=True)
    cv2.imwrite(out_mask_path, filtered_mask)
    cv2.imwrite(out_heatmap_path, overlay)
    return {
        "cell_count": cell_count,
        "positive_pixels": positive_pixels,
        "noise_instance_count": noise_instances,
        "noise_pixels": noise_pixels,
        "min_detection_area_pixels": min_a,
        "min_detection_short_side_pixels": min_ss,
        "speckle_morph_open_kernel": morph_k,
        "mask_path": out_mask_path,
        "heatmap_path": out_heatmap_path,
    }
