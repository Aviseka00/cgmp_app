"""
Sliding-window inference on full-resolution images.

Usage (from d:\\IPA\\cell_detection):
  python predict.py --checkpoint runs/fold_0/best_model.pth --image "..\\QS_2834.jpg" --out pred.png
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch

from modeling import build_model


def gaussian_window(h: int, w: int) -> np.ndarray:
    yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    d2 = yy * yy + xx * xx
    return np.exp(-d2 * 3.0)


@torch.no_grad()
def predict_full_image(
    model: torch.nn.Module,
    bgr: np.ndarray,
    device: torch.device,
    tile: int = 512,
    stride: int = 256,
) -> np.ndarray:
    """Return float32 probability map (H,W) same size as input image."""
    h, w = bgr.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)
    wgt = np.zeros((h, w), dtype=np.float32)
    gw = gaussian_window(tile, tile)
    model.eval()
    for y0 in range(0, max(1, h - tile + 1), stride):
        for x0 in range(0, max(1, w - tile + 1), stride):
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            crop = bgr[y0:y1, x0:x1]
            ch, cw = crop.shape[:2]
            pad_h = tile - ch
            pad_w = tile - cw
            if pad_h > 0 or pad_w > 0:
                crop = cv2.copyMakeBorder(
                    crop, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
                )
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            chw = np.transpose(rgb, (2, 0, 1))[None, ...]
            t = torch.from_numpy(chw).to(device)
            logit = model(t)
            prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
            prob = prob[:ch, :cw]
            g = gw[:ch, :cw]
            acc[y0:y1, x0:x1] += prob * g
            wgt[y0:y1, x0:x1] += g
    wgt[wgt < 1e-6] = 1.0
    return acc / wgt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out", type=str, default="prediction.png")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--binary_thresh", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder = ckpt.get("encoder", "mit_b0")
    model = build_model(encoder=encoder).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")

    prob = predict_full_image(model, bgr, device, tile=args.tile, stride=args.stride)
    binary = (prob >= args.binary_thresh).astype(np.uint8) * 255
    overlay = bgr.copy()
    heat = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(bgr, 0.65, heat_color, 0.35, 0)
    out_path = args.out
    stem, ext = os.path.splitext(out_path)
    cv2.imwrite(out_path, binary)
    cv2.imwrite(stem + "_heatmap" + ext, blended)
    print(f"Wrote {out_path} and {stem + '_heatmap' + ext}")


if __name__ == "__main__":
    main()
