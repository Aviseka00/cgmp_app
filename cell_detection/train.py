"""
Train a U-Net to segment positive cells from microscopy JPG + Cellpose *_seg.npy pairs.

Usage (from d:\\IPA\\cell_detection):
  python train.py --data_dir .. --epochs 60 --encoder mit_b0

Defaults use mit_b0 (strong pretrained) and 5-fold CV summary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FullImageTensorDataset, PositiveCellCropDataset
from io_utils import dedupe_pairs_by_image_hash, discover_pairs
from metrics import dice_score_binary, iou_binary
from modeling import build_model
from qc_report import embed_qc_into_checkpoint, run_val_qc

try:
    import segmentation_models_pytorch as smp
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install: pip install segmentation-models-pytorch") from e


def kfold_indices(n: int, k: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train_idx, val_idx))
    return out


def train_one_fold(
    pairs: list[tuple[str, str]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]

    train_ds = PositiveCellCropDataset(
        train_pairs,
        patch=args.patch,
        crops_per_image=args.crops_per_image,
        pos_fraction=args.pos_fraction,
        seed=args.seed,
    )
    val_ds = FullImageTensorDataset(val_pairs, long_side=args.val_long_side)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = build_model(encoder=args.encoder).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss = smp.losses.SoftBCEWithLogitsLoss()
    use_amp = device.type == "cuda"
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    best_dice = 0.0
    best_iou_at_best_dice = 0.0
    best_path = os.path.join(args.out_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Ep{epoch}/{args.epochs} train", leave=False)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = dice_loss(logits, y) + args.bce_weight * bce_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses[-20:])))
        scheduler.step()

        model.eval()
        dices, ious = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["mask"].to(device)
                logits = model(x)
                prob = torch.sigmoid(logits)
                dices.append(dice_score_binary(prob, y).item())
                ious.append(iou_binary(prob, y).item())
        mean_dice = float(np.mean(dices)) if dices else 0.0
        mean_iou = float(np.mean(ious)) if ious else 0.0
        tqdm.write(
            f"  epoch {epoch:03d} train_loss={np.mean(losses):.4f} "
            f"val_dice={mean_dice:.4f} val_iou={mean_iou:.4f}"
        )

        if mean_dice >= best_dice:
            best_dice = mean_dice
            best_iou_at_best_dice = mean_iou
            torch.save(
                {
                    "model": model.state_dict(),
                    "encoder": args.encoder,
                    "val_dice": mean_dice,
                    "val_iou": mean_iou,
                    "train_paths": [p[0] for p in train_pairs],
                    "val_paths": [p[0] for p in val_pairs],
                },
                best_path,
            )

    ckpt_best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_best["model"])
    qc_report = run_val_qc(
        model,
        val_loader,
        device,
        out_dir=args.out_dir,
        threshold=args.qc_threshold,
        tag="val",
    )
    embed_qc_into_checkpoint(best_path, qc_report, args.qc_threshold)
    scalars = qc_report["scalars"]
    roc_s = f"{scalars['roc_auc']:.4f}" if scalars["roc_auc"] is not None else "n/a"
    ap_s = f"{scalars['average_precision']:.4f}" if scalars["average_precision"] is not None else "n/a"
    tqdm.write(
        f"  QC (val pixels, thr={args.qc_threshold}): "
        f"F1={scalars['f1_score']:.4f} Dice={scalars['dice_coefficient']:.4f} "
        f"IoU={scalars['iou_jaccard']:.4f} ROC_AUC={roc_s} AP={ap_s}"
    )

    return {
        "best_val_dice": best_dice,
        "best_val_iou": best_iou_at_best_dice,
        "checkpoint": best_path,
        "qc_metrics": scalars,
        "qc_report_path": os.path.join(args.out_dir, "val_qc_metrics.json"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), ".."))
    p.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), "runs"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder", type=str, default="mit_b0")
    p.add_argument("--patch", type=int, default=512)
    p.add_argument("--crops_per_image", type=int, default=36)
    p.add_argument("--pos_fraction", type=float, default=0.72)
    p.add_argument("--val_long_side", type=int, default=1024)
    p.add_argument("--bce_weight", type=float, default=0.5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--single_fold", type=int, default=-1, help="If >=0, train only this fold id")
    p.add_argument(
        "--qc_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for confusion matrix / F1 / etc.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    root_out = os.path.abspath(args.out_dir)
    os.makedirs(root_out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = dedupe_pairs_by_image_hash(discover_pairs(data_dir))
    if len(pairs) < 3:
        print("Need at least 3 image/seg pairs after dedupe.", file=sys.stderr)
        sys.exit(1)

    meta = {
        "n_pairs": len(pairs),
        "pairs": [{"image": a, "seg": b} for a, b in pairs],
        "encoder": args.encoder,
    }
    with open(os.path.join(root_out, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Device: {device} | Unique image pairs: {len(pairs)}")
    splits = kfold_indices(len(pairs), min(args.folds, len(pairs)), seed=args.seed)

    results = []
    fold_range = (
        [args.single_fold]
        if args.single_fold >= 0
        else list(range(len(splits)))
    )
    for fi in fold_range:
        if fi < 0 or fi >= len(splits):
            continue
        tr, va = splits[fi]
        print(f"\n=== Fold {fi + 1}/{len(splits)} | train={len(tr)} val={len(va)} ===")
        fold_dir = os.path.join(root_out, f"fold_{fi}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_args = argparse.Namespace(**vars(args))
        fold_args.out_dir = fold_dir
        out = train_one_fold(pairs, tr, va, fold_args, device)
        out["fold"] = fi
        results.append(out)

    if results:
        mean_dice = float(np.mean([r["best_val_dice"] for r in results]))
        print("\n=== Summary ===")
        for r in results:
            q = r.get("qc_metrics") or {}
            roc = q.get("roc_auc")
            f1 = q.get("f1_score")
            roc_p = f"{float(roc):.4f}" if roc is not None else "n/a"
            f1_p = f"{float(f1):.4f}" if f1 is not None else "n/a"
            print(
                f"  fold {r['fold']}: best val dice={r['best_val_dice']:.4f} "
                f"F1={f1_p} ROC_AUC={roc_p}"
            )
        print(f"Mean best val dice (across trained folds): {mean_dice:.4f}")

        def _auc_key(rec: dict) -> float:
            v = (rec.get("qc_metrics") or {}).get("roc_auc")
            return float(v) if v is not None else -1.0

        best = max(results, key=_auc_key)
        if _auc_key(best) < 0:
            best = max(results, key=lambda r: r["best_val_dice"])
        prod_ckpt = os.path.join(root_out, "production_model.pth")
        shutil.copy2(best["checkpoint"], prod_ckpt)
        for name in (
            "val_qc_metrics.json",
            "val_confusion_matrix.png",
            "val_roc_curve.png",
            "val_pr_curve.png",
        ):
            src = os.path.join(os.path.dirname(best["checkpoint"]), name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(root_out, "production_" + name))
        print(
            f"\nProduction bundle (best fold by ROC-AUC, else Dice): fold {best['fold']}\n"
            f"  {prod_ckpt}\n"
            f"  {os.path.join(root_out, 'production_val_qc_metrics.json')}"
        )

        def _json_safe(o: object) -> object:
            if isinstance(o, dict):
                return {k: _json_safe(v) for k, v in o.items()}
            if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
                return None
            return o

        with open(os.path.join(root_out, "cv_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                _json_safe(
                    {
                        "folds": results,
                        "mean_best_val_dice": mean_dice,
                        "production_fold": best["fold"],
                        "production_checkpoint": prod_ckpt,
                    }
                ),
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
