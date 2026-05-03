"""Pixel-level QC: ROC/AUC, confusion matrix, F1, PR-AUC, plots, JSON."""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _collect_pixels(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_chunks: list[np.ndarray] = []
    y_score_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)
            y_true_chunks.append(y.detach().cpu().numpy().ravel().astype(np.uint8))
            y_score_chunks.append(prob.detach().cpu().numpy().ravel().astype(np.float32))
    y_true = np.concatenate(y_true_chunks)
    y_score = np.concatenate(y_score_chunks)
    return y_true, y_score


def _scalar_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(np.uint8)
    labels_present = np.unique(y_true)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    denom = tp + fp
    precision = float(tp / denom) if denom > 0 else 0.0
    denom_r = tp + fn
    recall = float(tp / denom_r) if denom_r > 0 else 0.0
    denom_s = tn + fp
    specificity = float(tn / denom_s) if denom_s > 0 else 0.0
    accuracy = float((tp + tn) / max(1, (tp + tn + fp + fn)))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    iou = float(jaccard_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0
    kappa = float(cohen_kappa_score(y_true, y_pred))

    roc_auc: float | None
    if len(labels_present) < 2:
        roc_auc = None
    else:
        roc_auc = float(roc_auc_score(y_true, y_score))

    pr_auc: float | None
    if len(labels_present) < 2:
        pr_auc = None
    else:
        pr_auc = float(average_precision_score(y_true, y_score))

    dice = float((2 * tp) / max(1, 2 * tp + fp + fn))

    return {
        "threshold": float(threshold),
        "pixel_counts": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "n_positive_pixels": n_pos,
        "n_negative_pixels": n_neg,
        "positive_pixel_fraction": float(n_pos / max(1, n_pos + n_neg)),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
        "f1_score": f1,
        "dice_coefficient": dice,
        "iou_jaccard": iou,
        "matthews_correlation_coefficient": mcc,
        "cohen_kappa": kappa,
        "roc_auc": roc_auc,
        "average_precision": pr_auc,
    }


def _plot_confusion(cm: np.ndarray, path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=("Pred 0", "Pred 1"),
        yticklabels=("True 0", "True 1"),
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Receiver operating characteristic (pixel level)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_pr(rec: np.ndarray, prec: np.ndarray, ap: float, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rec, prec, lw=2, label=f"PR (AP = {ap:.4f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curve (pixel level)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_val_qc(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_dir: str,
    threshold: float = 0.5,
    tag: str = "val",
) -> dict[str, Any]:
    """
    Aggregate pixel predictions over the validation loader (same tensors as training val),
    compute metrics, save JSON + PNGs under out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    y_true, y_score = _collect_pixels(model, val_loader, device)
    scalars = _scalar_metrics(y_true, y_score, threshold)

    y_pred = (y_score >= threshold).astype(np.uint8)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    _plot_confusion(
        cm,
        os.path.join(out_dir, f"{tag}_confusion_matrix.png"),
        title=f"Confusion matrix ({tag}, thr={threshold})",
    )

    if scalars["roc_auc"] is not None and scalars["average_precision"] is not None:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        _plot_roc(fpr, tpr, scalars["roc_auc"], os.path.join(out_dir, f"{tag}_roc_curve.png"))
        prec_c, rec_c, _ = precision_recall_curve(y_true, y_score)
        ap = float(scalars["average_precision"])
        _plot_pr(rec_c, prec_c, ap, os.path.join(out_dir, f"{tag}_pr_curve.png"))

    def _sanitize(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _sanitize(v) for k, v in x.items()}
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        if isinstance(x, (np.floating,)):
            v = float(x)
            return None if np.isnan(v) or np.isinf(v) else v
        return x

    report = {
        "scalars": _sanitize(scalars),
        "confusion_matrix_2x2": cm.tolist(),
        "notes": (
            "Pixel-level metrics on validation tensors (same resize as training val). "
            "ROC/PR require both classes in the pooled validation set."
        ),
    }
    json_path = os.path.join(out_dir, f"{tag}_qc_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def embed_qc_into_checkpoint(ckpt_path: str, qc_report: dict[str, Any], threshold: float) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["qc_metrics"] = qc_report["scalars"]
    ckpt["qc_confusion_matrix"] = qc_report["confusion_matrix_2x2"]
    ckpt["qc_threshold"] = float(threshold)
    ckpt["qc_notes"] = qc_report.get("notes", "")
    torch.save(ckpt, ckpt_path)
