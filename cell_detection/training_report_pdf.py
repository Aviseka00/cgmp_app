"""
Generate a PDF training summary from a run directory (cv_summary.json, dataset_meta.json,
production plots). Uses matplotlib only (already in requirements).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

# Theme — cohesive palette for an attractive “dashboard” look
C = {
    "bg_top": "#0b1220",
    "bg_mid": "#111d32",
    "card": "#1a2744",
    "card_border": "#2d3f5c",
    "text": "#f1f5f9",
    "muted": "#94a3b8",
    "accent1": "#38bdf8",  # sky
    "accent2": "#34d399",  # emerald
    "accent3": "#a78bfa",  # violet
    "accent4": "#fbbf24",  # amber
    "accent5": "#fb7185",  # rose
    "header_grad_a": "#2563eb",
    "header_grad_b": "#7c3aed",
}


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v or abs(v) == float("inf"):
            return "—"
        return f"{v:.{digits}f}"
    return str(v)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _gradient_ax(ax: plt.Axes, y0: float = 0.0, y1: float = 1.0) -> None:
    """Vertical gradient background."""
    n = 120
    z = np.linspace(0, 1, n).reshape(-1, 1)
    z = np.tile(z, (1, 80))
    cmap = LinearSegmentedColormap.from_list("bg", [C["bg_top"], C["bg_mid"], "#162038"], N=256)
    ax.imshow(z, extent=[0, 1, y0, y1], origin="lower", aspect="auto", cmap=cmap, zorder=0)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _kpi_box(
    fig: plt.Figure,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    value: str,
    face: str,
    z: int = 5,
) -> None:
    ax = fig.add_axes([x, y, w, h])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fancy = FancyBboxPatch(
        (0.02, 0.08),
        0.96,
        0.84,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=face,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.92,
        zorder=z,
    )
    ax.add_patch(fancy)
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=16, fontweight="bold", color="white", zorder=z + 1)
    ax.text(0.5, 0.28, label, ha="center", va="center", fontsize=8, color=(1, 1, 1, 0.92), zorder=z + 1)


def _section_card(
    ax: plt.Axes,
    x: float,
    y_bottom: float,
    width: float,
    height: float,
    title: str,
    title_color: str,
    body_lines: list[str],
    fontsize: float = 7.2,
) -> None:
    fancy = FancyBboxPatch(
        (x, y_bottom),
        width,
        height,
        transform=ax.transAxes,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=C["card"],
        edgecolor=C["card_border"],
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(fancy)
    ax.text(
        x + 0.02,
        y_bottom + height - 0.028,
        title,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=title_color,
        zorder=3,
        va="top",
    )
    body = "\n".join(body_lines)
    ax.text(
        x + 0.03,
        y_bottom + height - 0.065,
        body,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=C["text"],
        va="top",
        ha="left",
        family="sans-serif",
        zorder=3,
        linespacing=1.35,
    )


def add_cover_page(
    pdf: PdfPages,
    *,
    title: str,
    subtitle: str,
    generated: str,
    run_dir: Path,
    kpi_roc: str,
    kpi_dice: str,
    kpi_ap: str,
    sections: list[tuple[str, str, list[str]]],
) -> None:
    """
    sections: (accent_color_hex, section_title, lines)
    """
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    _gradient_ax(ax)

    # Top decorative bar (gradient strip)
    grad_bar = np.linspace(0, 1, 64).reshape(1, -1)
    ax.imshow(
        grad_bar,
        extent=[0, 1, 0.91, 0.985],
        origin="lower",
        aspect="auto",
        cmap=LinearSegmentedColormap.from_list("bar", [C["header_grad_a"], C["header_grad_b"]]),
        zorder=1,
    )

    fig.text(0.5, 0.945, title, ha="center", va="center", fontsize=22, fontweight="bold", color="white", zorder=4)
    fig.text(0.5, 0.898, subtitle, ha="center", va="center", fontsize=10, color=C["muted"], style="italic", zorder=4)
    fig.text(0.5, 0.865, f"Generated {generated}  ·  {run_dir.name}", ha="center", va="center", fontsize=7.5, color=C["muted"], zorder=4)

    # KPI row
    _kpi_box(fig, 0.07, 0.72, 0.26, 0.11, "ROC AUC", kpi_roc, C["accent1"])
    _kpi_box(fig, 0.37, 0.72, 0.26, 0.11, "Mean val Dice", kpi_dice, C["accent2"])
    _kpi_box(fig, 0.67, 0.72, 0.26, 0.11, "Average precision", kpi_ap, C["accent3"])

    # Section cards — stack from bottom of KPI area (height scales with content)
    y_cursor = 0.68
    gap = 0.014
    for accent, sec_title, lines in sections:
        card_h = min(0.28, max(0.095, 0.036 + len(lines) * 0.0175))
        fs = max(6.25, min(7.4, 8.0 - len(lines) * 0.07))
        _section_card(ax, 0.06, y_cursor - card_h, 0.88, card_h, sec_title, accent, lines, fontsize=fs)
        y_cursor -= card_h + gap
        if y_cursor < 0.05:
            break

    pdf.savefig(fig, facecolor=C["bg_top"], dpi=150)
    plt.close(fig)


def add_config_table_page(
    pdf: PdfPages,
    title: str,
    rows: list[tuple[str, str]],
    accent: str,
) -> None:
    """Two-column key/value config table with colorful styling."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    fig.patch.set_facecolor(C["bg_top"])
    _gradient_ax(ax)

    fig.text(0.5, 0.94, title, ha="center", fontsize=15, fontweight="bold", color="white")
    fig.text(0.5, 0.905, "Hyperparameters & pipeline defaults", ha="center", fontsize=9, color=C["muted"])

    headers = ["Setting", "Value"]
    data = [[k, str(v)] for k, v in rows]
    tbl = ax.table(
        cellText=data,
        colLabels=headers,
        bbox=[0.08, 0.08, 0.84, 0.78],
        cellLoc="left",
        colWidths=[0.48, 0.52],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.05, 1.65)

    header_bg = accent
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["card_border"])
        cell.set_linewidth(1)
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(0.06)
        else:
            alt = (r % 2 == 0)
            cell.set_facecolor("#1e2d47" if alt else "#152238")
            cell.set_text_props(color=C["text"])
    pdf.savefig(fig, facecolor=C["bg_top"], dpi=150)
    plt.close(fig)


def add_table_page(
    pdf: PdfPages,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    highlight_fold_col: int | None = None,
    highlight_value: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    fig.patch.set_facecolor(C["bg_top"])
    _gradient_ax(ax)

    fig.text(0.5, 0.94, title, ha="center", fontsize=15, fontweight="bold", color="white")
    fig.text(0.5, 0.905, "Cross-validation fold comparison", ha="center", fontsize=9, color=C["muted"])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        bbox=[0.04, 0.1, 0.92, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.45)

    header_cmap = LinearSegmentedColormap.from_list("h", [C["header_grad_a"], C["header_grad_b"]])
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(C["card_border"])
        cell.set_linewidth(0.8)
        if row == 0:
            # gradient-ish header: vary by column
            t = col / max(len(headers) - 1, 1)
            cell.set_facecolor(header_cmap(0.35 + 0.5 * t))
            cell.set_text_props(color="white", fontweight="bold")
        else:
            row_is_prod = (
                highlight_fold_col is not None
                and highlight_value is not None
                and rows[row - 1][highlight_fold_col] == highlight_value
            )
            is_highlight = row_is_prod
            if is_highlight:
                cell.set_facecolor("#14532d")
                cell.set_text_props(color="#bbf7d0", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#1a2744")
                cell.set_text_props(color=C["text"])
            else:
                cell.set_facecolor("#152238")
                cell.set_text_props(color=C["text"])

    pdf.savefig(fig, facecolor=C["bg_top"], dpi=150)
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path, accent: str) -> None:
    if not image_path.is_file():
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0, 0, 1, 1])
        _gradient_ax(ax)
        fig.text(0.5, 0.5, f"Missing: {image_path.name}", ha="center", color=C["muted"])
        pdf.savefig(fig, facecolor=C["bg_top"], dpi=150)
        plt.close(fig)
        return

    img = mpimg.imread(str(image_path))
    fig = plt.figure(figsize=(8.5, 11))
    ax_bg = fig.add_axes([0, 0, 1, 1])
    _gradient_ax(ax_bg)

    # Title ribbon
    ax_bg.add_patch(
        Rectangle(
            (0, 0.91),
            1,
            0.09,
            transform=ax_bg.transAxes,
            facecolor=accent,
            alpha=0.95,
            zorder=1,
        )
    )
    fig.text(0.5, 0.945, title, ha="center", va="center", fontsize=13, fontweight="bold", color="white", zorder=3)

    # Image with colored frame (axes border)
    ax_img = fig.add_axes([0.08, 0.08, 0.84, 0.80])
    ax_img.imshow(img)
    ax_img.axis("off")
    frame = FancyBboxPatch(
        (0, 0),
        1,
        1,
        transform=ax_img.transAxes,
        boxstyle="square,pad=0",
        fill=False,
        edgecolor=accent,
        linewidth=3.5,
        zorder=10,
    )
    ax_img.add_patch(frame)

    pdf.savefig(fig, facecolor=C["bg_top"], dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Build training summary PDF from a run folder.")
    p.add_argument(
        "--run_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "runs_cv_full"),
        help="Directory containing cv_summary.json and dataset_meta.json",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PDF path (default: <run_dir>/Training_Summary_Report.pdf)",
    )
    args = p.parse_args()
    run_dir = Path(os.path.abspath(args.run_dir))
    out_pdf = Path(args.out) if args.out else run_dir / "Training_Summary_Report.pdf"

    cv_path = run_dir / "cv_summary.json"
    meta_path = run_dir / "dataset_meta.json"
    prod_json = run_dir / "production_val_qc_metrics.json"

    if not cv_path.is_file():
        raise SystemExit(f"Missing {cv_path}")

    cv = _load_json(cv_path)
    meta = _load_json(meta_path) if meta_path.is_file() else {}
    prod = _load_json(prod_json) if prod_json.is_file() else {}

    train_defaults = {
        "epochs": 60,
        "batch_size": 8,
        "learning_rate": "3e-4",
        "encoder_backbone": meta.get("encoder", "mit_b0"),
        "patch_size": 512,
        "crops_per_image": 36,
        "pos_fraction": 0.72,
        "val_long_side": 1024,
        "bce_weight": 0.5,
        "folds": 5,
        "seed": 42,
        "qc_threshold": 0.5,
        "optimizer": "AdamW + cosine scheduler",
        "loss": "Dice + BCE (weighted)",
        "amp": "enabled when CUDA",
    }

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_pairs = meta.get("n_pairs", "?")
    folds = cv.get("folds") or []
    mean_dice = cv.get("mean_best_val_dice")
    prod_fold = cv.get("production_fold")
    prod_ckpt = cv.get("production_checkpoint", "")

    scalars = (prod.get("scalars") or {}) if prod else {}
    if not scalars and folds and prod_fold is not None:
        for f in folds:
            if f.get("fold") == prod_fold:
                scalars = f.get("qc_metrics") or {}
                break

    kpi_roc = _fmt(scalars.get("roc_auc"))
    kpi_ap = _fmt(scalars.get("average_precision"))
    kpi_dice = _fmt(mean_dice)

    dataset_lines = [
        f"Unique image / mask pairs: {n_pairs}",
        f"Encoder backbone: {meta.get('encoder', '—')}",
        f"Model family: U-Net (segmentation-models-pytorch)",
    ]
    cv_lines = [
        f"Folds: {len(folds)}",
        f"Mean best validation Dice: {_fmt(mean_dice)}",
        f"Production fold (best ROC-AUC): {prod_fold}",
        f"Checkpoint: {Path(str(prod_ckpt)).name}" if prod_ckpt else "—",
    ]
    qc_lines = [
        f"Threshold: {_fmt(scalars.get('threshold'), 2)}  ·  Accuracy: {_fmt(scalars.get('accuracy'))}",
        f"F1 / Dice: {_fmt(scalars.get('f1_score'))}  ·  IoU: {_fmt(scalars.get('iou_jaccard'))}",
        f"Precision / Recall: {_fmt(scalars.get('precision'))} / {_fmt(scalars.get('recall'))}",
        f"Specificity: {_fmt(scalars.get('specificity'))}  ·  Balanced acc.: {_fmt(scalars.get('balanced_accuracy'))}",
        f"MCC: {_fmt(scalars.get('matthews_correlation_coefficient'))}  ·  Cohen κ: {_fmt(scalars.get('cohen_kappa'))}",
    ]
    pc = scalars.get("pixel_counts") or {}
    if pc:
        qc_lines.append(f"Pixels — TN {pc.get('TN')}  FP {pc.get('FP')}  FN {pc.get('FN')}  TP {pc.get('TP')}")
    qc_lines.append(f"Positive pixel fraction: {_fmt(scalars.get('positive_pixel_fraction'))}")
    if prod.get("notes"):
        qc_lines.append("")
        qc_lines.append(str(prod["notes"])[:220])

    # Page 1 sections (compact)
    sections_p1: list[tuple[str, str, list[str]]] = [
        (C["accent2"], "Dataset", dataset_lines),
        (C["accent3"], "Cross-validation summary", cv_lines),
        (C["accent4"], "Production model — validation QC", qc_lines),
    ]

    config_kv = [(k.replace("_", " ").title(), v) for k, v in train_defaults.items()]

    headers = [
        "Fold",
        "Best val Dice",
        "Best val IoU",
        "ROC AUC",
        "AP",
        "F1",
        "Prec",
        "Rec",
        "Spec",
        "Bal acc",
    ]
    table_rows: list[list[str]] = []
    for f in sorted(folds, key=lambda x: x.get("fold", 0)):
        q = f.get("qc_metrics") or {}
        table_rows.append(
            [
                str(f.get("fold", "")),
                _fmt(f.get("best_val_dice")),
                _fmt(f.get("best_val_iou")),
                _fmt(q.get("roc_auc")),
                _fmt(q.get("average_precision")),
                _fmt(q.get("f1_score")),
                _fmt(q.get("precision")),
                _fmt(q.get("recall")),
                _fmt(q.get("specificity")),
                _fmt(q.get("balanced_accuracy")),
            ]
        )

    detail_headers = ["Fold", "Dice", "IoU", "ROC AUC", "AP", "F1", "MCC", "κ", "IoU (QC)"]
    detail_rows: list[list[str]] = []
    for f in sorted(folds, key=lambda x: x.get("fold", 0)):
        q = f.get("qc_metrics") or {}
        detail_rows.append(
            [
                str(f.get("fold", "")),
                _fmt(f.get("best_val_dice")),
                _fmt(f.get("best_val_iou")),
                _fmt(q.get("roc_auc")),
                _fmt(q.get("average_precision")),
                _fmt(q.get("f1_score")),
                _fmt(q.get("matthews_correlation_coefficient")),
                _fmt(q.get("cohen_kappa")),
                _fmt(q.get("iou_jaccard")),
            ]
        )

    prod_fold_str = str(prod_fold) if prod_fold is not None else ""

    with PdfPages(out_pdf) as pdf:
        add_cover_page(
            pdf,
            title="Training Summary Report",
            subtitle="Cell segmentation · Cross-validation & production bundle · IPA",
            generated=generated,
            run_dir=run_dir,
            kpi_roc=kpi_roc,
            kpi_dice=kpi_dice,
            kpi_ap=kpi_ap,
            sections=sections_p1,
        )

        add_config_table_page(
            pdf,
            "Training configuration",
            config_kv,
            accent=C["accent1"],
        )

        add_table_page(
            pdf,
            "Per-fold metrics (validation)",
            headers,
            table_rows,
            highlight_fold_col=0,
            highlight_value=prod_fold_str,
        )
        add_table_page(
            pdf,
            "Per-fold extended QC metrics",
            detail_headers,
            detail_rows,
            highlight_fold_col=0,
            highlight_value=prod_fold_str,
        )

        add_image_page(pdf, "Production validation — ROC curve", run_dir / "production_val_roc_curve.png", C["accent1"])
        add_image_page(pdf, "Production validation — Precision–Recall curve", run_dir / "production_val_pr_curve.png", C["accent3"])
        add_image_page(pdf, "Production validation — Confusion matrix (pixels)", run_dir / "production_val_confusion_matrix.png", C["accent5"])

    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
