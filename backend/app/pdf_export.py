"""Generate printable PDF summaries for analysis archive records (fpdf2)."""

from __future__ import annotations

from typing import Any

from fpdf import FPDF


def _txt(val: Any) -> str:
    s = "" if val is None else str(val)
    return s.encode("latin-1", "replace").decode("latin-1")


def build_analysis_archive_pdf(doc: dict[str, Any]) -> bytes:
    """Build a PDF bytes blob from an analysis_archives document."""
    pdf = FPDF()
    pdf.set_margins(14, 14, 14)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _txt("Cell Segmentation Analysis Report"), ln=True)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 5, _txt("cGMP Cell Analysis Platform — exported record"), ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, _txt("Batch"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, _txt(f"Code: {doc.get('batch_code') or '—'}"), ln=True)
    pdf.cell(0, 5, _txt(f"Name: {doc.get('batch_name') or '—'}"), ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, _txt("Run metadata"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, _txt(f"Archive / report ID: {doc.get('_id') or '—'}"), ln=True)
    rn = doc.get("analysis_run_number")
    if rn is not None:
        pdf.cell(0, 5, _txt(f"Analysis run number: {rn}"), ln=True)
    pdf.cell(0, 5, _txt(f"Analyzed at (UTC): {doc.get('analyzed_at') or '—'}"), ln=True)
    pdf.cell(0, 5, _txt(f"Total images: {doc.get('image_count', '—')}"), ln=True)
    pdf.cell(0, 5, _txt(f"Total cells (batch): {doc.get('total_cells', '—')}"), ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, _txt("People"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, _txt(f"Operator (batch owner): {doc.get('created_by_username') or '—'}"), ln=True)
    pdf.cell(0, 5, _txt(f"Analyst (run by): {doc.get('analyzed_by_username') or '—'}"), ln=True)
    pdf.ln(3)

    outputs = doc.get("outputs") or []
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, _txt("Per-image results"), ln=True)
    pdf.set_font("Helvetica", "", 9)

    if not outputs:
        pdf.cell(0, 5, _txt("No image rows stored."), ln=True)
    else:
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(95, 7, _txt("Filename"), border=1, fill=True)
        pdf.cell(0, 7, _txt("Cell count"), border=1, ln=True, fill=True)
        pdf.set_font("Helvetica", "", 8)
        for o in outputs:
            fn = (o.get("filename") or "")[:80]
            cc = o.get("cell_count", "")
            pdf.cell(95, 6, _txt(fn), border=1)
            pdf.cell(0, 6, _txt(str(cc)), border=1, ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _txt("This PDF summarizes the stored analysis run. The authoritative JSON report and checksum chain remain on the server."))

    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1")
    return bytes(out)
