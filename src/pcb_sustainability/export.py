"""Export helpers for reports."""

from __future__ import annotations

from dataclasses import asdict
from io import BytesIO

from fpdf import FPDF

from .models import ComponentScore, PCBFeatures, PCBScore, RecoveryEstimate
from .scoring import component_scores_to_dataframe


def _pdf_safe(value: object) -> str:
    """Convert arbitrary report text into built-in Helvetica-safe PDF text."""

    text = "" if value is None else str(value)
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00b2": "2",
        "\u03a9": "Ohm",
        "\u03bc": "u",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_json_report(
    component_scores: list[ComponentScore],
    bom_summary: dict,
    pcb_features: PCBFeatures,
    pcb_score: PCBScore,
    recovery: RecoveryEstimate,
) -> dict:
    return {
        "bom_summary": bom_summary,
        "component_scores": [asdict(item) for item in component_scores],
        "pcb_features": asdict(pcb_features),
        "pcb_score": asdict(pcb_score),
        "recovery_estimate": asdict(recovery),
    }


def component_csv_bytes(component_scores: list[ComponentScore]) -> bytes:
    return component_scores_to_dataframe(component_scores).to_csv(index=False).encode("utf-8")


def pdf_report_bytes(report: dict) -> bytes:
    """Generate a concise PDF suitable for academic demo export."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _pdf_safe("PCB Sustainability & Recyclability Report"), ln=True)
    pdf.set_font("Helvetica", "", 10)

    summary = report["bom_summary"]
    pcb_score = report["pcb_score"]
    recovery = report["recovery_estimate"]
    lines = [
        f"BoM sustainability score: {summary.get('summary_score', 0)}/100",
        f"PCB design-for-recycling score: {pcb_score.get('score', 0)}/100",
        f"Final recyclability score: {recovery.get('final_score', 0)}/100",
        f"Estimated recovery: {recovery.get('recovery_percent', 0)}%",
        f"Components analyzed: {summary.get('component_count', 0)}",
    ]
    for line in lines:
        pdf.cell(0, 7, _pdf_safe(line), ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _pdf_safe("Key Recommendations"), ln=True)
    pdf.set_font("Helvetica", "", 9)
    shown = 0
    for component in report["component_scores"]:
        for rec in component.get("recommendations", [])[:2]:
            pdf.multi_cell(
                0,
                5,
                _pdf_safe(f"- {component.get('part_number')}: {rec.get('label')} - {rec.get('reason')}"),
                new_x="LMARGIN",
                new_y="NEXT",
            )
            shown += 1
            if shown >= 10:
                break
        if shown >= 10:
            break
    for rec in report["pcb_score"].get("recommendations", []):
        pdf.multi_cell(
            0,
            5,
            _pdf_safe(f"- PCB: {rec.get('label')} - {rec.get('reason')}"),
            new_x="LMARGIN",
            new_y="NEXT",
        )

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _pdf_safe("Estimated Recoverable Materials"), ln=True)
    pdf.set_font("Helvetica", "", 9)
    for material, value in recovery.get("materials", {}).items():
        pdf.cell(0, 5, _pdf_safe(f"{material}: {value}"), ln=True)

    data = pdf.output(dest="S")
    if isinstance(data, str):
        return data.encode("latin-1")
    buffer = BytesIO(data)
    return buffer.getvalue()
