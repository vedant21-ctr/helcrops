from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
from typing import Optional
import os

def create_pdf(report_text: str, crop_name: str, generated_at: Optional[str] = None):
    """
    Build a structured, professional PDF advisory report.
    """
    os.makedirs("reports", exist_ok=True)
    safe_crop = "".join(c for c in crop_name if c.isalnum() or c in (" ", "-", "_")).strip() or "Crop"
    file_path = f"reports/Farm_Advisory_{safe_crop.replace(' ', '_')}.pdf"
    when = generated_at or datetime.now().strftime("%B %d, %Y at %H:%M")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=22,
        textColor=HexColor("#00C853"),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontSize=11,
        textColor=HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=18,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=HexColor("#2979FF"),
        spaceBefore=14,
        spaceAfter=8,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=14,
        textColor=HexColor("#1A1A1A"),
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=18,
        bulletIndent=8,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=9,
        textColor=HexColor("#B45309"),
        backColor=HexColor("#FFF7ED"),
        borderPadding=10,
        spaceBefore=12,
    )

    doc = SimpleDocTemplate(
        file_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    story = []

    # Header band (table simulates colored header)
    header_data = [[Paragraph("<b>AI Advisory Report</b><br/><font size=9 color='#666666'>HaveCrops Analytics</font>", styles["Normal"])]]
    header_table = Table(header_data, colWidths=[7 * inch])
    header_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#E8F5E9")),
                ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#C8E6C9")),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
                ("LEFTPADDING", (0, 0), (-1, -1), 16),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 0.2 * inch))

    meta = Table(
        [
            [
                Paragraph(f"<b>Crop:</b> {crop_name}", body_style),
                Paragraph(f"<b>Generated:</b> {when}", body_style),
            ]
        ],
        colWidths=[3.2 * inch, 3.2 * inch],
    )
    meta.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(meta)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Executive summary", h2_style))
    story.append(
        Paragraph(
            "This document summarizes model-assisted yield context and agent-generated agronomic guidance. "
            "Verify recommendations with local extension services and soil testing.",
            body_style,
        )
    )
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Detailed advisory", h2_style))

    lines = report_text.split("\n")
    for raw in lines:
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 0.06 * inch))
            continue
        clean = line.replace("**", "").replace("#", "").strip()
        if line.startswith("#"):
            story.append(Paragraph(f"<b>{clean.lstrip('#').strip()}</b>", h2_style))
        elif (line.startswith("-") or line.startswith("*")) and len(line) > 1:
            txt = line.lstrip("-*").strip().replace("**", "")
            story.append(Paragraph(f"• {txt}", bullet_style))
        else:
            story.append(Paragraph(clean, body_style))

    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This AI-generated report supports educational and preliminary planning. "
            "It does not replace certified agronomists, regulatory guidance, or field-specific validation.",
            disclaimer_style,
        )
    )

    doc.build(story)
    return file_path
