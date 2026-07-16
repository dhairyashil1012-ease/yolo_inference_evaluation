import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(output_pdf_path, backend, config_data, performance_data, predictions):

    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2C5282"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748")
    )
    
    header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.whitesmoke,
        fontName='Helvetica-Bold'
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph(f"YOLO Classification {backend.upper()} Model Inference Report", title_style))
    story.append(Paragraph(f"<b>Backend Engine:</b> {backend.upper()}", body_style))
    story.append(Spacer(1, 15))

    # PHASE 1: Configuration & System Metadata
    story.append(Paragraph("Phase 1: Configuration & System Metadata", h2_style))
    
    class_names_value = config_data.pop("Class Names", None)
    
    config_table_data = []
    for k, v in config_data.items():
        if k == "Python Version" and isinstance(v, list):
            v = " ".join(v[:2])
            
        config_table_data.append([
            Paragraph(f"<b>{k}</b>", body_style), 
            Paragraph(str(v), body_style)
        ])
        
    t1 = Table(config_table_data, colWidths=[150, 380])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t1)
    story.append(Spacer(1, 10))
    
    if class_names_value:
        story.append(Paragraph("<b>Model Trained Class Names:</b>", body_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph(str(class_names_value), body_style))
        story.append(Spacer(1, 15))

    # PHASE 2: Inference Performance Metrics (Includes Split Preprocess/Inference/Postprocess Times)
    story.append(Paragraph("Phase 2: Inference Performance Metrics", h2_style))
    perf_table_data = [
        [Paragraph(f"<b>{k}</b>", body_style), Paragraph(str(v), body_style)] 
        for k, v in performance_data.items()
    ]
    t2 = Table(perf_table_data, colWidths=[150, 380])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t2)
    story.append(Spacer(1, 15))

    # PHASE 3: Detailed Predictions Table
    story.append(Paragraph("Phase 3: Detailed Predictions Summary", h2_style))
    
    pred_table_data = [[
        Paragraph("<b>Image File</b>", header_style), 
        Paragraph("<b>Predicted Class</b>", header_style), 
        Paragraph("<b>Confidence</b>", header_style)
    ]]
    
    for p in predictions:
        pred_table_data.append([
            Paragraph(p['file_name'], body_style),
            Paragraph(p['class_name'], body_style),
            Paragraph(f"{p['confidence']:.1f}%", body_style)
        ])
        
    t3 = Table(pred_table_data, colWidths=[230, 170, 130])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    
    story.append(t3)

    doc.build(story)
    print(f"\nPDF Report successfully generated at: {output_pdf_path}")