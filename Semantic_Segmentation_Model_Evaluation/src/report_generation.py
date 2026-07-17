import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import VerticalBarChart

def generate_pdf_report(output_pdf_path, backend, config_data, performance_data, predictions):
    """
    Generates a structured PDF report containing metadata, performance metrics, and predictions.
    """
    # 1. Page setup
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
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

    story = []

    # Title & Subtitle
    story.append(Paragraph(f"YOLO Classification {backend.upper()} Model Inference Report", title_style))
    story.append(Paragraph(f"<b>Backend Engine:</b> {backend.upper()}", body_style))
    story.append(Spacer(1, 15))

    # PHASE 1: Configuration Metadata
    story.append(Paragraph("Phase 1: Configuration Metadata", h2_style))
    config_table_data = [
        [Paragraph(f"<b>{k}</b>", body_style), Paragraph(str(v), body_style)] 
        for k, v in config_data.items()
    ]
    t1 = Table(config_table_data, colWidths=[150, 380])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t1)
    story.append(Spacer(1, 15))

    # PHASE 2: Inference Performance Metrics
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

    # PHASE 3: Chart Visualization
    story.append(Paragraph("Phase 3: Confidence Score Analytics", h2_style))
    
    # Extract data for the chart
    names = [p['file_name'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    
    if confidences:
        # Drawing canvas for ReportLab Graphics
        d = Drawing(530, 160)
        chart = VerticalBarChart()
        chart.x = 40
        chart.y = 25
        chart.height = 110
        chart.width = 460
        chart.data = [confidences]
        chart.categoryAxis.categoryNames = [n if len(n) < 12 else n[:10]+"..." for n in names]
        chart.categoryAxis.labels.fontSize = 8
        chart.categoryAxis.labels.dy = -10
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = 100
        chart.valueAxis.valueStep = 20
        chart.valueAxis.labels.fontSize = 8
        chart.bars[0].fillColor = colors.HexColor("#3182CE")
        d.add(chart)
        story.append(d)
    else:
        story.append(Paragraph("No prediction data available to chart.", body_style))
    story.append(Spacer(1, 15))

    # PHASE 4: Detailed Predictions Table
    story.append(Paragraph("Phase 4: Detailed Predictions Summary", h2_style))
    pred_table_data = [[
        Paragraph("<b>Image File</b>", body_style), 
        Paragraph("<b>Predicted Class</b>", body_style), 
        Paragraph("<b>Confidence</b>", body_style)
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
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    
    # We wrap the last table in a KeepTogether so it doesn't awkwardly fracture pages if unnecessary
    story.append(KeepTogether(t3))

    # Build the document
    doc.build(story)
    print(f"\nPDF Report successfully generated at: {output_pdf_path}")
