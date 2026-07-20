import datetime
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(output_pdf_path, backend, config_data, performance_data, predictions):
    # Initialize Document Geometry
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Define Custom Corporate Color Profiles
    primary_color = colors.HexColor("#1A365D")    # Dark Slate Blue
    secondary_color = colors.HexColor("#2B6CB0")  # Teal Accent Blue
    text_dark = colors.HexColor("#2D3748")        # Charcoal
    bg_light = colors.HexColor("#F7FAFC")         # Soft Gray Backdrops
    accent_green = colors.HexColor("#2F855A")     # Forest Green
    
    # Establish Standardized Layout Typography Styles
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'],
        fontSize=24, leading=28, textColor=primary_color, spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=colors.HexColor("#718096"), spaceAfter=20
    )
    h1_style = ParagraphStyle(
        'SectionH1', parent=styles['Heading2'],
        fontSize=14, leading=18, textColor=primary_color, spaceBefore=14, spaceAfter=8,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'TableBody', parent=styles['Normal'],
        fontSize=9, leading=12, textColor=text_dark
    )
    header_style = ParagraphStyle(
        'TableHeader', parent=styles['Normal'],
        fontSize=9, leading=12, textColor=colors.white, fontName='Helvetica-Bold'
    )

    story = []

    # Title & Header Segment
    story.append(Paragraph("YOLO26n-sem Performance & Inference Report", title_style))
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated on: {current_time} | Execution Engine: {backend.upper()}", subtitle_style))
    story.append(Spacer(1, 10))

    # SECTION 1: SYSTEM ENVIRONMENT & MODEL SPECIFICATIONS
    story.append(Paragraph("1. System Configuration & Architecture Specs", h1_style))
    
    sys_table_data = []
    # Pivot dictionary key-values into dual columns for clean display
    for label, val in config_data.items():
        sys_table_data.append([
            Paragraph(f"<b>{label}</b>", body_style),
            Paragraph(str(val), body_style)
        ])
        
    sys_table = Table(sys_table_data, colWidths=[200, 332])
    sys_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(sys_table)
    story.append(Spacer(1, 15))

    # SECTION 2: PERFORMANCE METRICS
    story.append(Paragraph("2. Pipeline Speed & Throughput Performance", h1_style))
    
    perf_table_data = []
    for label, val in performance_data.items():
        perf_table_data.append([
            Paragraph(f"<b>{label}</b>", body_style),
            Paragraph(str(val), body_style)
        ])
        
    perf_table = Table(perf_table_data, colWidths=[200, 332])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 15))

    # SECTION 3: IMAGE-LEVEL SEMANTIC SEGMENTATION BREAKDOWN
    story.append(Paragraph("3. Image-Level Class Mask Breakdown", h1_style))
    
    # Table Header Design Layout
    predictions_table_data = [[
        Paragraph("Target File Context", header_style),
        Paragraph("Segmented Class Name", header_style),
        Paragraph("Pixel Spatial Area", header_style),
        Paragraph("Total Area Coverage", header_style)
    ]]

    for img_record in predictions:
        filename = img_record["file_name"]
        segmentations = img_record.get("segmentations", [])
        
        if not segmentations:
            predictions_table_data.append([
                Paragraph(filename, body_style),
                Paragraph("<i>No Active Mask Detections</i>", body_style),
                Paragraph("0.0%", body_style),
                Paragraph("0 px", body_style),
                Paragraph("0.00%", body_style)
            ])
            continue

        for idx, seg in enumerate(segmentations):
            # Show the filename only on the first row of its segmentations for visual clarity
            display_name = filename if idx == 0 else ""
            
            predictions_table_data.append([
                Paragraph(display_name, body_style),
                Paragraph(f"<b>{seg['class_name']}</b>", body_style),
                Paragraph(f"{seg['pixel_area']:,} px", body_style),
                Paragraph(str(seg['coverage']), body_style)
            ])

    # Instantiate predictions overview table grid bounds
    pred_table = Table(predictions_table_data, colWidths=[140, 112, 80, 100, 100], repeatRows=1)
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light])
    ]))
    
    story.append(pred_table)

    # Compile flowables cleanly into target destination
    doc.build(story)
    print(f"PDF Report successfully compiled and written to: {output_pdf_path}")