import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script usage
import matplotlib.pyplot as plt
import numpy as np
import librosa

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_waveform_plot(audio_data, output_path):
    """
    Generates a matplotlib plot showing the audio waveform and its RMS energy envelope.
    Saves it as a temporary PNG to embed in the PDF.
    """
    y = np.array(audio_data["waveform_y"])
    sr = audio_data["waveform_sr"]
    rms = np.array(audio_data["rms_levels"])
    rms_times = np.array(audio_data["rms_times"])
    pauses = audio_data["pauses"]
    
    # Calculate duration
    duration = audio_data["duration"]
    times = np.linspace(0, duration, len(y))
    
    plt.figure(figsize=(7, 2.5), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot waveform amplitude
    plt.plot(times, y, color='#a0aec0', alpha=0.6, label='Waveform')
    
    # Highlight pauses
    pause_labeled = False
    for p in pauses:
        plt.axvspan(p["start"], p["end"], color='#fed7d7', alpha=0.8, 
                    label='Pauses' if not pause_labeled else "")
        pause_labeled = True
        
    plt.title("Audio Waveform & Pause Analysis", fontsize=10, fontweight='bold', color='#1a202c', pad=10)
    plt.xlabel("Time (seconds)", fontsize=8, color='#4a5568')
    plt.ylabel("Normalized Amplitude", fontsize=8, color='#4a5568')
    plt.tick_params(labelsize=8)
    
    if pause_labeled:
        plt.legend(loc='upper right', fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0')
        
    plt.tight_layout()
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.close()

def build_pdf_report(pdf_path, concept_name, transcript, concept_results, fluency_results, temp_plot_path):
    """
    Builds a beautifully styled PDF report containing evaluation results.
    """
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette styles
    primary_color = colors.HexColor("#1e3a8a")    # Deep Navy
    secondary_color = colors.HexColor("#3b82f6")  # Bright Blue
    text_color = colors.HexColor("#1f2937")       # Dark Gray
    accent_green = colors.HexColor("#10b981")     # Emerald Green
    accent_orange = colors.HexColor("#f59e0b")    # Amber Orange
    accent_red = colors.HexColor("#ef4444")       # Rose Red
    bg_light = colors.HexColor("#f9fafb")         # Light Off-white
    
    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4b5563"),
        spaceAfter=20
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=text_color,
        spaceAfter=8
    )
    
    transcript_style = ParagraphStyle(
        'Transcript_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#374151"),
        spaceAfter=8
    )
    
    meta_label_style = ParagraphStyle(
        'MetaLabel',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        textColor=primary_color
    )
    
    meta_val_style = ParagraphStyle(
        'MetaVal',
        fontName='Helvetica',
        fontSize=10,
        leading=12,
        textColor=text_color
    )
    
    story = []
    
    # 1. Header (Title & Subtitle)
    story.append(Paragraph("Voice-Based Concept Understanding Analyser", title_style))
    story.append(Paragraph(f"<b>Concept Evaluation Report:</b> {concept_name} &nbsp;&nbsp;|&nbsp;&nbsp; Generated automatically by VBCUA Platform", subtitle_style))
    
    # Divider line
    divider = Table([[""]], colWidths=[doc.width], rowHeights=[2])
    divider.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), secondary_color),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(divider)
    story.append(Spacer(1, 15))
    
    # 2. Performance Summary Card (Table)
    sim_pct = int(concept_results["similarity_score"] * 100)
    fluency_score = int(fluency_results["fluency_score"])
    overall_score = int((sim_pct * 0.6) + (fluency_score * 0.4))
    
    summary_data = [
        [
            Paragraph("<b>Overall Score</b>", meta_label_style),
            Paragraph(f"<b>{overall_score} / 100</b>", ParagraphStyle('Ovr', fontName='Helvetica-Bold', fontSize=12, textColor=primary_color)),
            Paragraph("<b>Understanding Level</b>", meta_label_style),
            Paragraph(f"<b>{concept_results['understanding_level']}</b>", ParagraphStyle('Level', fontName='Helvetica-Bold', fontSize=10, textColor=accent_green if "Strong" in concept_results['understanding_level'] else (accent_orange if "Moderate" in concept_results['understanding_level'] else accent_red)))
        ],
        [
            Paragraph("Semantic Similarity", meta_label_style),
            Paragraph(f"{sim_pct}%", meta_val_style),
            Paragraph("Speaking Rate", meta_label_style),
            Paragraph(f"{int(fluency_results['wpm'])} WPM", meta_val_style)
        ],
        [
            Paragraph("Speech Fluency Score", meta_label_style),
            Paragraph(f"{fluency_score}%", meta_val_style),
            Paragraph("Pause Ratio / Count", meta_label_style),
            Paragraph(f"{int(fluency_results['pause_ratio'] * 100)}% ({fluency_results['pause_count']} pauses)", meta_val_style)
        ],
        [
            Paragraph("Total Duration", meta_label_style),
            Paragraph(f"{fluency_results['duration']:.2f} seconds", meta_val_style),
            Paragraph("Total Filler Words", meta_label_style),
            Paragraph(f"{fluency_results['total_fillers']} words", meta_val_style)
        ]
    ]
    
    summary_table = Table(summary_data, colWidths=[1.5*inch, 1.8*inch, 1.6*inch, 2.3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_light),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e5e7eb")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))
    
    # 3. Transcript Segment
    story.append(Paragraph("Spoken Transcription", h1_style))
    transcript_box_data = [[Paragraph(f'"{transcript}"', transcript_style)]]
    transcript_table = Table(transcript_box_data, colWidths=[doc.width])
    transcript_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f3f4f6")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#d1d5db")),
        ('PADDING', (0,0), (-1,-1), 12),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(transcript_table)
    story.append(Spacer(1, 15))
    
    # 4. Waveform Image Section
    if os.path.exists(temp_plot_path):
        story.append(Paragraph("Visual Waveform and Pause Plot", h1_style))
        story.append(Image(temp_plot_path, width=7*inch, height=2.5*inch))
        story.append(Spacer(1, 10))
        
    # 5. Concept Points Checklist (Page break safeguard)
    story.append(Paragraph("Conceptual Key Points Coverage", h1_style))
    
    checklist_rows = []
    # Header
    checklist_rows.append([
        Paragraph("<b>Expected Key Concept Point</b>", meta_label_style),
        Paragraph("<b>Coverage Status</b>", meta_label_style),
        Paragraph("<b>Similarity Score</b>", meta_label_style)
    ])
    
    for kp in concept_results["key_points_status"]:
        status = kp["status"]
        if status == "Covered":
            status_color = "#10b981" # green
        elif status == "Partially Covered":
            status_color = "#f59e0b" # orange
        else:
            status_color = "#ef4444" # red
            
        status_para = Paragraph(f"<font color='{status_color}'><b>{status}</b></font>", meta_val_style)
        
        checklist_rows.append([
            Paragraph(kp["key_point"], body_style),
            status_para,
            Paragraph(f"{int(kp['similarity'] * 100)}%", meta_val_style)
        ])
        
    checklist_table = Table(checklist_rows, colWidths=[4.2*inch, 1.8*inch, 1.2*inch])
    checklist_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e5e7eb")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    
    story.append(KeepTogether([checklist_table]))
    story.append(Spacer(1, 15))
    
    # 6. Speech Fluency & Filler Words Breakdown
    story.append(Paragraph("Fluency & Communication Insight", h1_style))
    story.append(Paragraph(f"<b>Fluency Feedback:</b> {fluency_results['fluency_feedback']}", body_style))
    story.append(Spacer(1, 8))
    
    # Filler word frequency table
    filler_data = [
        [Paragraph(f"<b>Filler Word</b>", meta_label_style), Paragraph("<b>Count</b>", meta_label_style)],
    ]
    for w, count in fluency_results["filler_counts"].items():
        if count > 0:
            filler_data.append([Paragraph(f"'{w}'", body_style), Paragraph(str(count), meta_val_style)])
            
    if len(filler_data) > 1:
        # Create double columns table or a smaller table
        filler_table = Table(filler_data, colWidths=[2.5*inch, 1.5*inch])
        filler_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f3f4f6")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
            ('PADDING', (0,0), (-1,-1), 6),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ]))
        story.append(KeepTogether([
            Paragraph("<b>Filler Words Breakdown</b>", meta_label_style),
            Spacer(1, 5),
            filler_table
        ]))
    else:
        story.append(Paragraph("Excellent! No standard filler words detected in your audio.", body_style))
        
    # Build Document
    doc.build(story)
