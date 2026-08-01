from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from xml.sax.saxutils import escape as xml_escape
import io


def generate_pdf_report(video_title, audio_metrics, nlp_metrics, summary_text, chapters,
                         radar_chart_bytes, radar_caption, transcript_text,
                         comments_paragraphs):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], textColor=colors.HexColor('#1DB954'))
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], textColor=colors.HexColor('#1DB954'))
    caption_style = ParagraphStyle('Caption', parent=styles['Normal'], fontSize=8, textColor=colors.grey)

    elements = []

    elements.append(Paragraph("YouTube Vibe & Audio Intelligence Analyzer", title_style))
    if video_title:
        elements.append(Paragraph(f"Vidéo : {xml_escape(video_title)}", styles['Normal']))
    elements.append(Spacer(1, 0.5 * cm))

    elements.append(Paragraph("Métriques Audio & Style", heading_style))
    metrics_data = [
        ["Tempo (BPM)", f"{audio_metrics['tempo']:.2f}"],
        ["Énergie Globale", f"{audio_metrics['energy']:.4f}"],
        ["Texture Spectrale", audio_metrics['profile']],
        ["Langue détectée", nlp_metrics['language'].upper()],
        ["Score de Positivité (Valence)", f"{nlp_metrics['valence']:.2f}"],
    ]
    table = Table(metrics_data, colWidths=[8 * cm, 8 * cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f0f0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.5 * cm))

    if radar_chart_bytes:
        elements.append(Paragraph("Radar de Vibe & Répartition des Commentaires", heading_style))
        elements.append(Image(io.BytesIO(radar_chart_bytes), width=17 * cm, height=9 * cm))
        if radar_caption:
            elements.append(Spacer(1, 0.2 * cm))
            elements.append(Paragraph(xml_escape(radar_caption), caption_style))
        elements.append(Spacer(1, 0.5 * cm))

    if comments_paragraphs:
        elements.append(Paragraph("Analyse des Commentaires du Public", heading_style))
        for para in comments_paragraphs:
            elements.append(Paragraph(xml_escape(para), styles['Normal']))
            elements.append(Spacer(1, 0.2 * cm))
        elements.append(Spacer(1, 0.3 * cm))

    if summary_text:
        elements.append(Paragraph("Résumé Automatique de l'IA", heading_style))
        clean_summary = summary_text.replace('#', '').replace('*', '')
        for line in clean_summary.split('\n'):
            if line.strip():
                elements.append(Paragraph(xml_escape(line.strip()), styles['Normal']))
        elements.append(Spacer(1, 0.5 * cm))

    if chapters:
        elements.append(Paragraph("Chapitres par Vibe", heading_style))
        chapter_data = [["Timestamp", "Label"]] + [
            [xml_escape(c.get("timestamp", "")), xml_escape(c.get("label", ""))] for c in chapters
        ]
        chapter_table = Table(chapter_data, colWidths=[4 * cm, 12 * cm])
        chapter_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1DB954')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(chapter_table)
        elements.append(Spacer(1, 0.5 * cm))

    if transcript_text:
        elements.append(Paragraph("Texte Intégral Transcrit par l'IA", heading_style))
        elements.append(Paragraph(xml_escape(transcript_text), styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()