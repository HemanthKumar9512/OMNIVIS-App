"""
OMNIVIS — PDF Report Generator
Creates professional session reports with detection summaries and metrics.
"""
import io
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates PDF reports for inference sessions."""

    def __init__(self):
        self.has_reportlab = False
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            self.has_reportlab = True
        except ImportError:
            logger.warning("reportlab not installed — PDF reports will be text-only fallback")

    def generate_report(
        self,
        session_data: Dict[str, Any],
        detections: List[Dict],
        anomalies: List[Dict],
        metrics: List[Dict],
        output_path: Optional[str] = None,
    ) -> bytes:
        """Generate a session report as PDF."""
        if self.has_reportlab:
            return self._generate_pdf(session_data, detections, anomalies, metrics, output_path)
        else:
            return self._generate_text_report(session_data, detections, anomalies, metrics)

    def _generate_pdf(self, session_data, detections, anomalies, metrics, output_path):
        """Generate PDF using ReportLab."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                                leftMargin=2 * cm, rightMargin=2 * cm)

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            'OmniTitle', parent=styles['Title'],
            fontSize=28, textColor=colors.HexColor("#6366f1"),
            spaceAfter=20
        ))
        styles.add(ParagraphStyle(
            'SectionHead', parent=styles['Heading2'],
            fontSize=16, textColor=colors.HexColor("#1e293b"),
            spaceBefore=15, spaceAfter=8
        ))
        styles.add(ParagraphStyle(
            'MetricLabel', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor("#64748b")
        ))

        elements = []

        # ─── Title ─────────────────────────────────
        elements.append(Paragraph("OMNIVIS", styles['OmniTitle']))
        elements.append(Paragraph("Session Inference Report", styles['Heading3']))
        elements.append(HRFlowable(width="100%", color=colors.HexColor("#6366f1"), thickness=2))
        elements.append(Spacer(1, 20))

        # ─── Session Info ──────────────────────────
        elements.append(Paragraph("Session Overview", styles['SectionHead']))
        session_table_data = [
            ["Parameter", "Value"],
            ["Session ID", str(session_data.get("id", "N/A"))],
            ["Source Type", str(session_data.get("source_type", "N/A"))],
            ["Started At", str(session_data.get("started_at", "N/A"))],
            ["Ended At", str(session_data.get("ended_at", "In Progress"))],
            ["Total Frames", str(session_data.get("frame_count", 0))],
            ["Average FPS", f"{session_data.get('avg_fps', 0):.1f}"],
            ["Active Modules", ", ".join(session_data.get("active_modules", []))],
        ]
        t = Table(session_table_data, colWidths=[4 * cm, 12 * cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#6366f1")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))

        # ─── Detection Summary ────────────────────
        if detections:
            elements.append(Paragraph("Detection Summary", styles['SectionHead']))
            # Count by class
            class_counts = {}
            for d in detections:
                cls = d.get("class_name", "unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1

            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            det_table = [["Class", "Count", "Avg Confidence"]]
            for cls, count in sorted_classes:
                avg_conf = sum(
                    d.get("confidence", 0) for d in detections if d.get("class_name") == cls
                ) / count
                det_table.append([cls, str(count), f"{avg_conf:.3f}"])

            t = Table(det_table, colWidths=[6 * cm, 4 * cm, 6 * cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#10b981")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f0fdf4")),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d1fae5")),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 20))

        # ─── Anomalies ────────────────────────────
        if anomalies:
            elements.append(Paragraph("Anomaly Events", styles['SectionHead']))
            anom_table = [["Time", "Type", "Severity", "Score", "Description"]]
            for a in anomalies[:50]:
                severity_color = {
                    "green": "#10b981", "yellow": "#f59e0b", "red": "#ef4444"
                }.get(a.get("severity", "green"), "#64748b")
                anom_table.append([
                    str(a.get("timestamp", "")),
                    a.get("anomaly_type", ""),
                    a.get("severity", "").upper(),
                    f"{a.get('score', 0):.3f}",
                    a.get("description", "")[:60],
                ])

            t = Table(anom_table, colWidths=[3 * cm, 3 * cm, 2 * cm, 2 * cm, 6 * cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#ef4444")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#fecaca")),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 20))

        # ─── Metrics Summary ──────────────────────
        if metrics:
            elements.append(Paragraph("Performance Metrics", styles['SectionHead']))
            avg_fps = sum(m.get("fps", 0) for m in metrics) / len(metrics)
            avg_inf = sum(m.get("inference_ms", 0) for m in metrics) / len(metrics)
            avg_gpu = sum(m.get("gpu_util", 0) for m in metrics) / len(metrics)

            met_table = [
                ["Metric", "Average", "Min", "Max"],
                ["FPS", f"{avg_fps:.1f}",
                 f"{min(m.get('fps', 0) for m in metrics):.1f}",
                 f"{max(m.get('fps', 0) for m in metrics):.1f}"],
                ["Inference (ms)", f"{avg_inf:.1f}",
                 f"{min(m.get('inference_ms', 0) for m in metrics):.1f}",
                 f"{max(m.get('inference_ms', 0) for m in metrics):.1f}"],
                ["GPU Util (%)", f"{avg_gpu:.1f}",
                 f"{min(m.get('gpu_util', 0) for m in metrics):.1f}",
                 f"{max(m.get('gpu_util', 0) for m in metrics):.1f}"],
            ]
            t = Table(met_table, colWidths=[4 * cm, 4 * cm, 4 * cm, 4 * cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3b82f6")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#bfdbfe")),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ]))
            elements.append(t)

        # ─── Footer ───────────────────────────────
        elements.append(Spacer(1, 40))
        elements.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f0"), thickness=1))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(
            f"Generated by OMNIVIS v1.0.0 — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            styles['MetricLabel']
        ))

        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)

        return pdf_bytes

    def _generate_text_report(self, session_data, detections, anomalies, metrics) -> bytes:
        """Fallback text report when reportlab is not available."""
        lines = [
            "=" * 60,
            "OMNIVIS — Session Inference Report",
            "=" * 60,
            f"Generated: {datetime.utcnow().isoformat()}",
            "",
            "SESSION OVERVIEW",
            "-" * 40,
            f"  ID: {session_data.get('id', 'N/A')}",
            f"  Source: {session_data.get('source_type', 'N/A')}",
            f"  Frames: {session_data.get('frame_count', 0)}",
            f"  Avg FPS: {session_data.get('avg_fps', 0):.1f}",
            "",
            f"DETECTIONS: {len(detections)} total",
            f"ANOMALIES: {len(anomalies)} events",
            f"METRIC SAMPLES: {len(metrics)}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines).encode("utf-8")
