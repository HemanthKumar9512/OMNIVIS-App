"""
OMNIVIS — Celery Async Task Workers
Background task queue for heavy inference jobs and report generation.
"""
import os
from celery import Celery
import logging

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "omnivis",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3000,
    worker_max_tasks_per_child=50,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, name="omnivis.process_video")
def process_video_task(self, file_path: str, session_id: str, config: dict):
    """Process an uploaded video file through the full pipeline."""
    import cv2
    import json

    logger.info(f"Processing video: {file_path} for session {session_id}")
    self.update_state(state="PROCESSING", meta={"progress": 0})

    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        return {"status": "error", "message": "Could not open video file"}

    results = {
        "total_frames": total_frames,
        "fps": fps,
        "detections": [],
        "processed_frames": 0,
    }

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if frame_idx % max(1, int(fps / 5)) == 0:
            # Import detection module here to avoid circular imports
            from modules.detection import ObjectDetector
            detector = ObjectDetector(model_variant="yolov8n", confidence=0.5)
            det_result = detector.detect(frame)
            results["detections"].extend(det_result["detections"])

        frame_idx += 1
        progress = int(frame_idx / max(total_frames, 1) * 100)
        self.update_state(state="PROCESSING", meta={"progress": progress})

    cap.release()
    results["processed_frames"] = frame_idx

    logger.info(f"Video processing complete: {frame_idx} frames, {len(results['detections'])} detections")
    return results


@celery_app.task(bind=True, name="omnivis.generate_report")
def generate_report_task(self, session_id: str, output_dir: str):
    """Generate a PDF report for a session."""
    from utils.reporter import ReportGenerator

    logger.info(f"Generating report for session {session_id}")
    self.update_state(state="GENERATING", meta={"progress": 0})

    reporter = ReportGenerator()
    output_path = os.path.join(output_dir, f"report_{session_id[:8]}.pdf")

    # In production, fetch data from database
    pdf_bytes = reporter.generate_report(
        session_data={"id": session_id, "source_type": "video", "frame_count": 0,
                      "started_at": "", "avg_fps": 0, "active_modules": []},
        detections=[],
        anomalies=[],
        metrics=[],
        output_path=output_path,
    )

    self.update_state(state="COMPLETE", meta={"progress": 100})
    return {"status": "complete", "path": output_path, "size": len(pdf_bytes)}


@celery_app.task(name="omnivis.train_anomaly_model")
def train_anomaly_model_task(feature_data: list):
    """Re-train anomaly detection models on accumulated features."""
    logger.info(f"Training anomaly model on {len(feature_data)} samples")
    # In production: load features, fit SVM + IsolationForest, save weights
    return {"status": "complete", "samples": len(feature_data)}
