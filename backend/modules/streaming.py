"""
OMNIVIS - Video Streaming Module
MJPEG and HLS streaming for webcam, video files, RTSP streams, and web streams.
Supports continuous streaming with real-time inference overlay.
"""
import cv2
import numpy as np
import time
import logging
import threading
import asyncio
import base64
import os
from typing import Dict, Any, Optional, List, Callable, Generator
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StreamSource(Enum):
    WEBCAM = "webcam"
    VIDEO_FILE = "video_file"
    RTSP = "rtsp"
    YOUTUBE = "youtube"
    HTTP_STREAM = "http_stream"


@dataclass
class StreamConfig:
    source_type: StreamSource
    source: str = ""
    width: int = 640
    height: int = 480
    fps: int = 30
    quality: int = 80
    enable_inference: bool = True
    modules: List[str] = None

    def __post_init__(self):
        if self.modules is None:
            self.modules = ["detection", "face", "tracking"]


class VideoCapture:
    """Unified video capture for all source types."""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.cap = None
        self.is_opened = False
        self.frame_count = 0
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self._lock = threading.Lock()

    def open(self) -> bool:
        try:
            if self.config.source_type == StreamSource.WEBCAM:
                source = int(self.config.source) if self.config.source.isdigit() else 0
                self.cap = cv2.VideoCapture(source)
            elif self.config.source_type == StreamSource.VIDEO_FILE:
                self.cap = cv2.VideoCapture(self.config.source)
            elif self.config.source_type in (StreamSource.RTSP, StreamSource.HTTP_STREAM):
                self.cap = cv2.VideoCapture(self.config.source, cv2.CAP_FFMPEG)
            elif self.config.source_type == StreamSource.YOUTUBE:
                self.cap = cv2.VideoCapture(self.config.source, cv2.CAP_FFMPEG)

            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.config.fps

                self.is_opened = True
                logger.info(f"Video capture opened: {self.config.source_type.value} ({self.width}x{self.height} @ {self.fps}fps)")
                return True
            else:
                logger.error(f"Failed to open video source: {self.config.source}")
                return False
        except Exception as e:
            logger.error(f"Video capture error: {e}")
            return False

    def read(self) -> tuple:
        if not self.is_opened or self.cap is None:
            return False, None

        with self._lock:
            ret, frame = self.cap.read()

        if ret:
            self.frame_count += 1
        return ret, frame

    def release(self):
        if self.cap:
            with self._lock:
                self.cap.release()
            self.is_opened = False
            logger.info("Video capture released")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()


class StreamProcessor:
    """Processes video frames through inference pipeline."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.processing = False
        self._last_frame = None
        self._last_result = None
        self._lock = threading.Lock()
        self._fps_counter = 0
        self._fps_start = time.time()
        self._current_fps = 0

    def process_frame(self, frame: np.ndarray) -> tuple:
        if frame is None or self.pipeline is None:
            return frame, {}

        try:
            result = self.pipeline.process(frame)

            annotated = frame.copy()
            detections = result.get("detection", {}).get("detections", [])
            annotated = self._draw_detections(annotated, detections)

            faces = result.get("face", {}).get("faces", [])
            if faces:
                annotated = self._draw_faces(annotated, faces)

            tracks = result.get("tracking", {}).get("tracks", [])
            if tracks:
                annotated = self._draw_tracks(annotated, tracks)

            self._last_result = result
            self._update_fps()

            return annotated, result
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame, {}

    def _draw_detections(self, frame, detections):
        for det in detections:
            bbox = det.get("bbox", {})
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            cls_name = det.get("class_name", "unknown")
            conf = det.get("confidence", 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return frame

    def _draw_faces(self, frame, faces):
        for face in faces:
            bbox = face.get("bbox", [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                labels = []
                if "age" in face and face["age"]:
                    labels.append(f"Age:{face['age']}")
                if "gender" in face:
                    labels.append(face["gender"])
                if "emotion" in face:
                    labels.append(face["emotion"])

                label = " | ".join(labels)
                if label:
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return frame

    def _draw_tracks(self, frame, tracks):
        for track in tracks:
            track_id = track.get("id", 0)
            bbox = track.get("bbox", [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                color = (255, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    def _update_fps(self):
        self._fps_counter += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_start = time.time()


class MJPEGStream:
    """MJPEG stream generator for HTTP streaming."""

    def __init__(self, processor: StreamProcessor):
        self.processor = processor
        self.frame_buffer = None
        self._lock = threading.Lock()

    def generate_frames(self, capture: VideoCapture) -> Generator[bytes, None, None]:
        if not capture.is_opened:
            capture.open()

        while capture.is_opened:
            ret, frame = capture.read()
            if not ret or frame is None:
                if capture.config.source_type == StreamSource.VIDEO_FILE:
                    capture.release()
                    break
                continue

            annotated, _ = self.processor.process_frame(frame)

            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, self.processor.config.quality])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def generate_frames_with_overlay(self, capture: VideoCapture) -> Generator[bytes, None, None]:
        if not capture.is_opened:
            capture.open()

        frame_count = 0
        while capture.is_opened:
            ret, frame = capture.read()
            if not ret or frame is None:
                if capture.config.source_type == StreamSource.VIDEO_FILE:
                    break
                continue

            annotated, result = self.processor.process_frame(frame)

            h, w = annotated.shape[:2]

            fps = self.processor._current_fps
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            det_count = result.get("detection", {}).get("count", 0)
            cv2.putText(annotated, f"Detections: {det_count}", (10, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            face_count = result.get("face", {}).get("face_count", 0)
            cv2.putText(annotated, f"Faces: {face_count}", (10, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            frame_count += 1
            cv2.putText(annotated, f"Frame: {frame_count}", (w - 150, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, self.processor.config.quality])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


class HLSStream:
    """HLS stream generator for adaptive bitrate streaming."""

    def __init__(self, output_dir: str = "hls_output", segment_duration: float = 2.0):
        self.output_dir = output_dir
        self.segment_duration = segment_duration
        self.segments = []
        self.sequence = 0
        self._lock = threading.Lock()
        self.writer = None
        os.makedirs(output_dir, exist_ok=True)

    def start_stream(self, capture: VideoCapture) -> bool:
        try:
            output_path = os.path.join(self.output_dir, "stream.m3u8")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self.writer = cv2.VideoWriter(
                os.path.join(self.output_dir, "temp.mp4"),
                fourcc, capture.fps, (capture.width, capture.height)
            )
            return self.writer.isOpened()
        except Exception as e:
            logger.error(f"HLS start failed: {e}")
            return False

    def write_segment(self, frame: np.ndarray) -> Optional[str]:
        if self.writer is None:
            return None

        self.writer.write(frame)
        self.sequence += 1

        if self.sequence % int(capture.fps * self.segment_duration) == 0:
            segment_name = f"segment_{self.sequence:06d}.ts"
            segment_path = os.path.join(self.output_dir, segment_name)

            self.writer.release()

            self._convert_to_ts(os.path.join(self.output_dir, "temp.mp4"), segment_path)

            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self.writer = cv2.VideoWriter(
                os.path.join(self.output_dir, "temp.mp4"),
                fourcc, capture.fps, (capture.width, capture.height)
            )

            with self._lock:
                self.segments.append(segment_name)
                if len(self.segments) > 5:
                    old = self.segments.pop(0)
                    old_path = os.path.join(self.output_dir, old)
                    if os.path.exists(old_path):
                        os.remove(old_path)

            self._update_playlist()
            return segment_name

        return None

    def _convert_to_ts(self, input_path: str, output_path: str):
        try:
            import subprocess
            subprocess.run([
                'ffmpeg', '-y', '-i', input_path,
                '-c', 'copy', '-f', 'mpegts', output_path
            ], capture_output=True, timeout=30)
        except Exception:
            import shutil
            shutil.copy(input_path, output_path)

    def _update_playlist(self):
        playlist_path = os.path.join(self.output_dir, "stream.m3u8")
        with open(playlist_path, 'w') as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:3\n")
            f.write(f"#EXT-X-TARGETDURATION:{int(self.segment_duration)}\n")
            f.write("#EXT-X-MEDIA-SEQUENCE:0\n")
            f.write("#EXT-X-PLAYLIST-TYPE:EVENT\n")

            for seg in self.segments:
                f.write(f"#EXTINF:{self.segment_duration:.3f},\n")
                f.write(f"{seg}\n")

    def get_playlist(self) -> Optional[str]:
        playlist_path = os.path.join(self.output_dir, "stream.m3u8")
        if os.path.exists(playlist_path):
            with open(playlist_path, 'r') as f:
                return f.read()
        return None

    def cleanup(self):
        if self.writer:
            self.writer.release()


class StreamManager:
    """Manages multiple video streams."""

    def __init__(self):
        self.active_streams: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self.stream_id_counter = 0

    def create_stream(self, config: StreamConfig, pipeline=None) -> str:
        with self._lock:
            self.stream_id_counter += 1
            stream_id = f"stream_{self.stream_id_counter}"

            capture = VideoCapture(config)
            processor = StreamProcessor(pipeline)

            self.active_streams[stream_id] = {
                "config": config,
                "capture": capture,
                "processor": processor,
                "running": False,
                "thread": None,
            }

            logger.info(f"Stream created: {stream_id} ({config.source_type.value})")
            return stream_id

    def start_stream(self, stream_id: str) -> bool:
        with self._lock:
            if stream_id not in self.active_streams:
                return False

            stream = self.active_streams[stream_id]
            if stream["running"]:
                return True

            if not stream["capture"].open():
                return False

            stream["running"] = True
            thread = threading.Thread(
                target=self._stream_loop,
                args=(stream_id,),
                daemon=True
            )
            stream["thread"] = thread
            thread.start()
            return True

    def stop_stream(self, stream_id: str) -> bool:
        with self._lock:
            if stream_id not in self.active_streams:
                return False

            stream = self.active_streams[stream_id]
            stream["running"] = False

            if stream["thread"] and stream["thread"].is_alive():
                stream["thread"].join(timeout=5)

            stream["capture"].release()
            logger.info(f"Stream stopped: {stream_id}")
            return True

    def get_stream_frame(self, stream_id: str) -> tuple:
        with self._lock:
            if stream_id not in self.active_streams:
                return None, None

            stream = self.active_streams[stream_id]
            if not stream["running"]:
                return None, None

        capture = stream["capture"]
        processor = stream["processor"]

        ret, frame = capture.read()
        if not ret or frame is None:
            if capture.config.source_type == StreamSource.VIDEO_FILE:
                self.stop_stream(stream_id)
            return None, None

        annotated, result = processor.process_frame(frame)
        return annotated, result

    def stream_generator(self, stream_id: str) -> Generator[bytes, None, None]:
        with self._lock:
            if stream_id not in self.active_streams:
                return

            stream = self.active_streams[stream_id]
            if not stream["running"]:
                if not stream["capture"].open():
                    return
                stream["running"] = True

        capture = stream["capture"]
        mjpeg = MJPEGStream(stream["processor"])

        for frame_bytes in mjpeg.generate_frames_with_overlay(capture):
            yield frame_bytes
            if not stream["running"]:
                break

    def _stream_loop(self, stream_id: str):
        while True:
            with self._lock:
                if stream_id not in self.active_streams:
                    break
                stream = self.active_streams[stream_id]
                if not stream["running"]:
                    break

            ret, _ = stream["capture"].read()
            if not ret:
                if stream["capture"].config.source_type == StreamSource.VIDEO_FILE:
                    self.stop_stream(stream_id)
                    break
                time.sleep(0.01)
            else:
                time.sleep(1.0 / max(1, stream["capture"].fps))

    def get_stream_info(self, stream_id: str) -> Optional[Dict]:
        with self._lock:
            if stream_id not in self.active_streams:
                return None

            stream = self.active_streams[stream_id]
            capture = stream["capture"]
            return {
                "stream_id": stream_id,
                "source_type": capture.config.source_type.value,
                "source": capture.config.source,
                "resolution": f"{capture.width}x{capture.height}",
                "fps": capture.fps,
                "frame_count": capture.frame_count,
                "running": stream["running"],
                "modules": capture.config.modules,
            }

    def list_streams(self) -> List[Dict]:
        with self._lock:
            return [
                self.get_stream_info(sid)
                for sid in self.active_streams
            ]

    def cleanup(self):
        with self._lock:
            for stream_id in list(self.active_streams.keys()):
                self.stop_stream(stream_id)
            self.active_streams.clear()


stream_manager = StreamManager()
