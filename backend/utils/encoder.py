"""
OMNIVIS — Frame Encoder / Compressor
Handles frame encoding for WebSocket transport and storage.
"""
import cv2
import numpy as np
import base64
from typing import Optional, Tuple
import io


class FrameEncoder:
    """Encodes and compresses video frames for transport."""

    FORMATS = {
        "jpeg": ".jpg",
        "webp": ".webp",
        "png": ".png",
    }

    def __init__(self, format: str = "jpeg", quality: int = 85):
        self.format = format
        self.quality = quality

    def encode_frame(self, frame: np.ndarray, format: Optional[str] = None,
                     quality: Optional[int] = None) -> bytes:
        """Encode frame to bytes."""
        fmt = format or self.format
        q = quality or self.quality
        ext = self.FORMATS.get(fmt, ".jpg")

        params = []
        if fmt == "jpeg":
            params = [cv2.IMWRITE_JPEG_QUALITY, q]
        elif fmt == "webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, q]
        elif fmt == "png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, (100 - q) // 11))]

        success, buffer = cv2.imencode(ext, frame, params)
        if not success:
            raise RuntimeError(f"Failed to encode frame as {fmt}")
        return buffer.tobytes()

    def encode_to_base64(self, frame: np.ndarray, format: Optional[str] = None,
                         quality: Optional[int] = None) -> str:
        """Encode frame to base64 string for JSON transport."""
        raw = self.encode_frame(frame, format, quality)
        return base64.b64encode(raw).decode("utf-8")

    def decode_base64(self, data: str) -> np.ndarray:
        """Decode base64 string back to frame."""
        raw = base64.b64decode(data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def resize_frame(frame: np.ndarray, target_size: Tuple[int, int],
                     keep_aspect: bool = True) -> np.ndarray:
        """Resize frame with optional aspect ratio preservation (letterbox)."""
        if not keep_aspect:
            return cv2.resize(frame, target_size)

        h, w = frame.shape[:2]
        tw, th = target_size
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Letterbox padding
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)  # Gray padding
        dx, dy = (tw - nw) // 2, (th - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas

    @staticmethod
    def normalize_frame(frame: np.ndarray) -> np.ndarray:
        """Normalize frame to [0, 1] float32 for model input."""
        return frame.astype(np.float32) / 255.0

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: list,
                        show_labels: bool = True,
                        show_confidence: bool = True) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        annotated = frame.copy()
        colors = {}

        for det in detections:
            cls = det.get("class_name", "unknown")
            conf = det.get("confidence", 0)
            bbox = det.get("bbox", {})
            track_id = det.get("track_id")

            # Consistent color per class
            if cls not in colors:
                hash_val = hash(cls) % 360
                hsv_color = np.array([[[hash_val // 2, 200, 255]]], dtype=np.uint8)
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                colors[cls] = tuple(int(c) for c in bgr_color)

            color = colors[cls]
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label
            if show_labels or show_confidence:
                parts = []
                if show_labels:
                    parts.append(cls)
                if track_id is not None:
                    parts.append(f"#{track_id}")
                if show_confidence:
                    parts.append(f"{conf:.2f}")
                label = " ".join(parts)

                (tw, th_), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th_ - baseline - 4), (x1 + tw, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - baseline - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    @staticmethod
    def draw_tracks(frame: np.ndarray, tracks: dict) -> np.ndarray:
        """Draw trajectory trails on frame."""
        annotated = frame.copy()
        for track_id, positions in tracks.items():
            if len(positions) < 2:
                continue
            hash_val = hash(str(track_id)) % 360
            hsv = np.array([[[hash_val // 2, 200, 255]]], dtype=np.uint8)
            color = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0])

            for i in range(1, len(positions)):
                alpha = i / len(positions)  # Fade trail
                pt1 = tuple(int(c) for c in positions[i - 1])
                pt2 = tuple(int(c) for c in positions[i])
                thickness = max(1, int(3 * alpha))
                cv2.line(annotated, pt1, pt2, color, thickness, cv2.LINE_AA)
        return annotated
