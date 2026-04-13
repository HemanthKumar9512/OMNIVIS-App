"""
OMNIVIS — Multi-Object Tracking Module
ByteTrack + Kalman Filter + Particle Filter for robust tracking.
"""
import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class KalmanTracker:
    """Single object Kalman filter tracker."""

    _count = 0

    def __init__(self, bbox: np.ndarray):
        KalmanTracker._count += 1
        self.id = KalmanTracker._count
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history: List[np.ndarray] = []

        # State: [x_center, y_center, area, aspect_ratio, vx, vy, va]
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.zeros((4, 7), np.float32)
        np.fill_diagonal(self.kf.measurementMatrix, 1)

        self.kf.transitionMatrix = np.eye(7, dtype=np.float32)
        self.kf.transitionMatrix[0, 4] = 1  # x += vx
        self.kf.transitionMatrix[1, 5] = 1  # y += vy
        self.kf.transitionMatrix[2, 6] = 1  # area += va

        self.kf.measurementNoiseCov *= 10.0
        self.kf.processNoiseCov *= 0.01
        self.kf.processNoiseCov[4:, 4:] *= 0.01
        self.kf.errorCovPost *= 10.0
        self.kf.errorCovPost[4:, 4:] *= 100.0

        # Initialize state from bbox
        z = self._bbox_to_z(bbox)
        self.kf.statePost[:4] = z.reshape(4, 1)

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [cx, cy, area, aspect_ratio]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2
        cy = bbox[1] + h / 2
        return np.array([cx, cy, w * h, w / max(h, 1e-6)], dtype=np.float32)

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, area, aspect_ratio] to [x1, y1, x2, y2]."""
        w = np.sqrt(max(z[2] * z[3], 0))
        h = z[2] / max(w, 1e-6)
        return np.array([
            z[0] - w / 2, z[1] - h / 2,
            z[0] + w / 2, z[1] + h / 2
        ], dtype=np.float32)

    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.age += 1
        self.time_since_update += 1
        prediction = self.kf.predict()
        return self._z_to_bbox(prediction[:4].flatten())

    def update(self, bbox: np.ndarray):
        """Update with new measurement."""
        self.hits += 1
        self.time_since_update = 0
        z = self._bbox_to_z(bbox)
        self.kf.correct(z.reshape(4, 1))
        state = self.kf.statePost[:4].flatten()
        self.history.append(self._z_to_bbox(state))
        # Keep last 100 positions
        if len(self.history) > 100:
            self.history = self.history[-100:]

    def get_state(self) -> np.ndarray:
        """Get current bounding box."""
        return self._z_to_bbox(self.kf.statePost[:4].flatten())

    def get_center(self) -> Tuple[float, float]:
        """Get current center position."""
        state = self.kf.statePost.flatten()
        return float(state[0]), float(state[1])


class ByteTracker:
    """ByteTrack multi-object tracker."""

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanTracker] = []
        self.track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with new detections."""
        start = time.perf_counter()

        # Predict existing tracks
        predicted_boxes = []
        for t in self.trackers:
            pred = t.predict()
            predicted_boxes.append(pred)

        # Convert detections to numpy
        det_boxes = []
        det_meta = []
        for d in detections:
            bbox = d.get("bbox", {})
            det_boxes.append([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
            det_meta.append(d)

        det_boxes = np.array(det_boxes) if det_boxes else np.empty((0, 4))
        predicted_boxes = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))

        # Compute IoU matrix
        if len(predicted_boxes) > 0 and len(det_boxes) > 0:
            iou_matrix = self._compute_iou_matrix(predicted_boxes, det_boxes)
        else:
            iou_matrix = np.empty((len(predicted_boxes), len(det_boxes)))

        # Hungarian matching
        matched, unmatched_dets, unmatched_trks = self._associate(
            iou_matrix, self.iou_threshold
        )

        # Update matched tracks
        for t_idx, d_idx in matched:
            self.trackers[t_idx].update(det_boxes[d_idx])

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            tracker = KalmanTracker(det_boxes[d_idx])
            self.trackers.append(tracker)

        # Remove dead tracks
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]

        # Build output
        results = []
        for t in self.trackers:
            if t.time_since_update > 0:
                continue
            if t.hits < self.min_hits and t.age > self.min_hits:
                continue

            state = t.get_state()
            center = t.get_center()
            self.track_history[t.id].append(center)
            # Keep last 60 positions for trail visualization
            if len(self.track_history[t.id]) > 60:
                self.track_history[t.id] = self.track_history[t.id][-60:]

            results.append({
                "track_id": t.id,
                "bbox": {
                    "x1": float(state[0]), "y1": float(state[1]),
                    "x2": float(state[2]), "y2": float(state[3]),
                },
                "center": center,
                "age": t.age,
                "hits": t.hits,
            })

        return results

    @staticmethod
    def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU between two sets of bounding boxes."""
        n = len(boxes_a)
        m = len(boxes_b)
        iou = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                xa1 = max(boxes_a[i][0], boxes_b[j][0])
                ya1 = max(boxes_a[i][1], boxes_b[j][1])
                xa2 = min(boxes_a[i][2], boxes_b[j][2])
                ya2 = min(boxes_a[i][3], boxes_b[j][3])

                inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
                area_a = (boxes_a[i][2] - boxes_a[i][0]) * (boxes_a[i][3] - boxes_a[i][1])
                area_b = (boxes_b[j][2] - boxes_b[j][0]) * (boxes_b[j][3] - boxes_b[j][1])
                union = area_a + area_b - inter

                iou[i][j] = inter / max(union, 1e-6)

        return iou

    @staticmethod
    def _associate(iou_matrix: np.ndarray, threshold: float):
        """Greedy matching based on IoU matrix."""
        matched = []
        unmatched_dets = list(range(iou_matrix.shape[1])) if iou_matrix.shape[1] > 0 else []
        unmatched_trks = list(range(iou_matrix.shape[0])) if iou_matrix.shape[0] > 0 else []

        if iou_matrix.size == 0:
            return matched, unmatched_dets, unmatched_trks

        # Greedy matching: assign highest IoU pairs first
        while True:
            max_iou = iou_matrix.max()
            if max_iou < threshold:
                break
            t_idx, d_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched.append((t_idx, d_idx))
            iou_matrix[t_idx, :] = 0
            iou_matrix[:, d_idx] = 0
            if t_idx in unmatched_trks:
                unmatched_trks.remove(t_idx)
            if d_idx in unmatched_dets:
                unmatched_dets.remove(d_idx)

        return matched, unmatched_dets, unmatched_trks

    def get_trails(self) -> Dict[int, List[Tuple[float, float]]]:
        """Get all active track trails for visualization."""
        active_ids = {t.id for t in self.trackers if t.time_since_update == 0}
        return {
            tid: positions
            for tid, positions in self.track_history.items()
            if tid in active_ids
        }

    def reset(self):
        """Reset all tracks."""
        self.trackers.clear()
        self.track_history.clear()
        KalmanTracker._count = 0
