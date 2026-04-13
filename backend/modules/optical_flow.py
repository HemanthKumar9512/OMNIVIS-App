"""
OMNIVIS — Optical Flow & Motion Intelligence
RAFT deep optical flow + Farneback classical fallback + motion segmentation.
"""
import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class OpticalFlowEngine:
    """Dense optical flow with RAFT and Farneback fallback."""

    def __init__(self, method: str = "auto", device: str = "auto"):
        self.method = method
        self.device = device
        self.raft_model = None
        self.prev_gray = None
        self.prev_frame = None
        self.bg_subtractor_mog = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.bg_subtractor_knn = cv2.createBackgroundSubtractorKNN(
            history=500, dist2Threshold=400.0, detectShadows=True
        )
        self.bg_method = "mog2"
        self.loaded = False
        self._load_model()

    def _load_model(self):
        """Try to load RAFT model."""
        if self.method == "farneback":
            self.loaded = True
            return

        try:
            import torch
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
            self.raft_model.to(self.device)
            self.raft_model.eval()
            self.method = "raft"
            self.loaded = True
            logger.info(f"RAFT optical flow loaded on {self.device}")
        except Exception as e:
            logger.warning(f"RAFT not available: {e}. Using Farneback.")
            self.method = "farneback"
            self.loaded = True

    def compute_flow(self, frame: np.ndarray) -> Dict[str, Any]:
        """Compute dense optical flow between consecutive frames."""
        start = time.perf_counter()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_frame = frame
            h, w = frame.shape[:2]
            return {
                "flow": np.zeros((h, w, 2), dtype=np.float32),
                "visualization": np.zeros((h, w, 3), dtype=np.uint8),
                "magnitude": np.zeros((h, w), dtype=np.float32),
                "motion_mask": np.zeros((h, w), dtype=np.uint8),
                "inference_ms": 0,
                "method": self.method,
            }

        if self.method == "raft" and self.raft_model is not None:
            try:
                flow = self._compute_raft(self.prev_frame, frame)
            except Exception:
                flow = self._compute_farneback(self.prev_gray, gray)
        else:
            flow = self._compute_farneback(self.prev_gray, gray)

        # Compute magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # HSV visualization
        hsv = np.zeros((*frame.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction
        hsv[..., 1] = 255  # Saturation = full
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_viz = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Motion segmentation via K-means
        motion_mask = self._segment_motion(mag)

        # Background subtraction
        bg_mask = self._background_subtract(frame)

        self.prev_gray = gray
        self.prev_frame = frame

        inference_ms = (time.perf_counter() - start) * 1000

        return {
            "flow": flow,
            "visualization": flow_viz,
            "magnitude": mag,
            "angle": ang,
            "motion_mask": motion_mask,
            "bg_mask": bg_mask,
            "mean_magnitude": float(np.mean(mag)),
            "max_magnitude": float(np.max(mag)),
            "inference_ms": inference_ms,
            "method": self.method,
        }

    def _compute_raft(self, prev_frame: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Compute flow using RAFT."""
        import torch
        import torchvision.transforms.functional as F

        prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
        curr_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to multiple of 8
        h, w = prev_rgb.shape[:2]
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        prev_rgb = cv2.resize(prev_rgb, (new_w, new_h))
        curr_rgb = cv2.resize(curr_rgb, (new_w, new_h))

        prev_tensor = F.to_tensor(prev_rgb).unsqueeze(0).to(self.device)
        curr_tensor = F.to_tensor(curr_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            flow_predictions = self.raft_model(prev_tensor, curr_tensor)
            flow = flow_predictions[-1][0].permute(1, 2, 0).cpu().numpy()

        # Resize back
        flow = cv2.resize(flow, (w, h))
        flow[..., 0] *= w / new_w
        flow[..., 1] *= h / new_h

        return flow

    def _compute_farneback(self, prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Compute flow using Farneback algorithm."""
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=5, winsize=15,
            iterations=3, poly_n=7, poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        return flow

    def _segment_motion(self, magnitude: np.ndarray, k: int = 3) -> np.ndarray:
        """K-means clustering on flow magnitude for motion segmentation."""
        mag_flat = magnitude.flatten().reshape(-1, 1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(mag_flat, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

        # Highest magnitude cluster = foreground motion
        max_center_idx = np.argmax(centers)
        motion_mask = (labels.reshape(magnitude.shape) == max_center_idx).astype(np.uint8) * 255

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

        return motion_mask

    def _background_subtract(self, frame: np.ndarray) -> np.ndarray:
        """Background subtraction using MOG2 or KNN."""
        if self.bg_method == "knn":
            return self.bg_subtractor_knn.apply(frame)
        return self.bg_subtractor_mog.apply(frame)

    def draw_flow_arrows(self, frame: np.ndarray, flow: np.ndarray,
                         step: int = 20, scale: float = 3.0) -> np.ndarray:
        """Draw sparse motion arrows on frame."""
        annotated = frame.copy()
        h, w = flow.shape[:2]

        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                mag = np.sqrt(fx ** 2 + fy ** 2)
                if mag < 1.0:
                    continue
                end_pt = (int(x + fx * scale), int(y + fy * scale))
                # Color by magnitude
                color_val = min(255, int(mag * 20))
                color = (0, color_val, 255 - color_val)
                cv2.arrowedLine(annotated, (x, y), end_pt, color, 1, cv2.LINE_AA, tipLength=0.3)

        return annotated

    def reset(self):
        """Reset state for new video."""
        self.prev_gray = None
        self.prev_frame = None
