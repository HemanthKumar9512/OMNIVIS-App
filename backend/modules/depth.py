"""
OMNIVIS — Depth Estimation Module
MiDaS v3.1 monocular depth + depth from defocus.
"""
import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Monocular depth estimation using MiDaS."""

    COLORMAPS = {
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
    }

    def __init__(self, model_type: str = "DPT_Large", device: str = "auto",
                 colormap: str = "plasma"):
        self.model_type = model_type
        self.device = device
        self.colormap = colormap
        self.model = None
        self.transform = None
        self.loaded = False
        self._load_model()

    def _load_model(self):
        """Load MiDaS depth estimation model."""
        try:
            import torch
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            self.loaded = True
            logger.info(f"MiDaS {self.model_type} loaded on {self.device}")
        except Exception as e:
            logger.warning(f"MiDaS load failed: {e}. Using disparity simulation.")
            self.loaded = False

    def estimate(self, frame: np.ndarray) -> Dict[str, Any]:
        """Estimate depth from a single frame."""
        start = time.perf_counter()
        h, w = frame.shape[:2]

        if self.loaded and self.model is not None:
            try:
                depth = self._run_midas(frame)
            except Exception as e:
                logger.error(f"MiDaS error: {e}")
                depth = self._simulate_depth(frame)
        else:
            depth = self._simulate_depth(frame)

        # Normalize to [0, 1]
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max - depth_min > 0:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth)

        # Colorize
        depth_colored = self._colorize_depth(depth_norm)

        # Depth from defocus (supplementary)
        dfd = self._depth_from_defocus(frame)

        return {
            "depth_map": depth_norm,
            "depth_colored": depth_colored,
            "depth_from_defocus": dfd,
            "min_depth": float(depth_min),
            "max_depth": float(depth_max),
            "mean_depth": float(np.mean(depth)),
            "inference_ms": (time.perf_counter() - start) * 1000,
            "model": self.model_type if self.loaded else "simulation",
        }

    def _run_midas(self, frame: np.ndarray) -> np.ndarray:
        """Run MiDaS inference."""
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy()

    def _simulate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Fast depth simulation using gradient."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Simple vertical gradient (bottom = closer)
        vert_gradient = np.linspace(0.5, 2.0, h).reshape(-1, 1)
        vert_gradient = np.tile(vert_gradient, (1, w))

        # Add noise
        noise = np.random.random((h, w)) * 0.3
        
        depth = vert_gradient + noise
        return depth.astype(np.float32)

    def _depth_from_defocus(self, frame: np.ndarray) -> np.ndarray:
        """Compute relative depth from image defocus (Laplacian variance)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Compute local blur measure in patches
        patch_size = 32
        depth_map = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h - patch_size, patch_size // 2):
            for x in range(0, w - patch_size, patch_size // 2):
                patch = gray[y:y + patch_size, x:x + patch_size]
                focus_measure = cv2.Laplacian(patch, cv2.CV_64F).var()
                depth_map[y:y + patch_size, x:x + patch_size] = focus_measure

        # Normalize
        if depth_map.max() > 0:
            depth_map = depth_map / depth_map.max()

        return depth_map

    def _colorize_depth(self, depth_norm: np.ndarray) -> np.ndarray:
        """Apply colormap to normalized depth."""
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        cmap = self.COLORMAPS.get(self.colormap, cv2.COLORMAP_PLASMA)
        return cv2.applyColorMap(depth_uint8, cmap)

    def draw_depth_overlay(self, frame: np.ndarray, depth_colored: np.ndarray,
                           alpha: float = 0.5) -> np.ndarray:
        """Overlay depth map on original frame."""
        depth_resized = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
        return cv2.addWeighted(frame, 1 - alpha, depth_resized, alpha, 0)
