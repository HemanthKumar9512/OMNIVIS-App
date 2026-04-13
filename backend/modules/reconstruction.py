"""
OMNIVIS — 3D Reconstruction Module
Feature matching + Structure from Motion + Dense reconstruction.
"""
import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SfMReconstructor:
    """Structure from Motion 3D reconstruction pipeline."""

    def __init__(self, feature_method: str = "orb", device: str = "auto"):
        self.feature_method = feature_method
        self.device = device
        self.frames: List[np.ndarray] = []
        self.keypoints_list: List[list] = []
        self.descriptors_list: List[np.ndarray] = []
        self.matches_list: List[list] = []
        self.point_cloud: Optional[np.ndarray] = None
        self.point_colors: Optional[np.ndarray] = None
        self.camera_poses: List[np.ndarray] = []

        # Camera intrinsics (default — override with calibration)
        self.K = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float64)

        self._init_feature_detector()

    def _init_feature_detector(self):
        """Initialize feature detector and matcher."""
        if self.feature_method == "sift":
            self.detector = cv2.SIFT_create(nfeatures=2000)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif self.feature_method == "orb":
            self.detector = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.detector = cv2.ORB_create(nfeatures=3000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def add_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Add a frame to the reconstruction pipeline."""
        start = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        result = {
            "keypoints_count": len(keypoints),
            "matches_count": 0,
            "inliers_count": 0,
            "frame_index": len(self.frames),
            "inference_ms": 0,
        }

        if descriptors is not None and len(self.frames) > 0 and self.descriptors_list[-1] is not None:
            # Match with previous frame
            prev_descriptors = self.descriptors_list[-1]
            prev_keypoints = self.keypoints_list[-1]

            matches = self.matcher.knnMatch(prev_descriptors, descriptors, k=2)

            # Lowe's ratio test
            good_matches = []
            for m_list in matches:
                if len(m_list) == 2:
                    m, n = m_list
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            result["matches_count"] = len(good_matches)

            if len(good_matches) >= 8:
                # Compute fundamental matrix with RANSAC
                pts1 = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

                E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                inlier_count = int(mask.sum()) if mask is not None else 0
                result["inliers_count"] = inlier_count

                if E is not None and inlier_count > 10:
                    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

                    # Triangulate points
                    pts1_inlier = pts1[mask_pose.ravel() > 0]
                    pts2_inlier = pts2[mask_pose.ravel() > 0]

                    if len(pts1_inlier) > 0:
                        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
                        P2 = self.K @ np.hstack([R, t])

                        points_4d = cv2.triangulatePoints(
                            P1, P2,
                            pts1_inlier.T, pts2_inlier.T
                        )
                        points_3d = (points_4d[:3] / points_4d[3:]).T

                        # Filter invalid points
                        valid = np.all(np.isfinite(points_3d), axis=1)
                        points_3d = points_3d[valid]
                        depths = points_3d[:, 2]
                        valid_depth = (depths > 0) & (depths < 100)
                        points_3d = points_3d[valid_depth]

                        # Get colors from frame
                        if len(points_3d) > 0:
                            colors = []
                            for pt2d in pts2_inlier[:len(points_3d)]:
                                x, y = int(pt2d[0]), int(pt2d[1])
                                x = max(0, min(x, frame.shape[1] - 1))
                                y = max(0, min(y, frame.shape[0] - 1))
                                colors.append(frame[y, x][::-1])  # BGR to RGB

                            if self.point_cloud is None:
                                self.point_cloud = points_3d
                                self.point_colors = np.array(colors)
                            else:
                                self.point_cloud = np.vstack([self.point_cloud, points_3d])
                                self.point_colors = np.vstack([self.point_colors, np.array(colors)])

                            # Keep point cloud manageable
                            if len(self.point_cloud) > 50000:
                                indices = np.random.choice(len(self.point_cloud), 50000, replace=False)
                                self.point_cloud = self.point_cloud[indices]
                                self.point_colors = self.point_colors[indices]

                    self.camera_poses.append(np.hstack([R, t]))

                self.matches_list.append(good_matches)

        self.frames.append(frame)
        self.keypoints_list.append(keypoints)
        self.descriptors_list.append(descriptors)

        result["inference_ms"] = (time.perf_counter() - start) * 1000
        result["total_points"] = len(self.point_cloud) if self.point_cloud is not None else 0
        result["total_frames"] = len(self.frames)

        return result

    def get_point_cloud(self) -> Dict[str, Any]:
        """Get current point cloud data for 3D visualization."""
        if self.point_cloud is None or len(self.point_cloud) == 0:
            return {"points": [], "colors": [], "count": 0}

        return {
            "points": self.point_cloud.tolist(),
            "colors": (self.point_colors / 255.0).tolist() if self.point_colors is not None else [],
            "count": len(self.point_cloud),
            "camera_poses": [p.tolist() for p in self.camera_poses],
        }

    def draw_matches(self, frame: np.ndarray) -> np.ndarray:
        """Draw feature keypoints on frame."""
        if len(self.keypoints_list) == 0:
            return frame
        return cv2.drawKeypoints(frame, self.keypoints_list[-1], None,
                                color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def set_camera_matrix(self, K: np.ndarray):
        """Update camera intrinsic matrix from calibration."""
        self.K = K

    def reset(self):
        """Reset the reconstruction pipeline."""
        self.frames.clear()
        self.keypoints_list.clear()
        self.descriptors_list.clear()
        self.matches_list.clear()
        self.point_cloud = None
        self.point_colors = None
        self.camera_poses.clear()


class WaveletAnalyzer:
    """2D Wavelet decomposition for texture analysis."""

    def __init__(self, wavelet: str = "haar", levels: int = 3):
        self.wavelet = wavelet
        self.levels = levels

    def decompose(self, frame: np.ndarray) -> Dict[str, Any]:
        """Perform 2D DWT decomposition."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        try:
            import pywt
            coeffs = pywt.wavedec2(gray, self.wavelet, level=self.levels)

            subbands = {"LL": coeffs[0]}
            for i, (LH, HL, HH) in enumerate(coeffs[1:], 1):
                subbands[f"LH_{i}"] = LH
                subbands[f"HL_{i}"] = HL
                subbands[f"HH_{i}"] = HH

            return {
                "subbands": {k: v.tolist() for k, v in subbands.items()},
                "energy": {k: float(np.sum(v ** 2)) for k, v in subbands.items()},
            }
        except ImportError:
            # Fallback: manual Haar wavelet using averaging/differencing
            return self._manual_haar(gray)

    def _manual_haar(self, gray: np.ndarray) -> Dict[str, Any]:
        """Manual Haar-like decomposition using OpenCV."""
        h, w = gray.shape
        # Downsample for LL (approximation)
        LL = cv2.pyrDown(gray)
        # Compute details via subtraction
        LL_up = cv2.pyrUp(LL, dstsize=(w, h))
        detail = gray - LL_up

        return {
            "subbands": {
                "LL": LL.tolist(),
                "detail": detail.tolist(),
            },
            "energy": {
                "LL": float(np.sum(LL ** 2)),
                "detail": float(np.sum(detail ** 2)),
            },
        }

    def compute_glcm_features(self, frame: np.ndarray) -> Dict[str, float]:
        """Compute GLCM texture features."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Quantize to fewer levels for GLCM
        gray_q = (gray // 16).astype(np.uint8)  # 16 levels

        h, w = gray_q.shape
        levels = 16
        glcm = np.zeros((levels, levels), dtype=np.float64)

        # Compute co-occurrence for distance=1, angle=0
        for y in range(h):
            for x in range(w - 1):
                i, j = gray_q[y, x], gray_q[y, x + 1]
                glcm[i, j] += 1
                glcm[j, i] += 1  # Symmetric

        # Normalize
        glcm_sum = glcm.sum()
        if glcm_sum > 0:
            glcm /= glcm_sum

        # Compute features
        contrast = 0.0
        energy = 0.0
        homogeneity = 0.0
        entropy_val = 0.0

        for i in range(levels):
            for j in range(levels):
                contrast += (i - j) ** 2 * glcm[i, j]
                energy += glcm[i, j] ** 2
                homogeneity += glcm[i, j] / (1 + abs(i - j))
                if glcm[i, j] > 0:
                    entropy_val -= glcm[i, j] * np.log2(glcm[i, j])

        return {
            "contrast": contrast,
            "energy": energy,
            "homogeneity": homogeneity,
            "entropy": entropy_val,
        }
