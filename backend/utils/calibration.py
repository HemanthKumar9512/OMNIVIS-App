"""
OMNIVIS — Camera Calibration Utilities
Chessboard-based camera calibration + stereo calibration.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import json
import logging

logger = logging.getLogger(__name__)


class CameraCalibrator:
    """Handles camera calibration using chessboard patterns."""

    def __init__(self, pattern_size: Tuple[int, int] = (9, 6),
                 square_size_mm: float = 25.0):
        self.pattern_size = pattern_size
        self.square_size = square_size_mm
        self.obj_points: List[np.ndarray] = []
        self.img_points: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None

        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_mm

        # Calibration results
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.rvecs: Optional[List] = None
        self.tvecs: Optional[List] = None
        self.rms_error: Optional[float] = None

    def add_frame(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Process a calibration frame. Returns (found, annotated_frame)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        self.image_size = gray.shape[::-1]

        # Find chessboard corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)

        if found:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            self.obj_points.append(self.objp)
            self.img_points.append(corners_refined)

            # Draw and return annotated frame
            annotated = frame.copy()
            cv2.drawChessboardCorners(annotated, self.pattern_size, corners_refined, found)
            return True, annotated

        return False, None

    def calibrate(self) -> Dict:
        """Run camera calibration on collected frames."""
        if len(self.obj_points) < 5:
            return {
                "success": False,
                "message": f"Need at least 5 valid frames, got {len(self.obj_points)}",
            }

        self.rms_error, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(
                self.obj_points, self.img_points, self.image_size, None, None
            )

        logger.info(f"Camera calibration RMS error: {self.rms_error:.4f}")

        return {
            "success": True,
            "rms_error": float(self.rms_error),
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coeffs": self.dist_coeffs.ravel().tolist(),
            "num_frames_used": len(self.obj_points),
            "message": f"Calibration successful with RMS error {self.rms_error:.4f}",
        }

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Remove lens distortion from frame."""
        if self.camera_matrix is None:
            return frame
        h, w = frame.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_mtx)
        x, y, w, h = roi
        return undistorted[y:y + h, x:x + w]

    def save(self, path: str):
        """Save calibration data to JSON."""
        if self.camera_matrix is None:
            raise ValueError("No calibration data to save")
        data = {
            "rms_error": float(self.rms_error),
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coeffs": self.dist_coeffs.ravel().tolist(),
            "image_size": list(self.image_size),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load calibration data from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        self.camera_matrix = np.array(data["camera_matrix"])
        self.dist_coeffs = np.array(data["distortion_coeffs"])
        self.image_size = tuple(data["image_size"])
        self.rms_error = data["rms_error"]


class StereoCalibrator:
    """Stereo camera calibration and rectification."""

    def __init__(self, left_calibrator: CameraCalibrator,
                 right_calibrator: CameraCalibrator):
        self.left = left_calibrator
        self.right = right_calibrator
        self.R: Optional[np.ndarray] = None
        self.T: Optional[np.ndarray] = None

    def calibrate_stereo(self, left_points: List[np.ndarray],
                         right_points: List[np.ndarray],
                         obj_points: List[np.ndarray]) -> Dict:
        """Perform stereo calibration."""
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        rms, _, _, _, _, self.R, self.T, E, F = cv2.stereoCalibrate(
            obj_points, left_points, right_points,
            self.left.camera_matrix, self.left.dist_coeffs,
            self.right.camera_matrix, self.right.dist_coeffs,
            self.left.image_size, criteria=criteria, flags=flags
        )

        return {
            "success": True,
            "rms_error": float(rms),
            "rotation": self.R.tolist(),
            "translation": self.T.ravel().tolist(),
        }

    def compute_disparity(self, left_frame: np.ndarray,
                          right_frame: np.ndarray) -> np.ndarray:
        """Compute disparity map using SGBM."""
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=128, blockSize=5,
            P1=8 * 3 * 5 ** 2, P2=32 * 3 * 5 ** 2,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32,
            preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        disparity = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disparity
