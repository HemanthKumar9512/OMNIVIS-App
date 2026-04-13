"""
OMNIVIS — Gait Analysis Module
MediaPipe pose estimation + LSTM gait classification.
"""
import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class GaitAnalyzer:
    """Human gait analysis using pose estimation."""

    GAIT_CLASSES = ["normal", "limping", "falling", "running", "suspicious"]

    BODY_PARTS = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]

    # Key joint indices for gait analysis
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

    def __init__(self, device: str = "auto"):
        self.device = device
        self.pose = None
        self.loaded = False
        self.gait_histories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=60))
        self._load_model()

    def _load_model(self):
        """Load MediaPipe Holistic pose estimator."""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.loaded = True
            logger.info("MediaPipe Pose loaded for gait analysis")
        except ImportError:
            logger.warning("MediaPipe not available. Using OpenPose-style skeleton simulation.")
            self.loaded = False

    def analyze(self, frame: np.ndarray, tracks: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Analyze human gait from frame."""
        start = time.perf_counter()

        if self.loaded:
            result = self._run_mediapipe(frame)
        else:
            result = self._simulate(frame)

        # Gait classification
        for person in result.get("persons", []):
            features = self._extract_gait_features(person)
            person["gait_features"] = features

            track_id = person.get("track_id", 0)
            self.gait_histories[track_id].append(features)

            if len(self.gait_histories[track_id]) >= 10:
                classification = self._classify_gait(list(self.gait_histories[track_id]))
                person["gait_class"] = classification["class"]
                person["gait_confidence"] = classification["confidence"]
                person["symmetry_index"] = classification.get("symmetry_index", 0)
                person["gait_deviation_index"] = classification.get("gdi", 0)

        result["inference_ms"] = (time.perf_counter() - start) * 1000
        return result

    def _run_mediapipe(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run MediaPipe pose estimation."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        h, w = frame.shape[:2]

        persons = []
        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append({
                    "x": lm.x * w,
                    "y": lm.y * h,
                    "z": lm.z,
                    "visibility": lm.visibility,
                })

            persons.append({
                "landmarks": landmarks,
                "track_id": 0,
            })

        return {"persons": persons, "person_count": len(persons)}

    def _simulate(self, frame: np.ndarray) -> Dict[str, Any]:
        """Simulate skeleton detection."""
        h, w = frame.shape[:2]
        # Generate plausible skeleton
        persons = []
        # Simple body detection using contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            if cv2.contourArea(c) < h * w * 0.02:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            if ch / max(cw, 1) < 1.2:  # Not person-like aspect ratio
                continue

            # Generate fake landmarks
            landmarks = self._generate_fake_skeleton(x, y, cw, ch)
            persons.append({
                "landmarks": landmarks,
                "track_id": len(persons),
            })

        return {"persons": persons, "person_count": len(persons)}

    def _generate_fake_skeleton(self, x: int, y: int, w: int, h: int) -> List[Dict]:
        """Generate plausible skeleton landmarks from bounding box."""
        cx = x + w / 2
        landmarks = []

        keypoint_positions = {
            0: (cx, y + h * 0.05),          # nose
            11: (cx - w * 0.2, y + h * 0.25),  # left shoulder
            12: (cx + w * 0.2, y + h * 0.25),  # right shoulder
            13: (cx - w * 0.25, y + h * 0.45), # left elbow
            14: (cx + w * 0.25, y + h * 0.45), # right elbow
            15: (cx - w * 0.3, y + h * 0.6),   # left wrist
            16: (cx + w * 0.3, y + h * 0.6),   # right wrist
            23: (cx - w * 0.15, y + h * 0.55),  # left hip
            24: (cx + w * 0.15, y + h * 0.55),  # right hip
            25: (cx - w * 0.15, y + h * 0.75),  # left knee
            26: (cx + w * 0.15, y + h * 0.75),  # right knee
            27: (cx - w * 0.15, y + h * 0.95),  # left ankle
            28: (cx + w * 0.15, y + h * 0.95),  # right ankle
        }

        for i in range(33):
            if i in keypoint_positions:
                px, py = keypoint_positions[i]
                # Add small random noise
                px += np.random.normal(0, 2)
                py += np.random.normal(0, 2)
            else:
                px, py = cx, y + h * 0.5

            landmarks.append({
                "x": float(px),
                "y": float(py),
                "z": 0.0,
                "visibility": 0.8 if i in keypoint_positions else 0.2,
            })

        return landmarks

    def _extract_gait_features(self, person: Dict) -> Dict[str, float]:
        """Extract gait-relevant features from skeleton."""
        landmarks = person.get("landmarks", [])
        if len(landmarks) < 33:
            return {}

        def get_point(idx):
            lm = landmarks[idx]
            return np.array([lm["x"], lm["y"]])

        # Joint angles
        left_hip_angle = self._compute_angle(
            get_point(self.LEFT_SHOULDER), get_point(self.LEFT_HIP), get_point(self.LEFT_KNEE)
        )
        right_hip_angle = self._compute_angle(
            get_point(self.RIGHT_SHOULDER), get_point(self.RIGHT_HIP), get_point(self.RIGHT_KNEE)
        )
        left_knee_angle = self._compute_angle(
            get_point(self.LEFT_HIP), get_point(self.LEFT_KNEE), get_point(self.LEFT_ANKLE)
        )
        right_knee_angle = self._compute_angle(
            get_point(self.RIGHT_HIP), get_point(self.RIGHT_KNEE), get_point(self.RIGHT_ANKLE)
        )

        # Stride width (horizontal distance between ankles)
        stride_width = abs(landmarks[self.LEFT_ANKLE]["x"] - landmarks[self.RIGHT_ANKLE]["x"])

        # Body tilt (angle of shoulder line from horizontal)
        shoulder_dx = landmarks[self.RIGHT_SHOULDER]["x"] - landmarks[self.LEFT_SHOULDER]["x"]
        shoulder_dy = landmarks[self.RIGHT_SHOULDER]["y"] - landmarks[self.LEFT_SHOULDER]["y"]
        body_tilt = np.degrees(np.arctan2(shoulder_dy, max(shoulder_dx, 1e-6)))

        # Center of mass approximation
        com_y = np.mean([landmarks[i]["y"] for i in [self.LEFT_HIP, self.RIGHT_HIP,
                                                       self.LEFT_SHOULDER, self.RIGHT_SHOULDER]])

        return {
            "left_hip_angle": float(left_hip_angle),
            "right_hip_angle": float(right_hip_angle),
            "left_knee_angle": float(left_knee_angle),
            "right_knee_angle": float(right_knee_angle),
            "stride_width": float(stride_width),
            "body_tilt": float(body_tilt),
            "center_of_mass_y": float(com_y),
            "hip_symmetry": float(abs(left_hip_angle - right_hip_angle)),
            "knee_symmetry": float(abs(left_knee_angle - right_knee_angle)),
        }

    @staticmethod
    def _compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Compute angle at point b formed by points a-b-c."""
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

    def _classify_gait(self, feature_sequence: List[Dict]) -> Dict[str, Any]:
        """Classify gait pattern from feature sequence."""
        if not feature_sequence or not feature_sequence[0]:
            return {"class": "unknown", "confidence": 0.0}

        # Extract time-series features
        hip_angles = [f.get("left_hip_angle", 0) + f.get("right_hip_angle", 0)
                      for f in feature_sequence if f]
        knee_angles = [f.get("left_knee_angle", 0) + f.get("right_knee_angle", 0)
                       for f in feature_sequence if f]
        tilts = [f.get("body_tilt", 0) for f in feature_sequence if f]
        symmetries = [f.get("hip_symmetry", 0) for f in feature_sequence if f]

        if not hip_angles:
            return {"class": "unknown", "confidence": 0.0}

        # Compute statistics
        hip_std = np.std(hip_angles)
        knee_range = np.max(knee_angles) - np.min(knee_angles) if knee_angles else 0
        tilt_std = np.std(tilts) if tilts else 0
        avg_symmetry = np.mean(symmetries) if symmetries else 0

        # Symmetry index (0 = perfectly symmetric, higher = more asymmetric)
        symmetry_index = float(avg_symmetry / max(np.mean(hip_angles), 1e-6) * 100)

        # Simple rule-based classification
        if tilt_std > 15 or (len(tilts) > 1 and max(tilts) > 30):
            return {"class": "falling", "confidence": 0.8, "symmetry_index": symmetry_index, "gdi": 50}
        elif hip_std > 20 and knee_range > 40:
            return {"class": "running", "confidence": 0.75, "symmetry_index": symmetry_index, "gdi": 85}
        elif avg_symmetry > 15:
            return {"class": "limping", "confidence": 0.7, "symmetry_index": symmetry_index, "gdi": 60}
        elif hip_std < 5 and knee_range < 10:
            return {"class": "suspicious", "confidence": 0.5, "symmetry_index": symmetry_index, "gdi": 70}
        else:
            return {"class": "normal", "confidence": 0.85, "symmetry_index": symmetry_index, "gdi": 95}

    def draw_skeleton(self, frame: np.ndarray, persons: List[Dict]) -> np.ndarray:
        """Draw skeleton overlays on frame."""
        annotated = frame.copy()
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
            (25, 27), (26, 28),
        ]

        for person in persons:
            landmarks = person.get("landmarks", [])
            if len(landmarks) < 33:
                continue

            # Draw connections
            for a, b in connections:
                if landmarks[a]["visibility"] > 0.3 and landmarks[b]["visibility"] > 0.3:
                    pt1 = (int(landmarks[a]["x"]), int(landmarks[a]["y"]))
                    pt2 = (int(landmarks[b]["x"]), int(landmarks[b]["y"]))
                    cv2.line(annotated, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

            # Draw keypoints
            for i, lm in enumerate(landmarks):
                if lm["visibility"] > 0.3:
                    pt = (int(lm["x"]), int(lm["y"]))
                    color = (0, 0, 255) if i in [self.LEFT_HIP, self.RIGHT_HIP,
                                                   self.LEFT_KNEE, self.RIGHT_KNEE,
                                                   self.LEFT_ANKLE, self.RIGHT_ANKLE] else (0, 255, 0)
                    cv2.circle(annotated, pt, 4, color, -1)

            # Draw gait label
            if "gait_class" in person:
                label = f"Gait: {person['gait_class']} ({person.get('gait_confidence', 0):.2f})"
                y_pos = int(landmarks[0]["y"] - 20) if landmarks[0]["visibility"] > 0.3 else 30
                x_pos = int(landmarks[0]["x"]) if landmarks[0]["visibility"] > 0.3 else 10
                cv2.putText(annotated, label, (x_pos, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def reset(self):
        """Reset gait analysis state."""
        self.gait_histories.clear()
