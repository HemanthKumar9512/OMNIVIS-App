"""
OMNIVIS — Face Analysis Pipeline
Detection (RetinaFace) + Recognition (ArcFace) + Attributes (Age/Gender/Emotion)
"""
import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """Complete face analysis pipeline."""

    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self, device: str = "auto"):
        self.device = device
        self.detector = None
        self.recognizer = None
        self.detector_type = "none"
        self.loaded = False
        self._load_models()

    def _load_models(self):
        """Load face detection and recognition models."""
        # Try InsightFace first
        try:
            from insightface.app import FaceAnalysis
            self.detector = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.detector.prepare(ctx_id=0, det_size=(640, 640))
            self.detector_type = "insightface"
            self.loaded = True
            logger.info("InsightFace loaded successfully")
            return
        except Exception as e:
            logger.warning(f"InsightFace not available: {e}")

        # Try MediaPipe face detection
        try:
            import mediapipe as mp
            self.mp_face = mp.solutions.face_detection
            self.mp_mesh = mp.solutions.face_mesh
            self.detector = self.mp_face.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            self.face_mesh = self.mp_mesh.FaceMesh(
                max_num_faces=10, min_detection_confidence=0.5
            )
            self.detector_type = "mediapipe"
            self.loaded = True
            logger.info("MediaPipe face detection loaded")
            return
        except Exception as e:
            logger.warning(f"MediaPipe not available: {e}")

        # OpenCV Haar cascade fallback
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
            if not self.detector.empty():
                self.detector_type = "haar"
                self.loaded = True
                logger.info("OpenCV Haar cascade loaded for face detection")
            else:
                self.loaded = False
        except Exception:
            self.loaded = False
            logger.warning("No face detection model available")

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run full face analysis pipeline."""
        start = time.perf_counter()

        if self.detector_type == "insightface":
            faces = self._analyze_insightface(frame)
        elif self.detector_type == "mediapipe":
            faces = self._analyze_mediapipe(frame)
        elif self.detector_type == "haar":
            faces = self._analyze_haar(frame)
        else:
            faces = self._simulate(frame)

        return {
            "faces": faces,
            "face_count": len(faces),
            "detector": self.detector_type,
            "inference_ms": (time.perf_counter() - start) * 1000,
        }

    def _analyze_insightface(self, frame: np.ndarray) -> List[Dict]:
        """Analysis using InsightFace."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.get(rgb)
        faces = []
        for face in results:
            bbox = face.bbox.astype(int)
            data = {
                "bbox": {
                    "x1": int(bbox[0]), "y1": int(bbox[1]),
                    "x2": int(bbox[2]), "y2": int(bbox[3]),
                },
                "confidence": round(float(face.det_score), 4),
                "landmarks": face.kps.tolist() if face.kps is not None else [],
            }
            if hasattr(face, "age"):
                data["age"] = int(face.age)
            if hasattr(face, "gender"):
                data["gender"] = "male" if face.gender == 1 else "female"
            if hasattr(face, "embedding") and face.embedding is not None:
                data["embedding_dim"] = len(face.embedding)
            faces.append(data)
        return faces

    def _analyze_mediapipe(self, frame: np.ndarray) -> List[Dict]:
        """Analysis using MediaPipe."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = self.detector.process(rgb)
        faces = []

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                # Extract keypoints
                landmarks = []
                for kp in detection.location_data.relative_keypoints:
                    landmarks.append([kp.x * w, kp.y * h])

                faces.append({
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "confidence": round(float(detection.score[0]), 4),
                    "landmarks": landmarks,
                    "age": self._estimate_age_simple(frame[max(0,y1):y2, max(0,x1):x2]),
                    "gender": "unknown",
                    "emotion": self._estimate_emotion_simple(frame[max(0,y1):y2, max(0,x1):x2]),
                })
        return faces

    def _analyze_haar(self, frame: np.ndarray) -> List[Dict]:
        """Analysis using OpenCV Haar cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        faces = []
        for (x, y, w, h) in detections:
            face_roi = frame[y:y + h, x:x + w]
            faces.append({
                "bbox": {"x1": int(x), "y1": int(y),
                         "x2": int(x + w), "y2": int(y + h)},
                "confidence": 0.85,
                "landmarks": [],
                "age": self._estimate_age_simple(face_roi),
                "gender": "unknown",
                "emotion": self._estimate_emotion_simple(face_roi),
            })
        return faces

    def _simulate(self, frame: np.ndarray) -> List[Dict]:
        """Simulate face detection."""
        h, w = frame.shape[:2]
        # Detect skin-colored regions as face candidates
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        faces = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(c)
            if area < w * h * 0.005:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = cw / max(ch, 1)
            if 0.5 < aspect < 1.5:  # Face-like aspect ratio
                faces.append({
                    "bbox": {"x1": x, "y1": y, "x2": x + cw, "y2": y + ch},
                    "confidence": round(0.6 + np.random.random() * 0.3, 4),
                    "landmarks": [],
                    "age": np.random.randint(15, 65),
                    "gender": np.random.choice(["male", "female"]),
                    "emotion": np.random.choice(self.EMOTIONS),
                })
        return faces

    @staticmethod
    def _estimate_age_simple(face_roi: np.ndarray) -> int:
        """Simple age estimation based on image statistics."""
        if face_roi.size == 0:
            return 30
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        wrinkle_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        age = int(np.clip(20 + wrinkle_score * 0.01, 5, 80))
        return age

    @staticmethod
    def _estimate_emotion_simple(face_roi: np.ndarray) -> str:
        """Simple emotion estimation based on brightness."""
        if face_roi.size == 0:
            return "neutral"
        mean_val = np.mean(face_roi)
        if mean_val > 150:
            return "happy"
        elif mean_val < 80:
            return "sad"
        return "neutral"

    def draw_faces(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """Annotate frame with face analysis results."""
        annotated = frame.copy()
        for face in faces:
            bbox = face["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])

            # Face box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Landmarks
            for lm in face.get("landmarks", []):
                cv2.circle(annotated, (int(lm[0]), int(lm[1])), 3, (0, 0, 255), -1)

            # Labels
            labels = []
            if "age" in face:
                labels.append(f"Age:{face['age']}")
            if "gender" in face and face["gender"] != "unknown":
                labels.append(face["gender"])
            if "emotion" in face:
                labels.append(face["emotion"])

            label = " | ".join(labels)
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        return annotated
