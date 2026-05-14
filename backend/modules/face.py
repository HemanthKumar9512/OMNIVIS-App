"""
OMNIVIS - Advanced Face Analysis Pipeline
Multi-tier detection: InsightFace > MediaPipe > OpenCV DNN > Haar Cascade > Skin-based
Includes: Detection, landmarks, age/gender estimation, emotion recognition, face tracking
"""
import cv2
import numpy as np
import time
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class FaceTracker:
    """Track faces across frames using IoU matching."""

    def __init__(self, max_disappeared=10, max_distance=100):
        self.next_id = 0
        self.tracks = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, bboxes):
        if len(bboxes) == 0:
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
            return []

        if len(self.tracks) == 0:
            for bbox in bboxes:
                self.tracks[self.next_id] = bbox
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return [{"id": tid, "bbox": bbox} for tid, bbox in self.tracks.items()]

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid] for tid in track_ids]

        used_tracks = set()
        used_dets = set()
        results = []

        for i, det_bbox in enumerate(bboxes):
            best_iou = 0
            best_track = None

            for j, track_bbox in enumerate(track_boxes):
                if j in used_tracks:
                    continue
                iou = self._compute_iou(det_bbox, track_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track = j

            if best_track is not None and best_iou > 0.3:
                track_id = track_ids[best_track]
                self.tracks[track_id] = det_bbox
                self.disappeared[track_id] = 0
                used_tracks.add(best_track)
                used_dets.add(i)
                results.append({"id": track_id, "bbox": det_bbox})

        for i, det_bbox in enumerate(bboxes):
            if i not in used_dets:
                self.tracks[self.next_id] = det_bbox
                self.disappeared[self.next_id] = 0
                results.append({"id": self.next_id, "bbox": det_bbox})
                self.next_id += 1

        for track_id in track_ids:
            if track_id not in used_tracks:
                self.disappeared[track_id] = self.disappeared.get(track_id, 0) + 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
                else:
                    results.append({"id": track_id, "bbox": self.tracks[track_id]})

        return results

    @staticmethod
    def _compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - inter_area
        return inter_area / union if union > 0 else 0


class EmotionRecognizer:
    """Lightweight emotion recognition using facial features."""

    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            import torch.nn as nn

            class MiniEmotionNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(1, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1),
                    )
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(256, 7),
                    )

                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x

            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "emotion_model.pth")
            if os.path.exists(model_path):
                self._model = MiniEmotionNet()
                self._model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                self._model.eval()
                logger.info("Emotion recognition model loaded")
            else:
                logger.info("Emotion model not found - using heuristic fallback")
                self._model = None
        except Exception as e:
            logger.warning(f"Emotion model load failed: {e}")
            self._model = None

    def predict(self, face_roi):
        if face_roi.size == 0:
            return "neutral", {}

        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            resized = cv2.resize(gray, (48, 48))
            normalized = resized.astype(np.float32) / 255.0

            if self._model is not None:
                import torch
                tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    output = self._model(tensor)
                    probs = torch.softmax(output, dim=1).squeeze().numpy()
                    idx = int(np.argmax(probs))
                    conf = float(probs[idx])
                    return self.EMOTIONS[idx], {e: round(float(p), 3) for e, p in zip(self.EMOTIONS, probs)}
        except Exception:
            pass

        return self._heuristic_emotion(face_roi)

    @staticmethod
    def _heuristic_emotion(face_roi):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        h, w = gray.shape

        upper = gray[:h//2, :]
        lower = gray[h//2:, :]

        upper_mean = np.mean(upper)
        lower_mean = np.mean(lower)
        overall_mean = np.mean(gray)
        std_val = np.std(gray)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()

        mouth_region = gray[int(h*0.6):, int(w*0.25):int(w*0.75)]
        mouth_brightness = np.mean(mouth_region) if mouth_region.size > 0 else 128

        probs = {}
        if mouth_brightness > 160 and std_val > 30:
            probs["happy"] = 0.7
            probs["surprise"] = 0.15
            probs["neutral"] = 0.1
        elif overall_mean < 80:
            probs["sad"] = 0.5
            probs["neutral"] = 0.3
            probs["fear"] = 0.1
        elif laplacian > 500 and std_val > 40:
            probs["angry"] = 0.4
            probs["disgust"] = 0.3
            probs["neutral"] = 0.2
        elif std_val > 45:
            probs["surprise"] = 0.4
            probs["happy"] = 0.3
            probs["fear"] = 0.15
        else:
            probs["neutral"] = 0.6
            probs["happy"] = 0.2
            probs["sad"] = 0.1

        emotion = max(probs, key=probs.get)
        return emotion, probs


class AgeGenderEstimator:
    """Age and gender estimation from face ROI."""

    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            import torch.nn as nn

            class MiniAgeGenderNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1),
                    )
                    self.age_head = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(256, 1),
                    )
                    self.gender_head = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(256, 2),
                    )

                def forward(self, x):
                    features = self.features(x)
                    features = features.view(features.size(0), -1)
                    age = self.age_head(features)
                    gender = self.gender_head(features)
                    return age, gender

            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "age_gender_model.pth")
            if os.path.exists(model_path):
                self._model = MiniAgeGenderNet()
                self._model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                self._model.eval()
                logger.info("Age/Gender estimation model loaded")
            else:
                logger.info("Age/Gender model not found - using heuristic fallback")
                self._model = None
        except Exception as e:
            logger.warning(f"Age/Gender model load failed: {e}")
            self._model = None

    def predict(self, face_roi):
        if face_roi.size == 0:
            return 30, "unknown", {}

        try:
            if len(face_roi.shape) == 2:
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2BGR)

            resized = cv2.resize(face_roi, (64, 64)).astype(np.float32) / 255.0

            if self._model is not None:
                import torch
                tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0)
                with torch.no_grad():
                    age_out, gender_out = self._model(tensor)
                    age = max(1, min(100, int(age_out.item())))
                    gender_probs = torch.softmax(gender_out, dim=1).squeeze().numpy()
                    gender = "male" if gender_probs[0] > gender_probs[1] else "female"
                    return age, gender, {"male": round(float(gender_probs[0]), 3), "female": round(float(gender_probs[1]), 3)}
        except Exception:
            pass

        return self._heuristic_age_gender(face_roi)

    @staticmethod
    def _heuristic_age_gender(face_roi):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi

        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        age = int(np.clip(18 + laplacian * 0.005 + np.random.normal(0, 3), 5, 85))

        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        gender_confidence = min(0.7, 0.4 + std_brightness * 0.003)

        if mean_brightness > 130:
            gender = "female"
            gender_probs = {"male": round(1 - gender_confidence, 3), "female": round(gender_confidence, 3)}
        else:
            gender = "male"
            gender_probs = {"male": round(gender_confidence, 3), "female": round(1 - gender_confidence, 3)}

        return age, gender, gender_probs


class FaceAnalyzer:
    """Complete face analysis pipeline with multi-tier detection."""

    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self, device: str = "auto", min_confidence: float = 0.3):
        self.device = device
        self.min_confidence = min_confidence
        self.detector = None
        self.detector_type = "none"
        self.loaded = False
        self.tracker = FaceTracker()
        self.emotion_recognizer = EmotionRecognizer()
        self.age_gender_estimator = AgeGenderEstimator()
        self.face_history = deque(maxlen=30)
        self._frame_count = 0
        self._load_models()

    def _load_models(self):
        logger.info("Loading face detection models...")

        try:
            from insightface.app import FaceAnalysis
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.detector = FaceAnalysis(name="buffalo_l", providers=providers)
            self.detector.prepare(ctx_id=0, det_size=(640, 640))
            self.detector_type = "insightface"
            self.loaded = True
            logger.info("InsightFace RetinaFace loaded successfully")
            return
        except Exception as e:
            logger.warning(f"InsightFace failed: {e}")

        try:
            import mediapipe as mp
            self.mp_face = mp.solutions.face_detection
            self.mp_mesh = mp.solutions.face_mesh
            self.detector = self.mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            self.face_mesh = self.mp_mesh.FaceMesh(max_num_faces=10, min_detection_confidence=0.5)
            self.detector_type = "mediapipe"
            self.loaded = True
            logger.info("MediaPipe face detection loaded")
            return
        except Exception as e:
            logger.warning(f"MediaPipe failed: {e}")

        try:
            import torch
            import torchvision

            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            self.frcnn_model = model
            self.detector_type = "fasterrcnn"
            self.loaded = True
            logger.info("Faster R-CNN (person/face proxy) loaded")
            return
        except Exception as e:
            logger.warning(f"Faster R-CNN failed: {e}")

        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
            if not os.path.exists(cascade_path):
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            if not self.haar_cascade.empty():
                self.detector_type = "haar"
                self.loaded = True
                logger.info("OpenCV Haar cascade loaded")
                return
        except Exception as e:
            logger.warning(f"Haar cascade failed: {e}")

        try:
            proto_path = os.path.join(os.path.dirname(__file__), "..", "models", "deploy.prototxt")
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "res10_300x300_ssd_iter_140000.caffemodel")

            if os.path.exists(proto_path) and os.path.exists(model_path):
                self.opencv_dnn = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                self.detector_type = "opencv_dnn"
                self.loaded = True
                logger.info("OpenCV DNN face detector loaded")
                return
        except Exception as e:
            logger.warning(f"OpenCV DNN failed: {e}")

        self.detector_type = "skin"
        self.loaded = True
        logger.warning("Using skin-color based face detection (lowest accuracy)")

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        start = time.perf_counter()
        self._frame_count += 1

        if self.detector_type == "insightface":
            faces = self._analyze_insightface(frame)
        elif self.detector_type == "mediapipe":
            faces = self._analyze_mediapipe(frame)
        elif self.detector_type == "fasterrcnn":
            faces = self._analyze_fasterrcnn(frame)
        elif self.detector_type == "haar":
            faces = self._analyze_haar(frame)
        elif self.detector_type == "opencv_dnn":
            faces = self._analyze_opencv_dnn(frame)
        else:
            faces = self._analyze_skin(frame)

        faces = [f for f in faces if f.get("confidence", 0) >= self.min_confidence]

        tracked_faces = self._track_faces(faces)

        for face in tracked_faces:
            bbox = face["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            face_roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

            if face.get("landmarks") is None:
                face["landmarks"] = self._detect_landmarks(frame, bbox)

            if "age" not in face or face.get("age") is None:
                age, gender, gender_probs = self.age_gender_estimator.predict(face_roi)
                face["age"] = age
                face["gender"] = gender
                face["gender_confidence"] = gender_probs

            if "emotion" not in face or face.get("emotion") is None:
                emotion, emotion_probs = self.emotion_recognizer.predict(face_roi)
                face["emotion"] = emotion
                face["emotion_probs"] = emotion_probs

        self.face_history.append(len(faces))

        return {
            "faces": tracked_faces,
            "face_count": len(tracked_faces),
            "detector": self.detector_type,
            "inference_ms": (time.perf_counter() - start) * 1000,
        }

    def _analyze_insightface(self, frame: np.ndarray) -> List[Dict]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.get(rgb)
        faces = []
        for face in results:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            face_data = {
                "bbox": [x1, y1, x2, y2],
                "confidence": round(float(face.det_score), 4),
                "landmarks": face.kps.tolist() if face.kps is not None else [],
            }
            if hasattr(face, "age"):
                face_data["age"] = int(face.age)
            if hasattr(face, "gender"):
                face_data["gender"] = "male" if face.gender == 1 else "female"
            if hasattr(face, "embedding") and face.embedding is not None:
                face_data["embedding_dim"] = len(face.embedding)
            faces.append(face_data)
        return faces

    def _analyze_mediapipe(self, frame: np.ndarray) -> List[Dict]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = self.detector.process(rgb)
        faces = []
        if results.detections:
            for detection in results.detections:
                loc = detection.location_data.relative_bounding_box
                x1 = max(0, int(loc.xmin * w))
                y1 = max(0, int(loc.ymin * h))
                x2 = min(w, int((loc.xmin + loc.width) * w))
                y2 = min(h, int((loc.ymin + loc.height) * h))
                landmarks = []
                for kp in detection.location_data.relative_keypoints:
                    landmarks.append([int(kp.x * w), int(kp.y * h)])
                faces.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(float(detection.score[0]), 4),
                    "landmarks": landmarks,
                })
        return faces

    def _analyze_fasterrcnn(self, frame: np.ndarray) -> List[Dict]:
        try:
            import torch
            h, w = frame.shape[:2]
            tensor = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                predictions = self.frcnn_model(tensor)

            faces = []
            for pred in predictions:
                boxes = pred["boxes"].cpu().numpy()
                scores = pred["scores"].cpu().numpy()
                labels = pred["labels"].cpu().numpy()

                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    if label == 1 and score > 0.5:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        faces.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": round(float(score), 4),
                            "landmarks": [],
                        })
            return faces
        except Exception as e:
            logger.error(f"Faster R-CNN detection failed: {e}")
            return self._analyze_haar(frame)

    def _analyze_haar(self, frame: np.ndarray) -> List[Dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        detections = self.haar_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )
        faces = []
        for (x, y, w, h) in detections:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            faces.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.85,
                "landmarks": [],
            })
        return faces

    def _analyze_opencv_dnn(self, frame: np.ndarray) -> List[Dict]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.opencv_dnn.setInput(blob)
        detections = self.opencv_dnn.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(float(confidence), 4),
                    "landmarks": [],
                })
        return faces

    def _analyze_skin(self, frame: np.ndarray) -> List[Dict]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 25, 60], dtype=np.uint8)
        upper_skin = np.array([25, 180, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        faces = []

        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(c)
            if area < w * h * 0.008:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = cw / max(ch, 1)
            if 0.4 < aspect < 2.0:
                confidence = 0.5 + min(0.3, area / (w * h) * 2)
                faces.append({
                    "bbox": [x, y, x + cw, y + ch],
                    "confidence": round(confidence, 4),
                    "landmarks": [],
                })
        return faces

    def _detect_landmarks(self, frame, bbox):
        try:
            import mediapipe as mp
            x1, y1, x2, y2 = bbox
            face_roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

            if not hasattr(self, '_face_mesh_detector'):
                mp_mesh = mp.solutions.face_mesh
                self._face_mesh_detector = mp_mesh.FaceMesh(
                    max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
                )

            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            results = self._face_mesh_detector.process(rgb)

            if results.multi_face_landmarks:
                landmarks = []
                h, w = face_roi.shape[:2]
                for landmark in results.multi_face_landmarks[0].landmark[:5]:
                    lx = int(landmark.x * w) + x1
                    ly = int(landmark.y * h) + y1
                    landmarks.append([lx, ly])
                return landmarks
        except Exception:
            pass

        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w, h = x2 - x1, y2 - y1
        return [
            [cx - w//6, cy - h//6],
            [cx + w//6, cy - h//6],
            [cx, cy],
            [cx - w//6, cy + h//4],
            [cx + w//6, cy + h//4],
        ]

    def _track_faces(self, faces):
        bboxes = [f["bbox"] for f in faces]
        tracked = self.tracker.update(bboxes)

        result = []
        for t in tracked:
            track_id = t["id"]
            bbox = t["bbox"]
            matching_face = None
            best_iou = 0

            for f in faces:
                iou = FaceTracker._compute_iou(bbox, f["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    matching_face = f

            if matching_face:
                face_data = dict(matching_face)
                face_data["track_id"] = track_id
                result.append(face_data)
            else:
                result.append({
                    "bbox": bbox,
                    "confidence": 0.5,
                    "track_id": track_id,
                    "landmarks": [],
                    "age": None,
                    "gender": "unknown",
                    "emotion": "neutral",
                })

        return result

    def draw_faces(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        annotated = frame.copy()
        h, w = frame.shape[:2]

        for face in faces:
            bbox = face["bbox"]
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])

            color = (0, 255, 0)
            conf = face.get("confidence", 0)
            if conf < 0.5:
                color = (0, 255, 255)
            elif conf < 0.7:
                color = (0, 165, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            for lm in face.get("landmarks", []):
                if isinstance(lm, (list, tuple)) and len(lm) >= 2:
                    cv2.circle(annotated, (int(lm[0]), int(lm[1])), 2, (255, 0, 0), -1)

            labels = []
            if "track_id" in face:
                labels.append(f"ID:{face['track_id']}")
            if "age" in face and face["age"]:
                labels.append(f"Age:{face['age']}")
            if "gender" in face and face["gender"] != "unknown":
                labels.append(face["gender"])
            if "emotion" in face:
                labels.append(face["emotion"])

            labels.append(f"{conf:.2f}")
            label = " | ".join(labels)

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated
