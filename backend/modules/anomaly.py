"""
OMNIVIS — Anomaly Detection Module
One-Class SVM + Autoencoder + Isolation Forest ensemble anomaly detection.
"""
import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional
from collections import deque

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Ensemble anomaly detection system."""

    ALERT_LEVELS = {
        0: "green",    # Normal
        1: "yellow",   # Suspicious
        2: "red",      # Anomaly confirmed
    }

    def __init__(self, feature_window: int = 100, threshold_svm: float = -0.5,
                 threshold_ae: float = 0.3, threshold_if: float = -0.3):
        self.feature_window = feature_window
        self.threshold_svm = threshold_svm
        self.threshold_ae = threshold_ae
        self.threshold_if = threshold_if

        # Feature history
        self.feature_history = deque(maxlen=feature_window)
        self.is_fitted = False
        self.frame_count = 0

        # Models
        self.svm = None
        self.iso_forest = None
        self.ae_model = None

        self._init_models()

    def _init_models(self):
        """Initialize anomaly detection models."""
        try:
            from sklearn.svm import OneClassSVM
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler

            self.svm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
            self.iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
            self.scaler = StandardScaler()
            logger.info("Scikit-learn anomaly detectors initialized")
        except ImportError:
            logger.warning("scikit-learn not available — using simple threshold-based anomaly detection")
            self.svm = None
            self.iso_forest = None

    def detect(self, detections: List[Dict], tracks: List[Dict],
               flow_magnitude: float = 0.0, frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run anomaly detection on current frame data."""
        start = time.perf_counter()
        self.frame_count += 1

        # Extract features
        features = self._extract_features(detections, tracks, flow_magnitude, frame)
        self.feature_history.append(features)

        anomalies = []
        overall_score = 0.0
        alert_level = "green"

        if len(self.feature_history) >= 30:
            # Fit models if not yet fitted
            if not self.is_fitted and len(self.feature_history) >= self.feature_window:
                self._fit_models()

            if self.is_fitted:
                scores = self._score_anomaly(features)
                overall_score = float(np.mean(list(scores.values())))

                # Determine alert level
                anomaly_votes = sum(1 for s in scores.values() if s > 0.5)

                if anomaly_votes >= 2:
                    alert_level = "red"
                elif anomaly_votes >= 1:
                    alert_level = "yellow"
                else:
                    alert_level = "green"

                if alert_level != "green":
                    anomaly_desc = self._describe_anomaly(features, scores)
                    anomalies.append({
                        "type": anomaly_desc["type"],
                        "severity": alert_level,
                        "score": round(overall_score, 4),
                        "description": anomaly_desc["description"],
                        "scores": {k: round(v, 4) for k, v in scores.items()},
                    })
            else:
                # Simple threshold-based detection during warm-up
                simple_result = self._simple_anomaly_check(features)
                if simple_result:
                    anomalies.append(simple_result)
                    alert_level = simple_result["severity"]

        return {
            "anomalies": anomalies,
            "alert_level": alert_level,
            "overall_score": round(overall_score, 4),
            "features": {k: round(float(v), 4) for k, v in features.items()},
            "is_fitted": self.is_fitted,
            "samples_collected": len(self.feature_history),
            "inference_ms": (time.perf_counter() - start) * 1000,
        }

    def _extract_features(self, detections: List[Dict], tracks: List[Dict],
                          flow_magnitude: float, frame: Optional[np.ndarray]) -> Dict[str, float]:
        """Extract anomaly-relevant features from current frame data."""
        features = {
            "detection_count": float(len(detections)),
            "avg_confidence": float(np.mean([d.get("confidence", 0) for d in detections])) if detections else 0.0,
            "avg_bbox_area": 0.0,
            "flow_magnitude": flow_magnitude,
            "track_count": float(len(tracks)),
            "avg_velocity": 0.0,
            "max_velocity": 0.0,
            "acceleration": 0.0,
        }

        # Bounding box area statistics
        if detections:
            areas = []
            for d in detections:
                bbox = d.get("bbox", {})
                area = abs((bbox.get("x2", 0) - bbox.get("x1", 0)) *
                          (bbox.get("y2", 0) - bbox.get("y1", 0)))
                areas.append(area)
            features["avg_bbox_area"] = float(np.mean(areas))
            features["std_bbox_area"] = float(np.std(areas))

        # Track velocity statistics
        if tracks and len(self.feature_history) > 0:
            velocities = []
            for t in tracks:
                center = t.get("center", (0, 0))
                prev_features = self.feature_history[-1] if self.feature_history else None
                if prev_features:
                    vel = np.sqrt(center[0] ** 2 + center[1] ** 2) * 0.01
                    velocities.append(vel)

            if velocities:
                features["avg_velocity"] = float(np.mean(velocities))
                features["max_velocity"] = float(np.max(velocities))

        # Frame brightness anomaly
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            features["brightness_mean"] = float(np.mean(gray))
            features["brightness_std"] = float(np.std(gray))

        return features

    def _fit_models(self):
        """Fit anomaly detection models on feature history."""
        feature_matrix = self._features_to_matrix(list(self.feature_history))

        if self.svm is not None and self.iso_forest is not None:
            try:
                scaled = self.scaler.fit_transform(feature_matrix)
                self.svm.fit(scaled)
                self.iso_forest.fit(scaled)
                self.is_fitted = True
                logger.info(f"Anomaly models fitted on {len(feature_matrix)} samples")
            except Exception as e:
                logger.error(f"Model fitting failed: {e}")
        else:
            self.is_fitted = True

    def _score_anomaly(self, features: Dict[str, float]) -> Dict[str, float]:
        """Score anomaly using all methods."""
        scores = {}
        feature_vec = self._features_to_matrix([features])

        if self.svm is not None:
            try:
                scaled = self.scaler.transform(feature_vec)
                svm_score = -self.svm.decision_function(scaled)[0]  # More negative = more anomalous
                scores["svm"] = float(max(0, min(1, svm_score)))
            except Exception:
                scores["svm"] = 0.0

        if self.iso_forest is not None:
            try:
                scaled = self.scaler.transform(feature_vec)
                if_score = -self.iso_forest.score_samples(scaled)[0]
                scores["isolation_forest"] = float(max(0, min(1, if_score)))
            except Exception:
                scores["isolation_forest"] = 0.0

        # Statistical anomaly (Z-score based)
        if len(self.feature_history) > 10:
            feature_matrix = self._features_to_matrix(list(self.feature_history))
            means = np.mean(feature_matrix, axis=0)
            stds = np.std(feature_matrix, axis=0) + 1e-6
            z_scores = np.abs((feature_vec[0] - means) / stds)
            max_z = float(np.max(z_scores))
            scores["statistical"] = float(max(0, min(1, max_z / 3.0)))

        return scores

    def _simple_anomaly_check(self, features: Dict[str, float]) -> Optional[Dict]:
        """Simple threshold-based anomaly check for warm-up period."""
        if features.get("flow_magnitude", 0) > 20:
            return {
                "type": "high_motion",
                "severity": "yellow",
                "score": 0.6,
                "description": "Unusually high motion detected",
            }
        if features.get("detection_count", 0) > 20:
            return {
                "type": "crowd",
                "severity": "yellow",
                "score": 0.5,
                "description": "High object density detected",
            }
        return None

    def _describe_anomaly(self, features: Dict[str, float], scores: Dict[str, float]) -> Dict:
        """Generate human-readable anomaly description."""
        max_score_method = max(scores, key=scores.get)
        desc_parts = []

        if features.get("flow_magnitude", 0) > 15:
            desc_parts.append("abnormal motion intensity")
        if features.get("detection_count", 0) > 15:
            desc_parts.append("unusual object count")
        if features.get("max_velocity", 0) > 50:
            desc_parts.append("high-speed movement detected")

        description = "; ".join(desc_parts) if desc_parts else f"Anomaly detected by {max_score_method}"

        return {
            "type": "ensemble_anomaly",
            "description": description,
        }

    @staticmethod
    def _features_to_matrix(features_list: List[Dict[str, float]]) -> np.ndarray:
        """Convert list of feature dicts to numpy matrix."""
        keys = sorted(features_list[0].keys())
        return np.array([[f.get(k, 0) for k in keys] for f in features_list])

    def reset(self):
        """Reset detector state."""
        self.feature_history.clear()
        self.is_fitted = False
        self.frame_count = 0
        self._init_models()
