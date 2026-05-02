"""
OMNIVIS — Advanced Anomaly Detection Module
Multi-layer detection: traffic violations, crowd anomalies, behavioral patterns,
statistical outliers, and rule-based violation detection for real-world scenarios.
"""
import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ViolationRecord:
    type: str
    severity: str
    score: float
    description: str
    confidence: float
    affected_objects: List[int] = field(default_factory=list)


class AnomalyDetector:
    """Multi-layer anomaly detection for real-world traffic and scene analysis."""

    ALERT_LEVELS = {0: "green", 1: "yellow", 2: "red"}

    TRAFFIC_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "traffic light", "stop sign"}
    PEDESTRIAN_CLASSES = {"person"}
    DANGEROUS_COMBOS = {
        ("person", "car"), ("person", "truck"), ("person", "bus"),
        ("person", "motorcycle"), ("bicycle", "car"), ("bicycle", "truck"),
    }

    def __init__(self, feature_window: int = 60):
        self.feature_window = feature_window
        self.feature_history = deque(maxlen=feature_window)
        self.violation_history = deque(maxlen=120)
        self.object_trajectory = {}
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps_estimate = 8.0

        self._init_models()

    def _init_models(self):
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import OneClassSVM
            self.iso_forest = IsolationForest(n_estimators=150, contamination=0.08, random_state=42, max_samples=0.8)
            self.scaler = StandardScaler()
            self.svm = OneClassSVM(kernel="rbf", nu=0.08, gamma="scale")
            self.is_fitted = False
            logger.info("Anomaly detection models initialized")
        except ImportError:
            self.iso_forest = None
            self.scaler = None
            self.svm = None
            self.is_fitted = True
            logger.warning("scikit-learn not available — using rule-based detection only")

    def detect(self, detections: List[Dict], tracks: List[Dict],
               flow_magnitude: float = 0.0, frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        start = time.perf_counter()
        self.frame_count += 1

        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            self.fps_estimate = 0.9 * self.fps_estimate + 0.1 * (1.0 / dt)

        violations = []

        violations.extend(self._detect_traffic_violations(detections, tracks, frame))
        violations.extend(self._detect_crowd_anomalies(detections, tracks, frame))
        violations.extend(self._detect_behavioral_anomalies(detections, tracks))
        violations.extend(self._detect_spatial_anomalies(detections, frame))

        if self.frame_count > 10:
            violations.extend(self._detect_temporal_anomalies(detections, tracks, flow_magnitude))

        if len(self.feature_history) >= 15:
            features = self._extract_features(detections, tracks, flow_magnitude, frame, violations)
            self.feature_history.append(features)

            ml_violations = self._ml_anomaly_check(features, violations)
            violations.extend(ml_violations)

        alert_level, overall_score = self._compute_alert_level(violations)

        anomaly_list = []
        for v in violations:
            anomaly_list.append({
                "type": v.type,
                "severity": v.severity,
                "score": round(v.score, 4),
                "description": v.description,
                "confidence": round(v.confidence, 3),
            })

        return {
            "anomalies": anomaly_list,
            "alert_level": alert_level,
            "overall_score": round(overall_score, 4),
            "violation_count": len(violations),
            "features": self._extract_features(detections, tracks, flow_magnitude, frame, violations),
            "is_fitted": self.is_fitted,
            "samples_collected": len(self.feature_history),
            "inference_ms": (time.perf_counter() - start) * 1000,
        }

    def _detect_traffic_violations(self, detections: List[Dict], tracks: List[Dict],
                                    frame: Optional[np.ndarray]) -> List[ViolationRecord]:
        violations = []
        if not frame is None:
            h, w = frame.shape[:2]
        else:
            h, w = 480, 640

        vehicles = [d for d in detections if d.get("class_name", "") in self.TRAFFIC_VEHICLE_CLASSES]
        pedestrians = [d for d in detections if d.get("class_name", "") in self.PEDESTRIAN_CLASSES]

        for ped in pedestrians:
            ped_bbox = ped.get("bbox", {})
            ped_center = ((ped_bbox.get("x1", 0) + ped_bbox.get("x2", 0)) / 2,
                         (ped_bbox.get("y1", 0) + ped_bbox.get("y2", 0)) / 2)
            ped_area = abs((ped_bbox.get("x2", 0) - ped_bbox.get("x1", 0)) *
                          (ped_bbox.get("y2", 0) - ped_bbox.get("y1", 0)))

            for veh in vehicles:
                veh_bbox = veh.get("bbox", {})
                veh_center = ((veh_bbox.get("x1", 0) + veh_bbox.get("x2", 0)) / 2,
                             (veh_bbox.get("y1", 0) + veh_bbox.get("y2", 0)) / 2)

                dist = np.sqrt((ped_center[0] - veh_center[0])**2 + (ped_center[1] - veh_center[1])**2)

                iou = self._compute_iou(ped_bbox, veh_bbox)
                overlap_ratio = iou / (ped_area + 1e-6) if ped_area > 0 else 0

                if iou > 0.15:
                    severity = "red" if iou > 0.3 else "yellow"
                    score = min(1.0, iou * 2.0)
                    violations.append(ViolationRecord(
                        type="pedestrian_vehicle_collision_risk",
                        severity=severity,
                        score=score,
                        description=f"Pedestrian too close to {veh['class_name']} (overlap: {iou:.0%})",
                        confidence=min(0.95, 0.5 + iou),
                    ))
                elif dist < 50 and ped_bbox.get("y2", 0) > h * 0.5:
                    violations.append(ViolationRecord(
                        type="pedestrian_in_traffic_lane",
                        severity="yellow",
                        score=0.55,
                        description=f"Pedestrian detected in active traffic lane near {veh['class_name']}",
                        confidence=0.7,
                    ))

        lane_road_area = h * w * 0.4
        vehicle_area = sum(abs(d.get("bbox", {}).get("x2", 0) - d.get("bbox", {}).get("x1", 0)) *
                          abs(d.get("bbox", {}).get("y2", 0) - d.get("bbox", {}).get("y1", 0))
                          for d in vehicles)
        density_ratio = vehicle_area / (lane_road_area + 1e-6)

        if density_ratio > 0.6:
            violations.append(ViolationRecord(
                type="traffic_congestion",
                severity="yellow",
                score=min(1.0, density_ratio),
                description=f"High traffic density detected ({density_ratio:.0%} road coverage)",
                confidence=0.8,
            ))

        fast_objects = []
        for track in tracks:
            center = track.get("center", (0, 0))
            prev_center = track.get("prev_center", center)
            velocity = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
            if velocity > 30:
                fast_objects.append((track, velocity))

        for track, vel in fast_objects:
            violations.append(ViolationRecord(
                type="speeding_vehicle",
                severity="yellow" if vel < 60 else "red",
                score=min(1.0, vel / 80.0),
                description=f"Vehicle #{track.get('track_id', '?')} moving at high speed (velocity: {vel:.1f}px/frame)",
                confidence=min(0.9, 0.4 + vel / 150.0),
            ))

        upper_road_vehicles = [v for v in vehicles
                              if v.get("bbox", {}).get("y1", 0) < h * 0.25
                              and v.get("bbox", {}).get("y2", 0) > h * 0.1]
        if len(upper_road_vehicles) > 2:
            violations.append(ViolationRecord(
                type="wrong_lane_usage",
                severity="yellow",
                score=0.5,
                description=f"{len(upper_road_vehicles)} vehicles in restricted/opposite lane area",
                confidence=0.65,
            ))

        return violations

    def _detect_crowd_anomalies(self, detections: List[Dict], tracks: List[Dict],
                                 frame: Optional[np.ndarray]) -> List[ViolationRecord]:
        violations = []
        if frame is None:
            return violations

        h, w = frame.shape[:2]

        persons = [d for d in detections if d.get("class_name") == "person"]
        if len(persons) < 5:
            return violations

        person_centers = []
        for p in persons:
            bbox = p.get("bbox", {})
            cx = (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2
            cy = (bbox.get("y1", 0) + bbox.get("y2", 0)) / 2
            person_centers.append((cx, cy))

        person_centers = np.array(person_centers)

        close_pairs = 0
        total_pairs = 0
        for i in range(len(person_centers)):
            for j in range(i + 1, len(person_centers)):
                dist = np.linalg.norm(person_centers[i] - person_centers[j])
                total_pairs += 1
                if dist < 40:
                    close_pairs += 1

        if total_pairs > 0:
            crowding_ratio = close_pairs / total_pairs
            if crowding_ratio > 0.3:
                violations.append(ViolationRecord(
                    type="dangerous_crowding",
                    severity="red" if crowding_ratio > 0.5 else "yellow",
                    score=min(1.0, crowding_ratio * 1.5),
                    description=f"Dangerous crowd density ({crowding_ratio:.0%} objects in close proximity)",
                    confidence=min(0.9, 0.5 + crowding_ratio),
                ))

        frame_area = h * w
        person_area = sum(abs(p.get("bbox", {}).get("x2", 0) - p.get("bbox", {}).get("x1", 0)) *
                         abs(p.get("bbox", {}).get("y2", 0) - p.get("bbox", {}).get("y1", 0))
                         for p in persons)
        area_density = person_area / (frame_area + 1e-6)

        if area_density > 0.3 and len(persons) > 10:
            violations.append(ViolationRecord(
                type="overcrowding",
                severity="yellow",
                score=min(1.0, area_density),
                description=f"Area overcrowding: {len(persons)} people occupying {area_density:.0%} of scene",
                confidence=0.75,
            ))

        return violations

    def _detect_behavioral_anomalies(self, detections: List[Dict], tracks: List[Dict]) -> List[ViolationRecord]:
        violations = []

        erratic_tracks = []
        for track in tracks:
            history = track.get("history", [])
            if len(history) >= 4:
                directions = []
                for i in range(1, len(history)):
                    dx = history[i][0] - history[i-1][0]
                    dy = history[i][1] - history[i-1][1]
                    angle = np.arctan2(dy, dx)
                    directions.append(angle)

                if len(directions) >= 3:
                    direction_changes = 0
                    for i in range(1, len(directions)):
                        angle_diff = abs(directions[i] - directions[i-1])
                        if angle_diff > np.pi:
                            angle_diff = 2 * np.pi - angle_diff
                        if angle_diff > np.pi / 3:
                            direction_changes += 1

                    if direction_changes >= 2:
                        erratic_tracks.append(track)

        for track in erratic_tracks:
            violations.append(ViolationRecord(
                type="erratic_movement",
                severity="yellow",
                score=0.6,
                description=f"Vehicle/person #{track.get('track_id', '?')} showing erratic movement pattern",
                confidence=0.7,
            ))

        stationary_near_traffic = []
        for track in tracks:
            history = track.get("history", [])
            if len(history) >= 5:
                recent = history[-5:]
                movement = np.std([np.sqrt((recent[i][0]-recent[i-1][0])**2 + (recent[i][1]-recent[i-1])**2)
                                  for i in range(1, len(recent))])
                if movement < 2:
                    center = track.get("center", (0, 0))
                    if center[1] > 0:
                        y_ratio = center[1] / 480
                        if 0.3 < y_ratio < 0.8:
                            stationary_near_traffic.append(track)

        return violations

    def _detect_spatial_anomalies(self, detections: List[Dict], frame: Optional[np.ndarray]) -> List[ViolationRecord]:
        violations = []
        if frame is None:
            return violations

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        horizontal_lines = self._detect_hough_lines(edges, h, w, theta_range=(0.85*np.pi/2, 1.15*np.pi/2))
        vertical_lines = self._detect_hough_lines(edges, h, w, theta_range=(0, 0.15*np.pi/2))

        for det in detections:
            bbox = det.get("bbox", {})
            x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
            det_center_y = (y1 + y2) / 2

            if det.get("class_name") in self.TRAFFIC_VEHICLE_CLASSES:
                for ly, theta in horizontal_lines:
                    if abs(det_center_y - ly) < 15 and 0.2*w < x1 < 0.8*w:
                        violations.append(ViolationRecord(
                            type="lane_violation",
                            severity="yellow",
                            score=0.45,
                            description=f"{det['class_name']} crossing lane boundary",
                            confidence=0.6,
                        ))
                        break

        return violations

    def _detect_hough_lines(self, edges, h, w, theta_range=(0, np.pi)):
        lines = cv2.HoughLines(edges, 1, np.pi/180, 80)
        if lines is None:
            return []

        filtered = []
        for line in lines:
            rho, theta = line[0]
            if theta_range[0] <= theta <= theta_range[1]:
                y = int(rho / np.sin(theta)) if np.sin(theta) != 0 else 0
                if 0 < y < h:
                    filtered.append((y, theta))
        return filtered[:10]

    def _detect_temporal_anomalies(self, detections: List[Dict], tracks: List[Dict],
                                    flow_magnitude: float) -> List[ViolationRecord]:
        violations = []

        if len(self.feature_history) < 10:
            return violations

        recent_detections = list(self.feature_history)[-10:]
        avg_det_count = np.mean([f.get("detection_count", 0) for f in recent_detections])
        std_det_count = np.std([f.get("detection_count", 0) for f in recent_detections]) + 1e-6

        current_count = len(detections)
        z_score = abs(current_count - avg_det_count) / std_det_count

        if z_score > 2.5 and current_count > avg_det_count * 1.5:
            violations.append(ViolationRecord(
                type="sudden_object_surge",
                severity="yellow",
                score=min(1.0, z_score / 4.0),
                description=f"Sudden increase in objects: {current_count} vs avg {avg_det_count:.0f} (z={z_score:.1f})",
                confidence=min(0.85, 0.5 + z_score / 6.0),
            ))

        recent_flow = [f.get("flow_magnitude", 0) for f in recent_detections if "flow_magnitude" in f]
        if recent_flow:
            avg_flow = np.mean(recent_flow)
            if flow_magnitude > avg_flow * 2.5 and flow_magnitude > 10:
                violations.append(ViolationRecord(
                    type="sudden_motion_spike",
                    severity="yellow",
                    score=min(1.0, flow_magnitude / (avg_flow * 4)),
                    description=f"Motion spike detected: {flow_magnitude:.1f} vs avg {avg_flow:.1f}",
                    confidence=0.75,
                ))

        return violations

    def _extract_features(self, detections: List[Dict], tracks: List[Dict],
                          flow_magnitude: float, frame: Optional[np.ndarray],
                          violations: List[ViolationRecord]) -> Dict[str, float]:
        features = {
            "detection_count": float(len(detections)),
            "vehicle_count": float(len([d for d in detections if d.get("class_name", "") in self.TRAFFIC_VEHICLE_CLASSES])),
            "pedestrian_count": float(len([d for d in detections if d.get("class_name") == "person"])),
            "avg_confidence": float(np.mean([d.get("confidence", 0) for d in detections])) if detections else 0.0,
            "flow_magnitude": flow_magnitude,
            "track_count": float(len(tracks)),
            "avg_velocity": 0.0,
            "max_velocity": 0.0,
            "velocity_variance": 0.0,
            "violation_count": float(len(violations)),
            "violation_score": float(sum(v.score for v in violations)),
            "scene_complexity": float(len(detections) * (1 + len(tracks) * 0.1)),
        }

        if detections:
            areas = [abs(d.get("bbox", {}).get("x2", 0) - d.get("bbox", {}).get("x1", 0)) *
                    abs(d.get("bbox", {}).get("y2", 0) - d.get("bbox", {}).get("y1", 0))
                    for d in detections]
            features["avg_bbox_area"] = float(np.mean(areas))
            features["std_bbox_area"] = float(np.std(areas))

        if tracks:
            velocities = []
            for t in tracks:
                center = t.get("center", (0, 0))
                prev_center = t.get("prev_center", center)
                vel = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                velocities.append(vel)
            if velocities:
                features["avg_velocity"] = float(np.mean(velocities))
                features["max_velocity"] = float(np.max(velocities))
                features["velocity_variance"] = float(np.var(velocities))

        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            features["brightness_mean"] = float(np.mean(gray))
            features["brightness_std"] = float(np.std(gray))
            features["edge_density"] = float(np.mean(cv2.Canny(gray, 50, 150) > 0) * 100)

        return features

    def _ml_anomaly_check(self, features: Dict[str, float],
                          current_violations: List[ViolationRecord]) -> List[ViolationRecord]:
        violations = []

        if self.iso_forest is None or not self.is_fitted:
            return violations

        try:
            if not self.is_fitted and len(self.feature_history) >= self.feature_window:
                feature_matrix = self._features_to_matrix(list(self.feature_history))
                scaled = self.scaler.fit_transform(feature_matrix)
                self.iso_forest.fit(scaled)
                self.is_fitted = True
                logger.info(f"ML models fitted on {len(feature_matrix)} samples")

            feature_vec = self._features_to_matrix([features])
            scaled = self.scaler.transform(feature_vec)
            if_score = -self.iso_forest.score_samples(scaled)[0]

            normalized_score = float(np.clip(if_score * 5, 0, 1))

            if normalized_score > 0.5 and len(current_violations) == 0:
                violations.append(ViolationRecord(
                    type="ml_statistical_anomaly",
                    severity="yellow" if normalized_score < 0.7 else "red",
                    score=normalized_score,
                    description=f"Statistical anomaly detected (score: {normalized_score:.2f}) — scene deviates from learned normal pattern",
                    confidence=min(0.8, normalized_score),
                ))
        except Exception as e:
            logger.error(f"ML anomaly check failed: {e}")

        return violations

    def _compute_alert_level(self, violations: List[ViolationRecord]) -> Tuple[str, float]:
        if not violations:
            return "green", 0.0

        red_count = sum(1 for v in violations if v.severity == "red")
        yellow_count = sum(1 for v in violations if v.severity == "yellow")

        max_score = max(v.score for v in violations) if violations else 0
        avg_score = np.mean([v.score for v in violations])

        weighted_score = (red_count * 1.0 + yellow_count * 0.5) / max(1, len(violations)) * max_score

        if red_count >= 1 or weighted_score > 0.6:
            return "red", float(max(max_score, weighted_score))
        elif yellow_count >= 1 or weighted_score > 0.3:
            return "yellow", float(max(max_score, weighted_score))

        return "green", float(avg_score * 0.3)

    def _compute_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        x1 = max(bbox1.get("x1", 0), bbox2.get("x1", 0))
        y1 = max(bbox1.get("y1", 0), bbox2.get("y1", 0))
        x2 = min(bbox1.get("x2", 0), bbox2.get("x2", 0))
        y2 = min(bbox1.get("y2", 0), bbox2.get("y2", 0))

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = abs(bbox1.get("x2", 0) - bbox1.get("x1", 0)) * abs(bbox1.get("y2", 0) - bbox1.get("y1", 0))
        area2 = abs(bbox2.get("x2", 0) - bbox2.get("x1", 0)) * abs(bbox2.get("y2", 0) - bbox2.get("y1", 0))
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def _features_to_matrix(features_list: List[Dict[str, float]]) -> np.ndarray:
        keys = sorted(features_list[0].keys())
        return np.array([[f.get(k, 0) for k in keys] for f in features_list])

    def reset(self):
        self.feature_history.clear()
        self.violation_history.clear()
        self.object_trajectory.clear()
        self.frame_count = 0
        self.is_fitted = False
        self._init_models()
