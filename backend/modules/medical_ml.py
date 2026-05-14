"""
OMNIVIS - Medical Scan Analysis with Deep Learning
Transfer learning models for medical image classification and abnormality detection.
Supports: X-Ray, MRI, CT, Ultrasound classification + pathology detection.
"""
import cv2
import numpy as np
import time
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ScanType(Enum):
    XRAY = "x-ray"
    MRI = "mri"
    CT = "ct_scan"
    ULTRASOUND = "ultrasound"
    UNKNOWN = "unknown"


@dataclass
class MedicalFinding:
    finding_type: str
    severity: str
    risk_level: str
    score: float
    description: str
    recommendation: str
    region: Optional[Dict[str, int]] = None
    confidence: float = 0.0


class MedicalScanClassifier:
    """Deep learning model for medical scan type classification."""

    SCAN_TYPES = ["x-ray", "mri", "ct_scan", "ultrasound"]

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.loaded = False
        self._load_model()

    def _load_model(self):
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            class MedicalClassifier(nn.Module):
                def __init__(self, num_classes=4):
                    super().__init__()
                    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
                    self.features = backbone.features
                    self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Dropout(0.3),
                        nn.Linear(1280, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, num_classes),
                    )

                def forward(self, x):
                    x = self.features(x)
                    x = self.classifier(x)
                    return x

            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "medical_scan_classifier.pth")
            if os.path.exists(model_path):
                self.model = MedicalClassifier(num_classes=4)
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.loaded = True
                logger.info(f"Medical scan classifier loaded on {self.device}")
            else:
                logger.info("Medical scan classifier weights not found - using heuristic classification")
                self.model = None
        except Exception as e:
            logger.warning(f"Medical classifier load failed: {e}")
            self.model = None

    def predict(self, image: np.ndarray) -> Tuple[str, Dict[str, float]]:
        if self.model is None:
            return self._heuristic_classify(image)

        try:
            import torch

            resized = cv2.resize(image, (224, 224))
            tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                idx = int(np.argmax(probs))
                scan_type = self.SCAN_TYPES[idx]
                conf_dict = {st: round(float(p), 4) for st, p in zip(self.SCAN_TYPES, probs)}
                return scan_type, conf_dict
        except Exception as e:
            logger.error(f"Medical classification failed: {e}")
            return self._heuristic_classify(image)

    @staticmethod
    def _heuristic_classify(image: np.ndarray) -> Tuple[str, Dict[str, float]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        aspect_ratio = w / h

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total = np.sum(hist)
        dark_ratio = np.sum(hist[:50]) / total
        bright_ratio = np.sum(hist[200:]) / total
        mid_ratio = np.sum(hist[80:180]) / total

        std_val = np.std(gray)

        scores = {"x-ray": 0.0, "mri": 0.0, "ct_scan": 0.0, "ultrasound": 0.0}

        if aspect_ratio > 1.5:
            scores["x-ray"] += 0.4
            if dark_ratio > 0.3:
                scores["x-ray"] += 0.3
        elif aspect_ratio > 1.2:
            scores["mri"] += 0.3
            if mid_ratio > 0.5:
                scores["mri"] += 0.3
        elif aspect_ratio > 1.0:
            scores["ct_scan"] += 0.3
            if bright_ratio > 0.2:
                scores["ct_scan"] += 0.2

        if aspect_ratio < 1.0:
            scores["ultrasound"] += 0.4
            if dark_ratio > 0.4:
                scores["ultrasound"] += 0.3

        if std_val < 30:
            scores["x-ray"] += 0.2

        total_score = sum(scores.values())
        if total_score > 0:
            probs = {k: round(v / total_score, 4) for k, v in scores.items()}
        else:
            probs = {k: 0.25 for k in scores}

        scan_type = max(scores, key=scores.get)
        return scan_type, probs


class PathologyDetector:
    """Deep learning model for pathology/abnormality detection in medical scans."""

    def __init__(self):
        self.models = {}
        self.device = "cpu"
        self.loaded = False
        self._load_models()

    def _load_models(self):
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            class PathologyNet(nn.Module):
                def __init__(self, num_classes=2):
                    super().__init__()
                    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                    backbone.fc = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(512, 128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, num_classes),
                    )
                    self.backbone = backbone

                def forward(self, x):
                    return self.backbone(x)

            model_dir = os.path.join(os.path.dirname(__file__), "..", "models")

            for scan_type in ["xray", "mri", "ct"]:
                model_path = os.path.join(model_dir, f"pathology_{scan_type}.pth")
                if os.path.exists(model_path):
                    model = PathologyNet(num_classes=2)
                    state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                    model.load_state_dict(state_dict)
                    model.to(self.device)
                    model.eval()
                    self.models[scan_type] = model
                    logger.info(f"Pathology detector for {scan_type} loaded on {self.device}")

            if self.models:
                self.loaded = True
            else:
                logger.info("Pathology detector weights not found - using heuristic detection")
        except Exception as e:
            logger.warning(f"Pathology detector load failed: {e}")

    def detect(self, image: np.ndarray, scan_type: str) -> Tuple[bool, float, Dict[str, float]]:
        model_key = scan_type.replace("-", "").replace("_", "")
        if model_key in self.models:
            return self._predict_with_model(image, model_key)
        return self._heuristic_detect(image, scan_type)

    def _predict_with_model(self, image: np.ndarray, model_key: str) -> Tuple[bool, float, Dict[str, float]]:
        try:
            import torch

            model = self.models[model_key]
            resized = cv2.resize(image, (224, 224))
            tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                abnormal_prob = float(probs[1])
                is_abnormal = abnormal_prob > 0.5
                conf_dict = {"normal": round(float(probs[0]), 4), "abnormal": round(float(probs[1]), 4)}
                return is_abnormal, abnormal_prob, conf_dict
        except Exception as e:
            logger.error(f"Pathology prediction failed: {e}")
            return self._heuristic_detect(image, model_key)

    @staticmethod
    def _heuristic_detect(image: np.ndarray, scan_type: str) -> Tuple[bool, float, Dict[str, float]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = np.std(gray)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_regions = [c for c in contours if cv2.contourArea(c) > h * w * 0.02]

        edges = cv2.Canny(enhanced, 30, 100)
        edge_density = np.sum(edges > 0) / (h * w)

        abnormality_score = 0.0

        if len(large_regions) > 3:
            abnormality_score += 0.3
        if edge_density > 0.15:
            abnormality_score += 0.2
        if contrast < 20:
            abnormality_score += 0.15

        mean_val = np.mean(gray)
        std_val = np.std(gray)
        hyper_ratio = np.sum(enhanced > (mean_val + 2 * std_val)) / (h * w)
        hypo_ratio = np.sum(enhanced < (mean_val - 2 * std_val)) / (h * w)

        if hyper_ratio > 0.15:
            abnormality_score += 0.25
        if hypo_ratio > 0.15:
            abnormality_score += 0.15

        mid = w // 2
        left = gray[:, :mid]
        right = cv2.flip(gray[:, mid:], 1)
        min_w = min(left.shape[1], right.shape[1])
        diff = np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float))
        asymmetry = np.mean(diff) / 255.0

        if asymmetry > 0.15:
            abnormality_score += 0.2

        abnormality_score = min(1.0, abnormality_score)
        is_abnormal = abnormality_score > 0.35

        return is_abnormal, abnormality_score, {
            "normal": round(1.0 - abnormality_score, 4),
            "abnormal": round(abnormality_score, 4),
        }


class MedicalScanAnalyzer:
    """Complete medical scan analysis with deep learning + heuristic fallback."""

    def __init__(self):
        self.classifier = MedicalScanClassifier()
        self.pathology_detector = PathologyDetector()
        self.analysis_history = []

    def analyze(self, image: np.ndarray, scan_type_hint: str = "auto") -> Dict[str, Any]:
        start = time.perf_counter()

        if scan_type_hint == "auto":
            scan_type, scan_probs = self.classifier.predict(image)
        else:
            type_map = {
                "x-ray": "x-ray", "xray": "x-ray",
                "mri": "mri", "MRI": "mri",
                "ct": "ct_scan", "ct_scan": "ct_scan", "ct-scan": "ct_scan",
                "ultrasound": "ultrasound", "usg": "ultrasound",
            }
            scan_type = type_map.get(scan_type_hint.lower(), "unknown")
            scan_probs = {scan_type: 1.0}

        findings = []

        findings.extend(self._analyze_image_quality(image))

        is_abnormal, abnormality_score, pathology_probs = self.pathology_detector.detect(image, scan_type)

        if scan_type == "x-ray":
            findings.extend(self._analyze_xray(image, abnormality_score, pathology_probs))
        elif scan_type == "mri":
            findings.extend(self._analyze_mri(image, abnormality_score, pathology_probs))
        elif scan_type == "ct_scan":
            findings.extend(self._analyze_ct(image, abnormality_score, pathology_probs))
        elif scan_type == "ultrasound":
            findings.extend(self._analyze_ultrasound(image, abnormality_score, pathology_probs))

        findings.extend(self._analyze_general_abnormalities(image, scan_type))
        findings.extend(self._analyze_texture_patterns(image))

        overall_risk, risk_score = self._compute_overall_risk(findings)

        annotated = self._annotate_findings(image.copy(), findings)

        inference_ms = (time.perf_counter() - start) * 1000

        self.analysis_history.append({
            "scan_type": scan_type,
            "risk_level": overall_risk,
            "timestamp": time.time(),
        })

        return {
            "scan_type": scan_type,
            "scan_type_confidence": scan_probs,
            "findings": [
                {
                    "type": f.finding_type,
                    "severity": f.severity,
                    "risk_level": f.risk_level,
                    "score": round(f.score, 4),
                    "description": f.description,
                    "recommendation": f.recommendation,
                    "confidence": round(f.confidence, 3),
                    "region": f.region,
                }
                for f in findings
            ],
            "overall_risk": overall_risk,
            "risk_score": round(risk_score, 4),
            "abnormality_probability": round(abnormality_score, 4),
            "pathology_confidence": pathology_probs,
            "finding_count": len(findings),
            "high_risk_count": sum(1 for f in findings if f.risk_level in ("high", "critical")),
            "annotated_image": self._encode_image(annotated),
            "inference_ms": round(inference_ms, 1),
            "summary": self._generate_summary(findings, overall_risk),
            "model_used": "deep_learning" if self.pathology_detector.loaded else "heuristic",
        }

    def _analyze_image_quality(self, image: np.ndarray) -> List[MedicalFinding]:
        findings = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 50:
            findings.append(MedicalFinding(
                finding_type="poor_image_quality",
                severity="medium",
                risk_level="medium",
                score=0.4,
                description=f"Image appears blurred (sharpness: {blur_score:.1f}). Critical details may be obscured.",
                recommendation="Re-scan with better focus and positioning.",
                confidence=0.85,
            ))

        contrast = np.std(gray)
        if contrast < 20:
            findings.append(MedicalFinding(
                finding_type="low_contrast",
                severity="medium",
                risk_level="medium",
                score=0.35,
                description=f"Low contrast detected (std: {contrast:.1f}). Tissue differentiation may be compromised.",
                recommendation="Adjust imaging parameters or apply contrast enhancement.",
                confidence=0.8,
            ))

        brightness = np.mean(gray)
        if brightness < 30:
            findings.append(MedicalFinding(
                finding_type="underexposed",
                severity="low",
                risk_level="low",
                score=0.25,
                description=f"Underexposed (brightness: {brightness:.1f}). Dark regions may hide abnormalities.",
                recommendation="Increase exposure or apply brightness correction.",
                confidence=0.75,
            ))
        elif brightness > 220:
            findings.append(MedicalFinding(
                finding_type="overexposed",
                severity="low",
                risk_level="low",
                score=0.25,
                description=f"Overexposed (brightness: {brightness:.1f}). Bright areas may obscure details.",
                recommendation="Decrease exposure settings.",
                confidence=0.75,
            ))

        noise_level = self._estimate_noise(gray)
        if noise_level > 30:
            findings.append(MedicalFinding(
                finding_type="high_noise",
                severity="medium",
                risk_level="medium",
                score=0.4,
                description=f"High noise (noise: {noise_level:.1f}). May mask subtle abnormalities.",
                recommendation="Apply noise reduction or re-scan.",
                confidence=0.7,
            ))

        return findings

    def _analyze_xray(self, image: np.ndarray, abnormality_score: float, pathology_probs: Dict) -> List[MedicalFinding]:
        findings = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_opacities = [c for c in contours if cv2.contourArea(c) > h * w * 0.02]

        if len(large_opacities) > 3:
            conf = min(0.85, pathology_probs.get("abnormal", 0.5) + 0.15)
            findings.append(MedicalFinding(
                finding_type="multiple_opacities",
                severity="high",
                risk_level="high",
                score=min(0.85, abnormality_score + 0.15),
                description=f"Multiple opaque regions ({len(large_opacities)}). May indicate pulmonary infiltrates, masses, or fluid accumulation.",
                recommendation="URGENT: Clinical correlation required. Consider CT chest.",
                confidence=conf,
            ))

        edges = cv2.Canny(enhanced, 30, 100)
        edge_density = np.sum(edges > 0) / (h * w)
        if edge_density > 0.15:
            findings.append(MedicalFinding(
                finding_type="increased_interstitial_markings",
                severity="medium",
                risk_level="medium",
                score=min(0.7, edge_density * 3),
                description=f"Increased interstitial markings (density: {edge_density:.1%}). May suggest ILD or edema.",
                recommendation="Correlate with pulmonary function tests.",
                confidence=0.65,
            ))

        asymmetry = self._compute_symmetry(gray)
        if asymmetry > 0.15:
            findings.append(MedicalFinding(
                finding_type="anatomical_asymmetry",
                severity="medium",
                risk_level="medium",
                score=min(0.7, asymmetry * 3),
                description=f"Anatomical asymmetry (score: {asymmetry:.2f}). May indicate mass effect or pleural effusion.",
                recommendation="Compare with prior imaging. Consider CT.",
                confidence=0.7,
            ))

        if abnormality_score > 0.6:
            findings.append(MedicalFinding(
                finding_type="ai_detected_abnormality",
                severity="high" if abnormality_score > 0.75 else "medium",
                risk_level="high" if abnormality_score > 0.75 else "medium",
                score=abnormality_score,
                description=f"AI model detected potential abnormality (confidence: {abnormality_score:.2%}).",
                recommendation="Clinical review strongly recommended.",
                confidence=abnormality_score,
            ))

        return findings

    def _analyze_mri(self, image: np.ndarray, abnormality_score: float, pathology_probs: Dict) -> List[MedicalFinding]:
        findings = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        mean_val = np.mean(enhanced)
        std_val = np.std(enhanced)

        hyperintense_mask = enhanced > (mean_val + 2 * std_val)
        hypointense_mask = enhanced < (mean_val - 2 * std_val)

        hyper_ratio = np.sum(hyperintense_mask) / (h * w)
        hypo_ratio = np.sum(hypointense_mask) / (h * w)

        if hyper_ratio > 0.1:
            findings.append(MedicalFinding(
                finding_type="hyperintense_regions",
                severity="high" if hyper_ratio > 0.2 else "medium",
                risk_level="high" if hyper_ratio > 0.2 else "medium",
                score=min(0.8, hyper_ratio * 3),
                description=f"Hyperintense regions ({hyper_ratio:.1%}). May indicate edema, inflammation, or tumor on T2.",
                recommendation="Correlate with clinical history and other sequences.",
                confidence=0.7,
            ))

        if hypo_ratio > 0.1:
            findings.append(MedicalFinding(
                finding_type="hypointense_regions",
                severity="medium",
                risk_level="medium",
                score=min(0.7, hypo_ratio * 3),
                description=f"Hypointense regions ({hypo_ratio:.1%}). May indicate hemorrhage or calcification.",
                recommendation="Evaluate with T2*, SWI sequences.",
                confidence=0.65,
            ))

        texture = self._compute_glcm_features(gray)
        if texture["entropy"] > 5.0:
            findings.append(MedicalFinding(
                finding_type="heterogeneous_tissue",
                severity="medium",
                risk_level="medium",
                score=min(0.7, texture["entropy"] / 8),
                description=f"Heterogeneous texture (entropy: {texture['entropy']:.2f}). May indicate tumor infiltration.",
                recommendation="Multi-parametric MRI analysis recommended.",
                confidence=0.6,
            ))

        if abnormality_score > 0.5:
            findings.append(MedicalFinding(
                finding_type="ai_detected_abnormality",
                severity="high" if abnormality_score > 0.7 else "medium",
                risk_level="high" if abnormality_score > 0.7 else "medium",
                score=abnormality_score,
                description=f"AI model detected potential abnormality (confidence: {abnormality_score:.2%}).",
                recommendation="Clinical review recommended with additional sequences.",
                confidence=abnormality_score,
            ))

        return findings

    def _analyze_ct(self, image: np.ndarray, abnormality_score: float, pathology_probs: Dict) -> List[MedicalFinding]:
        findings = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        density_levels = {
            "very_low": np.sum(enhanced < 30) / (h * w),
            "low": np.sum((enhanced >= 30) & (enhanced < 80)) / (h * w),
            "medium": np.sum((enhanced >= 80) & (enhanced < 160)) / (h * w),
            "high": np.sum((enhanced >= 160) & (enhanced < 220)) / (h * w),
            "very_high": np.sum(enhanced >= 220) / (h * w),
        }

        if density_levels["very_high"] > 0.05:
            findings.append(MedicalFinding(
                finding_type="hyperdense_areas",
                severity="high" if density_levels["very_high"] > 0.1 else "medium",
                risk_level="high" if density_levels["very_high"] > 0.1 else "medium",
                score=min(0.8, density_levels["very_high"] * 5),
                description=f"Hyperdense areas ({density_levels['very_high']:.1%}). May indicate calcification or hemorrhage.",
                recommendation="Correlate with non-contrast images. HU measurement recommended.",
                confidence=0.7,
            ))

        if density_levels["very_low"] > 0.1:
            findings.append(MedicalFinding(
                finding_type="hypodense_areas",
                severity="medium",
                risk_level="medium",
                score=min(0.7, density_levels["very_low"] * 3),
                description=f"Hypodense areas ({density_levels['very_low']:.1%}). May indicate edema or infarction.",
                recommendation="Evaluate with windowing adjustments.",
                confidence=0.65,
            ))

        if abnormality_score > 0.5:
            findings.append(MedicalFinding(
                finding_type="ai_detected_abnormality",
                severity="high" if abnormality_score > 0.7 else "medium",
                risk_level="high" if abnormality_score > 0.7 else "medium",
                score=abnormality_score,
                description=f"AI model detected potential abnormality (confidence: {abnormality_score:.2%}).",
                recommendation="Contrast-enhanced CT or biopsy may be required.",
                confidence=abnormality_score,
            ))

        return findings

    def _analyze_ultrasound(self, image: np.ndarray, abnormality_score: float, pathology_probs: Dict) -> List[MedicalFinding]:
        findings = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        speckle_noise = self._estimate_noise(gray)
        if speckle_noise > 40:
            findings.append(MedicalFinding(
                finding_type="speckle_artifact",
                severity="low",
                risk_level="low",
                score=0.3,
                description="Significant speckle noise. May reduce diagnostic accuracy.",
                recommendation="Apply speckle reduction filtering.",
                confidence=0.7,
            ))

        echogenicity_std = np.std(gray)
        if echogenicity_std > 60:
            findings.append(MedicalFinding(
                finding_type="heterogeneous_echotexture",
                severity="medium",
                risk_level="medium",
                score=min(0.7, echogenicity_std / 100),
                description="Heterogeneous echotexture. May indicate diffuse parenchymal disease.",
                recommendation="Compare with normal tissue. Doppler may help.",
                confidence=0.6,
            ))

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        if edge_density > 0.08:
            findings.append(MedicalFinding(
                finding_type="suspicious_borders",
                severity="medium",
                risk_level="medium",
                score=min(0.6, edge_density * 5),
                description="Irregular borders detected. May suggest malignancy.",
                recommendation="Biopsy or follow-up imaging recommended.",
                confidence=0.65,
            ))

        if abnormality_score > 0.5:
            findings.append(MedicalFinding(
                finding_type="ai_detected_abnormality",
                severity="high" if abnormality_score > 0.7 else "medium",
                risk_level="high" if abnormality_score > 0.7 else "medium",
                score=abnormality_score,
                description=f"AI model detected potential abnormality (confidence: {abnormality_score:.2%}).",
                recommendation="Further diagnostic evaluation recommended.",
                confidence=abnormality_score,
            ))

        return findings

    def _analyze_general_abnormalities(self, image: np.ndarray, scan_type: str) -> List[MedicalFinding]:
        findings = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        kernel = np.ones((5, 5), np.float32) / 25
        smoothed = cv2.filter2D(gray, -1, kernel)
        difference = cv2.absdiff(gray, smoothed)
        _, high_freq = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_freq_ratio = np.sum(high_freq > 0) / (h * w)

        if high_freq_ratio > 0.12:
            findings.append(MedicalFinding(
                finding_type="high_frequency_anomaly",
                severity="medium",
                risk_level="medium",
                score=min(0.7, high_freq_ratio * 4),
                description=f"Unusual high-frequency patterns ({high_freq_ratio:.1%}). May indicate microcalcifications.",
                recommendation="Magnification views recommended.",
                confidence=0.55,
            ))

        return findings

    def _analyze_texture_patterns(self, image: np.ndarray) -> List[MedicalFinding]:
        findings = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        glcm = self._compute_glcm_features(gray)

        if glcm["contrast"] > 500:
            findings.append(MedicalFinding(
                finding_type="high_texture_contrast",
                severity="medium",
                risk_level="medium",
                score=min(0.6, glcm["contrast"] / 800),
                description=f"High texture contrast ({glcm['contrast']:.1f}). May indicate structural disruption.",
                recommendation="Evaluate in clinical context.",
                confidence=0.55,
            ))

        if glcm["homogeneity"] < 0.2:
            findings.append(MedicalFinding(
                finding_type="low_texture_homogeneity",
                severity="medium",
                risk_level="medium",
                score=min(0.6, (1 - glcm["homogeneity"]) * 0.8),
                description=f"Low texture homogeneity ({glcm['homogeneity']:.2f}). Non-uniform tissue.",
                recommendation="Multi-region analysis recommended.",
                confidence=0.5,
            ))

        return findings

    def _compute_overall_risk(self, findings: List[MedicalFinding]) -> Tuple[str, float]:
        if not findings:
            return "low", 0.0

        risk_scores = {"low": 0.1, "medium": 0.4, "high": 0.7, "critical": 0.95}
        max_risk = "low"
        max_score = 0.0
        weighted_sum = 0.0
        total_weight = 0.0

        for f in findings:
            score = f.score * f.confidence
            weighted_sum += score
            total_weight += f.confidence
            if f.risk_level == "critical":
                max_risk = "critical"
                max_score = max(max_score, score)
            elif f.risk_level == "high" and max_risk not in ("critical",):
                max_risk = "high"
                max_score = max(max_score, score)
            elif f.risk_level == "medium" and max_risk == "low":
                max_risk = "medium"
                max_score = max(max_score, score)

        avg_score = weighted_sum / total_weight if total_weight > 0 else 0
        high_risk_count = sum(1 for f in findings if f.risk_level in ("high", "critical"))

        if high_risk_count >= 2 or avg_score > 0.6:
            return "critical", float(max(max_score, avg_score))
        elif high_risk_count >= 1 or avg_score > 0.4:
            return "high", float(max(max_score, avg_score))
        elif avg_score > 0.2:
            return "medium", float(avg_score)
        else:
            return "low", float(avg_score)

    def _generate_summary(self, findings: List[MedicalFinding], overall_risk: str) -> str:
        if not findings:
            return "No significant abnormalities detected. Image appears within normal limits."

        critical = [f for f in findings if f.risk_level == "critical"]
        high = [f for f in findings if f.risk_level == "high"]
        medium = [f for f in findings if f.risk_level == "medium"]
        low = [f for f in findings if f.risk_level == "low"]

        parts = []
        if overall_risk in ("critical", "high"):
            parts.append(f"*** {overall_risk.upper()} RISK DETECTED ***")
            parts.append(f"{len(critical) + len(high)} significant finding(s) requiring immediate attention:")
            for f in critical + high:
                parts.append(f"  - {f.description}")
                parts.append(f"    Recommendation: {f.recommendation}")

        if medium:
            parts.append(f"\n{len(medium)} moderate finding(s) for clinical review:")
            for f in medium[:3]:
                parts.append(f"  - {f.description}")

        if low:
            parts.append(f"\n{len(low)} minor observation(s):")
            for f in low[:2]:
                parts.append(f"  - {f.description}")

        parts.append(f"\nOverall: {overall_risk.upper()} RISK")
        parts.append("\nNOTE: This is an AI-assisted screening analysis. All findings require clinical correlation.")

        return "\n".join(parts)

    def _annotate_findings(self, image: np.ndarray, findings: List[MedicalFinding]) -> np.ndarray:
        annotated = image.copy()
        h, w = annotated.shape[:2]
        risk_colors = {
            "critical": (0, 0, 255),
            "high": (0, 140, 255),
            "medium": (0, 255, 255),
            "low": (0, 255, 0),
        }

        y_offset = 30
        for finding in findings:
            color = risk_colors.get(finding.risk_level, (255, 255, 255))
            if finding.region:
                r = finding.region
                cv2.rectangle(annotated, (r["x"], r["y"]), (r["x"] + r["w"], r["y"] + r["h"]), color, 2)
            text = f"[{finding.risk_level.upper()}] {finding.finding_type}"
            cv2.putText(annotated, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
            if y_offset > h - 20:
                break

        _, overall_risk, risk_score = None, *self._compute_overall_risk(findings)
        banner_color = risk_colors.get(overall_risk, (255, 255, 255))
        cv2.rectangle(annotated, (0, 0), (w, 25), banner_color, -1)
        cv2.putText(annotated, f"RISK: {overall_risk.upper()} | Score: {risk_score:.2f} | Findings: {len(findings)}",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return annotated

    @staticmethod
    def _encode_image(image: np.ndarray) -> Optional[str]:
        try:
            import base64
            _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buf).decode()
        except Exception:
            return None

    @staticmethod
    def _compute_symmetry(gray: np.ndarray) -> float:
        h, w = gray.shape
        mid = w // 2
        left = gray[:, :mid]
        right = cv2.flip(gray[:, mid:], 1)
        min_w = min(left.shape[1], right.shape[1])
        diff = np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float))
        return float(np.mean(diff) / 255.0)

    @staticmethod
    def _estimate_noise(gray: np.ndarray) -> float:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blurred)
        return float(np.std(noise))

    @staticmethod
    def _compute_glcm_features(gray: np.ndarray) -> Dict[str, float]:
        gray_q = (gray // 32).astype(np.uint8)
        h, w = gray_q.shape
        levels = 8
        glcm = np.zeros((levels, levels), dtype=np.float64)

        for y in range(0, h - 1, 2):
            for x in range(0, w - 1, 2):
                i, j = gray_q[y, x], gray_q[y, x + 1]
                glcm[i, j] += 1
                glcm[j, i] += 1

        glcm_sum = glcm.sum()
        if glcm_sum > 0:
            glcm /= glcm_sum

        contrast = energy = homogeneity = entropy_val = 0.0
        for i in range(levels):
            for j in range(levels):
                p = glcm[i, j]
                contrast += (i - j) ** 2 * p
                energy += p ** 2
                homogeneity += p / (1 + abs(i - j))
                if p > 0:
                    entropy_val -= p * np.log2(p)

        return {
            "contrast": float(contrast),
            "energy": float(energy),
            "homogeneity": float(homogeneity),
            "entropy": float(entropy_val),
        }
