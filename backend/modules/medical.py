"""
OMNIVIS — Medical Scan Analysis Module
Analyzes MRI, CT, X-ray, and other medical imaging scans to detect risks,
abnormalities, and potential pathologies with clear risk reporting.
"""
import cv2
import numpy as np
import time
import logging
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


class MedicalScanAnalyzer:
    """Multi-modal medical scan analysis with risk detection and reporting."""

    def __init__(self):
        self.analysis_history = []
        self._init_classifiers()

    def _init_classifiers(self):
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            self.feature_scaler = StandardScaler()
            self.texture_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.has_sklearn = True
            logger.info("Medical analysis ML classifiers initialized")
        except ImportError:
            self.has_sklearn = False
            logger.warning("scikit-learn not available — using heuristic-based medical analysis")

    def analyze(self, image: np.ndarray, scan_type_hint: str = "auto") -> Dict[str, Any]:
        start = time.perf_counter()

        scan_type = self._detect_scan_type(image, scan_type_hint)

        findings = []

        findings.extend(self._analyze_image_quality(image))

        if scan_type == ScanType.XRAY:
            findings.extend(self._analyze_xray(image))
        elif scan_type == ScanType.MRI:
            findings.extend(self._analyze_mri(image))
        elif scan_type == ScanType.CT:
            findings.extend(self._analyze_ct(image))
        elif scan_type == ScanType.ULTRASOUND:
            findings.extend(self._analyze_ultrasound(image))

        findings.extend(self._analyze_general_abnormalities(image, scan_type))

        findings.extend(self._analyze_texture_patterns(image))

        overall_risk, risk_score = self._compute_overall_risk(findings)

        annotated = self._annotate_findings(image.copy(), findings)

        inference_ms = (time.perf_counter() - start) * 1000

        return {
            "scan_type": scan_type.value,
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
            "finding_count": len(findings),
            "high_risk_count": sum(1 for f in findings if f.risk_level in ("high", "critical")),
            "annotated_image": self._encode_image(annotated),
            "inference_ms": round(inference_ms, 1),
            "summary": self._generate_summary(findings, overall_risk),
        }

    def _detect_scan_type(self, image: np.ndarray, hint: str) -> ScanType:
        if hint != "auto":
            type_map = {
                "x-ray": ScanType.XRAY, "xray": ScanType.XRAY,
                "mri": ScanType.MRI, "MRI": ScanType.MRI,
                "ct": ScanType.CT, "ct_scan": ScanType.CT, "ct-scan": ScanType.CT,
                "ultrasound": ScanType.ULTRASOUND, "usg": ScanType.ULTRASOUND,
            }
            return type_map.get(hint.lower(), ScanType.UNKNOWN)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        aspect_ratio = w / h

        mean_val = np.mean(gray)
        std_val = np.std(gray)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        dark_ratio = np.sum(hist[:50]) / np.sum(hist)
        bright_ratio = np.sum(hist[200:]) / np.sum(hist)
        mid_ratio = np.sum(hist[80:180]) / np.sum(hist)

        is_grayscale = std_val < 30 or np.allclose(image[:, :, 0], image[:, :, 1])

        if aspect_ratio > 1.5 and dark_ratio > 0.3:
            return ScanType.XRAY
        elif aspect_ratio > 1.2 and mid_ratio > 0.5:
            return ScanType.MRI
        elif aspect_ratio > 1.0 and bright_ratio > 0.2:
            return ScanType.CT
        elif aspect_ratio < 1.0 and dark_ratio > 0.4:
            return ScanType.ULTRASOUND
        elif is_grayscale and dark_ratio > 0.2:
            return ScanType.XRAY
        else:
            return ScanType.UNKNOWN

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
                description=f"Image appears blurred (sharpness score: {blur_score:.1f}). Low quality scans may obscure critical details.",
                recommendation="Consider re-scanning with better focus and patient positioning for accurate diagnosis.",
                confidence=0.85,
            ))

        contrast = np.std(gray)
        if contrast < 20:
            findings.append(MedicalFinding(
                finding_type="low_contrast",
                severity="medium",
                risk_level="medium",
                score=0.35,
                description=f"Low image contrast detected (std: {contrast:.1f}). Tissue differentiation may be compromised.",
                recommendation="Adjust imaging parameters or apply contrast enhancement for better tissue visualization.",
                confidence=0.8,
            ))

        brightness = np.mean(gray)
        if brightness < 30:
            findings.append(MedicalFinding(
                finding_type="underexposed",
                severity="low",
                risk_level="low",
                score=0.25,
                description=f"Image appears underexposed (mean brightness: {brightness:.1f}). Dark regions may hide abnormalities.",
                recommendation="Increase exposure settings or apply brightness correction.",
                confidence=0.75,
            ))
        elif brightness > 220:
            findings.append(MedicalFinding(
                finding_type="overexposed",
                severity="low",
                risk_level="low",
                score=0.25,
                description=f"Image appears overexposed (mean brightness: {brightness:.1f}). Bright areas may obscure details.",
                recommendation="Decrease exposure settings to preserve detail in bright regions.",
                confidence=0.75,
            ))

        noise_level = self._estimate_noise(gray)
        if noise_level > 30:
            findings.append(MedicalFinding(
                finding_type="high_noise",
                severity="medium",
                risk_level="medium",
                score=0.4,
                description=f"High noise level detected (noise: {noise_level:.1f}). May mask subtle abnormalities.",
                recommendation="Apply noise reduction filtering or re-scan with optimized settings.",
                confidence=0.7,
            ))

        return findings

    def _analyze_xray(self, image: np.ndarray) -> List[MedicalFinding]:
        findings = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        opacity_regions = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = opacity_regions[0] if len(opacity_regions) == 2 else opacity_regions[1]

        large_opacities = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (h * w * 0.02):
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect = cw / ch if ch > 0 else 0
                large_opacities.append({
                    "area": area,
                    "x": x, "y": y, "w": cw, "h": ch,
                    "aspect": aspect,
                })

        if len(large_opacities) > 3:
            findings.append(MedicalFinding(
                finding_type="multiple_opacities",
                severity="high",
                risk_level="high",
                score=0.7,
                description=f"Multiple large opaque regions detected ({len(large_opacities)}). May indicate pulmonary infiltrates, masses, or fluid accumulation.",
                recommendation="URGENT: Clinical correlation required. Consider CT chest for further evaluation.",
                confidence=0.65,
            ))

        for op in large_opacities:
            if op["aspect"] > 2.5 or op["aspect"] < 0.4:
                findings.append(MedicalFinding(
                    finding_type="irregular_opacity",
                    severity="medium",
                    risk_level="medium",
                    score=0.5,
                    description=f"Irregular shaped opacity detected (aspect ratio: {op['aspect']:.2f}). Unusual shape may indicate pathological process.",
                    recommendation="Further imaging recommended to characterize the nature of this finding.",
                    region={"x": op["x"], "y": op["y"], "w": op["w"], "h": op["h"]},
                    confidence=0.6,
                ))

        edges = cv2.Canny(enhanced, 30, 100)
        edge_density = np.sum(edges > 0) / (h * w)

        if edge_density > 0.15:
            findings.append(MedicalFinding(
                finding_type="increased_interstitial_markings",
                severity="medium",
                risk_level="medium",
                score=0.55,
                description=f"Increased interstitial markings detected (density: {edge_density:.1%}). May suggest interstitial lung disease or edema.",
                recommendation="Clinical correlation with pulmonary function tests recommended.",
                confidence=0.6,
            ))

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        asymmetry = self._compute_symmetry(gray)
        if asymmetry > 0.15:
            findings.append(MedicalFinding(
                finding_type="anatomical_asymmetry",
                severity="medium",
                risk_level="medium",
                score=0.5,
                description=f"Significant anatomical asymmetry detected (score: {asymmetry:.2f}). May indicate mass effect, atelectasis, or pleural effusion.",
                recommendation="Compare with previous imaging. Consider CT for 3D assessment.",
                confidence=0.65,
            ))

        return findings

    def _analyze_mri(self, image: np.ndarray) -> List[MedicalFinding]:
        findings = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        hyperintense_mask = enhanced > (np.mean(enhanced) + 2 * np.std(enhanced))
        hypointense_mask = enhanced < (np.mean(enhanced) - 2 * np.std(enhanced))

        hyper_ratio = np.sum(hyperintense_mask) / (h * w)
        hypo_ratio = np.sum(hypointense_mask) / (h * w)

        if hyper_ratio > 0.1:
            findings.append(MedicalFinding(
                finding_type="hyperintense_regions",
                severity="high" if hyper_ratio > 0.2 else "medium",
                risk_level="high" if hyper_ratio > 0.2 else "medium",
                score=min(0.8, hyper_ratio * 3),
                description=f"Significant hyperintense regions detected ({hyper_ratio:.1%} of image). On T2-weighted MRI, this may indicate edema, inflammation, demyelination, or tumor.",
                recommendation="Correlate with clinical history and other MRI sequences (T1, FLAIR, DWI). Gadolinium contrast may be helpful.",
                confidence=0.7,
            ))

        if hypo_ratio > 0.1:
            findings.append(MedicalFinding(
                finding_type="hypointense_regions",
                severity="medium",
                risk_level="medium",
                score=min(0.7, hypo_ratio * 3),
                description=f"Significant hypointense regions detected ({hypo_ratio:.1%} of image). May indicate hemorrhage, calcification, or fibrosis.",
                recommendation="Evaluate with additional sequences (T2*, SWI) to characterize the nature of these findings.",
                confidence=0.65,
            ))

        texture_features = self._compute_glcm_features(gray)
        if texture_features["entropy"] > 5.0:
            findings.append(MedicalFinding(
                finding_type="heterogeneous_tissue",
                severity="medium",
                risk_level="medium",
                score=min(0.7, texture_features["entropy"] / 8),
                description=f"Heterogeneous tissue texture detected (entropy: {texture_features['entropy']:.2f}). Heterogeneity may indicate tumor infiltration or mixed pathology.",
                recommendation="Multi-parametric MRI analysis recommended. Consider biopsy if clinical suspicion is high.",
                confidence=0.6,
            ))

        contours, _ = cv2.findContours(
            cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > h * w * 0.05:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                if circularity > 0.7:
                    findings.append(MedicalFinding(
                        finding_type="well_defined_lesion",
                        severity="medium",
                        risk_level="medium",
                        score=0.5,
                        description=f"Well-defined rounded lesion detected (circularity: {circularity:.2f}). May represent cyst, benign tumor, or other focal pathology.",
                        recommendation="Characterize with contrast-enhanced sequences. Follow-up imaging may be needed.",
                        confidence=0.55,
                    ))

        return findings

    def _analyze_ct(self, image: np.ndarray) -> List[MedicalFinding]:
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
                description=f"Hyperdense areas detected ({density_levels['very_high']:.1%} of image). May indicate calcification, hemorrhage, or contrast enhancement.",
                recommendation="Correlate with non-contrast images. Hounsfield unit measurement recommended.",
                confidence=0.7,
            ))

        if density_levels["very_low"] > 0.1:
            findings.append(MedicalFinding(
                finding_type="hypodense_areas",
                severity="medium",
                risk_level="medium",
                score=min(0.7, density_levels["very_low"] * 3),
                description=f"Extensive hypodense areas detected ({density_levels['very_low']:.1%}). May indicate edema, infarction, cystic change, or emphysema.",
                recommendation="Evaluate with windowing adjustments. Clinical correlation essential.",
                confidence=0.65,
            ))

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mass_like = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if h * w * 0.02 < area < h * w * 0.3:
                x, y, cw, ch = cv2.boundingRect(contour)
                mass_like.append({"x": x, "y": y, "w": cw, "h": ch, "area": area})

        if len(mass_like) > 0:
            findings.append(MedicalFinding(
                finding_type="suspected_mass",
                severity="high",
                risk_level="high",
                score=0.65,
                description=f"Mass-like structure detected ({len(mass_like)} region(s)). Requires characterization for benign vs malignant features.",
                recommendation="URGENT: Contrast-enhanced CT, biopsy, or PET-CT may be required. Immediate clinical review advised.",
                confidence=0.6,
            ))

        return findings

    def _analyze_ultrasound(self, image: np.ndarray) -> List[MedicalFinding]:
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
                description=f"Significant speckle noise detected. May reduce diagnostic accuracy.",
                recommendation="Apply speckle reduction filtering (e.g., anisotropic diffusion).",
                confidence=0.7,
            ))

        echogenicity_std = np.std(gray)
        if echogenicity_std > 60:
            findings.append(MedicalFinding(
                finding_type="heterogeneous_echotexture",
                severity="medium",
                risk_level="medium",
                score=min(0.7, echogenicity_std / 100),
                description=f"Heterogeneous echotexture detected. May indicate diffuse parenchymal disease.",
                recommendation="Compare with normal tissue. Doppler evaluation may be helpful.",
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
                description=f"Irregular/spiculated borders detected in tissue structure. Irregular margins may suggest malignancy.",
                recommendation="Biopsy or follow-up imaging strongly recommended.",
                confidence=0.65,
            ))

        return findings

    def _analyze_general_abnormalities(self, image: np.ndarray, scan_type: ScanType) -> List[MedicalFinding]:
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
                description=f"Unusual high-frequency patterns detected ({high_freq_ratio:.1%} of image). May indicate microcalcifications, fibrosis, or early pathological changes.",
                recommendation="Magnification views or higher resolution imaging recommended.",
                confidence=0.55,
            ))

        intensity_distribution = self._analyze_intensity_distribution(gray)
        if intensity_distribution["bimodality"] > 0.6:
            findings.append(MedicalFinding(
                finding_type="bimodal_intensity",
                severity="low",
                risk_level="low",
                score=0.35,
                description=f"Bimodal intensity distribution detected. Distinct tissue populations may indicate normal vs abnormal tissue separation.",
                recommendation="Segment analysis of distinct intensity regions may reveal localized pathology.",
                confidence=0.5,
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
                description=f"High texture contrast detected ({glcm['contrast']:.1f}). May indicate structural disruption or tissue heterogeneity.",
                recommendation="Evaluate texture patterns in context of clinical presentation.",
                confidence=0.55,
            ))

        if glcm["homogeneity"] < 0.2:
            findings.append(MedicalFinding(
                finding_type="low_texture_homogeneity",
                severity="medium",
                risk_level="medium",
                score=min(0.6, (1 - glcm["homogeneity"]) * 0.8),
                description=f"Low texture homogeneity ({glcm['homogeneity']:.2f}). Non-uniform tissue texture may indicate pathological changes.",
                recommendation="Multi-region texture analysis recommended.",
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

        parts.append(f"\nOverall: {overall_risk.upper()} RISK (Score: {self._compute_overall_risk(findings)[1]:.2f})")
        parts.append("\nNOTE: This is an AI-assisted screening analysis. All findings require clinical correlation and should not replace professional medical diagnosis.")

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
                cv2.rectangle(annotated, (r["x"], r["y"]),
                            (r["x"] + r["w"], r["y"] + r["h"]), color, 2)

            text = f"[{finding.risk_level.upper()}] {finding.finding_type}"
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
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
        except:
            return None

    @staticmethod
    def _compute_symmetry(gray: np.ndarray) -> float:
        h, w = gray.shape
        mid = w // 2
        left = gray[:, :mid]
        right = cv2.flip(gray[:, mid:], 1)
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]
        diff = np.abs(left.astype(float) - right.astype(float))
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

    @staticmethod
    def _analyze_intensity_distribution(gray: np.ndarray) -> Dict[str, float]:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()

        mean_val = np.sum(np.arange(256) * hist)
        variance = np.sum((np.arange(256) - mean_val) ** 2 * hist)
        skewness = np.sum((np.arange(256) - mean_val) ** 3 * hist) / (variance ** 1.5 + 1e-10)

        peaks = []
        smoothed = cv2.GaussianBlur(hist.reshape(-1, 1).astype(np.float32), (15, 1), 0).flatten()
        for i in range(5, len(smoothed) - 5):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1] and smoothed[i] > 0.01:
                peaks.append((i, smoothed[i]))

        peaks.sort(key=lambda x: x[1], reverse=True)
        bimodality = 0.0
        if len(peaks) >= 2:
            peak_distance = abs(peaks[0][0] - peaks[1][0])
            bimodality = min(1.0, peak_distance / 128.0) * (peaks[1][1] / peaks[0][1])

        return {
            "mean": float(mean_val),
            "variance": float(variance),
            "skewness": float(skewness),
            "bimodality": float(bimodality),
            "num_peaks": len(peaks),
        }
