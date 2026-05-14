"""
OMNIVIS - Model Evaluation & Benchmarking Script
Compares OMNIVIS models with baseline models across multiple metrics.
Generates comprehensive reports with accuracy, precision, recall, F1, and speed comparisons.
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("evaluate")

MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "datasets"
RESULTS_DIR = Path(__file__).parent / "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


@dataclass
class ModelMetrics:
    model_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    inference_time_ms: float = 0.0
    fps: float = 0.0
    params_millions: float = 0.0
    memory_mb: float = 0.0
    per_class_metrics: Dict = field(default_factory=dict)
    confusion_matrix: List = field(default_factory=list)


@dataclass
class BenchmarkReport:
    task: str
    dataset: str
    models: Dict[str, ModelMetrics] = field(default_factory=dict)
    best_model: str = ""
    timestamp: str = ""

    def summary(self) -> str:
        lines = [
            "=" * 70,
            f"OMNIVIS Model Benchmark Report",
            f"Task: {self.task}",
            f"Dataset: {self.dataset}",
            f"Timestamp: {self.timestamp}",
            "=" * 70,
            "",
            f"{'Model':<30} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'FPS':>10} {'Params(M)':>10}",
            "-" * 70,
        ]

        for name, metrics in sorted(self.models.items(), key=lambda x: x[1].accuracy, reverse=True):
            lines.append(
                f"{name:<30} {metrics.accuracy:>9.1%} {metrics.f1_score:>9.3f} "
                f"{metrics.roc_auc:>9.3f} {metrics.fps:>9.1f} {metrics.params_millions:>9.2f}"
            )

        if self.best_model:
            lines.append("")
            lines.append(f" Best Model: {self.best_model}")
            lines.append(f" Best Accuracy: {self.models[self.best_model].accuracy:.1%}")

        lines.append("=" * 70)
        return "\n".join(lines)


class BaselineModels:
    """Baseline models for comparison."""

    @staticmethod
    def get_haar_cascade():
        """OpenCV Haar Cascade baseline for face detection."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        return cv2.CascadeClassifier(cascade_path)

    @staticmethod
    def get_svm_classifier():
        """SVM baseline for classification."""
        from sklearn.svm import SVC
        return SVC(kernel="rbf", probability=True, random_state=42)

    @staticmethod
    def get_random_forest():
        """Random Forest baseline for classification."""
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    @staticmethod
    def get_xgboost():
        """XGBoost baseline for classification."""
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="mlogloss")
        except ImportError:
            return None

    @staticmethod
    def get_mobilenet_v2(num_classes: int):
        """MobileNetV2 baseline (lightweight)."""
        import torchvision.models as models
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    @staticmethod
    def get_resnet18(num_classes: int):
        """ResNet18 baseline."""
        import torchvision.models as models
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    @staticmethod
    def get_vgg16(num_classes: int):
        """VGG16 baseline (heavy)."""
        import torchvision.models as models
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model


class Evaluator:
    """Comprehensive model evaluation."""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Evaluation device: {self.device}")

    def evaluate_classification(self, model, dataloader: DataLoader, model_name: str, num_classes: int) -> ModelMetrics:
        """Evaluate a classification model with full metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            roc_auc_score, confusion_matrix, classification_report
        )

        model.eval()
        model.to(self.device)

        all_preds = []
        all_targets = []
        all_probs = []
        inference_times = []

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc=f"Evaluating {model_name}", leave=False):
                inputs = inputs.to(self.device)

                start = time.perf_counter()
                outputs = model(inputs)
                elapsed = (time.perf_counter() - start) * 1000
                inference_times.append(elapsed)

                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets)
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="weighted", zero_division=0
        )

        avg_inference_time = np.mean(inference_times)
        fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0

        roc_auc = 0.0
        if num_classes == 2:
            try:
                roc_auc = roc_auc_score(all_targets, all_probs[:, 1])
            except:
                pass
        else:
            try:
                roc_auc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="weighted")
            except:
                pass

        params_millions = sum(p.numel() for p in model.parameters()) / 1e6

        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )

        cm = confusion_matrix(all_targets, all_preds)

        return ModelMetrics(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            inference_time_ms=avg_inference_time,
            fps=fps,
            params_millions=params_millions,
            per_class_metrics={
                "precision": per_class_precision.tolist(),
                "recall": per_class_recall.tolist(),
                "f1": per_class_f1.tolist(),
            },
            confusion_matrix=cm.tolist(),
        )

    def evaluate_sklearn_model(self, model, X_test, y_test, model_name: str) -> ModelMetrics:
        """Evaluate a scikit-learn model."""
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            roc_auc_score, confusion_matrix
        )

        start = time.perf_counter()
        if hasattr(model, "predict_proba"):
            all_probs = model.predict_proba(X_test)
        else:
            all_probs = None
        all_preds = model.predict(X_test)
        elapsed = (time.perf_counter() - start) * 1000

        accuracy = accuracy_score(y_test, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, all_preds, average="weighted", zero_division=0
        )

        avg_inference_time = elapsed / len(y_test)
        fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0

        num_classes = len(np.unique(y_test))
        roc_auc = 0.0
        if all_probs is not None:
            if num_classes == 2:
                roc_auc = roc_auc_score(y_test, all_probs[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, all_probs, multi_class="ovr", average="weighted")

        cm = confusion_matrix(y_test, all_preds)

        return ModelMetrics(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            inference_time_ms=avg_inference_time,
            fps=fps,
            per_class_metrics={},
            confusion_matrix=cm.tolist(),
        )


def benchmark_face_models(test_dir: str = None) -> BenchmarkReport:
    """Benchmark face emotion recognition models."""
    logger.info("=" * 60)
    logger.info("Benchmarking Face Emotion Recognition Models")
    logger.info("=" * 60)

    if test_dir is None:
        test_dir = str(DATA_DIR / "synthetic" / "val")

    evaluator = Evaluator()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from train_face_model import FaceDataset
    test_dataset = FaceDataset(test_dir, transform=transform, task="emotion")

    if len(test_dataset) == 0:
        logger.error("No test data found. Run download_datasets.py --synthetic first.")
        return None

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    num_classes = len(test_dataset.label_to_idx)

    report = BenchmarkReport(
        task="Face Emotion Recognition",
        dataset="synthetic" if "synthetic" in test_dir else test_dir,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    models_to_test = {}

    if (MODELS_DIR / "emotion_model.pth").exists():
        from train_face_model import EmotionNet
        model = EmotionNet(num_classes=num_classes)
        model.load_state_dict(torch.load(MODELS_DIR / "emotion_model.pth", map_location=evaluator.device, weights_only=True))
        models_to_test["OMNIVIS-EmotionNet"] = model

    mobilenet = BaselineModels.get_mobilenet_v2(num_classes)
    models_to_test["Baseline-MobileNetV2"] = mobilenet

    resnet18 = BaselineModels.get_resnet18(num_classes)
    models_to_test["Baseline-ResNet18"] = resnet18

    for name, model in models_to_test.items():
        metrics = evaluator.evaluate_classification(model, test_loader, name, num_classes)
        report.models[name] = metrics
        logger.info(f"  {name}: Acc={metrics.accuracy:.4f}, F1={metrics.f1_score:.3f}, FPS={metrics.fps:.1f}")

    if report.models:
        report.best_model = max(report.models.keys(), key=lambda k: report.models[k].accuracy)

    report_path = RESULTS_DIR / "face_benchmark.json"
    report_dict = {
        "task": report.task,
        "dataset": report.dataset,
        "timestamp": report.timestamp,
        "best_model": report.best_model,
        "models": {k: {kk: vv for kk, vv in v.__dict__.items()} for k, v in report.models.items()},
    }
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"\n{report.summary()}")
    logger.info(f"Report saved to: {report_path}")

    return report


def benchmark_medical_models(test_dir: str = None) -> BenchmarkReport:
    """Benchmark medical image classification models."""
    logger.info("=" * 60)
    logger.info("Benchmarking Medical Image Classification Models")
    logger.info("=" * 60)

    if test_dir is None:
        test_dir = str(DATA_DIR / "synthetic" / "val")

    evaluator = Evaluator()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from train_medical_model import MedicalDataset, MedicalScanClassifier
    test_dataset = MedicalDataset(test_dir, transform=transform, task="scan_type")

    if len(test_dataset) == 0:
        logger.error("No test data found. Run download_datasets.py --synthetic first.")
        return None

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    num_classes = len(test_dataset.label_to_idx)

    report = BenchmarkReport(
        task="Medical Scan Classification",
        dataset="synthetic" if "synthetic" in test_dir else test_dir,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    models_to_test = {}

    if (MODELS_DIR / "medical_scan_classifier.pth").exists():
        checkpoint = torch.load(MODELS_DIR / "medical_scan_classifier.pth", map_location=evaluator.device, weights_only=True)
        config_data = checkpoint.get("config", {})
        model = MedicalScanClassifier(num_classes=num_classes, backbone=config_data.get("backbone", "efficientnet_b0"))
        model.load_state_dict(checkpoint["model_state_dict"])
        models_to_test["OMNIVIS-MedicalClassifier"] = model
    else:
        model = MedicalScanClassifier(num_classes=num_classes, backbone="efficientnet_b0")
        models_to_test["OMNIVIS-MedicalClassifier (untrained)"] = model

    mobilenet = BaselineModels.get_mobilenet_v2(num_classes)
    models_to_test["Baseline-MobileNetV2"] = mobilenet

    resnet18 = BaselineModels.get_resnet18(num_classes)
    models_to_test["Baseline-ResNet18"] = resnet18

    vgg16 = BaselineModels.get_vgg16(num_classes)
    models_to_test["Baseline-VGG16"] = vgg16

    for name, model in models_to_test.items():
        metrics = evaluator.evaluate_classification(model, test_loader, name, num_classes)
        report.models[name] = metrics
        logger.info(f"  {name}: Acc={metrics.accuracy:.4f}, F1={metrics.f1_score:.3f}, FPS={metrics.fps:.1f}")

    if report.models:
        report.best_model = max(report.models.keys(), key=lambda k: report.models[k].accuracy)

    report_path = RESULTS_DIR / "medical_benchmark.json"
    report_dict = {
        "task": report.task,
        "dataset": report.dataset,
        "timestamp": report.timestamp,
        "best_model": report.best_model,
        "models": {k: {kk: vv for kk, vv in v.__dict__.items()} for k, v in report.models.items()},
    }
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"\n{report.summary()}")
    logger.info(f"Report saved to: {report_path}")

    return report


def benchmark_detection_accuracy(image_dir: str = None) -> Dict[str, Any]:
    """Benchmark face detection accuracy using different detectors."""
    logger.info("=" * 60)
    logger.info("Benchmarking Face Detection Accuracy")
    logger.info("=" * 60)

    results = {}

    test_images = []
    if image_dir:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            test_images.extend(list(Path(image_dir).glob(ext)))
    else:
        logger.info("No test image directory specified. Using synthetic images.")
        from download_datasets import _generate_synthetic_face
        os.makedirs(RESULTS_DIR / "test_faces", exist_ok=True)
        for i in range(20):
            for emotion in ["happy", "sad", "neutral"]:
                img = _generate_synthetic_face(emotion, 128)
                path = RESULTS_DIR / "test_faces" / f"{emotion}_{i}.png"
                cv2.imwrite(str(path), img)
                test_images.append(path)

    detectors = {
        "InsightFace": "insightface",
        "MediaPipe": "mediapipe",
        "Haar Cascade": "haar",
        "OpenCV DNN": "opencv_dnn",
    }

    for name, det_type in detectors.items():
        try:
            start = time.perf_counter()
            detected = 0

            for img_path in test_images:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                if det_type == "haar":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cascade = BaselineModels.get_haar_cascade()
                    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    if len(faces) > 0:
                        detected += 1
                elif det_type == "mediapipe":
                    try:
                        import mediapipe as mp
                        mp_face = mp.solutions.face_detection
                        with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = detector.process(rgb)
                            if results.detections:
                                detected += 1
                    except:
                        pass

            elapsed = time.perf_counter() - start
            accuracy = detected / max(1, len(test_images))

            results[name] = {
                "accuracy": accuracy,
                "detected": detected,
                "total": len(test_images),
                "time_ms": elapsed * 1000,
                "fps": len(test_images) / elapsed if elapsed > 0 else 0,
            }

            logger.info(f"  {name}: Accuracy={accuracy:.1%} ({detected}/{len(test_images)}), FPS={results[name]['fps']:.1f}")

        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            results[name] = {"accuracy": 0, "error": str(e)}

    report_path = RESULTS_DIR / "detection_benchmark.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nDetection benchmark saved to: {report_path}")
    return results


def generate_comparison_chart(reports: List[BenchmarkReport]):
    """Generate visual comparison charts."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for report in reports:
            if not report.models:
                continue

            names = list(report.models.keys())
            accuracies = [report.models[n].accuracy for n in names]
            f1_scores = [report.models[n].f1_score for n in names]

            x = np.arange(len(names))
            width = 0.35

            bars1 = axes[0].bar(x - width/2, accuracies, width, label=report.task[:20], alpha=0.8)
            axes[0].set_xlabel("Model")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_title("Model Accuracy Comparison")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([n.split("-")[-1] for n in names], rotation=45, ha="right")
            axes[0].legend()
            axes[0].set_ylim(0, 1.1)

            bars2 = axes[1].bar(x - width/2, f1_scores, width, label=report.task[:20], alpha=0.8)
            axes[1].set_xlabel("Model")
            axes[1].set_ylabel("F1 Score")
            axes[1].set_title("Model F1 Score Comparison")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([n.split("-")[-1] for n in names], rotation=45, ha="right")
            axes[1].legend()
            axes[1].set_ylim(0, 1.1)

        plt.tight_layout()
        chart_path = RESULTS_DIR / "model_comparison.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Comparison chart saved to: {chart_path}")

    except Exception as e:
        logger.warning(f"Could not generate comparison chart: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate and benchmark OMNIVIS models")
    parser.add_argument("--task", type=str, default="all", choices=["face", "medical", "detection", "all"], help="Which task to benchmark")
    parser.add_argument("--face-test-dir", type=str, default=None, help="Face test dataset directory")
    parser.add_argument("--medical-test-dir", type=str, default=None, help="Medical test dataset directory")
    parser.add_argument("--detection-images", type=str, default=None, help="Directory with test images for detection benchmark")
    parser.add_argument("--report", action="store_true", help="Generate comparison report")

    args = parser.parse_args()

    reports = []

    if args.task in ["face", "all"]:
        face_report = benchmark_face_models(args.face_test_dir)
        if face_report:
            reports.append(face_report)

    if args.task in ["medical", "all"]:
        medical_report = benchmark_medical_models(args.medical_test_dir)
        if medical_report:
            reports.append(medical_report)

    if args.task in ["detection", "all"]:
        detection_results = benchmark_detection_accuracy(args.detection_images)

    if reports:
        generate_comparison_chart(reports)

        combined_path = RESULTS_DIR / "combined_benchmark.json"
        combined = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reports": [
                {
                    "task": r.task,
                    "dataset": r.dataset,
                    "best_model": r.best_model,
                    "models": {k: {kk: vv for kk, vv in v.__dict__.items()} for k, v in r.models.items()},
                }
                for r in reports
            ],
        }
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        logger.info(f"Combined benchmark saved to: {combined_path}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
