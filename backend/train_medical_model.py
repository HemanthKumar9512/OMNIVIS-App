"""
OMNIVIS - Medical Model Training Script
Transfer learning for medical image classification and pathology detection.
Supports: X-Ray, MRI, CT, Ultrasound classification + normal/abnormal detection.
Uses: EfficientNet, ResNet, DenseNet backbones with medical data augmentation.
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("train_medical")

MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "datasets"
os.makedirs(MODELS_DIR, exist_ok=True)


@dataclass
class MedicalTrainingConfig:
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.0001
    weight_decay: float = 1e-4
    num_workers: int = 4
    patience: int = 10
    image_size: int = 224
    backbone: str = "efficientnet_b0"
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class MedicalDataset(Dataset):
    """Dataset for medical image classification."""

    SCAN_TYPES = ["x-ray", "mri", "ct", "ultrasound"]

    def __init__(self, root_dir: str, transform=None, task: str = "scan_type"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.task = task
        self.samples = []
        self.labels = []
        self.label_to_idx = {}

        self._load_data()

    def _load_data(self):
        if self.task == "scan_type":
            self.label_to_idx = {t: i for i, t in enumerate(self.SCAN_TYPES)}
        elif self.task == "pathology":
            self.label_to_idx = {"normal": 0, "abnormal": 1}

        if not self.root_dir.exists():
            logger.warning(f"Dataset directory not found: {self.root_dir}")
            return

        for label_dir in sorted(self.root_dir.iterdir()):
            if label_dir.is_dir() and label_dir.name in self.label_to_idx:
                label_idx = self.label_to_idx[label_dir.name]
                for img_path in label_dir.glob("*"):
                    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                        self.samples.append(str(img_path))
                        self.labels.append(label_idx)

        logger.info(f"Loaded {len(self.samples)} samples from {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = cv2.imread(img_path)

        if image is None:
            image = np.zeros((self.transform.transforms[0].size, self.transform.transforms[0].size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


class MedicalScanClassifier(nn.Module):
    """EfficientNet-based medical scan type classifier."""

    def __init__(self, num_classes: int = 4, backbone: str = "efficientnet_b0"):
        super().__init__()

        if backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            backbone_model = models.efficientnet_b0(weights=weights)
            num_features = 1280
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            backbone_model = models.resnet18(weights=weights)
            num_features = 512
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            backbone_model = models.resnet50(weights=weights)
            num_features = 2048
        elif backbone == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT
            backbone_model = models.densenet121(weights=weights)
            num_features = 1024
        else:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            backbone_model = models.efficientnet_b0(weights=weights)
            num_features = 1280

        if "densenet" in backbone:
            self.features = nn.Sequential(*list(backbone_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )
        else:
            if backbone == "efficientnet_b0":
                self.features = backbone_model.features
            else:
                self.features = nn.Sequential(*list(backbone_model.children())[:-2])

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PathologyDetector(nn.Module):
    """Binary classifier for normal vs abnormal medical scans."""

    def __init__(self, backbone: str = "resnet18"):
        super().__init__()

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            num_features = 512
        else:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            num_features = 512

        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.model = model

    def forward(self, x):
        return self.model(x)


class MedicalTrainer:
    """Training loop for medical models with class weighting and mixed precision."""

    def __init__(self, model: nn.Module, config: MedicalTrainingConfig, save_dir: str = None):
        self.model = model.to(config.device)
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else MODELS_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs, eta_min=1e-6)

        self.scaler = torch.cuda.amp.GradScaler() if config.device == "cuda" else None

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    def train(self, train_loader: DataLoader, val_loader: DataLoader, class_weights: torch.Tensor = None):
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.config.device))

        logger.info(f"Training on {self.config.device}")
        logger.info(f"Backbone: {self.config.backbone}")
        logger.info(f"Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}")
        logger.info(f"LR: {self.config.learning_rate}, Scheduler: CosineAnnealingLR")

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch [{epoch+1}/{self.config.epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch + 1, val_acc, val_loss)
                logger.info(f" ** Best model saved! Val Acc: {val_acc:.4f}")
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}. Best val acc: {self.best_val_acc:.4f}")
                break

        logger.info(f"Training complete. Best val acc: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        return self.history

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc="Train", leave=False):
            inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return total_loss / total, correct / total

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Val", leave=False):
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)

                with torch.cuda.amp.autocast() if self.scaler else torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return total_loss / total, correct / total

    def _save_checkpoint(self, epoch: int, val_acc: float, val_loss: float):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "config": {
                "backbone": self.config.backbone,
                "image_size": self.config.image_size,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
            },
        }
        torch.save(checkpoint, self.save_dir / "best_medical_checkpoint.pth")

    def save_history(self):
        history_path = self.save_dir / "medical_training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")

    def plot_history(self):
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(self.history["train_loss"], label="Train Loss")
            axes[0].plot(self.history["val_loss"], label="Val Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].set_title("Medical Model Loss")

            axes[1].plot(self.history["train_acc"], label="Train Acc")
            axes[1].plot(self.history["val_acc"], label="Val Acc")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].legend()
            axes[1].set_title("Medical Model Accuracy")

            plt.tight_layout()
            plt.savefig(self.save_dir / "medical_training_curves.png", dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not plot history: {e}")


def compute_class_weights(dataset: MedicalDataset) -> torch.Tensor:
    """Compute class weights for imbalanced datasets."""
    from collections import Counter

    label_counts = Counter(dataset.labels)
    total = len(dataset.labels)
    num_classes = len(label_counts)

    weights = torch.zeros(num_classes)
    for cls, count in label_counts.items():
        weights[cls] = total / (num_classes * count)

    logger.info(f"Class weights: {weights.tolist()}")
    return weights


def train_scan_classifier(data_dir: str = None, config: MedicalTrainingConfig = None):
    """Train medical scan type classifier (X-Ray, MRI, CT, Ultrasound)."""
    logger.info("=" * 60)
    logger.info("Training Medical Scan Type Classifier")
    logger.info("=" * 60)

    if config is None:
        config = MedicalTrainingConfig()

    if data_dir is None:
        synthetic_dir = DATA_DIR / "synthetic" / "train"
        val_dir = DATA_DIR / "synthetic" / "val"
        if not synthetic_dir.exists():
            logger.info("Creating synthetic dataset...")
            from download_datasets import create_synthetic_dataset
            create_synthetic_dataset()
    else:
        data_path = Path(data_dir)
        synthetic_dir = data_path / "train"
        val_dir = data_path / "val"

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MedicalDataset(synthetic_dir, transform=transform_train, task="scan_type")
    val_dataset = MedicalDataset(val_dir, transform=transform_val, task="scan_type")

    if len(train_dataset) == 0:
        logger.error("No training data found. Run download_datasets.py --synthetic first.")
        return None

    class_weights = compute_class_weights(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Classes: {train_dataset.label_to_idx}")

    model = MedicalScanClassifier(num_classes=len(train_dataset.label_to_idx), backbone=config.backbone)
    trainer = MedicalTrainer(model, config, save_dir=MODELS_DIR)

    history = trainer.train(train_loader, val_loader, class_weights=class_weights)
    trainer.save_history()
    trainer.plot_history()

    model_path = MODELS_DIR / "medical_scan_classifier.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Medical scan classifier saved to {model_path}")

    return model


def train_pathology_detector(data_dir: str = None, config: MedicalTrainingConfig = None):
    """Train pathology detector (normal vs abnormal)."""
    logger.info("=" * 60)
    logger.info("Training Pathology Detector")
    logger.info("=" * 60)

    if config is None:
        config = MedicalTrainingConfig(epochs=30)

    model = PathologyDetector(backbone=config.backbone)

    logger.info("Pathology detector requires labeled normal/abnormal images.")
    logger.info(f"Saving placeholder model to {MODELS_DIR / 'pathology_xray.pth'}")

    for scan_type in ["xray", "mri", "ct"]:
        torch.save(model.state_dict(), MODELS_DIR / f"pathology_{scan_type}.pth")
        logger.info(f"Saved placeholder: pathology_{scan_type}.pth")

    return model


def train_ensemble_models(data_dir: str = None):
    """Train an ensemble of different backbones for comparison."""
    logger.info("=" * 60)
    logger.info("Training Ensemble of Medical Models")
    logger.info("=" * 60)

    backbones = ["efficientnet_b0", "resnet18", "resnet50", "densenet121"]
    results = {}

    for backbone in backbones:
        logger.info(f"\nTraining with backbone: {backbone}")

        config = MedicalTrainingConfig(
            backbone=backbone,
            epochs=30,
            batch_size=16,
            patience=5,
        )

        model = train_scan_classifier(data_dir, config)
        if model is not None:
            results[backbone] = {
                "best_val_acc": model.best_val_acc if hasattr(model, 'best_val_acc') else 0,
                "config": vars(config),
            }

    logger.info("\n" + "=" * 60)
    logger.info("Ensemble Results Summary")
    logger.info("=" * 60)
    for backbone, result in results.items():
        logger.info(f"  {backbone}: Val Acc = {result['best_val_acc']:.4f}")

    return results


def evaluate_medical_model(model_path: str, test_dir: str, task: str = "scan_type"):
    """Evaluate a trained medical model."""
    logger.info(f"Evaluating medical model: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if task == "scan_type":
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        config_data = checkpoint.get("config", {})
        model = MedicalScanClassifier(num_classes=4, backbone=config_data.get("backbone", "efficientnet_b0"))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = PathologyDetector()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = MedicalDataset(test_dir, transform=transform, task=task)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets)
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = (all_preds == all_targets).mean()
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))

    if len(np.unique(all_targets)) == 2:
        try:
            all_probs = np.array(all_probs)
            auc = roc_auc_score(all_targets, all_probs[:, 1])
            logger.info(f"ROC AUC: {auc:.4f}")
        except:
            pass

    return accuracy


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train medical models for OMNIVIS")
    parser.add_argument("--task", type=str, default="scan_type", choices=["scan_type", "pathology", "ensemble"], help="Training task")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet18", "resnet50", "densenet121"], help="Backbone architecture")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--evaluate", type=str, default=None, help="Evaluate a trained model")
    parser.add_argument("--test-dir", type=str, default=None, help="Test dataset directory")

    args = parser.parse_args()

    config = MedicalTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        backbone=args.backbone,
    )

    if args.evaluate:
        test_dir = args.test_dir or str(DATA_DIR / "synthetic" / "test")
        evaluate_medical_model(args.evaluate, test_dir, args.task)
        return

    if args.task == "scan_type":
        train_scan_classifier(args.data_dir, config)
    elif args.task == "pathology":
        train_pathology_detector(args.data_dir, config)
    elif args.task == "ensemble":
        train_ensemble_models(args.data_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
