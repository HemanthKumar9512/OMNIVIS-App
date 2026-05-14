"""
OMNIVIS - Face Model Training Script
Trains emotion recognition and age/gender estimation models.
Supports: FER2013, CelebA, and synthetic datasets.
Includes: Training, validation, early stopping, checkpointing, and evaluation.
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("train_face")

MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "datasets"
os.makedirs(MODELS_DIR, exist_ok=True)


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_workers: int = 4
    patience: int = 10
    image_size: int = 48
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class FaceDataset(Dataset):
    """Dataset for face emotion/age/gender classification."""

    def __init__(self, root_dir: str, transform=None, task: str = "emotion"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.task = task
        self.samples = []
        self.labels = []
        self.label_to_idx = {}

        self._load_data()

    def _load_data(self):
        if self.task == "emotion":
            self.label_to_idx = {
                "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
                "sad": 4, "surprise": 5, "neutral": 6
            }
        elif self.task == "gender":
            self.label_to_idx = {"male": 0, "female": 1}

        if not self.root_dir.exists():
            logger.warning(f"Dataset directory not found: {self.root_dir}")
            return

        for label_dir in sorted(self.root_dir.iterdir()):
            if label_dir.is_dir() and label_dir.name in self.label_to_idx:
                label_idx = self.label_to_idx[label_dir.name]
                for img_path in label_dir.glob("*"):
                    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
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

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


class EmotionNet(nn.Module):
    """CNN for facial emotion recognition."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AgeGenderNet(nn.Module):
    """CNN for age regression and gender classification."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.age_head = nn.Linear(256, 1)
        self.gender_head = nn.Linear(256, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.shared_fc(x)
        age = self.age_head(x)
        gender = self.gender_head(x)
        return age, gender


class Trainer:
    """Training loop with validation, early stopping, and checkpointing."""

    def __init__(self, model: nn.Module, config: TrainingConfig, save_dir: str = None):
        self.model = model.to(config.device)
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else MODELS_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=5)

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        logger.info(f"Training on {self.config.device}")
        logger.info(f"Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}")
        logger.info(f"LR: {self.config.learning_rate}, Patience: {self.config.patience}")

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_acc)

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
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "image_size": self.config.image_size,
            },
        }
        torch.save(checkpoint, self.save_dir / "best_checkpoint.pth")

    def save_history(self):
        history_path = self.save_dir / "training_history.json"
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
            axes[0].set_title("Loss")

            axes[1].plot(self.history["train_acc"], label="Train Acc")
            axes[1].plot(self.history["val_acc"], label="Val Acc")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].legend()
            axes[1].set_title("Accuracy")

            plt.tight_layout()
            plt.savefig(self.save_dir / "training_curves.png", dpi=150)
            plt.close()
            logger.info(f"Training curves saved to {self.save_dir / 'training_curves.png'}")
        except Exception as e:
            logger.warning(f"Could not plot history: {e}")


def train_emotion_model(data_dir: str = None, config: TrainingConfig = None):
    """Train emotion recognition model."""
    logger.info("=" * 60)
    logger.info("Training Emotion Recognition Model")
    logger.info("=" * 60)

    if config is None:
        config = TrainingConfig()

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
        transforms.RandomRotation(10),
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

    train_dataset = FaceDataset(synthetic_dir, transform=transform_train, task="emotion")
    val_dataset = FaceDataset(val_dir, transform=transform_val, task="emotion")

    if len(train_dataset) == 0:
        logger.error("No training data found. Run download_datasets.py --synthetic first.")
        return None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Classes: {train_dataset.label_to_idx}")

    model = EmotionNet(num_classes=len(train_dataset.label_to_idx))
    trainer = Trainer(model, config, save_dir=MODELS_DIR)

    history = trainer.train(train_loader, val_loader)
    trainer.save_history()
    trainer.plot_history()

    model_path = MODELS_DIR / "emotion_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Emotion model saved to {model_path}")

    return model


def train_age_gender_model(data_dir: str = None, config: TrainingConfig = None):
    """Train age and gender estimation model."""
    logger.info("=" * 60)
    logger.info("Training Age/Gender Estimation Model")
    logger.info("=" * 60)

    if config is None:
        config = TrainingConfig(image_size=64, epochs=30)

    model = AgeGenderNet()
    logger.info(f"Model: {model}")
    logger.info("Age/Gender training requires labeled dataset with age and gender annotations.")
    logger.info("Using CelebA or similar dataset recommended.")
    logger.info(f"Saving placeholder model to {MODELS_DIR / 'age_gender_model.pth'}")

    torch.save(model.state_dict(), MODELS_DIR / "age_gender_model.pth")
    return model


def evaluate_model(model_path: str, test_dir: str, task: str = "emotion"):
    """Evaluate a trained model on test data."""
    logger.info(f"Evaluating model: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if task == "emotion":
        model = EmotionNet(num_classes=7)
    else:
        model = AgeGenderNet()

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48) if task == "emotion" else (64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = FaceDataset(test_dir, transform=transform, task=task)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            if task == "emotion":
                _, predicted = outputs.max(1)
            else:
                _, predicted = outputs[1].max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = (all_preds == all_targets).mean()
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(all_targets, all_preds))
    print(confusion_matrix(all_targets, all_preds))

    return accuracy


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train face models for OMNIVIS")
    parser.add_argument("--task", type=str, default="emotion", choices=["emotion", "age_gender", "both"], help="Training task")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--evaluate", type=str, default=None, help="Evaluate a trained model (path to checkpoint)")
    parser.add_argument("--test-dir", type=str, default=None, help="Test dataset directory for evaluation")

    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )

    if args.evaluate:
        test_dir = args.test_dir or str(DATA_DIR / "synthetic" / "test")
        evaluate_model(args.evaluate, test_dir)
        return

    if args.task in ["emotion", "both"]:
        train_emotion_model(args.data_dir, config)

    if args.task in ["age_gender", "both"]:
        train_age_gender_model(args.data_dir, config)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
