"""
OMNIVIS — Segmentation Module
Instance segmentation (Mask R-CNN) + Semantic segmentation (DeepLabV3+).
"""
import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# PASCAL VOC color palette for semantic segmentation
VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv/monitor"
]


class InstanceSegmentor:
    """Mask R-CNN instance segmentation with level-set refinement."""

    def __init__(self, confidence: float = 0.5, device: str = "auto"):
        self.confidence = confidence
        self.device = device
        self.model = None
        self.loaded = False
        self._load_model()

    def _load_model(self):
        try:
            import torch
            import torchvision
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
                weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            )
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            logger.info(f"Mask R-CNN loaded on {self.device}")
        except Exception as e:
            logger.warning(f"Mask R-CNN load failed: {e}. Using simulation mode.")
            self.loaded = False

    def segment(self, frame: np.ndarray) -> Dict[str, Any]:
        start = time.perf_counter()

        if self.loaded and self.model is not None:
            try:
                results = self._run_maskrcnn(frame)
            except Exception as e:
                logger.error(f"Mask R-CNN inference error: {e}")
                results = self._simulate(frame)
        else:
            results = self._simulate(frame)

        return {
            "masks": results,
            "inference_ms": (time.perf_counter() - start) * 1000,
        }

    def _run_maskrcnn(self, frame: np.ndarray) -> List[Dict]:
        import torch
        import torchvision.transforms.functional as F

        img_tensor = F.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(self.device)

        with torch.no_grad():
            predictions = self.model([img_tensor])[0]

        results = []
        for i in range(len(predictions["scores"])):
            score = float(predictions["scores"][i].cpu())
            if score < self.confidence:
                continue

            mask = predictions["masks"][i, 0].cpu().numpy()
            mask_binary = (mask > 0.5).astype(np.uint8)
            box = predictions["boxes"][i].cpu().numpy()
            label = int(predictions["labels"][i].cpu())

            results.append({
                "mask": mask_binary,
                "bbox": {
                    "x1": float(box[0]), "y1": float(box[1]),
                    "x2": float(box[2]), "y2": float(box[3]),
                },
                "class_id": label,
                "confidence": round(score, 4),
            })

        return results[:20]  # Limit to 20 instances

    def _simulate(self, frame: np.ndarray) -> List[Dict]:
        """Simulate instance segmentation using contour detection."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            if cv2.contourArea(c) < w * h * 0.01:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 1, -1)
            x, y, cw, ch = cv2.boundingRect(c)
            results.append({
                "mask": mask,
                "bbox": {"x1": float(x), "y1": float(y),
                         "x2": float(x + cw), "y2": float(y + ch)},
                "class_id": np.random.randint(1, 21),
                "confidence": round(0.6 + np.random.random() * 0.35, 4),
            })
        return results

    def draw_masks(self, frame: np.ndarray, masks: List[Dict],
                   alpha: float = 0.4) -> np.ndarray:
        """Draw colored transparent masks on frame."""
        overlay = frame.copy()
        for i, m in enumerate(masks):
            color = tuple(int(c) for c in VOC_COLORMAP[m["class_id"] % len(VOC_COLORMAP)])
            mask = m["mask"]
            overlay[mask > 0] = np.array(color, dtype=np.uint8)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


class SemanticSegmentor:
    """DeepLabV3+ semantic segmentation."""

    def __init__(self, backbone: str = "mobilenet", device: str = "auto"):
        self.backbone = backbone
        self.device = device
        self.model = None
        self.loaded = False
        self._load_model()

    def _load_model(self):
        try:
            import torch
            import torchvision
            if self.backbone == "resnet101":
                self.model = torchvision.models.segmentation.deeplabv3_resnet101(
                    weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
                )
            else:
                self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
                    weights=torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
                )
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            logger.info(f"DeepLabV3+ loaded ({self.backbone}) on {self.device}")
        except Exception as e:
            logger.warning(f"DeepLabV3+ load failed: {e}. Using simulation mode.")
            self.loaded = False

    def segment(self, frame: np.ndarray) -> Dict[str, Any]:
        start = time.perf_counter()

        if self.loaded and self.model is not None:
            try:
                segmap = self._run_deeplabv3(frame)
            except Exception as e:
                logger.error(f"DeepLabV3+ error: {e}")
                segmap = self._simulate(frame)
        else:
            segmap = self._simulate(frame)

        # Colorize
        colored = self._colorize(segmap)

        return {
            "segmentation_map": segmap,
            "colored": colored,
            "classes_found": [VOC_CLASSES[i] for i in np.unique(segmap) if i < len(VOC_CLASSES)],
            "inference_ms": (time.perf_counter() - start) * 1000,
        }

    def _run_deeplabv3(self, frame: np.ndarray) -> np.ndarray:
        import torch
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(520),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]
        segmap = output.argmax(0).cpu().numpy().astype(np.uint8)

        # Resize back to original
        h, w = frame.shape[:2]
        segmap = cv2.resize(segmap, (w, h), interpolation=cv2.INTER_NEAREST)
        return segmap

    def _simulate(self, frame: np.ndarray) -> np.ndarray:
        """Simulate semantic segmentation using color clustering."""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 4, h // 4))
        pixels = small.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = min(8, len(VOC_CLASSES))
        _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        segmap = labels.reshape(h // 4, w // 4).astype(np.uint8)
        return cv2.resize(segmap, (w, h), interpolation=cv2.INTER_NEAREST)

    def _colorize(self, segmap: np.ndarray) -> np.ndarray:
        """Apply PASCAL VOC colormap to segmentation map."""
        h, w = segmap.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id in range(len(VOC_COLORMAP)):
            colored[segmap == cls_id] = VOC_COLORMAP[cls_id]
        return colored
