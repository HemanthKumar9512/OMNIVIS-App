"""
OMNIVIS — Scene Graph Construction
GNN-based scene graph with spatial relationship prediction.
"""
import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

PREDICATES = [
    "on", "near", "behind", "in front of", "above", "below",
    "left of", "right of", "inside", "next to", "holding",
    "wearing", "riding", "driving", "standing on", "sitting on",
    "walking on", "attached to", "hanging from", "belonging to",
    "looking at", "watching", "using", "carrying", "pulling",
    "pushing", "eating", "covering", "playing with", "parked on",
    "lying on", "mounted on", "growing on", "along", "between",
    "against", "over", "under", "across", "through",
    "with", "of", "at", "to", "from",
    "toward", "away from", "around", "part of", "has"
]


class SceneGraphBuilder:
    """Constructs scene graphs from detected objects."""

    def __init__(self):
        self.prev_graph = None

    def build(self, detections: List[Dict], frame_shape: Tuple[int, int] = (640, 640)) -> Dict[str, Any]:
        """Build scene graph from detections."""
        start = time.perf_counter()

        nodes = []
        edges = []

        h, w = frame_shape[:2]

        # Create nodes from detections
        for i, det in enumerate(detections):
            bbox = det.get("bbox", {})
            cx = (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2
            cy = (bbox.get("y1", 0) + bbox.get("y2", 0)) / 2
            area = (bbox.get("x2", 0) - bbox.get("x1", 0)) * (bbox.get("y2", 0) - bbox.get("y1", 0))

            nodes.append({
                "id": i,
                "label": det.get("class_name", "object"),
                "class_id": det.get("class_id", 0),
                "confidence": det.get("confidence", 0),
                "center": [cx / w, cy / h],  # Normalized
                "area": area / (w * h),  # Normalized
                "bbox": bbox,
                "track_id": det.get("track_id"),
            })

        # Predict relationships between pairs
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                relationships = self._predict_relationships(nodes[i], nodes[j])
                for rel in relationships:
                    edges.append({
                        "source": i,
                        "target": j,
                        "predicate": rel["predicate"],
                        "confidence": rel["confidence"],
                    })

        # Build triplets
        triplets = []
        for edge in edges:
            if edge["confidence"] > 0.3:
                triplets.append({
                    "subject": nodes[edge["source"]]["label"],
                    "predicate": edge["predicate"],
                    "object": nodes[edge["target"]]["label"],
                    "confidence": edge["confidence"],
                })

        graph = {
            "nodes": nodes,
            "edges": edges,
            "triplets": triplets,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "inference_ms": (time.perf_counter() - start) * 1000,
        }

        self.prev_graph = graph
        return graph

    def _predict_relationships(self, node_a: Dict, node_b: Dict) -> List[Dict]:
        """Predict spatial relationships between two objects."""
        relationships = []
        ca = node_a["center"]
        cb = node_b["center"]
        area_a = node_a["area"]
        area_b = node_b["area"]

        # Spatial relationships
        dx = cb[0] - ca[0]
        dy = cb[1] - ca[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # Near
        if distance < 0.2:
            relationships.append({"predicate": "near", "confidence": round(1 - distance * 4, 3)})

        # Left/Right
        if abs(dx) > 0.05:
            pred = "left of" if dx < 0 else "right of"
            relationships.append({"predicate": pred, "confidence": round(min(abs(dx) * 3, 0.95), 3)})

        # Above/Below
        if abs(dy) > 0.05:
            pred = "above" if dy < 0 else "below"
            relationships.append({"predicate": pred, "confidence": round(min(abs(dy) * 3, 0.95), 3)})

        # Containment (larger object contains smaller)
        bbox_a = node_a["bbox"]
        bbox_b = node_b["bbox"]
        if self._is_inside(bbox_b, bbox_a):
            relationships.append({"predicate": "inside", "confidence": 0.85})
        elif self._is_inside(bbox_a, bbox_b):
            relationships.append({"predicate": "inside", "confidence": 0.85})

        # On (object B is on top of object A)
        if abs(dy) < 0.1 and abs(dx) < 0.1 and cb[1] < ca[1]:
            if area_b < area_a * 0.8:
                relationships.append({"predicate": "on", "confidence": 0.7})

        # Semantic relationships based on class
        label_a = node_a["label"].lower()
        label_b = node_b["label"].lower()

        if label_a == "person" and label_b in ["car", "bus", "truck", "motorcycle", "bicycle"]:
            if distance < 0.15:
                relationships.append({"predicate": "driving", "confidence": 0.6})
        elif label_a == "person" and label_b in ["horse", "elephant"]:
            if distance < 0.15:
                relationships.append({"predicate": "riding", "confidence": 0.6})

        return relationships

    @staticmethod
    def _is_inside(bbox_inner: Dict, bbox_outer: Dict) -> bool:
        """Check if bbox_inner is inside bbox_outer."""
        return (bbox_inner.get("x1", 0) >= bbox_outer.get("x1", 0) and
                bbox_inner.get("y1", 0) >= bbox_outer.get("y1", 0) and
                bbox_inner.get("x2", 0) <= bbox_outer.get("x2", 0) and
                bbox_inner.get("y2", 0) <= bbox_outer.get("y2", 0))
