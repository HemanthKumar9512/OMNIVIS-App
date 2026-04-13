"""
OMNIVIS — Main FastAPI Application
WebSocket real-time inference + REST API + Module Pipeline Manager.
"""
import os
import sys
import json
import time
import asyncio
import base64
import logging
from datetime import datetime
from typing import Dict, Set, Any, Optional
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("omnivis")

# ── Pipeline Manager ──────────────────────────────────────────
class PipelineManager:
    """Manages all CV/ML modules and orchestrates inference pipeline."""

    def __init__(self):
        self.modules: Dict[str, Any] = {}
        self.enabled: Dict[str, bool] = {}
        self.configs: Dict[str, dict] = {}
        self.initialized = False

    def initialize(self):
        """Lazy-load all modules."""
        if self.initialized:
            return

        logger.info("Initializing OMNIVIS pipeline modules...")

        try:
            from modules.detection import ObjectDetector
            self.modules["detection"] = ObjectDetector()
            self.enabled["detection"] = True
            logger.info("✓ Detection module loaded")
        except Exception as e:
            logger.error(f"✗ Detection module failed: {e}")

        try:
            from modules.segmentation import InstanceSegmentor, SemanticSegmentor
            self.modules["instance_seg"] = InstanceSegmentor()
            self.modules["semantic_seg"] = SemanticSegmentor()
            self.enabled["segmentation"] = True
            logger.info("✓ Segmentation modules loaded")
        except Exception as e:
            logger.error(f"✗ Segmentation module failed: {e}")

        try:
            from modules.face import FaceAnalyzer
            self.modules["face"] = FaceAnalyzer()
            self.enabled["face"] = True
            logger.info("✓ Face analysis module loaded")
        except Exception as e:
            logger.error(f"✗ Face module failed: {e}")

        try:
            from modules.optical_flow import OpticalFlowEngine
            self.modules["optical_flow"] = OpticalFlowEngine()
            self.enabled["optical_flow"] = True
            logger.info("✓ Optical flow module loaded")
        except Exception as e:
            logger.error(f"✗ Optical flow module failed: {e}")

        try:
            from modules.depth import DepthEstimator
            self.modules["depth"] = DepthEstimator()
            self.enabled["depth"] = True
            logger.info("✓ Depth estimation module loaded")
        except Exception as e:
            logger.error(f"✗ Depth module failed: {e}")

        try:
            from modules.reconstruction import SfMReconstructor
            self.modules["reconstruction"] = SfMReconstructor()
            self.enabled["reconstruction"] = False  # Off by default
            logger.info("✓ 3D reconstruction module loaded")
        except Exception as e:
            logger.error(f"✗ Reconstruction module failed: {e}")

        try:
            from modules.tracking import ByteTracker
            self.modules["tracking"] = ByteTracker()
            self.enabled["tracking"] = True
            logger.info("✓ Tracking module loaded")
        except Exception as e:
            logger.error(f"✗ Tracking module failed: {e}")

        try:
            from modules.scene_graph import SceneGraphBuilder
            self.modules["scene_graph"] = SceneGraphBuilder()
            self.enabled["scene_graph"] = True
            logger.info("✓ Scene graph module loaded")
        except Exception as e:
            logger.error(f"✗ Scene graph module failed: {e}")

        try:
            from modules.trajectory import TrajectoryPredictor
            self.modules["trajectory"] = TrajectoryPredictor()
            self.enabled["trajectory"] = True
            logger.info("✓ Trajectory prediction module loaded")
        except Exception as e:
            logger.error(f"✗ Trajectory module failed: {e}")

        try:
            from modules.anomaly import AnomalyDetector
            self.modules["anomaly"] = AnomalyDetector()
            self.enabled["anomaly"] = True
            logger.info("✓ Anomaly detection module loaded")
        except Exception as e:
            logger.error(f"✗ Anomaly module failed: {e}")

        try:
            from modules.gait import GaitAnalyzer
            self.modules["gait"] = GaitAnalyzer()
            self.enabled["gait"] = False  # Off by default
            logger.info("✓ Gait analysis module loaded")
        except Exception as e:
            logger.error(f"✗ Gait module failed: {e}")

        try:
            from modules.action import ActionRecognizer
            self.modules["action"] = ActionRecognizer()
            self.enabled["action"] = False  # Off by default
            logger.info("✓ Action recognition module loaded")
        except Exception as e:
            logger.error(f"✗ Action module failed: {e}")

        self.initialized = True
        logger.info(f"Pipeline initialized: {sum(1 for v in self.enabled.values() if v)}/{len(self.enabled)} modules active")

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run all enabled modules on a single frame."""
        if not self.initialized:
            self.initialize()

        results = {"timestamp": time.time(), "frame_shape": frame.shape[:2]}
        total_start = time.perf_counter()

        # Module 1: Detection
        detections = []
        if self.enabled.get("detection"):
            det_result = self.modules["detection"].detect(frame)
            detections = det_result.get("detections", [])
            results["detection"] = {
                "detections": detections,
                "count": len(detections),
                "inference_ms": det_result.get("inference_ms", 0),
            }

        # Module 1B/C: Segmentation
        if self.enabled.get("segmentation"):
            if "semantic_seg" in self.modules:
                seg_result = self.modules["semantic_seg"].segment(frame)
                results["segmentation"] = {
                    "classes_found": seg_result.get("classes_found", []),
                    "inference_ms": seg_result.get("inference_ms", 0),
                }

        # Module 1D: Face Analysis
        if self.enabled.get("face"):
            face_result = self.modules["face"].analyze(frame)
            results["face"] = {
                "faces": face_result.get("faces", []),
                "face_count": face_result.get("face_count", 0),
                "inference_ms": face_result.get("inference_ms", 0),
            }

        # Module 2A: Optical Flow
        flow_magnitude = 0.0
        if self.enabled.get("optical_flow"):
            flow_result = self.modules["optical_flow"].compute_flow(frame)
            flow_magnitude = flow_result.get("mean_magnitude", 0)
            results["optical_flow"] = {
                "mean_magnitude": flow_result.get("mean_magnitude", 0),
                "max_magnitude": flow_result.get("max_magnitude", 0),
                "method": flow_result.get("method", "none"),
                "inference_ms": flow_result.get("inference_ms", 0),
            }

        # Module 2B: Depth
        if self.enabled.get("depth"):
            depth_result = self.modules["depth"].estimate(frame)
            results["depth"] = {
                "min_depth": depth_result.get("min_depth", 0),
                "max_depth": depth_result.get("max_depth", 0),
                "mean_depth": depth_result.get("mean_depth", 0),
                "model": depth_result.get("model", "none"),
                "inference_ms": depth_result.get("inference_ms", 0),
            }

        # Module 3: Tracking
        tracks = []
        if self.enabled.get("tracking") and detections:
            tracks = self.modules["tracking"].update(detections)
            # Assign track IDs back to detections
            for track in tracks:
                for det in detections:
                    det_bbox = det.get("bbox", {})
                    trk_bbox = track.get("bbox", {})
                    # Simple IoU-based matching
                    if (abs(det_bbox.get("x1", 0) - trk_bbox.get("x1", 0)) < 30 and
                        abs(det_bbox.get("y1", 0) - trk_bbox.get("y1", 0)) < 30):
                        det["track_id"] = track["track_id"]
                        break

            results["tracking"] = {
                "tracks": tracks,
                "track_count": len(tracks),
                "trails": {str(k): v for k, v in self.modules["tracking"].get_trails().items()},
            }

        # Module 3: Scene Graph
        if self.enabled.get("scene_graph") and detections:
            graph = self.modules["scene_graph"].build(detections, frame.shape[:2])
            results["scene_graph"] = {
                "nodes": graph.get("nodes", []),
                "edges": graph.get("edges", []),
                "triplets": graph.get("triplets", []),
            }

        # Module 3: Trajectory Prediction
        if self.enabled.get("trajectory") and tracks:
            traj_result = self.modules["trajectory"].update(tracks)
            results["trajectory"] = {
                "predictions": {
                    str(k): v for k, v in traj_result.get("predictions", {}).items()
                },
                "inference_ms": traj_result.get("inference_ms", 0),
            }

        # Module 3: Anomaly Detection
        if self.enabled.get("anomaly"):
            anom_result = self.modules["anomaly"].detect(
                detections, tracks, flow_magnitude, frame
            )
            results["anomaly"] = {
                "alert_level": anom_result.get("alert_level", "green"),
                "anomalies": anom_result.get("anomalies", []),
                "overall_score": anom_result.get("overall_score", 0),
            }

        # Module: Gait Analysis
        if self.enabled.get("gait"):
            gait_result = self.modules["gait"].analyze(frame, tracks)
            results["gait"] = {
                "persons": [
                    {k: v for k, v in p.items() if k != "landmarks"}
                    for p in gait_result.get("persons", [])
                ],
                "person_count": gait_result.get("person_count", 0),
            }

        # Module: Action Recognition
        if self.enabled.get("action"):
            action_result = self.modules["action"].process_frame(frame)
            results["action"] = {
                "actions": action_result.get("actions", []),
                "buffer_fill": action_result.get("buffer_fill", 0),
            }

        # 3D Reconstruction
        if self.enabled.get("reconstruction"):
            recon_result = self.modules["reconstruction"].add_frame(frame)
            results["reconstruction"] = {
                "keypoints_count": recon_result.get("keypoints_count", 0),
                "matches_count": recon_result.get("matches_count", 0),
                "total_points": recon_result.get("total_points", 0),
            }

        total_ms = (time.perf_counter() - total_start) * 1000
        results["total_inference_ms"] = round(total_ms, 1)
        results["fps"] = round(1000 / max(total_ms, 1), 1)

        return results

    def switch_model(self, module: str, model_name: str, variant: Optional[str] = None):
        """Hot-swap a module's model."""
        if module == "detection" and "detection" in self.modules:
            self.modules["detection"].update_config(model_variant=model_name)
        elif module == "depth" and "depth" in self.modules:
            from modules.depth import DepthEstimator
            self.modules["depth"] = DepthEstimator(model_type=model_name)
        logger.info(f"Model switched: {module} → {model_name}")

    def set_module_enabled(self, module: str, enabled: bool):
        """Enable or disable a module."""
        if module in self.enabled:
            self.enabled[module] = enabled
            logger.info(f"Module {module}: {'enabled' if enabled else 'disabled'}")

    def update_config(self, module: str, config: dict):
        """Update module configuration."""
        if module == "detection" and "detection" in self.modules:
            self.modules["detection"].update_config(**config)
        self.configs[module] = config

    def get_status(self) -> Dict:
        return {
            "modules": {
                name: {
                    "enabled": self.enabled.get(name, False),
                    "loaded": name in self.modules,
                }
                for name in ["detection", "segmentation", "face", "optical_flow",
                             "depth", "reconstruction", "tracking", "scene_graph",
                             "trajectory", "anomaly", "gait", "action"]
            }
        }


# ── Global Pipeline Manager ──────────────────────────────────
pipeline_manager = PipelineManager()


# ── Connection Manager ────────────────────────────────────────
class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "stream": set(),
            "alerts": set(),
            "metrics": set(),
        }

    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        self.active_connections[channel].add(websocket)
        logger.info(f"WS connected: {channel} ({len(self.active_connections[channel])} total)")

    def disconnect(self, websocket: WebSocket, channel: str):
        self.active_connections[channel].discard(websocket)
        logger.info(f"WS disconnected: {channel}")

    async def broadcast(self, channel: str, message: dict):
        dead = set()
        for conn in self.active_connections[channel]:
            try:
                await conn.send_json(message)
            except Exception:
                dead.add(conn)
        self.active_connections[channel] -= dead

    def get_counts(self) -> Dict[str, int]:
        return {ch: len(conns) for ch, conns in self.active_connections.items()}


ws_manager = ConnectionManager()


# ── Lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("=" * 60)
    logger.info("  OMNIVIS — Omniscient Vision Intelligence System")
    logger.info("  Starting up...")
    logger.info("=" * 60)

    # Initialize pipeline in background
    pipeline_manager.initialize()

    # Initialize database
    try:
        from db.session import init_db
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database init skipped: {e}")

    yield

    logger.info("OMNIVIS shutting down...")
    try:
        from db.session import close_db
        await close_db()
    except Exception:
        pass


# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="OMNIVIS — Omniscient Vision Intelligence System",
    description="Production-grade real-time autonomous perception engine",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST routes
from api.routes import router as api_router
app.include_router(api_router, prefix="/api")

# Ensure uploads directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), "uploads"), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)


# ── WebSocket: Main Stream ───────────────────────────────────
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Main real-time inference WebSocket channel."""
    await ws_manager.connect(websocket, "stream")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type", "")

            if msg_type == "frame":
                # Decode base64 frame
                frame_data = message.get("data", {}).get("frame", "")
                if not frame_data:
                    continue

                # Decode image
                try:
                    img_bytes = base64.b64decode(frame_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                except Exception as e:
                    logger.error(f"Frame decode error: {e}")
                    continue

                # Process through pipeline
                results = pipeline_manager.process_frame(frame)

                # Annotate frame
                from utils.encoder import FrameEncoder
                encoder = FrameEncoder()
                annotated = frame.copy()

                # Draw detection boxes
                if "detection" in results:
                    annotated = encoder.draw_detections(
                        annotated, results["detection"].get("detections", [])
                    )

                # Draw tracking trails
                if "tracking" in results:
                    trails = results["tracking"].get("trails", {})
                    annotated = encoder.draw_tracks(annotated, trails)

                # Encode annotated frame
                annotated_b64 = encoder.encode_to_base64(annotated, quality=80)

                # Build response
                response = {
                    "type": "result",
                    "data": {
                        "annotated_frame": annotated_b64,
                        **{k: v for k, v in results.items()
                           if k not in ["frame_shape"]},
                    }
                }

                await websocket.send_json(response)

                # Broadcast anomaly alerts
                if results.get("anomaly", {}).get("alert_level") in ["yellow", "red"]:
                    await ws_manager.broadcast("alerts", {
                        "type": "alert",
                        "data": results["anomaly"],
                        "timestamp": time.time(),
                    })

                # Broadcast metrics
                await ws_manager.broadcast("metrics", {
                    "type": "metrics",
                    "data": {
                        "fps": results.get("fps", 0),
                        "inference_ms": results.get("total_inference_ms", 0),
                        "detection_count": results.get("detection", {}).get("count", 0),
                        "track_count": results.get("tracking", {}).get("track_count", 0),
                        "alert_level": results.get("anomaly", {}).get("alert_level", "green"),
                        "timestamp": time.time(),
                    }
                })

            elif msg_type == "config":
                # Update pipeline configuration
                config_data = message.get("data", {})
                module = config_data.get("module")
                if module:
                    if "enabled" in config_data:
                        pipeline_manager.set_module_enabled(module, config_data["enabled"])
                    if "config" in config_data:
                        pipeline_manager.update_config(module, config_data["config"])

                await websocket.send_json({
                    "type": "config_ack",
                    "data": pipeline_manager.get_status(),
                })

            elif msg_type == "status":
                await websocket.send_json({
                    "type": "status",
                    "data": {
                        **pipeline_manager.get_status(),
                        "connections": ws_manager.get_counts(),
                    }
                })

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, "stream")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket, "stream")


# ── WebSocket: Alerts ─────────────────────────────────────────
@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """Anomaly alert broadcast channel."""
    await ws_manager.connect(websocket, "alerts")
    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, "alerts")


# ── WebSocket: Metrics ────────────────────────────────────────
@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """System performance metrics stream."""
    await ws_manager.connect(websocket, "metrics")
    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, "metrics")


@app.get("/")
async def root():
    return {
        "name": "OMNIVIS",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }
