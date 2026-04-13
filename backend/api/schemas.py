"""
OMNIVIS — Pydantic Schemas
Request/response models for all API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


# ── Enums ──────────────────────────────────────────────────────
class UserRoleEnum(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"

class SourceTypeEnum(str, Enum):
    WEBCAM = "webcam"
    RTSP = "rtsp"
    YOUTUBE = "youtube"
    FILE_IMAGE = "file_image"
    FILE_VIDEO = "file_video"

class AlertSeverityEnum(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"

class ModuleNameEnum(str, Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    FACE = "face"
    OPTICAL_FLOW = "optical_flow"
    DEPTH = "depth"
    RECONSTRUCTION = "reconstruction"
    TRACKING = "tracking"
    SCENE_GRAPH = "scene_graph"
    TRAJECTORY = "trajectory"
    ANOMALY = "anomaly"
    GAIT = "gait"
    ACTION = "action"
    GAN = "gan"


# ── Auth Schemas ───────────────────────────────────────────────
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 900  # 15 minutes

class TokenPayload(BaseModel):
    sub: str
    role: Optional[str] = None
    exp: int
    type: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    full_name: Optional[str] = None

class OAuthCallback(BaseModel):
    code: str
    provider: str  # google or github


# ── User Schemas ───────────────────────────────────────────────
class UserOut(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    role: UserRoleEnum
    tier: str = "free"
    avatar_url: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ── Session Schemas ────────────────────────────────────────────
class SessionCreate(BaseModel):
    source_type: SourceTypeEnum
    source_url: Optional[str] = None
    active_modules: List[str] = ["detection"]
    config: Dict[str, Any] = {}

class SessionOut(BaseModel):
    id: str
    source_type: SourceTypeEnum
    started_at: datetime
    ended_at: Optional[datetime] = None
    frame_count: int = 0
    total_detections: int = 0
    avg_fps: Optional[float] = None
    active_modules: List[str] = []

    class Config:
        from_attributes = True


# ── Detection Schemas ──────────────────────────────────────────
class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class DetectionOut(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: BBox
    track_id: Optional[int] = None
    mask: Optional[str] = None  # Base64 encoded RLE mask

class FrameDetections(BaseModel):
    frame_id: int
    timestamp: float
    detections: List[DetectionOut] = []
    fps: float = 0.0


# ── Module Config Schemas ──────────────────────────────────────
class ModuleConfig(BaseModel):
    enabled: bool = True
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1, le=1000)
    model_variant: Optional[str] = None  # e.g., "yolov8x", "yolov8s"

class PipelineConfig(BaseModel):
    modules: Dict[str, ModuleConfig] = {}
    input_resolution: str = "640x640"
    output_format: str = "jpeg"  # jpeg, webp, png
    output_quality: int = Field(default=85, ge=1, le=100)
    show_labels: bool = True
    show_confidence: bool = True
    show_tracking: bool = True


# ── Model Switch Schema ───────────────────────────────────────
class ModelSwitchRequest(BaseModel):
    module: ModuleNameEnum
    model_name: str
    variant: Optional[str] = None


# ── Metrics Schemas ────────────────────────────────────────────
class LiveMetrics(BaseModel):
    timestamp: float
    fps: float = 0.0
    inference_ms: float = 0.0
    gpu_util: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_util: float = 0.0
    ram_mb: float = 0.0
    map_50: Optional[float] = None
    iou: Optional[float] = None
    epe: Optional[float] = None
    ssim: Optional[float] = None
    detection_count: int = 0
    tracking_count: int = 0


# ── Anomaly Schemas ────────────────────────────────────────────
class AnomalyAlert(BaseModel):
    id: str
    timestamp: float
    anomaly_type: str
    severity: AlertSeverityEnum
    score: float
    description: str
    bbox: Optional[BBox] = None
    frame_id: Optional[int] = None


# ── Calibration Schemas ───────────────────────────────────────
class CalibrationRequest(BaseModel):
    pattern_type: str = "chessboard"  # chessboard, circles, asymmetric_circles
    pattern_size: List[int] = [9, 6]  # columns, rows
    square_size_mm: float = 25.0

class CalibrationResult(BaseModel):
    success: bool
    rms_error: Optional[float] = None
    camera_matrix: Optional[List[List[float]]] = None
    distortion_coeffs: Optional[List[float]] = None
    message: str = ""


# ── WebSocket Message Schema ──────────────────────────────────
class WSMessage(BaseModel):
    type: str  # frame, config, command
    data: Dict[str, Any] = {}


# ── Report Schema ─────────────────────────────────────────────
class ReportRequest(BaseModel):
    session_id: str
    include_charts: bool = True
    include_detections: bool = True
    include_anomalies: bool = True


# ── Health Check ──────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    active_sessions: int = 0
    uptime_seconds: float = 0.0
    modules_loaded: List[str] = []
