"""
OMNIVIS — Database ORM Models
SQLAlchemy models — works with SQLite (dev) and PostgreSQL (production).
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    ForeignKey, Text, JSON, Enum as SAEnum
)
from sqlalchemy.orm import relationship, declarative_base
import enum


def gen_uuid():
    return str(uuid.uuid4())


Base = declarative_base()


class UserRole(str, enum.Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"


class AlertSeverity(str, enum.Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class SourceType(str, enum.Enum):
    WEBCAM = "webcam"
    RTSP = "rtsp"
    YOUTUBE = "youtube"
    FILE_IMAGE = "file_image"
    FILE_VIDEO = "file_video"


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=True)  # Null for OAuth users
    full_name = Column(String(255), nullable=True)
    role = Column(SAEnum(UserRole), default=UserRole.VIEWER, nullable=False)
    api_key = Column(String(64), unique=True, nullable=True, index=True)
    api_key_hash = Column(String(255), nullable=True)
    oauth_provider = Column(String(50), nullable=True)  # google, github
    oauth_id = Column(String(255), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)
    tier = Column(String(20), default="free")  # free, pro, enterprise
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    source_type = Column(SAEnum(SourceType), nullable=False)
    source_url = Column(String(1000), nullable=True)
    resolution = Column(String(20), nullable=True)  # e.g., "1920x1080"
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    frame_count = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    avg_fps = Column(Float, nullable=True)
    active_modules = Column(JSON, default=list)  # List of enabled module names
    config = Column(JSON, default=dict)  # Session-specific settings
    is_recording = Column(Boolean, default=False)
    recording_path = Column(String(500), nullable=True)

    user = relationship("User", back_populates="sessions")
    detections = relationship("Detection", back_populates="session", cascade="all, delete-orphan")
    anomalies = relationship("Anomaly", back_populates="session", cascade="all, delete-orphan")
    metrics = relationship("Metric", back_populates="session", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False)
    frame_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    class_name = Column(String(100), nullable=False)
    class_id = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    track_id = Column(Integer, nullable=True)
    mask_rle = Column(Text, nullable=True)  # Run-length encoded mask
    attributes = Column(JSON, nullable=True)  # Extra per-detection data

    session = relationship("Session", back_populates="detections")


class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False)
    frame_id = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    anomaly_type = Column(String(100), nullable=False)  # motion, appearance, trajectory
    severity = Column(SAEnum(AlertSeverity), nullable=False)
    score = Column(Float, nullable=False)
    description = Column(Text, nullable=True)
    bbox = Column(JSON, nullable=True)  # {x1, y1, x2, y2}
    alert_sent = Column(Boolean, default=False)
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(36), nullable=True)

    session = relationship("Session", back_populates="anomalies")


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    map_50 = Column(Float, nullable=True)  # mAP@0.5
    map_50_95 = Column(Float, nullable=True)  # mAP@0.5:0.95
    iou = Column(Float, nullable=True)
    epe = Column(Float, nullable=True)  # End Point Error
    ssim = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    inference_ms = Column(Float, nullable=True)
    gpu_util = Column(Float, nullable=True)  # 0-100%
    gpu_memory_mb = Column(Float, nullable=True)
    cpu_util = Column(Float, nullable=True)
    ram_mb = Column(Float, nullable=True)
    detection_count = Column(Integer, nullable=True)
    tracking_count = Column(Integer, nullable=True)

    session = relationship("Session", back_populates="metrics")


class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(100), nullable=False, unique=True)
    module = Column(String(50), nullable=False)  # detection, segmentation, etc.
    version = Column(String(20), nullable=False)
    path = Column(String(500), nullable=False)
    framework = Column(String(50), nullable=True)  # pytorch, onnx, tensorrt
    active = Column(Boolean, default=False)
    accuracy = Column(Float, nullable=True)
    input_size = Column(String(20), nullable=True)  # e.g., "640x640"
    params_millions = Column(Float, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
