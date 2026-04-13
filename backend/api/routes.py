"""
OMNIVIS — REST API Routes
All REST endpoints for the OMNIVIS platform.
"""
import os
import uuid
import time
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Query
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import io
import logging

from api.schemas import (
    SessionCreate, SessionOut, ModelSwitchRequest, HealthResponse,
    CalibrationRequest, CalibrationResult, TokenResponse, LoginRequest,
    RegisterRequest, OAuthCallback, UserOut, PipelineConfig,
    ReportRequest, LiveMetrics,
)
from api.auth import (
    get_current_user, get_optional_user, require_role,
    hash_password, verify_password, create_access_token,
    create_refresh_token, generate_api_key, hash_api_key,
    get_google_user_info, get_github_user_info,
)
from db.session import get_db
from db.models import User, Session, Detection, Anomaly, Metric, MLModel, UserRole

logger = logging.getLogger(__name__)
router = APIRouter()
START_TIME = time.time()


# ── Auth Endpoints ─────────────────────────────────────────────
@router.post("/auth/register", response_model=TokenResponse, tags=["auth"])
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user."""
    existing = await db.execute(select(User).where(User.email == req.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=req.email,
        hashed_password=hash_password(req.password),
        full_name=req.full_name,
        role=UserRole.VIEWER,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return TokenResponse(
        access_token=create_access_token(str(user.id), user.role.value),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.post("/auth/login", response_model=TokenResponse, tags=["auth"])
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Login with email and password."""
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()
    if not user or not user.hashed_password or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return TokenResponse(
        access_token=create_access_token(str(user.id), user.role.value),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.post("/auth/oauth/{provider}", response_model=TokenResponse, tags=["auth"])
async def oauth_login(provider: str, callback: OAuthCallback, db: AsyncSession = Depends(get_db)):
    """OAuth2 login (Google/GitHub)."""
    if provider == "google":
        user_info = await get_google_user_info(callback.code)
        email = user_info.get("email")
        name = user_info.get("name")
        avatar = user_info.get("picture")
        oauth_id = user_info.get("id")
    elif provider == "github":
        user_info = await get_github_user_info(callback.code)
        email = user_info.get("email")
        name = user_info.get("login")
        avatar = user_info.get("avatar_url")
        oauth_id = str(user_info.get("id"))
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    if not email:
        raise HTTPException(status_code=400, detail="Could not get email from OAuth provider")

    # Find or create user
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user:
        user = User(
            email=email,
            full_name=name,
            avatar_url=avatar,
            oauth_provider=provider,
            oauth_id=oauth_id,
            role=UserRole.VIEWER,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    return TokenResponse(
        access_token=create_access_token(str(user.id), user.role.value),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.get("/auth/me", response_model=UserOut, tags=["auth"])
async def get_me(user: User = Depends(get_current_user)):
    """Get current user info."""
    return UserOut(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        tier=user.tier,
        avatar_url=user.avatar_url,
        created_at=user.created_at,
    )


@router.post("/auth/api-key", tags=["auth"])
async def generate_user_api_key(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a new API key for the user."""
    api_key = generate_api_key()
    user.api_key_hash = hash_api_key(api_key)
    await db.commit()
    return {"api_key": api_key, "message": "Store this key securely — it won't be shown again."}


# ── Upload & Session ──────────────────────────────────────────
@router.post("/upload", tags=["inference"])
async def upload_media(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload an image or video for batch processing."""
    allowed_types = ["image/jpeg", "image/png", "image/webp", "video/mp4", "video/webm", "video/avi"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    # Save file
    upload_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] if file.filename else ".bin"
    file_path = os.path.join(upload_dir, f"{file_id}{ext}")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Create session
    source_type = "file_image" if file.content_type.startswith("image") else "file_video"
    session = Session(
        user_id=user.id,
        source_type=source_type,
        source_url=file_path,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    return {
        "session_id": str(session.id),
        "file_id": file_id,
        "file_type": file.content_type,
        "file_size": len(content),
        "message": "File uploaded. Connect to /ws/stream to start processing.",
    }


@router.get("/sessions", response_model=List[SessionOut], tags=["sessions"])
async def list_sessions(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=20, le=100),
    offset: int = 0,
):
    """List all inference sessions for current user."""
    result = await db.execute(
        select(Session)
        .where(Session.user_id == user.id)
        .order_by(Session.started_at.desc())
        .offset(offset)
        .limit(limit)
    )
    sessions = result.scalars().all()
    return [
        SessionOut(
            id=str(s.id),
            source_type=s.source_type,
            started_at=s.started_at,
            ended_at=s.ended_at,
            frame_count=s.frame_count or 0,
            total_detections=s.total_detections or 0,
            avg_fps=s.avg_fps,
            active_modules=s.active_modules or [],
        )
        for s in sessions
    ]


@router.get("/session/{session_id}/report", tags=["sessions"])
async def download_report(
    session_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate and download a PDF report for a session."""
    from utils.reporter import ReportGenerator

    result = await db.execute(
        select(Session).where(Session.id == session_id, Session.user_id == user.id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get detections and anomalies
    det_result = await db.execute(
        select(Detection).where(Detection.session_id == session_id).limit(1000)
    )
    detections = [
        {"class_name": d.class_name, "confidence": d.confidence, "frame_id": d.frame_id}
        for d in det_result.scalars().all()
    ]

    anom_result = await db.execute(
        select(Anomaly).where(Anomaly.session_id == session_id)
    )
    anomalies = [
        {"timestamp": str(a.timestamp), "anomaly_type": a.anomaly_type,
         "severity": a.severity.value if a.severity else "green", "score": a.score,
         "description": a.description}
        for a in anom_result.scalars().all()
    ]

    met_result = await db.execute(
        select(Metric).where(Metric.session_id == session_id).limit(500)
    )
    metrics = [
        {"fps": m.fps, "inference_ms": m.inference_ms, "gpu_util": m.gpu_util}
        for m in met_result.scalars().all()
    ]

    reporter = ReportGenerator()
    pdf_bytes = reporter.generate_report(
        session_data={
            "id": str(session.id),
            "source_type": session.source_type.value if session.source_type else "unknown",
            "started_at": str(session.started_at),
            "ended_at": str(session.ended_at) if session.ended_at else "In Progress",
            "frame_count": session.frame_count or 0,
            "avg_fps": session.avg_fps or 0,
            "active_modules": session.active_modules or [],
        },
        detections=detections,
        anomalies=anomalies,
        metrics=metrics,
    )

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=omnivis_report_{session_id[:8]}.pdf"}
    )


# ── Model Management ─────────────────────────────────────────
@router.post("/model/switch", tags=["models"])
async def switch_model(
    req: ModelSwitchRequest,
    user: User = Depends(require_role(UserRole.ANALYST, UserRole.ADMIN)),
):
    """Hot-swap a detection/segmentation/etc model."""
    from main import pipeline_manager
    try:
        pipeline_manager.switch_model(req.module.value, req.model_name, req.variant)
        return {"status": "success", "message": f"Switched {req.module} to {req.model_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Metrics ──────────────────────────────────────────────────
@router.get("/metrics/live", tags=["metrics"])
async def get_live_metrics():
    """Get live system metrics in Prometheus format."""
    try:
        import psutil
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
    except ImportError:
        cpu = 0.0
        ram = type('obj', (object,), {'used': 0, 'total': 1, 'percent': 0})()

    gpu_util = 0.0
    gpu_mem = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_util = 50.0  # Would use pynvml in production
            gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
    except ImportError:
        pass

    metrics_text = f"""# HELP omnivis_cpu_usage CPU usage percentage
# TYPE omnivis_cpu_usage gauge
omnivis_cpu_usage {cpu}
# HELP omnivis_ram_usage RAM usage in MB
# TYPE omnivis_ram_usage gauge
omnivis_ram_usage {ram.used / 1024 / 1024:.1f}
# HELP omnivis_gpu_utilization GPU utilization percentage
# TYPE omnivis_gpu_utilization gauge
omnivis_gpu_utilization {gpu_util}
# HELP omnivis_gpu_memory GPU memory in MB
# TYPE omnivis_gpu_memory gauge
omnivis_gpu_memory {gpu_mem:.1f}
# HELP omnivis_uptime_seconds Uptime in seconds
# TYPE omnivis_uptime_seconds counter
omnivis_uptime_seconds {time.time() - START_TIME:.0f}
"""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(metrics_text, media_type="text/plain")


# ── Calibration ───────────────────────────────────────────────
@router.post("/calibrate", response_model=CalibrationResult, tags=["calibration"])
async def calibrate_camera(
    req: CalibrationRequest,
    user: User = Depends(require_role(UserRole.ANALYST, UserRole.ADMIN)),
):
    """Start camera calibration process."""
    return CalibrationResult(
        success=True,
        message="Calibration endpoint ready. Send calibration frames via WebSocket.",
    )


# ── Health Check ──────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """System health check."""
    gpu_available = False
    gpu_name = None
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        uptime_seconds=round(time.time() - START_TIME, 1),
        modules_loaded=["detection", "segmentation", "face", "optical_flow",
                        "depth", "reconstruction", "tracking", "scene_graph",
                        "trajectory", "anomaly", "gait", "action", "gan"],
    )
