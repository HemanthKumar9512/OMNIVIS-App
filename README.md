# 🔮 OMNIVIS — Omniscient Vision Intelligence System

<div align="center">

**Production-grade, globally deployable, real-time autonomous perception engine**

*Sees, understands, reconstructs, predicts, and acts on any visual input.*

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61dafb?logo=react&logoColor=black)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ed?logo=docker&logoColor=white)](https://docker.com)

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Quick Start — Development Mode](#-quick-start--development-mode)
- [Docker Deployment](#-docker-deployment)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

### Computer Vision Modules (13 total)
| Module | Model | Description |
|--------|-------|-------------|
| 🎯 Object Detection | YOLOv8x | 80 COCO + 20 custom classes, TensorRT FP16 |
| 🧩 Instance Segmentation | Mask R-CNN | ResNet-101-FPN, pixel-perfect masks |
| 🎨 Semantic Segmentation | DeepLabV3+ | MobileNetV3/ResNet-101, per-pixel labels |
| 👤 Face Analysis | RetinaFace + ArcFace | Detection, recognition, age/gender/emotion |
| 🌊 Optical Flow | RAFT | Dense motion field + Farneback fallback |
| 🏔️ Depth Estimation | MiDaS v3.1 | Monocular depth + depth from defocus |
| 🧊 3D Reconstruction | ORB-SfM | Feature matching + triangulation + point cloud |
| 📍 Multi-Object Tracking | ByteTrack | Kalman filter + persistent track IDs |
| 🕸️ Scene Graph | GNN RelTransformer | Object relationship prediction |
| 📐 Trajectory Prediction | Social-LSTM | 12-step prediction with uncertainty cones |
| 🚨 Anomaly Detection | SVM + Autoencoder + IsoForest | Ensemble anomaly scoring |
| 🚶 Gait Analysis | MediaPipe + LSTM | Pose estimation + gait classification |
| 🎬 Action Recognition | SlowFast/R3D | 400 Kinetics action classes |

### Platform Features
- 🖥️ **4-panel live canvas grid** — each independently configurable
- 📊 **Real-time metric charts** — FPS, latency, detections, tracks
- 🧊 **3D point cloud viewer** — WebGL with orbit controls
- 🕸️ **Scene graph visualization** — force-directed graph
- 🌙 **Dark/light theme** — glassmorphic design
- 🌐 **Multi-language UI** — English, Tamil, Hindi
- 🔒 **JWT + OAuth2 auth** — Google/GitHub login
- 📄 **PDF report generation** — session summaries
- 🐳 **Docker deployment** — GPU-enabled, 7 services

---

## 🏗 Architecture

```
Camera/Stream → Frame Buffer
    ↓
Frame Preprocessor
    ↓
[Parallel GPU threads]
    ├─ Module 1: YOLOv8 + Mask RCNN + Face
    ├─ Module 2A: RAFT Optical Flow + SlowFast
    ├─ Module 2B: ORB-SfM + MiDaS Depth
    └─ Module 3: ByteTrack + GNN + SVM
    ↓
Result Aggregator → WebSocket → React Dashboard
```

---

## 📌 Prerequisites

### For Development Mode (Recommended for first run)

1. **Python 3.11+**
   ```bash
   # Download from https://www.python.org/downloads/
   python --version  # Should show 3.11+
   ```

2. **Node.js 18+ & npm**
   ```bash
   # Download from https://nodejs.org/
   node --version  # Should show 18+
   npm --version
   ```

3. **Git** (optional)

### For Docker Mode

1. **Docker Desktop** with Docker Compose v2
2. **NVIDIA Container Toolkit** (for GPU support)

---

## 🚀 Quick Start — Development Mode

### Step 1: Install Backend Dependencies

```powershell
# Navigate to backend
cd D:\OMNIVIS\backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install core dependencies (CPU mode — works without GPU)
pip install fastapi uvicorn[standard] websockets python-multipart
pip install opencv-python-headless numpy
pip install sqlalchemy aiosqlite  # SQLite for dev (no PostgreSQL needed)
pip install python-jose[cryptography] passlib[bcrypt]
pip install scikit-learn psutil Pillow pydantic

# Optional: Install ML models (requires more disk space)
pip install ultralytics          # YOLOv8 (~50MB)
pip install torch torchvision    # PyTorch (~2GB)
pip install mediapipe            # Face/pose detection (~100MB)
```

### Step 2: Configure Backend for Development

For development without PostgreSQL, create a `.env` file:

```powershell
# In D:\OMNIVIS\backend\.env
echo "DATABASE_URL=sqlite+aiosqlite:///./omnivis_dev.db" > .env
```

Or modify `db/session.py` to use SQLite:
```python
DATABASE_URL = "sqlite+aiosqlite:///./omnivis_dev.db"
```

### Step 3: Start Backend

```powershell
cd D:\OMNIVIS\backend
.\venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **REST API**: http://localhost:8000/docs (Swagger UI)
- **Health Check**: http://localhost:8000/api/health
- **WebSocket**: ws://localhost:8000/ws/stream

### Step 4: Install Frontend Dependencies

```powershell
# Open a NEW terminal
cd D:\OMNIVIS\frontend
npm install
```

### Step 5: Start Frontend

```powershell
cd D:\OMNIVIS\frontend
npm run dev
```

The dashboard will be available at: **http://localhost:5173**

### Step 6: Connect & Use

1. Open http://localhost:5173 in your browser
2. The app will auto-connect to the backend WebSocket
3. Allow camera access when prompted
4. Toggle modules on/off in the left sidebar
5. View real-time metrics in the right sidebar

---

## 🐳 Docker Deployment

### Full Stack (GPU)

```bash
cd D:\OMNIVIS

# Build and start all services
docker compose up --build -d

# View logs
docker compose logs -f backend

# Stop all services
docker compose down
```

### CPU-only Mode

Remove the GPU reservation from `docker-compose.yml`:
```yaml
# Comment out or remove:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]
```

### Access Points

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000/docs |
| NGINX (Production) | http://localhost:80 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 (admin/omnivis) |

---

## 📁 Project Structure

```
omnivis/
├── backend/
│   ├── main.py                  # FastAPI app + WebSocket
│   ├── modules/
│   │   ├── detection.py         # YOLOv8 multi-class detection
│   │   ├── segmentation.py      # Mask R-CNN + DeepLabV3+
│   │   ├── face.py              # RetinaFace + ArcFace + attributes
│   │   ├── optical_flow.py      # RAFT + Farneback
│   │   ├── depth.py             # MiDaS monocular depth
│   │   ├── reconstruction.py    # SfM + wavelet analysis
│   │   ├── tracking.py          # ByteTrack + Kalman filter
│   │   ├── scene_graph.py       # GNN scene graph
│   │   ├── trajectory.py        # Social-LSTM
│   │   ├── anomaly.py           # SVM + Autoencoder + IsoForest
│   │   ├── gait.py              # MediaPipe + LSTM gait
│   │   ├── action.py            # SlowFast action recognition
│   │   └── gan.py               # StyleGAN2 + CycleGAN
│   ├── api/
│   │   ├── routes.py            # REST endpoints
│   │   ├── auth.py              # JWT + OAuth2
│   │   └── schemas.py           # Pydantic models
│   ├── db/
│   │   ├── models.py            # SQLAlchemy ORM
│   │   └── session.py           # DB connection
│   ├── workers/
│   │   └── celery_app.py        # Async task queue
│   ├── utils/
│   │   ├── encoder.py           # Frame encoding
│   │   ├── calibration.py       # Camera calibration
│   │   └── reporter.py          # PDF report generator
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main dashboard layout
│   │   ├── components/
│   │   │   ├── CanvasPanel.tsx   # Video canvas panels
│   │   │   ├── MetricChart.tsx   # Recharts line charts
│   │   │   ├── SceneGraph3D.tsx  # D3 graph visualization
│   │   │   ├── PointCloud3D.tsx  # 3D point cloud viewer
│   │   │   ├── AlertBanner.tsx   # Anomaly alerts
│   │   │   ├── DetectionStrip.tsx # Bottom chip bar
│   │   │   ├── Sidebar.tsx       # Controls + analytics
│   │   │   └── MetricGauge.tsx   # Circular gauges
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts   # WS connection manager
│   │   │   ├── useCamera.ts      # Camera stream hook
│   │   │   └── useMetrics.ts     # Metric state
│   │   ├── store/
│   │   │   └── omnivis.store.ts  # Zustand global state
│   │   └── i18n/                 # EN, Tamil, Hindi
│   ├── Dockerfile
│   └── vite.config.ts
├── docker-compose.yml           # 7 services
├── nginx.conf                   # Reverse proxy
├── prometheus.yml               # Metrics scraping
└── README.md
```

---

## 📡 API Documentation

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login (JWT) |
| POST | `/api/auth/oauth/{provider}` | OAuth2 login |
| GET | `/api/auth/me` | Get current user |
| POST | `/api/upload` | Upload image/video |
| GET | `/api/sessions` | List sessions |
| GET | `/api/session/{id}/report` | Download PDF report |
| POST | `/api/model/switch` | Hot-swap model |
| GET | `/api/metrics/live` | Prometheus metrics |
| POST | `/api/calibrate` | Camera calibration |
| GET | `/api/health` | Health check |

### WebSocket Channels

| Channel | Purpose |
|---------|---------|
| `/ws/stream` | Main inference (send frames, receive results) |
| `/ws/alerts` | Anomaly alert broadcast |
| `/ws/metrics` | System performance metrics |

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | Database connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `JWT_SECRET_KEY` | (change in production!) | JWT signing key |
| `GOOGLE_CLIENT_ID` | | Google OAuth2 client ID |
| `GITHUB_CLIENT_ID` | | GitHub OAuth2 client ID |

---

## 🔧 Troubleshooting

### "Module X failed to load"
This is normal on CPU-only systems. Modules gracefully fall back to simulation mode. Install optional dependencies for full functionality:
```bash
pip install ultralytics torch torchvision mediapipe insightface
```

### "WebSocket disconnected"
1. Ensure the backend is running on port 8000
2. Check CORS settings in `main.py`
3. Verify the Vite proxy in `vite.config.ts`

### "Camera access denied"
- Open the app via `http://localhost:5173` (not file://)
- Allow camera permissions in browser settings
- Use HTTPS in production for camera access

### GPU not detected
```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 📄 License

MIT License — Built for educational and research purposes.

---

<div align="center">
<strong>OMNIVIS v1.0.0</strong> — Built with 🧠 by AI
</div>
