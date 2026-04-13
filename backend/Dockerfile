FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ffmpeg wget curl git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
    && pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/uploads /app/reports

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--ws-max-size", "16777216"]
