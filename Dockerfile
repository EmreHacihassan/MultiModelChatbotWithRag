# ==============================================================================
# MyChatbot - Dockerfile
# ==============================================================================
# Multi-stage build for optimized production image
#
# Usage:
#   docker build -t mychatbot .
#   docker run -p 8000:8000 -p 3002:3002 mychatbot
#
# ==============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Python Backend Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim as backend-builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Node.js Frontend Builder
# -----------------------------------------------------------------------------
FROM node:20-alpine as frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm ci --only=production || npm install

# Copy frontend source
COPY frontend/ .

# Build frontend for production
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 3: Production Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DJANGO_SETTINGS_MODULE=backend.app.server.settings \
    PORT=8000

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos '' appuser

# Copy Python packages from builder
COPY --from=backend-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Copy built frontend from builder
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Create necessary directories
RUN mkdir -p /app/logs /app/data/sessions /app/rag/uploads && \
    chown -R appuser:appuser /app/logs /app/data /app/rag

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health/ || exit 1

# Expose ports
EXPOSE 8000 3002

# Default command: Run backend with Uvicorn
CMD ["uvicorn", "backend.app.server.asgi:application", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
