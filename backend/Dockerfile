# Production Dockerfile for AI Research Agent FastAPI Backend
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install curl for container health check
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create unprivileged system user and group
RUN groupadd -r appgroup && useradd -r -u 1000 -g appgroup appuser

# Install Python dependencies first for optimal layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ app/

# Set ownership to non-root user
RUN chown -R appuser:appgroup /app

# Switch to unprivileged user
USER appuser

# Expose default API port
EXPOSE 8000

# Container healthcheck using liveness probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production startup: single worker, no reload
# In-memory rate limiting is scoped per-worker.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
