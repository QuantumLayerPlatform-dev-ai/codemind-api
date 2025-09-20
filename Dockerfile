# =====================================================
# Builder Stage - Compile dependencies
# =====================================================
FROM python:3.10-slim AS builder

# Build arguments
ARG VERSION=0.1.0
ARG BUILD_DATE
ARG COMMIT_SHA

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels for better layer caching
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels --no-cache-dir -r requirements.txt

# =====================================================
# Runtime Stage - Minimal production image
# =====================================================
FROM python:3.10-slim AS runtime

# Build arguments (re-declare for runtime stage)
ARG VERSION=0.1.0
ARG BUILD_DATE
ARG COMMIT_SHA

# Metadata labels
LABEL version="${VERSION}"
LABEL build-date="${BUILD_DATE}"
LABEL commit-sha="${COMMIT_SHA}"
LABEL component="api"
LABEL project="codemind"
LABEL description="CodeMind API - Optimized Production Build"
LABEL maintainer="CodeMind Team"

WORKDIR /app

# Install only runtime dependencies (no gcc/build tools)
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/*.whl \
    && rm -rf /wheels

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]