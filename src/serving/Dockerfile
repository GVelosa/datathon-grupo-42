# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN pip install --upgrade pip --no-cache-dir

COPY pyproject.toml ./

RUN pip install --no-cache-dir --prefix=/install \
    fastapi \
    uvicorn[standard] \
    pydantic \
    anthropic \
    sentence-transformers \
    faiss-cpu \
    scikit-learn \
    torch \
    mlflow \
    prometheus-client \
    pyyaml \
    pandas \
    numpy

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Fraud Detection API"
LABEL org.opencontainers.image.description="FastAPI serving layer — Datathon Grupo 42"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/ ./src/
COPY configs/ ./configs/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "/dev/null"]
