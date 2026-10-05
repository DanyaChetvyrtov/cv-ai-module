FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    CV_MODEL_PATH=/app/models/yolo11n.pt \
    CV_DEVICE=cpu

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# CPU wheels avoid pulling large CUDA libraries into the demo image.
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . \
    && useradd --create-home app \
    && mkdir -p /app/models && chown app:app /app/models
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=120s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"
CMD ["uvicorn", "cv_test.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

