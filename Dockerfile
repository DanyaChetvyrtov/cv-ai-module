FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY licenses ./licenses
RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/models \
    && chown -R app:app /app

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=120s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"
CMD ["uvicorn", "cv_test.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
