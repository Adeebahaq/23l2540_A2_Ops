FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8080

WORKDIR /app

# dependencies first (layer cache), code and model last
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api_app.py inference.py ./
COPY models ./models

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

# Cloud Run injects $PORT; one worker (one copy of the model in RAM) with threads for concurrency
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 --access-logfile - --error-logfile - --log-level info api_app:app