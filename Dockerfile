ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HOST=0.0.0.0 \
    PORT=5000

WORKDIR /app

# 1) dependencies first: this layer is rebuilt only when requirements change
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# 2) application code last: it changes often, so it must not invalidate the pip layer
COPY app.py inference.py db.py ./
COPY templates ./templates
COPY static ./static

# non-root user; no secrets and no model are baked in
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=6s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=5).status == 200 else 1)"

CMD ["python", "app.py"]