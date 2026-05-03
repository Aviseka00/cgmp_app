# Build from repository root (this folder), not from backend/ alone:
#   docker build -t cgmp-app .
# Render "Docker" deploy uses this file by default when Docker Context is the repo root.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app /app/app
COPY frontend /app/frontend
COPY cell_detection /opt/cell_detection
COPY weights /app/weights

EXPOSE 8000

# Match inference imports (see backend/app/inference.py)
ENV CELL_DETECTION_DIR=/opt/cell_detection

# Render sets PORT; default 8000 for local runs.
ENV UVICORN_WORKERS=1
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-1}"]
