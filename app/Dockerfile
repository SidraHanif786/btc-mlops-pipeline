# Stage 1: Builder (installs dependencies into isolated prefix)
FROM python:3.10-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install dill
# Stage 2: Production Runtime (minimal, non-root)
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_DIR=/app/models \
    PORT=5000

RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m appuser

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup src/ ./src/

RUN mkdir -p /app/models /app/data && \
    chown -R appuser:appgroup /app/models /app/data

USER appuser
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "2", "app.app:app"]