FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY ppofolio ./ppofolio
COPY configs ./configs
COPY models ./models

RUN pip install --upgrade pip && pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p /app/data

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn ppofolio.serve.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
