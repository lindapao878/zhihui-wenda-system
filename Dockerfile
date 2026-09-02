# syntax=docker/dockerfile:1
FROM python:3.12-slim

# build-essential for any C extensions missing wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app source only; model weights and data come from volume mounts
COPY knowledge/ knowledge/
COPY batch_import.py .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8001

# Default: import API. Query API overrides command in compose.
CMD ["python", "-m", "uvicorn", "knowledge.api.import_router:app", "--host", "0.0.0.0", "--port", "8000"]
