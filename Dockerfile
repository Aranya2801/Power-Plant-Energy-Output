# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Aranya <github.com/Aranya2801>"
LABEL description="Power Plant Energy Output ML System"
LABEL version="1.0.0"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Generate dataset & train models
RUN python data/generate_dataset.py && \
    python scripts/run_pipeline.py

# Expose ports
EXPOSE 8000 8501

# Default: start API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
