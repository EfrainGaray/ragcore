FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Application source
COPY ragcore/ ragcore/

# Data directory (overridden by volume mount in docker-compose)
RUN mkdir -p /app/data/chroma

EXPOSE 8000 8001

CMD ["python", "-m", "ragcore.main"]
