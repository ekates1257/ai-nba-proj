FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install the project first to maximize Docker layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip setuptools wheel && pip install .

# Copy runtime data after dependencies so code changes are cheaper to rebuild.
COPY artifacts ./artifacts
COPY prediction.csv ./prediction.csv

EXPOSE 8000

CMD ["sh", "-c", "uvicorn nba_predictor.main:app --host 0.0.0.0 --port ${PORT}"]

