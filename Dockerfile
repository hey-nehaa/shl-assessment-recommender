FROM python:3.11-slim

# Render uses PORT env var; default to 10000
ENV PORT=10000
EXPOSE 10000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY app/ app/
COPY data/ data/

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
