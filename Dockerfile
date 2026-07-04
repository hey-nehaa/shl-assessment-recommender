FROM python:3.11-slim

# HF Spaces requires port 7860 and non-root user
ENV PORT=7860
EXPOSE 7860

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

# HF Spaces runs as uid 1000
RUN useradd -m -u 1000 user
USER user

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
