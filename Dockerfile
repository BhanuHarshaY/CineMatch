FROM python:3.11-slim

WORKDIR /app

# System deps for faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build-time: generate embeddings, FAISS index, metadata
RUN python scripts/build_index.py

# Runtime config
EXPOSE 80
ENV OPENAI_API_KEY=""

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
