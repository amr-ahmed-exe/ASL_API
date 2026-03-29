FROM python:3.10-slim

# Install system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    enchant-2 \
    hunspell-en-us \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (Docker caches this layer)
COPY api_requirements.txt .
RUN pip install --no-cache-dir -r api_requirements.txt

# Copy application files
COPY app.py .
COPY sign_language_model.tflite .
COPY white.jpg .

# Expose port (Render uses $PORT env variable)
EXPOSE 8000

# Start the server — Render sets $PORT automatically
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
