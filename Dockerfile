# Use Python 3.9 slim image (smaller size, faster builds)
FROM python:3.9-slim

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies needed for your audio processing libraries
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir tensorflow==2.13.0
RUN pip install --no-cache-dir flask flask-cors gunicorn
RUN pip install --no-cache-dir librosa scikit-learn scipy numpy

# Copy your application code
COPY . .

# Create model directory if it doesn't exist
RUN mkdir -p model

# Expose port (Railway/Render will override with $PORT)
EXPOSE 5000

# Use gunicorn for production serving
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --max-requests 100 --preload app:app