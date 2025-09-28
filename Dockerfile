FROM python:3.9-slim

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir tensorflow==2.13.0
RUN pip install --no-cache-dir flask flask-cors gunicorn
RUN pip install --no-cache-dir librosa scikit-learn scipy numpy

# Copy model files specifically
COPY model/ ./model/

# Copy the rest of your application
COPY *.py .

# Expose port
EXPOSE 8080

# Use gunicorn with proper port handling
CMD ["python", "app.py"]
