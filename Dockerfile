# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables to reduce pip warnings and improve stability
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies in layers for better caching
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install audio system dependencies separately
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install additional audio dependencies if needed
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/* || true

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install dependencies in stages
COPY requirements.txt .

# Install core scientific packages first
RUN pip install numpy==1.24.3 --no-deps

# Install scipy
RUN pip install scipy==1.10.1

# Install TensorFlow (this is often the problematic package)
RUN pip install tensorflow==2.13.0 --no-deps || pip install tensorflow-cpu==2.13.0

# Install scikit-learn
RUN pip install scikit-learn==1.3.0

# Install audio packages
RUN pip install librosa==0.10.1 soundfile==0.12.1

# Install Flask and other web dependencies
RUN pip install Flask==2.3.3 gunicorn==21.2.0

# Install any remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/model /app/temp

# Set Flask environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run the application
CMD ["python", "app.py"]