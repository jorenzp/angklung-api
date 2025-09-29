# Use Python 3.9 slim image for better compatibility with scientific libraries
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for audio processing and scientific libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create necessary directories
RUN mkdir -p /app/model /app/temp

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose the port that Railway will use
EXPOSE 5000

# Health check to ensure the application is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the Flask application
CMD ["python", "app.py"]