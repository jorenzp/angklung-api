FROM python:3.9-slim

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies with --root-user-action=ignore to suppress warnings
RUN pip install --upgrade pip --root-user-action=ignore
RUN pip install --no-cache-dir numpy==1.23.5 --root-user-action=ignore
RUN pip install --no-cache-dir tensorflow==2.13.0 --root-user-action=ignore
RUN pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

# Copy application files
COPY . .

# Expose port
EXPOSE 8080

# Use python directly - no gunicorn needed
CMD ["python", "app.py"]