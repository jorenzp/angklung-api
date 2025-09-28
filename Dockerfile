FROM python:3.9-slim

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

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

# Copy everything from capstone folder to /app
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip --root-user-action=ignore
RUN pip install --no-cache-dir numpy==1.23.5 --root-user-action=ignore
RUN pip install --no-cache-dir tensorflow==2.13.0 --root-user-action=ignore
RUN pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

ENV PORT=8080
EXPOSE $PORT

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "--preload", "app:app"]