FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for volumes
RUN mkdir -p /app/data /app/chroma_db /app/media_uploads

COPY ./entrypoint-web.sh /app/entrypoint-web.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint-web.sh

# Expose API port
EXPOSE 8000
