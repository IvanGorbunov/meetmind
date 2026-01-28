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

# Create a non-root user
RUN useradd -ms /bin/bash appuser

# Copy application code
COPY . .

# Create directories for volumes and set permissions for the app user
RUN mkdir -p /app/data /app/chroma_db /app/media_uploads && \
    chmod +x /app/entrypoint-web.sh && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Run entrypoint script
ENTRYPOINT ["./entrypoint-web.sh"]
