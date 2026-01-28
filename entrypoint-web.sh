#!/bin/bash
set -e

# Run alembic migrations if alembic directory exists
if [ -d "alembic" ]; then
    echo "Running database migrations..."
    alembic upgrade head
fi

# Start uvicorn server
echo "Starting uvicorn server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
