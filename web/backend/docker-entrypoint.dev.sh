#!/bin/bash

# Enable error handling
set -e

# Function to cleanup processes on exit
cleanup() {
    echo "Cleaning up processes..."
    kill $(jobs -p) 2>/dev/null || true
}

# Setup cleanup trap
trap cleanup EXIT

# Start Jupyter notebook server in background
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' &

# Wait for Jupyter to start
sleep 2

echo "Jupyter notebook server is running at http://localhost:8888"

# Install development tools if not present
if ! command -v watchfiles &> /dev/null; then
    echo "Installing development tools..."
    uv pip install watchfiles
fi

# Start FastAPI with hot reload using watchfiles
echo "Starting FastAPI development server..."
watchfiles "uvicorn main:app --host 0.0.0.0 --port 8000 --reload" --filter python
