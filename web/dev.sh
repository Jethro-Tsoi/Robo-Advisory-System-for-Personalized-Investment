#!/bin/bash

# Build and start containers
echo "Starting development environment..."
docker-compose up --build

# Cleanup on exit
cleanup() {
    echo "Stopping containers..."
    docker-compose down
    echo "Development environment stopped"
}

trap cleanup EXIT

# Keep script running
wait
