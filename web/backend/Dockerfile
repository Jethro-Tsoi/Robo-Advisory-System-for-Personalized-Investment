# Stage 1: Development
FROM python:3.9-slim AS development

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies using uv
RUN uv pip install -r requirements.txt

# Copy application code
COPY . .

# Development entrypoint script
COPY docker-entrypoint.dev.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.dev.sh

EXPOSE 8000

CMD ["docker-entrypoint.dev.sh"]

# Stage 2: Production
FROM python:3.9-slim AS production

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies using uv
RUN uv pip install -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
