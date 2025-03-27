# Financial Sentiment Analysis Project Makefile

.PHONY: up down build clean test logs

# Development commands
up:
	docker compose up

down:
	docker compose down

build:
	docker compose build

# Start development environment with optional tools
dev:
	docker compose --profile devtools up

# Clean up
clean:
	docker compose down -v
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name "*.pyc" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name "node_modules" -exec rm -r {} +
	find . -type d -name ".next" -exec rm -r {} +

# Create necessary directories and setup environment
setup:
	mkdir -p data/models logs
	test -f .env || cp .env.example .env
	@echo "Created directories and environment file"
	@echo "Please configure your API keys in .env"

# View logs
logs:
	docker compose logs -f

# Run tests
test-backend:
	docker compose exec backend pytest

test-frontend:
	docker compose exec frontend npm test

# Access Jupyter notebooks
jupyter:
	docker compose exec backend jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root

# Show help
help:
	@echo "Available commands:"
	@echo "  make up              - Start all services"
	@echo "  make down            - Stop all services"
	@echo "  make build           - Build/rebuild services"
	@echo "  make dev             - Start development environment with tools"
	@echo "  make clean           - Remove all containers and volumes"
	@echo "  make setup           - Create necessary directories and files"
	@echo "  make logs            - View service logs"
	@echo "  make test-backend    - Run backend tests"
	@echo "  make test-frontend   - Run frontend tests"
	@echo "  make jupyter         - Start Jupyter notebook server"
