# NeuroSynth Unified - Makefile
# ==============================
# Common development and deployment tasks

.PHONY: help install dev test lint format docker clean

# Default target
help:
	@echo "NeuroSynth Unified - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Development:"
	@echo "  make install     Install production dependencies"
	@echo "  make dev         Install development dependencies"
	@echo "  make run         Run development server"
	@echo ""
	@echo "Testing:"
	@echo "  make test        Run all tests"
	@echo "  make test-unit   Run unit tests only"
	@echo "  make test-cov    Run tests with coverage"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint        Run linters"
	@echo "  make format      Format code"
	@echo "  make typecheck   Run type checking"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-up     Start all services"
	@echo "  make docker-down   Stop all services"
	@echo "  make docker-logs   View logs"
	@echo ""
	@echo "Database:"
	@echo "  make db-init     Initialize database"
	@echo "  make db-migrate  Run migrations"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean       Remove build artifacts"

# =============================================================================
# Installation
# =============================================================================

install:
	pip install --upgrade pip
	pip install -r requirements.txt

dev: install
	pip install -r requirements-test.txt
	pip install black ruff mypy

# =============================================================================
# Development Server
# =============================================================================

run:
	uvicorn src.api.main:app --reload --port 8000

run-prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# =============================================================================
# Testing
# =============================================================================

test:
	pytest

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest -x -q

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

format:
	black src/ tests/

format-check:
	black src/ tests/ --check

typecheck:
	mypy src/ --ignore-missing-imports

quality: lint format-check typecheck

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t neurosynth:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-shell:
	docker-compose exec api bash

docker-test:
	docker-compose exec api pytest

docker-rebuild:
	docker-compose build --no-cache api
	docker-compose up -d api

# =============================================================================
# Database
# =============================================================================

db-init:
	python scripts/init_database.py

db-migrate:
	python scripts/migrate.py

db-shell:
	docker-compose exec postgres psql -U neurosynth -d neurosynth

# =============================================================================
# Index Building
# =============================================================================

build-indexes:
	python scripts/build_indexes.py --database $(DATABASE_URL) --output ./indexes

# =============================================================================
# Cleanup
# =============================================================================

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true

clean-docker:
	docker-compose down -v --remove-orphans
	docker system prune -f

# =============================================================================
# Production Deployment
# =============================================================================

deploy-check:
	@echo "Checking deployment requirements..."
	@test -f .env || (echo "ERROR: .env file missing" && exit 1)
	@grep -q "VOYAGE_API_KEY=" .env || (echo "ERROR: VOYAGE_API_KEY not set" && exit 1)
	@grep -q "ANTHROPIC_API_KEY=" .env || (echo "ERROR: ANTHROPIC_API_KEY not set" && exit 1)
	@echo "All checks passed!"

deploy: deploy-check docker-build docker-up
	@echo "Deployment complete!"
	@echo "API available at: http://localhost:8000"
	@echo "Docs available at: http://localhost:8000/docs"
