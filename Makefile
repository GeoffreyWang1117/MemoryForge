.PHONY: help install dev test lint format clean docker-up docker-down api benchmark

# Default target
help:
	@echo "MemoryForge - Hierarchical Context Memory System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@echo "  install     Install production dependencies"
	@echo "  dev         Install development dependencies"
	@echo "  test        Run tests"
	@echo "  lint        Run linter"
	@echo "  format      Format code"
	@echo "  clean       Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  docker-up   Start all services"
	@echo "  docker-down Stop all services"
	@echo "  docker-logs View logs"
	@echo ""
	@echo "Run:"
	@echo "  api         Run API server"
	@echo "  benchmark   Run performance benchmark"
	@echo "  demo        Run conversation demo"
	@echo "  analyze     Analyze codebase (use: make analyze PATH=./src)"

# Installation
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=memoryforge --cov-report=html

# Code Quality
lint:
	ruff check memoryforge tests

format:
	ruff format memoryforge tests

# Clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/

# Docker
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose build

# Run
api:
	python scripts/run_api.py

benchmark:
	python scripts/benchmark.py

demo:
	python scripts/demo_conversation.py

analyze:
	python -m memoryforge.cli analyze $(PATH)
