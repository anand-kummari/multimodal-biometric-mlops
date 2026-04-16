.PHONY: help install install-dev lint format typecheck test test-cov train infer preprocess benchmark clean docker-build

PYTHON := python
SRC_DIR := src
TEST_DIR := tests

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in production mode
	$(PYTHON) -m pip install .

install-dev: ## Install package in development mode with all dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

lint: ## Run linting with ruff
	ruff check $(SRC_DIR) $(TEST_DIR)

format: ## Format code with ruff
	ruff format $(SRC_DIR) $(TEST_DIR)
	ruff check --fix $(SRC_DIR) $(TEST_DIR)

typecheck: ## Run type checking with mypy
	mypy $(SRC_DIR)

test: ## Run unit tests
	pytest $(TEST_DIR) -v --tb=short

test-cov: ## Run tests with coverage report
	pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html

train: ## Run training pipeline (use CONFIG=<path> to override)
	$(PYTHON) scripts/train.py $(if $(CONFIG),--config-path=$(CONFIG),)

infer: ## Run inference pipeline
	$(PYTHON) scripts/infer.py

preprocess: ## Run data preprocessing with Ray
	$(PYTHON) scripts/preprocess.py

benchmark: ## Run data loading benchmarks
	$(PYTHON) benchmarks/benchmark_dataloader.py

clean: ## Clean build artifacts and caches
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov/ .coverage
	rm -rf outputs/ multirun/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build: ## Build Docker image
	docker build -t multimodal-biometric-mlops:latest .

all: lint typecheck test ## Run all checks (lint + typecheck + test)
