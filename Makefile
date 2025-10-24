.PHONY: help install dev run test test-coverage format lint mypy check clean

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync

dev:  ## Run development server with hot reload
	cd backend && uv run uvicorn app:app --reload --port 8000

run:  ## Run production server
	cd backend && uv run uvicorn app:app --port 8000

test:  ## Run all tests
	uv run pytest backend/tests/ -v

test-coverage:  ## Run tests with coverage report
	uv run pytest backend/tests/ -v --cov=backend --cov-report=html --cov-report=term

format:  ## Auto-format code with black and ruff
	@echo "Running black..."
	uv run black backend/
	@echo "Running ruff..."
	uv run ruff check backend/ --fix

lint:  ## Run linting with ruff (no auto-fix)
	uv run ruff check backend/

mypy:  ## Run type checking with mypy
	uv run mypy backend/

check:  ## Run all quality checks (format + lint + mypy)
	@echo "Step 1/3: Formatting code..."
	$(MAKE) format
	@echo ""
	@echo "Step 2/3: Running linter..."
	$(MAKE) lint
	@echo ""
	@echo "Step 3/3: Running type checker..."
	$(MAKE) mypy
	@echo ""
	@echo "✅ All quality checks passed!"

clean:  ## Remove cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov .coverage
	@echo "✅ Cleaned cache and temporary files"
