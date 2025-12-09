.PHONY: install install-dev install-mamba setup clean test lint format collect-data build-features train backtest live docker-build docker-up docker-down

# Environment setup
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-mamba:
	pip install -e ".[mamba]"

install-flash:
	pip install flash-attn --no-build-isolation

setup: install-dev
	pre-commit install

# PyTorch with CUDA
install-cuda:
	pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Code quality
lint:
	ruff check src/ tests/ scripts/
	mypy src/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

# Data pipeline
collect-data:
	python scripts/collect_data.py --config config/base.yaml

build-features:
	python scripts/build_features.py --config config/base.yaml

# Training
train:
	python scripts/train_models.py --config config/base.yaml

train-tft:
	python scripts/train_models.py --config config/base.yaml --model tft

train-mamba:
	python scripts/train_models.py --config config/base.yaml --model mamba

train-ensemble:
	python scripts/train_models.py --config config/base.yaml --model ensemble

# Optimization
optimize:
	python scripts/train_models.py --config config/base.yaml --optimize

# Backtesting
backtest:
	python scripts/run_backtest.py --config config/base.yaml

walk-forward:
	python scripts/run_backtest.py --config config/base.yaml --walk-forward

# Live trading
live:
	python scripts/live_trading.py --config config/production.yaml

live-paper:
	python scripts/live_trading.py --config config/production.yaml --paper

# MLflow
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///experiments/mlruns/mlflow.db --port 5000

# Docker
docker-build:
	docker-compose -f deployment/docker/docker-compose.yml build

docker-up:
	docker-compose -f deployment/docker/docker-compose.yml up -d

docker-down:
	docker-compose -f deployment/docker/docker-compose.yml down

docker-logs:
	docker-compose -f deployment/docker/docker-compose.yml logs -f

# Jupyter
notebook:
	jupyter lab --notebook-dir=notebooks

# Help
help:
	@echo "Available commands:"
	@echo "  install        - Install package"
	@echo "  install-dev    - Install with dev dependencies"
	@echo "  install-cuda   - Install PyTorch with CUDA support"
	@echo "  setup          - Full development setup"
	@echo "  clean          - Remove build artifacts"
	@echo "  test           - Run all tests"
	@echo "  lint           - Run linters"
	@echo "  format         - Format code"
	@echo "  collect-data   - Collect market data"
	@echo "  build-features - Build feature store"
	@echo "  train          - Train all models"
	@echo "  backtest       - Run backtests"
	@echo "  live           - Start live trading"
	@echo "  docker-up      - Start Docker containers"
