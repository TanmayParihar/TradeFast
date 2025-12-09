# Crypto Trading Bot

Multi-modal hybrid architecture for cryptocurrency algorithmic trading on Binance Futures.

## Features

- **Multi-Modal Data Fusion**: OHLCV, Order Book, On-Chain metrics, and Sentiment analysis
- **Hybrid Neural Networks**: Temporal Fusion Transformer (TFT) + Mamba State Space Model
- **Ensemble Learning**: Stacking with LightGBM/XGBoost meta-learner
- **Meta-Labeling**: Triple Barrier Method with position sizing via Kelly Criterion
- **Advanced Validation**: Purged K-Fold CV with embargo, Walk-Forward Optimization
- **Hardware Optimized**: Designed for RTX 4070 (8GB VRAM) + Intel i9-13900H

## Architecture

```
Data Ingestion → Feature Engineering → Multi-Modal Fusion → Model Ensemble
                                                              ↓
Live Execution ← Risk Management ← Meta-Labeling ← Signal Generation
```

## Quick Start

```bash
# Create conda environment
conda create -n crypto_bot python=3.11 -y
conda activate crypto_bot

# Install PyTorch with CUDA
make install-cuda

# Install package
make install-dev

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Collect data
make collect-data

# Build features
make build-features

# Train models
make train

# Run backtest
make backtest

# Start live trading (paper mode)
make live-paper
```

## Project Structure

```
crypto_trading_bot/
├── config/              # Configuration files
├── data/                # Data storage
├── src/                 # Source code
│   ├── data/           # Data ingestion & processing
│   ├── features/       # Feature engineering
│   ├── models/         # ML models
│   ├── training/       # Training infrastructure
│   ├── strategy/       # Trading strategy
│   ├── execution/      # Order execution
│   ├── backtest/       # Backtesting engine
│   └── utils/          # Utilities
├── scripts/            # Entry point scripts
├── notebooks/          # Jupyter notebooks
├── experiments/        # MLflow & Optuna results
├── tests/              # Unit & integration tests
└── deployment/         # Docker & systemd files
```

## Supported Assets

- BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, ADAUSDT
- AVAXUSDT, DOGEUSDT, DOTUSDT, MATICUSDT, XRPUSDT

## Requirements

- Python 3.11+
- CUDA 12.1+ (for GPU acceleration)
- 8GB+ VRAM (optimized for RTX 4070)
- 32GB+ RAM recommended

## License

MIT License
