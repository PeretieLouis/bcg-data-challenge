# BCG Data Challenge

<!-- Build & CI Status -->
![CI](https://github.com/auggy-ntn/urw-data-challenge/actions/workflows/ci.yaml/badge.svg?event=push)

<!-- Code Quality & Tools -->
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

<!-- Environment & Package Management -->
![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)


## Project Structure

```
bcg-data-challenge/
├── data/
│   ├── bronze/              # Raw data (CSV, parquet)
│   ├── silver/              # Cleaned data
│   └── gold/                # Feature-engineered data
├── src/
│   ├── data_processing/     # Data pipelines
│   │   ├── bronze_to_silver.py
│   │   └── silver_to_gold.py
│   ├── modeling/            # ML training & evaluation
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/               # Helper functions
├── results/                 # Model outputs & predictions
├── configs/                 # Configuration files
├── notebooks/               # Jupyter notebooks

```

## Setup

### Prerequisites

- Python 3.13+
- [UV](https://github.com/astral-sh/uv) for dependency management

### Installation

1. Install UV:
```bash
pip install pipx && pipx ensurepath
pipx install uv
```

2. Clone and install dependencies:
```bash
git clone <repo-url>
cd bcg-data-challenge
uv sync
```

3. Install pre-commit hooks:
```bash
uv run pre-commit install
```

## Usage

### 1. Data Processing Pipeline

**Bronze to Silver** (clean raw data):
```bash
uv run python src/data_processing/bronze_to_silver.py
```

**Silver to Gold** (feature engineering):
```bash
uv run python src/data_processing/silver_to_gold.py
```

### 2. Model Training

Train XGBoost with Optuna hyperparameter optimization:
```bash
uv run python src/modeling/train.py
```

For quick testing with fewer trials:
```bash
uv run python -c "from src.modeling.train import run_training; run_training(n_trials=20)"
```

### 3. Model Evaluation

View evaluation summary:
```bash
uv run python src/modeling/evaluate.py
```

### Full Pipeline

Run the complete pipeline:
```bash
# 1. Process data
uv run python src/data_processing/bronze_to_silver.py
uv run python src/data_processing/silver_to_gold.py

# 2. Train model (100 trials, ~10 min)
uv run python src/modeling/train.py

# 3. View results
uv run python src/modeling/evaluate.py
```

## Output Files

After training, results are saved to `results/`:

| File | Description |
|------|-------------|
| `model.joblib` | Trained XGBoost model |
| `best_params.json` | Optimized hyperparameters |
| `metrics.json` | Cross-validation metrics |
| `feature_importance.csv` | Feature rankings |
| `cv_predictions.csv` | CV predictions for analysis |
| `predictions_ssp*.csv` | Climate scenario predictions |


## Development

### Pre-commit Hooks

This project uses pre-commit hooks for code quality:

```bash
# Run on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff --all-files
```

Hooks include:
- `ruff`: Python linter with auto-fix
- `ruff-format`: Code formatter
- `nbstripout`: Strip notebook outputs
- `uv-lock`: Sync dependency lock file

## CI/CD

GitHub Actions automatically runs on every push and pull request:
- Sets up Python 3.13
- Installs dependencies via UV
- Runs all pre-commit hooks
