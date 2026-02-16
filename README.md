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
├── .github/
│   └── workflows/
│       └── ci.yaml          # GitHub Actions CI/CD pipeline
├── data/                     # Data files directory
├── src/                      # Source code directory
├── .gitignore               # Git ignore rules
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
└── README.md                # This file
```

## Setup

### Prerequisites

- Python 3.13
- [UV](https://github.com/astral-sh/uv) for dependency management

### Installation

1. Install UV:
```bash
pip install pipx && pipx ensurepath
pipx install uv
```

2. Install dependencies:
```bash
uv sync
```

3. Install pre-commit hooks:
```bash
source .venv/bin/activate # if Linux environment
source .venv/Scripts/activate # if windows environment
pip install pre-commit
pre-commit install
```

## Development

### Pre-commit Hooks

This project uses pre-commit hooks to maintain code quality:

- **Code Quality**:
  - `ruff`: Fast Python linter with auto-fix
  - `ruff-format`: Python code formatter

- **File Validation**:
  - `trailing-whitespace`: Remove trailing whitespace
  - `end-of-file-fixer`: Ensure files end with a newline
  - `check-toml`: Validate TOML files
  - `check-yaml`: Validate YAML files
  - `check-json`: Validate JSON files
  - `check-added-large-files`: Prevent large files from being committed

- **Notebook Management**:
  - `nbstripout`: Strip output from Jupyter notebooks

- **Dependency Management**:
  - `uv-lock`: Keep dependency lock file in sync

### Running Pre-commit Hooks Manually

```bash
# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
```

## CI/CD

GitHub Actions automatically runs on every push and pull request:
- Sets up Python 3.13
- Installs dependencies via UV
- Runs all pre-commit hooks with pre-push stage
