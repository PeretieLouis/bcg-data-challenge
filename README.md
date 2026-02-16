# BCG Data Challenge

A Python 3.13 project using UV for dependency management.

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
source .venv/bin/activate
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
