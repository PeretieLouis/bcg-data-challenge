"""
Utility Functions
=================
Shared helper functions for the pipeline.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_PATH = PROJECT_ROOT / "configs"


def load_config(config_name: str = "config") -> dict:
    """Load configuration from YAML file."""
    config_path = CONFIGS_PATH / f"{config_name}.yaml"

    if not config_path.exists():
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_name: str = "config") -> None:
    """Save configuration to YAML file."""
    CONFIGS_PATH.mkdir(parents=True, exist_ok=True)
    config_path = CONFIGS_PATH / f"{config_name}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✓ Config saved to {config_path}")
