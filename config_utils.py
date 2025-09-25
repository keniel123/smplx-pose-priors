#!/usr/bin/env python3
"""
Configuration utilities for loading YAML config files.
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import os


class ConfigLoader:
    """Utility class for loading and managing YAML configurations."""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_path.endswith('.yaml') and not config_path.endswith('.yml'):
            config_path += '.yaml'

        # Try relative to config_dir first, then absolute path
        if not Path(config_path).is_absolute():
            config_path = self.config_dir / config_path

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get_available_configs(self) -> list:
        """Get list of available config files."""
        if not self.config_dir.exists():
            return []

        configs = []
        for file in self.config_dir.glob("*.yaml"):
            configs.append(file.stem)
        for file in self.config_dir.glob("*.yml"):
            configs.append(file.stem)

        return sorted(configs)

    def save_config(self, config: Dict[str, Any], config_path: str):
        """Save configuration to YAML file."""
        if not config_path.endswith('.yaml') and not config_path.endswith('.yml'):
            config_path += '.yaml'

        if not Path(config_path).is_absolute():
            config_path = self.config_dir / config_path

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)


def create_config_parser() -> argparse.ArgumentParser:
    """Create argument parser for config-based training scripts."""
    parser = argparse.ArgumentParser(add_help=False)  # Don't add help to allow combining

    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file (e.g., configs/gaussian_prior.yaml or just gaussian_prior)')
    parser.add_argument('--override', type=str, nargs='*', default=[],
                       help='Override config values (e.g., --override training.learning_rate=1e-4 data.batch_size=64)')

    return parser


def override_config(config: Dict[str, Any], overrides: list) -> Dict[str, Any]:
    """Override configuration values from command line arguments."""
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}. Use key=value format.")

        key, value = override.split('=', 1)

        # Handle nested keys (e.g., training.learning_rate)
        keys = key.split('.')
        current = config

        # Navigate to parent dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the final value with type conversion
        final_key = keys[-1]

        # Try to convert value to appropriate type
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        elif value.lower() == 'null' or value.lower() == 'none':
            value = None
        else:
            try:
                # Try int first, then float
                if '.' not in value and 'e' not in value.lower():
                    value = int(value)
                else:
                    value = float(value)
            except ValueError:
                # Keep as string if conversion fails
                pass

        current[final_key] = value

    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations, with override_config taking precedence."""
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def print_config(config: Dict[str, Any], title: str = "Configuration"):
    """Pretty print configuration."""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print('='*60)

    def print_dict(d: Dict[str, Any], indent: int = 0):
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{'  '*indent}{key}:")
                print_dict(value, indent + 1)
            else:
                print(f"{'  '*indent}{key}: {value}")

    print_dict(config)
    print('='*60)


# Example usage and testing
if __name__ == "__main__":
    # Test config loader
    loader = ConfigLoader()

    print("Available configs:")
    for config in loader.get_available_configs():
        print(f"  - {config}")

    # Test loading a config
    try:
        config = loader.load_config("gaussian_prior")
        print_config(config, "Gaussian Prior Config")

        # Test override
        overrides = ["training.learning_rate=2e-3", "data.batch_size=64", "training.max_epochs=100"]
        config = override_config(config, overrides)
        print_config(config, "Gaussian Prior Config (with overrides)")

    except FileNotFoundError as e:
        print(f"Config not found: {e}")