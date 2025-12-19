"""Configuration loader utilities."""
import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables if present
    if os.getenv("RAW_DATA_DIR"):
        config["data"]["raw_dir"] = os.getenv("RAW_DATA_DIR")
    if os.getenv("PROCESSED_DATA_DIR"):
        config["data"]["processed_dir"] = os.getenv("PROCESSED_DATA_DIR")
    
    return config


def get_data_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Get data directory paths from config."""
    base_path = Path.cwd()
    return {
        "raw": base_path / config["data"]["raw_dir"],
        "processed": base_path / config["data"]["processed_dir"],
        "labels": base_path / config["data"]["labels_dir"],
        "models": base_path / config["data"]["models_dir"],
        "checkpoints": base_path / config["data"]["checkpoints_dir"],
    }

