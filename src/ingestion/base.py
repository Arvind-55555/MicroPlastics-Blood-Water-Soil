"""Base classes for data ingestion."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from loguru import logger


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, name: str, config: Dict[str, Any], output_dir: Path):
        self.name = name
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized data source: {name}")
    
    @abstractmethod
    def discover(self) -> List[Dict[str, Any]]:
        """Discover available datasets/resources."""
        pass
    
    @abstractmethod
    def ingest(self, resource: Dict[str, Any]) -> Path:
        """Ingest a specific resource."""
        pass
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str = "metadata.json"):
        """Save metadata about ingested data."""
        import json
        metadata_file = self.output_dir / filename
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")


class APISource(DataSource):
    """Base class for API-based data sources."""
    
    def __init__(self, name: str, config: Dict[str, Any], output_dir: Path):
        super().__init__(name, config, output_dir)
        self.base_url = config.get("base_url", "")
        self.api_key = config.get("api_key")
        self.session = None
    
    def _make_request(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict:
        """Make HTTP request with error handling."""
        import requests
        if headers is None:
            headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return {}


class DatasetSource(DataSource):
    """Base class for dataset repositories (Kaggle, Zenodo, etc.)."""
    
    def __init__(self, name: str, config: Dict[str, Any], output_dir: Path):
        super().__init__(name, config, output_dir)
        self.datasets = config.get("datasets", [])
        self.search_queries = config.get("search_queries", [])

