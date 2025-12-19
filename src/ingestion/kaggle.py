"""Kaggle dataset ingestion."""
from pathlib import Path
from typing import List, Dict, Any
import os
from loguru import logger
from .base import DatasetSource


class KaggleSource(DatasetSource):
    """Ingest data from Kaggle datasets."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__("kaggle", config, output_dir)
        # Check for Kaggle credentials
        kaggle_dir = Path.home() / ".kaggle"
        if not (kaggle_dir / "kaggle.json").exists():
            logger.warning("Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY in .env")
    
    def discover(self) -> List[Dict[str, Any]]:
        """Discover Kaggle datasets."""
        try:
            import kaggle
        except ImportError:
            logger.error("kaggle package not installed. Install with: pip install kaggle")
            return []
        
        all_results = []
        
        # Search for microplastic datasets
        search_queries = [
            "microplastic",
            "FTIR microplastic",
            "Raman microplastic",
            "microplastic images"
        ]
        
        for query in search_queries:
            logger.info(f"Searching Kaggle for: {query}")
            try:
                datasets = kaggle.api.dataset_list(search=query)
                for dataset in datasets:
                    result = {
                        "id": dataset.ref,
                        "title": dataset.title,
                        "size": dataset.size,
                        "download_count": dataset.downloadCount,
                        "source": "kaggle"
                    }
                    all_results.append(result)
                    logger.info(f"Found: {result['title']} ({result['id']})")
            except Exception as e:
                logger.error(f"Error searching Kaggle: {e}")
        
        # Also check configured datasets
        for dataset_id in self.datasets:
            try:
                dataset = kaggle.api.dataset_metadata(dataset_id)
                result = {
                    "id": dataset_id,
                    "title": dataset.title,
                    "size": dataset.size,
                    "download_count": dataset.downloadCount,
                    "source": "kaggle"
                }
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error fetching dataset {dataset_id}: {e}")
        
        logger.info(f"Discovered {len(all_results)} datasets from Kaggle")
        return all_results
    
    def ingest(self, resource: Dict[str, Any]) -> Path:
        """Download a Kaggle dataset."""
        try:
            import kaggle
        except ImportError:
            logger.error("kaggle package not installed")
            return None
        
        dataset_id = resource["id"]
        dataset_dir = self.output_dir / f"kaggle_{dataset_id.replace('/', '_')}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading Kaggle dataset: {dataset_id}")
        try:
            kaggle.api.dataset_download_files(
                dataset_id,
                path=str(dataset_dir),
                unzip=True
            )
            logger.info(f"Downloaded dataset to {dataset_dir}")
            return dataset_dir
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_id}: {e}")
            return None

