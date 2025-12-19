"""Zenodo API data ingestion."""
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import json
from loguru import logger
from .base import APISource


class ZenodoSource(APISource):
    """Ingest data from Zenodo repository."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__("zenodo", config, output_dir)
        self.search_queries = config.get("search_queries", [])
    
    def discover(self) -> List[Dict[str, Any]]:
        """Discover datasets on Zenodo."""
        all_results = []
        
        for query in self.search_queries:
            logger.info(f"Searching Zenodo for: {query}")
            url = f"{self.base_url}/records"
            params = {
                "q": query,
                "type": "dataset",
                "size": 100,
                "sort": "mostrecent"
            }
            
            response = self._make_request(url, params=params)
            
            if "hits" in response and "hits" in response["hits"]:
                for hit in response["hits"]["hits"]:
                    result = {
                        "id": hit.get("id"),
                        "title": hit.get("metadata", {}).get("title", ""),
                        "doi": hit.get("metadata", {}).get("doi", ""),
                        "creators": hit.get("metadata", {}).get("creators", []),
                        "publication_date": hit.get("metadata", {}).get("publication_date", ""),
                        "files": hit.get("files", []),
                        "keywords": hit.get("metadata", {}).get("keywords", []),
                        "description": hit.get("metadata", {}).get("description", ""),
                        "source": "zenodo"
                    }
                    all_results.append(result)
                    logger.info(f"Found: {result['title']} (ID: {result['id']})")
        
        logger.info(f"Discovered {len(all_results)} datasets from Zenodo")
        return all_results
    
    def ingest(self, resource: Dict[str, Any]) -> Path:
        """Download files from a Zenodo record."""
        record_id = resource["id"]
        files = resource.get("files", [])
        
        record_dir = self.output_dir / f"zenodo_{record_id}"
        record_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_file = record_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(resource, f, indent=2)
        
        # Download files
        downloaded_files = []
        for file_info in files:
            file_url = file_info.get("links", {}).get("self", "")
            filename = file_info.get("key", f"file_{len(downloaded_files)}")
            filepath = record_dir / filename
            
            if file_url:
                logger.info(f"Downloading {filename} from Zenodo...")
                try:
                    response = requests.get(file_url, stream=True, timeout=60)
                    response.raise_for_status()
                    
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    downloaded_files.append(str(filepath))
                    logger.info(f"Downloaded {filename}")
                except Exception as e:
                    logger.error(f"Failed to download {filename}: {e}")
        
        logger.info(f"Ingested {len(downloaded_files)} files from Zenodo record {record_id}")
        return record_dir

