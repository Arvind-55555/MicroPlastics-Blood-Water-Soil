"""PubMed API data ingestion for research publications."""
from pathlib import Path
from typing import List, Dict, Any
import requests
import time
from loguru import logger
from .base import APISource


class PubMedSource(APISource):
    """Ingest data from PubMed publications and supplemental materials."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__("pubmed", config, output_dir)
        self.search_queries = config.get("search_queries", [])
        self.email = config.get("email", "user@example.com")  # Required by NCBI
    
    def discover(self) -> List[Dict[str, Any]]:
        """Search PubMed for relevant publications."""
        all_results = []
        
        for query in self.search_queries:
            logger.info(f"Searching PubMed for: {query}")
            
            # Search for publications
            search_url = f"{self.base_url}/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": 100,
                "retmode": "json",
                "email": self.email
            }
            
            response = self._make_request(search_url, params=params)
            
            if "esearchresult" in response and "idlist" in response["esearchresult"]:
                pmids = response["esearchresult"]["idlist"]
                logger.info(f"Found {len(pmids)} publications")
                
                # Fetch details for each publication
                for pmid in pmids[:20]:  # Limit to first 20 to avoid rate limiting
                    time.sleep(0.34)  # NCBI rate limit: 3 requests/second
                    
                    fetch_url = f"{self.base_url}/efetch.fcgi"
                    fetch_params = {
                        "db": "pubmed",
                        "id": pmid,
                        "retmode": "xml"
                    }
                    
                    # Get summary instead (lighter)
                    summary_url = f"{self.base_url}/esummary.fcgi"
                    summary_params = {
                        "db": "pubmed",
                        "id": pmid,
                        "retmode": "json"
                    }
                    
                    summary = self._make_request(summary_url, params=summary_params)
                    
                    if "result" in summary and pmid in summary["result"]:
                        pub_data = summary["result"][pmid]
                        result = {
                            "pmid": pmid,
                            "title": pub_data.get("title", ""),
                            "authors": pub_data.get("authors", []),
                            "pub_date": pub_data.get("pubdate", ""),
                            "journal": pub_data.get("source", ""),
                            "doi": pub_data.get("elocationid", ""),
                            "source": "pubmed"
                        }
                        all_results.append(result)
                        logger.info(f"Found: {result['title']}")
        
        logger.info(f"Discovered {len(all_results)} publications from PubMed")
        return all_results
    
    def ingest(self, resource: Dict[str, Any]) -> Path:
        """Download supplemental data if available."""
        pmid = resource["pmid"]
        pub_dir = self.output_dir / f"pubmed_{pmid}"
        pub_dir.mkdir(parents=True, exist_ok=True)
        
        # Save publication metadata
        import json
        metadata_file = pub_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(resource, f, indent=2)
        
        # Note: Actual supplemental data download would require
        # parsing publication pages or using specific APIs
        # This is a placeholder for the structure
        logger.info(f"Saved metadata for PubMed ID {pmid}")
        logger.warning("Supplemental data download requires manual extraction or specific APIs")
        
        return pub_dir

