"""Main data ingestion orchestrator."""
import argparse
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
import yaml

from ..utils.config import load_config, get_data_paths
from .zenodo import ZenodoSource
from .kaggle import KaggleSource
from .pubmed import PubMedSource


def create_source(source_config: Dict[str, Any], output_dir: Path):
    """Create appropriate data source instance."""
    source_type = source_config.get("type", "api")
    name = source_config.get("name", "unknown")
    
    if name == "zenodo" and source_config.get("enabled", False):
        return ZenodoSource(source_config, output_dir)
    elif name == "kaggle" and source_config.get("enabled", False):
        return KaggleSource(source_config, output_dir)
    elif name == "pubmed" and source_config.get("enabled", False):
        return PubMedSource(source_config, output_dir)
    else:
        logger.warning(f"Unknown or disabled source: {name}")
        return None


def main():
    """Main ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Microplastic data ingestion pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--source", type=str, default=None,
                       help="Specific source to ingest (default: all enabled sources)")
    parser.add_argument("--discover-only", action="store_true",
                       help="Only discover resources, don't download")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    paths = get_data_paths(config)
    
    # Setup logging
    logger.add(
        config.get("logging", {}).get("file", "logs/ingestion.log"),
        rotation="10 MB",
        retention="7 days"
    )
    
    logger.info("Starting data ingestion pipeline")
    
    # Get enabled sources
    sources_config = config.get("ingestion", {}).get("sources", [])
    
    if args.source:
        sources_config = [s for s in sources_config if s.get("name") == args.source]
    
    # Process each source
    all_discovered = []
    for source_config in sources_config:
        if not source_config.get("enabled", False):
            continue
        
        source_name = source_config.get("name")
        logger.info(f"Processing source: {source_name}")
        
        source = create_source(source_config, paths["raw"] / source_name)
        if source is None:
            continue
        
        # Discover resources
        discovered = source.discover()
        all_discovered.extend(discovered)
        
        if args.discover_only:
            continue
        
        # Ingest resources
        for resource in discovered:
            try:
                output_path = source.ingest(resource)
                if output_path:
                    logger.info(f"Successfully ingested: {resource.get('title', resource.get('id', 'unknown'))}")
            except Exception as e:
                logger.error(f"Error ingesting resource: {e}")
    
    # Save discovery summary
    if all_discovered:
        import json
        summary_file = paths["raw"] / "discovery_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_discovered, f, indent=2)
        logger.info(f"Saved discovery summary to {summary_file}")
    
    logger.info("Data ingestion pipeline completed")


if __name__ == "__main__":
    main()

