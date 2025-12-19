"""Main preprocessing orchestrator."""
import argparse
from pathlib import Path
from loguru import logger

from ..utils.config import load_config, get_data_paths
from .spectra import SpectraPreprocessor
from .images import ImagePreprocessor
from .tabular import TabularPreprocessor


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Microplastic data preprocessing pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data-type", type=str, choices=["spectra", "images", "tabular", "all"],
                       default="all", help="Type of data to preprocess")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    paths = get_data_paths(config)
    
    # Setup logging
    logger.add(
        config.get("logging", {}).get("file", "logs/preprocessing.log"),
        rotation="10 MB",
        retention="7 days"
    )
    
    logger.info("Starting data preprocessing pipeline")
    
    # Process spectra
    if args.data_type in ["spectra", "all"]:
        logger.info("Preprocessing spectral data...")
        spectra_config = config.get("preprocessing", {}).get("spectra", {})
        spectra_preprocessor = SpectraPreprocessor(spectra_config)
        spectra_preprocessor.batch_preprocess(
            paths["raw"],
            paths["processed"] / "spectra"
        )
    
    # Process images
    if args.data_type in ["images", "all"]:
        logger.info("Preprocessing image data...")
        images_config = config.get("preprocessing", {}).get("images", {})
        image_preprocessor = ImagePreprocessor(images_config)
        image_preprocessor.batch_preprocess(
            paths["raw"],
            paths["processed"] / "images"
        )
    
    # Process tabular
    if args.data_type in ["tabular", "all"]:
        logger.info("Preprocessing tabular data...")
        tabular_config = config.get("preprocessing", {}).get("tabular", {})
        tabular_preprocessor = TabularPreprocessor(tabular_config)
        tabular_preprocessor.batch_preprocess(
            paths["raw"],
            paths["processed"] / "tabular"
        )
    
    logger.info("Data preprocessing pipeline completed")


if __name__ == "__main__":
    main()

