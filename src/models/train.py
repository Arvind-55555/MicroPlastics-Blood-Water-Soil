"""Model training script."""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from loguru import logger
import mlflow
import mlflow.pytorch

from ..utils.config import load_config, get_data_paths
from .spectra_models import Spectra1DCNN, SpectraRandomForest, SpectraXGBoost
from .image_models import EfficientNetClassifier, MicroplasticDetector
from .tabular_models import TabularXGBoost
from .multimodal import MultimodalFusion


def train_spectra_model(config, data_paths):
    """Train spectral classification model."""
    logger.info("Training spectral model...")
    
    # Load preprocessed spectra data
    spectra_dir = data_paths["processed"] / "spectra"
    if not spectra_dir.exists():
        logger.error(f"Spectra data not found at {spectra_dir}")
        return
    
    # Load spectra files (simplified - in practice, load from actual data)
    # This is a placeholder structure
    logger.warning("Spectra data loading needs to be implemented based on actual data format")
    
    model_config = config["models"]["spectra"]
    model_type = model_config.get("type", "1d_cnn")
    
    if model_type == "1d_cnn":
        # Example training (needs actual data)
        input_length = 3600  # Example: 400-4000 cm^-1 at 1 cm^-1 resolution
        num_classes = 2  # Binary classification
        
        model = Spectra1DCNN(
            input_length=input_length,
            num_classes=num_classes,
            filters=model_config["architecture"]["filters"],
            kernel_sizes=model_config["architecture"]["kernel_sizes"],
            dropout=model_config["architecture"]["dropout"],
            task="classification"
        )
        
        logger.info("Spectra CNN model created (training needs actual data)")
    
    elif model_type == "random_forest":
        model = SpectraRandomForest(task="classification")
        logger.info("RandomForest model created (training needs actual data)")
    
    elif model_type == "xgboost":
        model = SpectraXGBoost(task="classification", **model_config.get("params", {}))
        logger.info("XGBoost model created (training needs actual data)")


def train_image_model(config, data_paths):
    """Train image classification/detection model."""
    logger.info("Training image model...")
    
    images_dir = data_paths["processed"] / "images"
    if not images_dir.exists():
        logger.error(f"Image data not found at {images_dir}")
        return
    
    model_config = config["models"]["images"]
    
    # Classification model
    if "classification" in model_config:
        num_classes = 2
        model = EfficientNetClassifier(
            num_classes=num_classes,
            model_size=model_config["classification"].get("model_size", "b0"),
            pretrained=True
        )
        logger.info("Image classification model created (training needs actual data)")
    
    # Detection model (YOLOv8)
    if "detection" in model_config:
        detector = MicroplasticDetector(
            model_size=model_config["detection"].get("model_size", "n"),
            pretrained=True
        )
        logger.info("YOLOv8 detector created (training needs labeled data in YOLO format)")


def train_tabular_model(config, data_paths):
    """Train tabular model."""
    logger.info("Training tabular model...")
    
    tabular_dir = data_paths["processed"] / "tabular"
    combined_file = tabular_dir / "combined_processed.parquet"
    
    if not combined_file.exists():
        logger.error(f"Tabular data not found at {combined_file}")
        return
    
    # Load data
    df = pd.read_parquet(combined_file)
    logger.info(f"Loaded {len(df)} samples from tabular data")
    
    # Separate features and target (assuming 'concentration' or 'presence' column)
    # This is a placeholder - actual column names depend on data
    if "concentration" in df.columns:
        target_col = "concentration"
        task = "regression"
    elif "presence" in df.columns:
        target_col = "presence"
        task = "classification"
    else:
        logger.warning("No target column found, skipping training")
        return
    
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model_config = config["models"]["tabular"]
    model = TabularXGBoost(task=task, **model_config.get("params", {}))
    model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    y_pred = model.predict(X_test)
    if task == "classification":
        logger.info("\n" + classification_report(y_test, y_pred))
    else:
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Save model
    import joblib
    model_path = data_paths["models"] / "tabular_xgboost.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train microplastic detection models")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model-type", type=str, 
                       choices=["spectra", "images", "tabular", "multimodal", "all"],
                       default="all", help="Type of model to train")
    parser.add_argument("--mlflow", action="store_true",
                       help="Log to MLflow")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    paths = get_data_paths(config)
    
    # Setup logging
    logger.add(
        config.get("logging", {}).get("file", "logs/training.log"),
        rotation="10 MB",
        retention="7 days"
    )
    
    # Setup MLflow
    if args.mlflow:
        mlflow.set_tracking_uri(config.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(config.get("MLFLOW_EXPERIMENT_NAME", "microplastics_detection"))
    
    logger.info("Starting model training pipeline")
    
    # Train models
    if args.model_type in ["spectra", "all"]:
        if args.mlflow:
            with mlflow.start_run(run_name="spectra_training"):
                train_spectra_model(config, paths)
        else:
            train_spectra_model(config, paths)
    
    if args.model_type in ["images", "all"]:
        if args.mlflow:
            with mlflow.start_run(run_name="image_training"):
                train_image_model(config, paths)
        else:
            train_image_model(config, paths)
    
    if args.model_type in ["tabular", "all"]:
        if args.mlflow:
            with mlflow.start_run(run_name="tabular_training"):
                train_tabular_model(config, paths)
        else:
            train_tabular_model(config, paths)
    
    logger.info("Model training pipeline completed")


if __name__ == "__main__":
    main()

