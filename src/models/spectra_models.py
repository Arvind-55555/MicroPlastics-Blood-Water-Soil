"""Spectral data models (1D CNN, RandomForest, XGBoost)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import xgboost as xgb
from loguru import logger


class Spectra1DCNN(nn.Module):
    """1D CNN for spectral classification/regression."""
    
    def __init__(self, input_length: int, num_classes: int = 2, 
                 filters: list = [64, 128, 256], kernel_sizes: list = [3, 5, 7],
                 dropout: float = 0.5, task: str = "classification"):
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for i, (filters_out, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, filters_out, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(filters_out),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
                )
            )
            in_channels = filters_out
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            dummy_output = self._forward_conv(dummy_input)
            flattened_size = dummy_output.numel()
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        if task == "classification":
            self.output = nn.Linear(64, num_classes)
        else:
            self.output = nn.Linear(64, 1)
    
    def _forward_conv(self, x):
        """Forward through convolutional layers."""
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x
    
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch, 1, length)
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        x = self.output(x)
        
        if self.task == "classification":
            return F.log_softmax(x, dim=1)
        else:
            return x


class SpectraRandomForest:
    """Random Forest for spectral data using engineered features."""
    
    def __init__(self, task: str = "classification", n_estimators: int = 100):
        self.task = task
        if task == "classification":
            self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    
    def extract_features(self, spectra: np.ndarray) -> np.ndarray:
        """Extract features from spectra (peaks, statistics, etc.)."""
        from scipy.signal import find_peaks
        
        features = []
        for spectrum in spectra:
            spec_features = []
            
            # Statistical features
            spec_features.extend([
                np.mean(spectrum),
                np.std(spectrum),
                np.median(spectrum),
                np.min(spectrum),
                np.max(spectrum),
                np.percentile(spectrum, 25),
                np.percentile(spectrum, 75),
            ])
            
            # Peak features
            peaks, properties = find_peaks(spectrum, height=np.mean(spectrum))
            spec_features.extend([
                len(peaks),
                np.mean(spectrum[peaks]) if len(peaks) > 0 else 0,
                np.max(spectrum[peaks]) if len(peaks) > 0 else 0,
            ])
            
            # Spectral moments
            wavelengths = np.arange(len(spectrum))
            spec_features.extend([
                np.sum(spectrum * wavelengths) / (np.sum(spectrum) + 1e-8),  # Centroid
                np.sqrt(np.sum(spectrum * (wavelengths - spec_features[-1])**2) / (np.sum(spectrum) + 1e-8)),  # Spread
            ])
            
            features.append(spec_features)
        
        return np.array(features)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        X_features = self.extract_features(X)
        self.model.fit(X_features, y)
        logger.info(f"Trained RandomForest on {len(X)} samples")
    
    def predict(self, X: np.ndarray):
        """Make predictions."""
        X_features = self.extract_features(X)
        return self.model.predict(X_features)
    
    def predict_proba(self, X: np.ndarray):
        """Predict probabilities (classification only)."""
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        X_features = self.extract_features(X)
        return self.model.predict_proba(X_features)


class SpectraXGBoost:
    """XGBoost for spectral data."""
    
    def __init__(self, task: str = "classification", **params):
        self.task = task
        self.params = params
        
        if task == "classification":
            self.model = xgb.XGBClassifier(**params, random_state=42)
        else:
            self.model = xgb.XGBRegressor(**params, random_state=42)
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train the model."""
        # Use first N features or downsample if too many
        if X.shape[1] > 1000:
            # Downsample or use feature selection
            step = X.shape[1] // 1000
            X = X[:, ::step]
            if X_val is not None:
                X_val = X_val[:, ::step]
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X, y)
        
        logger.info(f"Trained XGBoost on {len(X)} samples")
    
    def predict(self, X: np.ndarray):
        """Make predictions."""
        if X.shape[1] > 1000:
            step = X.shape[1] // 1000
            X = X[:, ::step]
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray):
        """Predict probabilities (classification only)."""
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        if X.shape[1] > 1000:
            step = X.shape[1] // 1000
            X = X[:, ::step]
        return self.model.predict_proba(X)

