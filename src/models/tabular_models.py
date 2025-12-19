"""Tabular data models (XGBoost, CatBoost, LightGBM)."""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from loguru import logger
from sklearn.metrics import classification_report, mean_absolute_error, r2_score


class TabularXGBoost:
    """XGBoost for tabular data."""
    
    def __init__(self, task: str = "classification", **params):
        self.task = task
        default_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42
        }
        default_params.update(params)
        
        if task == "classification":
            self.model = xgb.XGBClassifier(**default_params)
        else:
            self.model = xgb.XGBRegressor(**default_params)
    
    def train(self, X: pd.DataFrame, y: np.ndarray,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None):
        """Train the model."""
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X, y)
        
        logger.info(f"Trained XGBoost on {len(X)} samples")
    
    def predict(self, X: pd.DataFrame):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame):
        """Predict probabilities (classification only)."""
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_
        else:
            feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
        
        return dict(zip(feature_names, self.model.feature_importances_))


class TabularCatBoost:
    """CatBoost for tabular data."""
    
    def __init__(self, task: str = "classification", **params):
        self.task = task
        default_params = {
            "iterations": 200,
            "learning_rate": 0.1,
            "depth": 6,
            "random_seed": 42,
            "verbose": False
        }
        default_params.update(params)
        
        if task == "classification":
            self.model = cb.CatBoostClassifier(**default_params)
        else:
            self.model = cb.CatBoostRegressor(**default_params)
    
    def train(self, X: pd.DataFrame, y: np.ndarray,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None,
              cat_features: Optional[list] = None):
        """Train the model."""
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=(X_val, y_val),
                cat_features=cat_features
            )
        else:
            self.model.fit(X, y, cat_features=cat_features)
        
        logger.info(f"Trained CatBoost on {len(X)} samples")
    
    def predict(self, X: pd.DataFrame):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame):
        """Predict probabilities (classification only)."""
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)


class TabularLightGBM:
    """LightGBM for tabular data."""
    
    def __init__(self, task: str = "classification", **params):
        self.task = task
        default_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
            "verbose": -1
        }
        default_params.update(params)
        
        if task == "classification":
            self.model = lgb.LGBMClassifier(**default_params)
        else:
            self.model = lgb.LGBMRegressor(**default_params)
    
    def train(self, X: pd.DataFrame, y: np.ndarray,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None,
              cat_features: Optional[list] = None):
        """Train the model."""
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                categorical_feature=cat_features
            )
        else:
            self.model.fit(X, y, categorical_feature=cat_features)
        
        logger.info(f"Trained LightGBM on {len(X)} samples")
    
    def predict(self, X: pd.DataFrame):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame):
        """Predict probabilities (classification only)."""
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)

