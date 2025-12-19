"""Multimodal fusion models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
from loguru import logger


class MultimodalFusion(nn.Module):
    """Multimodal fusion model combining spectra, images, and tabular data."""
    
    def __init__(self, 
                 spectra_embedding_dim: int = 128,
                 image_embedding_dim: int = 512,
                 tabular_dim: int = 64,
                 fusion_type: Literal["early", "late", "attention"] = "late",
                 num_classes: int = 2,
                 task: str = "classification",
                 dropout: float = 0.3):
        super().__init__()
        self.fusion_type = fusion_type
        self.task = task
        self.num_classes = num_classes
        
        # Embedding layers
        self.spectra_embedding = nn.Sequential(
            nn.Linear(spectra_embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64)
        )
        
        self.image_embedding = nn.Sequential(
            nn.Linear(image_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        
        self.tabular_embedding = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )
        
        if fusion_type == "early":
            # Concatenate all embeddings
            fused_dim = 64 + 128 + 32
            self.fusion_layer = nn.Sequential(
                nn.Linear(fused_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128)
            )
        elif fusion_type == "late":
            # Process each modality separately, then combine
            self.fusion_layer = nn.Sequential(
                nn.Linear(64 + 128 + 32, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128)
            )
        elif fusion_type == "attention":
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=64,  # Use smallest embedding dim
                num_heads=4,
                batch_first=True
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(64 + 128 + 32, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Output layer
        if task == "classification":
            self.output = nn.Linear(128, num_classes)
        else:
            self.output = nn.Linear(128, 1)
    
    def forward(self, spectra_emb: torch.Tensor, 
                image_emb: torch.Tensor,
                tabular_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with multimodal inputs."""
        # Embed each modality
        spectra_emb = self.spectra_embedding(spectra_emb)
        image_emb = self.image_embedding(image_emb)
        tabular_emb = self.tabular_embedding(tabular_emb)
        
        if self.fusion_type == "early":
            # Concatenate early
            fused = torch.cat([spectra_emb, image_emb, tabular_emb], dim=1)
            fused = self.fusion_layer(fused)
        
        elif self.fusion_type == "late":
            # Concatenate after processing
            fused = torch.cat([spectra_emb, image_emb, tabular_emb], dim=1)
            fused = self.fusion_layer(fused)
        
        elif self.fusion_type == "attention":
            # Attention mechanism
            # Reshape for attention (batch, seq_len, embed_dim)
            # Use spectra_emb as query, others as key/value
            seq_emb = torch.stack([spectra_emb, image_emb, tabular_emb], dim=1)
            # Pad to same dimension for attention
            # For simplicity, use concatenation with attention-like weighting
            fused = torch.cat([spectra_emb, image_emb, tabular_emb], dim=1)
            fused = self.fusion_layer(fused)
        
        # Output
        output = self.output(fused)
        
        if self.task == "classification":
            return F.log_softmax(output, dim=1)
        else:
            return output


class StackedEnsemble:
    """Stacked ensemble combining multiple models."""
    
    def __init__(self, base_models: list, meta_model, task: str = "classification"):
        self.base_models = base_models
        self.meta_model = meta_model
        self.task = task
    
    def fit(self, X_spectra, X_images, X_tabular, y, 
            X_val_spectra=None, X_val_images=None, X_val_tabular=None, y_val=None):
        """Train stacked ensemble."""
        # Train base models
        base_predictions = []
        for model in self.base_models:
            if hasattr(model, 'train'):
                model.train(X_spectra, X_images, X_tabular, y)
            
            # Get predictions for meta-model
            if hasattr(model, 'predict_proba') and self.task == "classification":
                preds = model.predict_proba(X_spectra, X_images, X_tabular)
            else:
                preds = model.predict(X_spectra, X_images, X_tabular)
                if self.task == "classification":
                    # Convert to probabilities
                    preds = np.eye(self.num_classes)[preds]
            
            base_predictions.append(preds)
        
        # Stack predictions
        meta_X = np.hstack(base_predictions)
        
        # Train meta-model
        if X_val_spectra is not None:
            val_predictions = []
            for model in self.base_models:
                if hasattr(model, 'predict_proba') and self.task == "classification":
                    preds = model.predict_proba(X_val_spectra, X_val_images, X_val_tabular)
                else:
                    preds = model.predict(X_val_spectra, X_val_images, X_val_tabular)
                    if self.task == "classification":
                        preds = np.eye(self.num_classes)[preds]
                val_predictions.append(preds)
            meta_X_val = np.hstack(val_predictions)
            self.meta_model.fit(meta_X, y, eval_set=[(meta_X_val, y_val)])
        else:
            self.meta_model.fit(meta_X, y)
    
    def predict(self, X_spectra, X_images, X_tabular):
        """Make predictions."""
        base_predictions = []
        for model in self.base_models:
            if hasattr(model, 'predict_proba') and self.task == "classification":
                preds = model.predict_proba(X_spectra, X_images, X_tabular)
            else:
                preds = model.predict(X_spectra, X_images, X_tabular)
                if self.task == "classification":
                    preds = np.eye(self.num_classes)[preds]
            base_predictions.append(preds)
        
        meta_X = np.hstack(base_predictions)
        return self.meta_model.predict(meta_X)

