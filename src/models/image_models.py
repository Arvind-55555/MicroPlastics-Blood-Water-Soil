"""Image models (EfficientNet, YOLOv8, ResNet)."""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple
from ultralytics import YOLO
from loguru import logger
import numpy as np


class EfficientNetClassifier(nn.Module):
    """EfficientNet-based classifier for microplastic images."""
    
    def __init__(self, num_classes: int = 2, model_size: str = "b0", 
                 pretrained: bool = True):
        super().__init__()
        
        # Load pretrained EfficientNet
        model_name = f"efficientnet_{model_size}"
        if model_size == "b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = 1280
        elif model_size == "b1":
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            in_features = 1280
        elif model_size == "b2":
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
            in_features = 1408
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)


class MicroplasticDetector:
    """YOLOv8-based object detector for microplastic particles."""
    
    def __init__(self, model_size: str = "n", pretrained: bool = True):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            pretrained: Use pretrained weights
        """
        model_name = f"yolov8{model_size}.pt" if pretrained else None
        self.model = YOLO(model_name)
        logger.info(f"Initialized YOLOv8 detector (size: {model_size})")
    
    def train(self, data_yaml: str, epochs: int = 50, imgsz: int = 640,
              batch_size: int = 16, **kwargs):
        """Train YOLOv8 model."""
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            **kwargs
        )
        logger.info(f"Training completed: {results}")
        return results
    
    def predict(self, image_path: str, conf_threshold: float = 0.5) -> dict:
        """Detect microplastics in an image."""
        results = self.model.predict(
            image_path,
            conf=conf_threshold,
            save=False
        )
        
        # Parse results
        result = results[0]
        detections = {
            "boxes": result.boxes.xyxy.cpu().numpy() if result.boxes is not None else [],
            "scores": result.boxes.conf.cpu().numpy() if result.boxes is not None else [],
            "classes": result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else [],
            "count": len(result.boxes) if result.boxes is not None else 0
        }
        
        return detections
    
    def predict_batch(self, image_paths: list, conf_threshold: float = 0.5) -> list:
        """Detect microplastics in multiple images."""
        results = self.model.predict(
            image_paths,
            conf=conf_threshold,
            save=False
        )
        
        all_detections = []
        for result in results:
            detections = {
                "boxes": result.boxes.xyxy.cpu().numpy() if result.boxes is not None else [],
                "scores": result.boxes.conf.cpu().numpy() if result.boxes is not None else [],
                "classes": result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else [],
                "count": len(result.boxes) if result.boxes is not None else 0
            }
            all_detections.append(detections)
        
        return all_detections


class ResNetClassifier(nn.Module):
    """ResNet-based classifier."""
    
    def __init__(self, num_classes: int = 2, model_depth: int = 18,
                 pretrained: bool = True):
        super().__init__()
        
        if model_depth == 18:
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif model_depth == 34:
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = 512
        elif model_depth == 50:
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"Unsupported ResNet depth: {model_depth}")
        
        # Replace classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)

