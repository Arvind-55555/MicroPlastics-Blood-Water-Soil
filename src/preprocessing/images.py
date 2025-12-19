"""Image preprocessing for microscopy data."""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import cv2
from loguru import logger
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImagePreprocessor:
    """Preprocess microscopy images."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_size = tuple(config.get("target_size", [512, 512]))
        self.normalization = config.get("normalization", "imagenet")
        self.augmentation_config = config.get("augmentation", {})
        
        # ImageNet normalization stats
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
    
    def load_image(self, filepath: Path) -> np.ndarray:
        """Load image from file."""
        img = Image.open(filepath)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    
    def resize(self, image: np.ndarray, size: Tuple[int, int], 
              method: str = "bilinear") -> np.ndarray:
        """Resize image to target size."""
        if method == "bilinear":
            return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        elif method == "nearest":
            return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
        else:
            return cv2.resize(image, size)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values."""
        if self.normalization == "imagenet":
            # Convert to float and normalize
            image = image.astype(np.float32) / 255.0
            # Apply ImageNet normalization
            mean = np.array(self.imagenet_mean).reshape(1, 1, 3)
            std = np.array(self.imagenet_std).reshape(1, 1, 3)
            image = (image - mean) / std
        elif self.normalization == "custom":
            # Min-max normalization
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        else:
            # Just convert to float [0, 1]
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def background_subtraction(self, image: np.ndarray, 
                             method: str = "gaussian") -> np.ndarray:
        """Subtract background from image."""
        if method == "gaussian":
            # Apply Gaussian blur and subtract
            blurred = cv2.GaussianBlur(image, (21, 21), 0)
            return cv2.subtract(image, blurred)
        elif method == "morphological":
            # Morphological opening
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            return cv2.subtract(image, background)
        else:
            return image
    
    def enhance_contrast(self, image: np.ndarray, 
                        method: str = "clahe") -> np.ndarray:
        """Enhance image contrast."""
        if len(image.shape) == 3:
            # Convert to grayscale for CLAHE
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            if len(image.shape) == 3:
                # Convert back to RGB
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            return enhanced
        elif method == "histogram_eq":
            if len(image.shape) == 3:
                # Apply to each channel
                channels = cv2.split(image)
                eq_channels = [cv2.equalizeHist(ch) for ch in channels]
                return cv2.merge(eq_channels)
            else:
                return cv2.equalizeHist(image)
        else:
            return image
    
    def get_augmentation_transform(self, is_training: bool = True) -> A.Compose:
        """Get augmentation transform."""
        if not is_training or not self.augmentation_config:
            return A.Compose([
                A.Resize(self.target_size[0], self.target_size[1]),
                A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
                ToTensorV2()
            ])
        
        transforms = [A.Resize(self.target_size[0], self.target_size[1])]
        
        if self.augmentation_config.get("rotation"):
            transforms.append(A.Rotate(limit=self.augmentation_config["rotation"]))
        
        if self.augmentation_config.get("flip_horizontal"):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if self.augmentation_config.get("flip_vertical"):
            transforms.append(A.VerticalFlip(p=0.5))
        
        if self.augmentation_config.get("brightness"):
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=self.augmentation_config["brightness"],
                contrast_limit=self.augmentation_config.get("contrast", 0.2)
            ))
        
        transforms.extend([
            A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def preprocess(self, filepath: Path, apply_augmentation: bool = False) -> Dict[str, Any]:
        """Complete preprocessing pipeline for an image."""
        logger.info(f"Preprocessing image: {filepath}")
        
        # Load
        image = self.load_image(filepath)
        original_shape = image.shape
        
        # Resize
        image = self.resize(image, self.target_size)
        
        # Background subtraction (optional)
        # image = self.background_subtraction(image)
        
        # Contrast enhancement (optional)
        # image = self.enhance_contrast(image)
        
        # Normalize
        image = self.normalize_image(image)
        
        return {
            "image": image,
            "metadata": {
                "source_file": str(filepath),
                "original_shape": original_shape,
                "processed_shape": image.shape,
                "target_size": self.target_size
            }
        }
    
    def extract_patches(self, image: np.ndarray, patch_size: Tuple[int, int],
                       stride: Optional[int] = None) -> List[np.ndarray]:
        """Extract patches from image for particle detection."""
        if stride is None:
            stride = patch_size[0] // 2
        
        patches = []
        h, w = image.shape[:2]
        ph, pw = patch_size
        
        for y in range(0, h - ph + 1, stride):
            for x in range(0, w - pw + 1, stride):
                patch = image[y:y+ph, x:x+pw]
                patches.append(patch)
        
        return patches
    
    def batch_preprocess(self, input_dir: Path, output_dir: Path):
        """Preprocess all images in a directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.rglob(f"*{ext}"))
            image_files.extend(input_dir.rglob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} image files")
        
        processed_data = []
        for filepath in image_files:
            try:
                result = self.preprocess(filepath)
                
                # Save processed image
                output_file = output_dir / f"{filepath.stem}_processed.npy"
                np.save(output_file, result["image"])
                
                processed_data.append(result["metadata"])
                logger.info(f"Processed: {filepath.name}")
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
        
        # Save metadata
        if processed_data:
            import pandas as pd
            metadata_df = pd.DataFrame(processed_data)
            metadata_df.to_csv(output_dir / "metadata.csv", index=False)
        
        return processed_data

