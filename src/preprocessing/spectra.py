"""Spectral data preprocessing (FTIR, Raman)."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from scipy import signal
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from loguru import logger


class SpectraPreprocessor:
    """Preprocess spectral data (FTIR, Raman)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.wavelength_range = config.get("target_wavelength_range", [400, 4000])
        self.resolution = config.get("target_resolution", 1.0)
        self.baseline_correction = config.get("baseline_correction", True)
        self.smoothing = config.get("smoothing", True)
        self.normalization = config.get("normalization", "minmax")
    
    def load_spectrum(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load spectrum from file (CSV, TXT, or JCAMP-DX)."""
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.csv':
            df = pd.read_csv(filepath)
            # Assume first column is wavelength, second is intensity
            wavelength = df.iloc[:, 0].values
            intensity = df.iloc[:, 1].values
        elif filepath.suffix.lower() in ['.txt', '.dat']:
            df = pd.read_csv(filepath, sep='\s+', header=None)
            wavelength = df.iloc[:, 0].values
            intensity = df.iloc[:, 1].values
        else:
            # Try generic CSV
            df = pd.read_csv(filepath)
            wavelength = df.iloc[:, 0].values
            intensity = df.iloc[:, 1].values
        
        return wavelength, intensity
    
    def baseline_correct(self, intensity: np.ndarray, 
                        method: str = "modpoly") -> np.ndarray:
        """Correct baseline using modified polynomial or asymmetric least squares."""
        if method == "modpoly":
            # Modified polynomial baseline correction
            degree = 3
            baseline = np.polyval(
                np.polyfit(np.arange(len(intensity)), intensity, degree),
                np.arange(len(intensity))
            )
            return intensity - baseline
        elif method == "als":
            # Asymmetric Least Squares
            lam = 1e5
            p = 0.001
            n_iter = 10
            
            L = len(intensity)
            D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            w = np.ones(L)
            
            for _ in range(n_iter):
                W = diags(w, 0, shape=(L, L))
                Z = W + lam * D.dot(D.transpose())
                z = spsolve(Z, w * intensity)
                w = p * (intensity > z) + (1 - p) * (intensity < z)
            
            return intensity - z
        else:
            return intensity
    
    def smooth(self, intensity: np.ndarray, 
              method: str = "savgol", window_length: int = 11) -> np.ndarray:
        """Smooth spectrum using Savitzky-Golay or moving average."""
        if method == "savgol":
            if len(intensity) < window_length:
                window_length = len(intensity) - 1 if len(intensity) % 2 == 0 else len(intensity)
                if window_length < 3:
                    return intensity
            return signal.savgol_filter(intensity, window_length, 3)
        elif method == "moving_average":
            return np.convolve(intensity, np.ones(window_length)/window_length, mode='same')
        else:
            return intensity
    
    def normalize(self, intensity: np.ndarray, method: str = "minmax") -> np.ndarray:
        """Normalize intensity values."""
        if method == "minmax":
            min_val = np.min(intensity)
            max_val = np.max(intensity)
            if max_val - min_val > 0:
                return (intensity - min_val) / (max_val - min_val)
            return intensity
        elif method == "zscore":
            mean = np.mean(intensity)
            std = np.std(intensity)
            if std > 0:
                return (intensity - mean) / std
            return intensity
        elif method == "robust":
            median = np.median(intensity)
            mad = np.median(np.abs(intensity - median))
            if mad > 0:
                return (intensity - median) / mad
            return intensity
        else:
            return intensity
    
    def resample(self, wavelength: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resample to common wavelength grid."""
        wmin, wmax = self.wavelength_range
        new_wavelength = np.arange(wmin, wmax + self.resolution, self.resolution)
        new_intensity = np.interp(new_wavelength, wavelength, intensity)
        return new_wavelength, new_intensity
    
    def preprocess(self, filepath: Path) -> Dict[str, Any]:
        """Complete preprocessing pipeline for a spectrum."""
        logger.info(f"Preprocessing spectrum: {filepath}")
        
        # Load
        wavelength, intensity = self.load_spectrum(filepath)
        
        # Baseline correction
        if self.baseline_correction:
            intensity = self.baseline_correct(intensity)
        
        # Smoothing
        if self.smoothing:
            intensity = self.smooth(intensity)
        
        # Resample
        if self.config.get("resample_wavelengths", True):
            wavelength, intensity = self.resample(wavelength, intensity)
        
        # Normalization
        intensity = self.normalize(intensity, method=self.normalization)
        
        return {
            "wavelength": wavelength,
            "intensity": intensity,
            "metadata": {
                "source_file": str(filepath),
                "n_points": len(wavelength),
                "wavelength_range": [float(wavelength.min()), float(wavelength.max())]
            }
        }
    
    def batch_preprocess(self, input_dir: Path, output_dir: Path):
        """Preprocess all spectra in a directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all spectrum files
        extensions = ['.csv', '.txt', '.dat', '.jdx', '.dx']
        spectrum_files = []
        for ext in extensions:
            spectrum_files.extend(input_dir.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(spectrum_files)} spectrum files")
        
        processed_data = []
        for filepath in spectrum_files:
            try:
                result = self.preprocess(filepath)
                
                # Save processed spectrum
                output_file = output_dir / f"{filepath.stem}_processed.csv"
                df = pd.DataFrame({
                    "wavelength": result["wavelength"],
                    "intensity": result["intensity"]
                })
                df.to_csv(output_file, index=False)
                
                processed_data.append(result["metadata"])
                logger.info(f"Processed: {filepath.name}")
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
        
        # Save metadata
        if processed_data:
            metadata_df = pd.DataFrame(processed_data)
            metadata_df.to_csv(output_dir / "metadata.csv", index=False)
        
        return processed_data

