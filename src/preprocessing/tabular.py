"""Tabular data preprocessing."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


class TabularPreprocessor:
    """Preprocess tabular sample metadata."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.unit_normalization = config.get("unit_normalization", True)
        self.outlier_removal = config.get("outlier_removal", True)
        self.outlier_method = config.get("outlier_method", "iqr")
        self.missing_data_strategy = config.get("missing_data_strategy", "median")
    
    def normalize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize concentration units to standard format."""
        # Common unit conversions
        unit_mapping = {
            "particles/kg": 1.0,
            "particles/g": 1000.0,
            "particles/mg": 1000000.0,
            "particles/L": 1.0,
            "particles/mL": 1000.0,
            "particles/μL": 1000000.0,
        }
        
        if "concentration" in df.columns and "unit" in df.columns:
            df = df.copy()
            for unit, factor in unit_mapping.items():
                mask = df["unit"].str.contains(unit.split("/")[0], case=False, na=False)
                df.loc[mask, "concentration"] *= factor
                df.loc[mask, "unit"] = "particles/kg" if "kg" in unit else "particles/L"
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, 
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove outliers using IQR or Z-score method."""
        if columns is None:
            # Auto-detect numeric columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_clean = df.copy()
        n_removed = 0
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if self.outlier_method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                n_removed += (~mask).sum()
                df_clean = df_clean[mask]
            
            elif self.outlier_method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                mask = z_scores < 3.0
                n_removed += (~mask).sum()
                df_clean = df_clean[mask]
        
        if n_removed > 0:
            logger.info(f"Removed {n_removed} outlier rows")
        
        return df_clean.reset_index(drop=True)
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data according to strategy."""
        df_clean = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                if self.missing_data_strategy == "median":
                    fill_value = df[col].median()
                elif self.missing_data_strategy == "mean":
                    fill_value = df[col].mean()
                elif self.missing_data_strategy == "drop":
                    df_clean = df_clean.dropna(subset=[col])
                    continue
                else:
                    fill_value = 0
                
                df_clean[col].fillna(fill_value, inplace=True)
                logger.info(f"Filled {col} missing values with {self.missing_data_strategy}: {fill_value}")
        
        # For categorical columns, fill with mode or "unknown"
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                if self.missing_data_strategy == "drop":
                    df_clean = df_clean.dropna(subset=[col])
                else:
                    df_clean[col].fillna("unknown", inplace=True)
        
        return df_clean
    
    def encode_categorical(self, df: pd.DataFrame, 
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Encode categorical variables."""
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        df_encoded = df.copy()
        
        for col in columns:
            if col in df.columns:
                # One-hot encode if low cardinality, otherwise label encode
                n_unique = df[col].nunique()
                if n_unique <= 10:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded = df_encoded.drop(columns=[col])
                else:
                    # Label encoding
                    df_encoded[col] = pd.Categorical(df[col]).codes
        
        return df_encoded
    
    def preprocess(self, filepath: Path) -> pd.DataFrame:
        """Complete preprocessing pipeline for tabular data."""
        logger.info(f"Preprocessing tabular data: {filepath}")
        
        # Load
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
        elif filepath.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        elif filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Unit normalization
        if self.unit_normalization:
            df = self.normalize_units(df)
        
        # Handle missing data
        df = self.handle_missing_data(df)
        
        # Remove outliers
        if self.outlier_removal:
            df = self.remove_outliers(df)
        
        # Encode categorical
        df = self.encode_categorical(df)
        
        logger.info(f"Preprocessed data: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def batch_preprocess(self, input_dir: Path, output_dir: Path):
        """Preprocess all tabular files in a directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all tabular files
        extensions = ['.csv', '.xlsx', '.xls', '.parquet']
        tabular_files = []
        for ext in extensions:
            tabular_files.extend(input_dir.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(tabular_files)} tabular files")
        
        processed_dfs = []
        for filepath in tabular_files:
            try:
                df = self.preprocess(filepath)
                
                # Save processed data
                output_file = output_dir / f"{filepath.stem}_processed.parquet"
                df.to_parquet(output_file, index=False)
                
                processed_dfs.append(df)
                logger.info(f"Processed: {filepath.name}")
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
        
        # Combine all dataframes if needed
        if processed_dfs:
            combined_df = pd.concat(processed_dfs, ignore_index=True)
            combined_file = output_dir / "combined_processed.parquet"
            combined_df.to_parquet(combined_file, index=False)
            logger.info(f"Saved combined dataset: {len(combined_df)} rows")
        
        return processed_dfs

