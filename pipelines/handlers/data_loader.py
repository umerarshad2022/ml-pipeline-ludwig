"""
Data Loader Handler
Loads data from various sources and applies configurable preprocessing.
"""
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoaderHandler:
    """
    Handler for loading data from various sources.

    Supported source types: csv, parquet, json.

    Optional preprocessing (configured under the step's ``preprocessing`` key):
        drop_columns  : list of column names to remove
        scaling       : "standard" | "minmax" | "none"  (default: "none")
        test_size     : float between 0 and 1 for train/test split (default: no split)

    When ``test_size`` is set, the handler stores X_train, X_test, y_train, y_test
    in the context (requires ``target_col`` or infers it from the first step result).
    """
    
    def __init__(self, config: Dict[str, Any], context: Dict[str, Any]):
        self.config = config
        self.context = context
    
    def execute(self) -> Dict[str, Any]:
        """Execute data loading and optional preprocessing"""
        logger.info("📥 Loading data...")
        
        source = self.config.get('source', {})
        source_type = source.get('type', 'csv')
        source_path = source.get('path')
        
        if not source_path:
            raise ValueError("Data source path is required")
        
        logger.info(f"   Type: {source_type}")
        logger.info(f"   Path: {source_path}")
        
        # Load based on type
        if source_type == 'csv':
            df = pd.read_csv(source_path)
        elif source_type == 'parquet':
            df = pd.read_parquet(source_path)
        elif source_type == 'json':
            df = pd.read_json(source_path)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        logger.info(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"   Columns: {list(df.columns)}")

        # Apply preprocessing if configured
        preprocessing = self.config.get('preprocessing', {})
        if preprocessing:
            df = self._apply_preprocessing(df, preprocessing)
        
        # Store in context
        self.context['data'] = df
        
        result: Dict[str, Any] = {'data': df, 'shape': df.shape}

        # Optional train/test split
        test_size = preprocessing.get('test_size') if preprocessing else None
        if test_size:
            result.update(self._split(df, preprocessing))

        return result

    # -----------------------------------------------------------------------

    def _apply_preprocessing(self, df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        """Apply column dropping and feature scaling."""
        # Drop columns
        drop_cols: List[str] = cfg.get('drop_columns', [])
        cols_to_drop = [c for c in drop_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"   Dropped columns: {cols_to_drop}")

        # Scaling
        scaling = cfg.get('scaling', 'none')
        if scaling and scaling != 'none':
            target_col = cfg.get('target_col')
            numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c != target_col
            ]
            if scaling == 'standard':
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                self.context['scaler'] = scaler
                logger.info(f"   Applied StandardScaler to: {numeric_cols}")
            elif scaling == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                self.context['scaler'] = scaler
                logger.info(f"   Applied MinMaxScaler to: {numeric_cols}")
            else:
                logger.warning(f"   Unknown scaling '{scaling}'; skipping.")

        return df

    def _split(self, df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a stratified train/test split and store results in context."""
        from sklearn.model_selection import train_test_split

        test_size: float = float(cfg.get('test_size', 0.2))
        target_col: Optional[str] = cfg.get('target_col')
        random_state: Optional[int] = cfg.get('random_state', 42)

        if target_col and target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
        else:
            X_train, X_test = train_test_split(
                df, test_size=test_size, random_state=random_state
            )
            y_train = y_test = None
            logger.warning(
                "train/test split performed without a target column; "
                "y_train and y_test are not available in context."
            )

        self.context['X_train'] = X_train
        self.context['X_test'] = X_test
        self.context['y_train'] = y_train
        self.context['y_test'] = y_test

        logger.info(
            f"✂️  Train/test split (test_size={test_size}): "
            f"{len(X_train)} train / {len(X_test)} test"
        )
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        }


class UnsupervisedLabelingHandler:
    """
    Handler for unsupervised labeling.
    
    Since Ludwig needs labels and we have unsupervised data,
    we use Isolation Forest to create pseudo-labels.
    """
    
    def __init__(self, config: Dict[str, Any], context: Dict[str, Any]):
        self.config = config
        self.context = context
    
    def execute(self) -> Dict[str, Any]:
        """Execute unsupervised labeling"""
        logger.info("🏷️  Creating pseudo-labels for unsupervised data...")
        
        from sklearn.ensemble import IsolationForest
        
        # Get data
        df = self.context.get('data')
        if df is None:
            raise ValueError("No data in context")
        
        method = self.config.get('method', 'isolation_forest')
        feature = self.config.get('feature')
        output_column = self.config.get('output_column', 'is_anomaly')
        contamination = self.config.get('contamination', 0.10)
        
        logger.info(f"   Method: {method}")
        logger.info(f"   Feature: {feature}")
        logger.info(f"   Contamination: {contamination}")
        
        if method == 'isolation_forest':
            X = df[[feature]].values
            
            logger.info("🎯 Training Isolation Forest for labeling...")
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X)
            
            # Predict: -1 = anomaly, 1 = normal
            predictions = model.predict(X)
            
            # Convert to binary: 1 = anomaly, 0 = normal
            df[output_column] = (predictions == -1).astype(int)
            
            n_anomalies = (predictions == -1).sum()
            logger.info(f"✅ Labeled {n_anomalies}/{len(df)} samples as anomalies ({n_anomalies/len(df)*100:.1f}%)")
        
        else:
            raise ValueError(f"Unsupported labeling method: {method}")
        
        # Update context
        self.context['data'] = df
        
        return {
            'labeled_data': df,
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(n_anomalies / len(df))
        }
