"""
Data Loader Handler
Loads data from various sources
"""
import logging
from typing import Dict, Any
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoaderHandler:
    """Handler for loading data from various sources"""
    
    def __init__(self, config: Dict[str, Any], context: Dict[str, Any]):
        self.config = config
        self.context = context
    
    def execute(self) -> Dict[str, Any]:
        """Execute data loading"""
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
        
        # Store in context
        self.context['data'] = df
        
        return {'data': df, 'shape': df.shape}


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
