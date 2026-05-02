"""
Ludwig Training Handler
Handles Ludwig model training within the pipeline
"""
import logging
from typing import Dict, Any
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class LudwigTrainHandler:
    """
    Handler for training Ludwig models.
    
    Ludwig automatically handles:
    - Model architecture selection
    - Feature preprocessing
    - Training loop
    - Validation
    - Checkpointing
    """
    
    def __init__(self, config: Dict[str, Any], context: Dict[str, Any], ludwig_config: Dict[str, Any]):
        """
        Initialize handler.
        
        Args:
            config: Step configuration
            context: Pipeline execution context
            ludwig_config: Ludwig model configuration
        """
        self.config = config
        self.context = context
        self.ludwig_config = ludwig_config
        
    def execute(self) -> Dict[str, Any]:
        """Execute Ludwig training"""
        logger.info("🚀 Starting Ludwig training...")
        
        try:
            from ludwig.api import LudwigModel
        except ImportError:
            raise ImportError(
                "Ludwig not installed. Install with: pip install ludwig"
            )
        
        # Get data from context
        if 'data' not in self.context:
            raise ValueError("No data in context. Run data_loader step first.")
        
        df = self.context['data']
        logger.info(f"📊 Training data: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Prepare Ludwig config
        config = self._prepare_ludwig_config()
        
        # Initialize model
        logger.info("🏗️  Initializing Ludwig model...")
        model = LudwigModel(config, logging_level=logging.INFO)
        
        # Train
        logger.info("🎯 Training model...")
        logger.info(f"   Epochs: {config.get('trainer', {}).get('epochs', 100)}")
        logger.info(f"   Batch size: {config.get('trainer', {}).get('batch_size', 32)}")
        
        train_stats, preprocessed_data, output_directory = model.train(
            dataset=df,
            output_directory=self.config.get('output_directory', 'artifacts/ludwig_model'),
            experiment_name=self.config.get('experiment_name', 'experiment'),
            skip_save_processed_input=False,
            skip_save_progress=False,
            skip_save_unprocessed_output=False
        )
        
        logger.info(f"✅ Training complete!")
        logger.info(f"   Model saved to: {output_directory}")
        
        # Save to context
        result = {
            'model': model,
            'train_stats': train_stats,
            'output_directory': output_directory,
            'preprocessed_data': preprocessed_data,
            'input_features': [feature['name'] for feature in config.get('input_features', [])],
            'output_feature_name': config.get('output_features', [{}])[0].get('name'),
        }
        
        return result
    
    def _prepare_ludwig_config(self) -> Dict[str, Any]:
        """Prepare Ludwig configuration dict"""
        config = {
            'input_features': self.ludwig_config['input_features'],
            'output_features': self.ludwig_config['output_features'],
        }
        
        # Add trainer config
        if 'trainer' in self.ludwig_config:
            config['trainer'] = self.ludwig_config['trainer']
        
        # Add defaults
        if 'defaults' in self.ludwig_config:
            config['defaults'] = self.ludwig_config['defaults']
        
        return config


class LudwigEvaluateHandler:
    """Handler for evaluating Ludwig models"""
    
    def __init__(self, config: Dict[str, Any], context: Dict[str, Any]):
        self.config = config
        self.context = context
    
    def execute(self) -> Dict[str, Any]:
        """Execute Ludwig evaluation"""
        logger.info("📊 Evaluating Ludwig model...")
        
        try:
            from ludwig.api import LudwigModel
        except ImportError:
            raise ImportError("Ludwig not installed")
        
        # Load model from context or path
        if 'train-ludwig' in self.context:
            model = self.context['train-ludwig']['model']
            logger.info("✅ Using model from training step")
        else:
            model_path = self.config.get('model_path')
            if not model_path:
                raise ValueError("No model available. Provide model_path or run training first.")
            logger.info(f"📦 Loading model from: {model_path}")
            model = LudwigModel.load(model_path)
        
        # Get data
        df = self.context.get('data')
        if df is None:
            raise ValueError("No data in context")
        
        # Evaluate
        logger.info("🔍 Running evaluation...")
        test_stats, predictions, output_directory = model.evaluate(
            dataset=df,
            output_directory=self.config.get('output_directory', 'artifacts/evaluation'),
            skip_save_unprocessed_output=False,
            collect_predictions=True,
            collect_overall_stats=True
        )
        
        logger.info("✅ Evaluation complete")
        
        # Log key metrics
        if test_stats:
            for feature, stats in test_stats.items():
                logger.info(f"\n📈 Metrics for {feature}:")
                for metric, value in stats.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"   {metric}: {value:.4f}")
        
        result = {
            'test_stats': test_stats,
            'predictions': predictions,
            'output_directory': output_directory
        }
        
        return result


class LudwigPredictHandler:
    """Handler for making predictions with Ludwig models"""
    
    def __init__(self, config: Dict[str, Any], context: Dict[str, Any]):
        self.config = config
        self.context = context
    
    def execute(self) -> Dict[str, Any]:
        """Execute Ludwig prediction"""
        logger.info("🔮 Making predictions with Ludwig model...")
        
        try:
            from ludwig.api import LudwigModel
        except ImportError:
            raise ImportError("Ludwig not installed")
        
        # Load model
        if 'train-ludwig' in self.context:
            model = self.context['train-ludwig']['model']
        else:
            model_path = self.config.get('model_path')
            if not model_path:
                raise ValueError("No model available")
            model = LudwigModel.load(model_path)
        
        # Get data
        df = self.context.get('data')
        if df is None:
            raise ValueError("No data in context")
        
        # Predict
        logger.info(f"🎯 Predicting on {len(df)} samples...")
        predictions, output_directory = model.predict(
            dataset=df,
            output_directory=self.config.get('output_directory', 'artifacts/predictions'),
            skip_save_predictions=False
        )
        
        logger.info("✅ Predictions complete")
        
        result = {
            'predictions': predictions,
            'output_directory': output_directory
        }
        
        return result
