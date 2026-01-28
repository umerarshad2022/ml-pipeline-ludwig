"""
Ludwig Visualization Handler
Creates visualizations from Ludwig training results
"""
import logging
from typing import Dict, Any
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class LudwigVisualizeHandler:
    """Handler for creating Ludwig visualizations"""
    
    def __init__(self, config: Dict[str, Any], context: Dict[str, Any]):
        self.config = config
        self.context = context
    
    def execute(self) -> Dict[str, Any]:
        """Execute visualization generation"""
        logger.info("📊 Generating visualizations...")
        
        try:
            from ludwig.visualize import (
                learning_curves,
                confusion_matrix,
                roc_curves,
                calibration_1_vs_all
            )
        except ImportError:
            logger.warning("Ludwig visualize module not available, skipping visualizations")
            return {'skipped': True}
        
        # Get model output directory
        if 'train-ludwig' in self.context:
            model_dir = self.context['train-ludwig']['output_directory']
        else:
            model_dir = self.config.get('model_path')
        
        if not model_dir:
            logger.warning("No model directory available, skipping visualizations")
            return {'skipped': True}
        
        output_dir = Path(self.config.get('output_directory', 'artifacts/visualizations'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        visualizations = self.config.get('visualizations', [])
        created = []
        
        # Generate each visualization
        for viz_type in visualizations:
            try:
                if viz_type == 'learning_curves':
                    logger.info("   Creating learning curves...")
                    learning_curves(
                        [str(model_dir)],
                        output_directory=str(output_dir)
                    )
                    created.append('learning_curves')
                
                elif viz_type == 'confusion_matrix':
                    logger.info("   Creating confusion matrix...")
                    confusion_matrix(
                        [str(model_dir)],
                        output_directory=str(output_dir)
                    )
                    created.append('confusion_matrix')
                
                elif viz_type == 'roc_curve':
                    logger.info("   Creating ROC curve...")
                    roc_curves(
                        [str(model_dir)],
                        output_directory=str(output_dir)
                    )
                    created.append('roc_curve')
                
                elif viz_type == 'feature_importance':
                    logger.info("   Feature importance (requires specific model type)...")
                    # Feature importance is model-specific
                    logger.warning("   Feature importance requires tree-based models, skipping...")
                
            except Exception as e:
                logger.warning(f"   Failed to create {viz_type}: {e}")
        
        logger.info(f"✅ Created {len(created)} visualizations in {output_dir}")
        
        return {
            'output_directory': str(output_dir),
            'visualizations_created': created
        }
