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
                learning_curves_cli,
                confusion_matrix_cli,
                roc_curves_from_test_statistics_cli,
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

        model_dir = Path(model_dir)
        training_stats_path = self._find_file(model_dir, 'training_statistics.json')
        if not training_stats_path:
            logger.warning("training_statistics.json not found in model directory, skipping visualizations")
            return {'skipped': True}

        metadata_path = self._find_file(model_dir, 'training_set_metadata.json')
        test_stats_path = self._find_file(model_dir, 'test_statistics.json')

        output_dir = Path(self.config.get('output_directory', 'artifacts/visualizations'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        visualizations = self.config.get('visualizations', [])
        created = []
        
        # Generate each visualization
        for viz_type in visualizations:
            try:
                if viz_type == 'learning_curves':
                    logger.info("   Creating learning curves...")
                    # learning_curves_cli expects a list of paths to training_statistics.json files
                    learning_curves_cli(
                        [str(training_stats_path)],
                        output_directory=str(output_dir)
                    )
                    created.append('learning_curves')
                
                elif viz_type == 'confusion_matrix':
                    if not test_stats_path or not metadata_path:
                        logger.warning("   test_statistics.json or training_set_metadata.json not found, skipping confusion matrix")
                        continue
                    logger.info("   Creating confusion matrix...")
                    confusion_matrix_cli(
                        [str(test_stats_path)],
                        str(metadata_path),
                        output_directory=str(output_dir)
                    )
                    created.append('confusion_matrix')
                
                elif viz_type == 'roc_curve':
                    if not test_stats_path:
                        logger.warning("   test_statistics.json not found, skipping ROC curve")
                        continue
                    logger.info("   Creating ROC curve...")
                    roc_curves_from_test_statistics_cli(
                        [str(test_stats_path)],
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

    def _find_file(self, root: Path, filename: str) -> Path | None:
        """Find a file directly in root or in any nested subdirectory."""
        direct = root / filename
        if direct.exists():
            return direct
        matches = sorted(root.rglob(filename))
        return matches[0] if matches else None
