"""
Ludwig Visualization Handler
Creates visualizations from Ludwig training results
"""
import logging
from typing import Dict, Any, Callable
from pathlib import Path
from datetime import datetime

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
            import ludwig.visualize as ludwig_visualize
        except Exception as e:
            logger.warning(f"Ludwig visualize module not available, skipping visualizations: {e}")
            return {'skipped': True}
        
        # Get model output directory
        if 'train-ludwig' in self.context:
            model_dir = self.context['train-ludwig']['output_directory']
        else:
            model_dir = self.config.get('model_path')
        
        if not model_dir:
            logger.warning("No model directory available, skipping visualizations")
            return {'skipped': True}

        experiment_dir = self._resolve_experiment_dir(Path(model_dir))
        if not experiment_dir:
            logger.warning("Could not locate Ludwig experiment directory, skipping visualizations")
            return {'skipped': True}
        
        output_dir = self._create_run_output_dir()
        
        visualizations = self.config.get('visualizations', [])
        created = []
        training_stats_path = self._find_first_file(experiment_dir, 'training_statistics.json')
        metadata_path = self._find_first_file(experiment_dir, 'training_set_metadata.json')
        test_stats_path = self._resolve_test_statistics_path(experiment_dir)

        if not training_stats_path and not test_stats_path:
            logger.warning("No Ludwig statistics artifacts found, skipping visualizations")
            return {'skipped': True}
        
        # Generate each visualization
        for viz_type in visualizations:
            try:
                if viz_type == 'learning_curves':
                    if not training_stats_path:
                        logger.warning("   training_statistics.json not found, skipping learning curves")
                        continue
                    logger.info("   Creating learning curves...")
                    learning_curves_cli = self._resolve_visualize_function(
                        ludwig_visualize,
                        viz_type,
                        ['learning_curves_cli']
                    )
                    learning_curves_cli(
                        str(training_stats_path),
                        output_directory=str(output_dir),
                        file_format='png'
                    )
                    created.append('learning_curves')
                
                elif viz_type == 'confusion_matrix':
                    if not test_stats_path or not metadata_path:
                        logger.warning("   test statistics or metadata missing, skipping confusion matrix")
                        continue
                    logger.info("   Creating confusion matrix...")
                    confusion_matrix_cli = self._resolve_visualize_function(
                        ludwig_visualize,
                        viz_type,
                        ['confusion_matrix_cli']
                    )
                    confusion_matrix_cli(
                        str(test_stats_path),
                        ground_truth_metadata=str(metadata_path),
                        output_feature_name=None,
                        top_n_classes=10,
                        normalize=True,
                        output_directory=str(output_dir),
                        file_format='png'
                    )
                    created.append('confusion_matrix')
                
                elif viz_type == 'roc_curve':
                    if not test_stats_path:
                        logger.warning("   test_statistics.json not found, skipping ROC curve")
                        continue
                    logger.info("   Creating ROC curve...")
                    roc_curves_cli = self._resolve_visualize_function(
                        ludwig_visualize,
                        viz_type,
                        ['roc_curves_from_test_statistics_cli', 'roc_curves_cli']
                    )
                    roc_curves_cli(
                        str(test_stats_path),
                        output_directory=str(output_dir),
                        file_format='png'
                    )
                    created.append('roc_curve')
                
                elif viz_type == 'feature_importance':
                    logger.info("   Feature importance (requires specific model type)...")
                    # Feature importance is model-specific
                    logger.warning("   Feature importance requires tree-based models, skipping...")
                
            except Exception as e:
                logger.exception(f"   Failed to create {viz_type}: {e}")
        
        logger.info(f"✅ Created {len(created)} visualizations in {output_dir}")
        
        return {
            'output_directory': str(output_dir),
            'visualizations_created': created
        }

    def _resolve_visualize_function(
        self,
        visualize_module: Any,
        viz_type: str,
        candidates: list[str]
    ) -> Callable[..., Any]:
        """Resolve Ludwig visualization function across supported API names."""
        for fn_name in candidates:
            fn = getattr(visualize_module, fn_name, None)
            if callable(fn):
                return fn
        raise AttributeError(
            f"No Ludwig visualization function found for {viz_type}. "
            f"Tried: {', '.join(candidates)}"
        )

    def _create_run_output_dir(self) -> Path:
        """Create an isolated per-run visualization output directory."""
        base_output_dir = Path(self.config.get('output_directory', 'artifacts/visualizations'))
        run_id = self.config.get('run_id') or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run_output_dir = base_output_dir / f'run_{run_id}'
        run_output_dir.mkdir(parents=True, exist_ok=True)
        return run_output_dir

    def _resolve_experiment_dir(self, model_dir: Path) -> Path | None:
        """Resolve the concrete Ludwig experiment directory containing training stats."""
        if model_dir.is_file():
            model_dir = model_dir.parent

        if not model_dir.exists():
            return None

        direct_stats = model_dir / 'training_statistics.json'
        if direct_stats.exists():
            return model_dir

        nested_stats = sorted(model_dir.rglob('training_statistics.json'))
        if nested_stats:
            return nested_stats[-1].parent

        return model_dir

    def _find_first_file(self, root: Path, filename: str) -> Path | None:
        """Find a file in root or nested directories."""
        direct = root / filename
        if direct.exists():
            return direct

        matches = sorted(root.rglob(filename))
        return matches[0] if matches else None

    def _resolve_test_statistics_path(self, experiment_dir: Path) -> Path | None:
        """Resolve test statistics from evaluate step output or experiment directory."""
        candidates = [experiment_dir]

        for value in self.context.values():
            if isinstance(value, dict) and value.get('test_stats') is not None:
                evaluate_dir = value.get('output_directory')
                if evaluate_dir:
                    candidates.append(Path(evaluate_dir))

        for candidate in candidates:
            if not candidate.exists():
                continue
            test_stats = self._find_first_file(candidate, 'test_statistics.json')
            if test_stats:
                return test_stats

        return None
