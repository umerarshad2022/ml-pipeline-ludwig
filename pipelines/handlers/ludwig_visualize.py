"""
Ludwig Visualization Handler
Creates visualizations from Ludwig training results.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

logger = logging.getLogger(__name__)


class LudwigVisualizeHandler:
    """Handler for creating Ludwig visualizations."""

    def __init__(self, config: Dict[str, Any], context: Dict[str, Any]):
        self.config = config
        self.context = context

    def execute(self) -> Dict[str, Any]:
        """Execute visualization generation."""
        logger.info("Generating visualizations...")

        try:
            from ludwig.visualize import (
                compare_performance,
                confusion_matrix,
                learning_curves,
                roc_curves_from_test_statistics,
            )
        except ImportError:
            logger.warning("Ludwig visualize module not available, skipping visualizations")
            return {"skipped": True}

        train_context = self.context.get("train-ludwig", {})
        evaluate_context = self.context.get("evaluate", {})

        train_stats = train_context.get("train_stats")
        test_stats = evaluate_context.get("test_stats")

        model_dir = self._path_or_none(
            train_context.get("output_directory") or self.config.get("model_path")
        )
        eval_dir = self._path_or_none(evaluate_context.get("output_directory"))

        if train_stats is None:
            train_stats = self._load_json_if_exists(self._find_file(model_dir, "training_statistics.json"))

        if test_stats is None:
            test_stats = self._load_json_if_exists(
                self._find_file(eval_dir, "test_statistics.json")
                or self._find_file(model_dir, "test_statistics.json")
            )

        metadata = getattr(train_context.get("model"), "training_set_metadata", None)
        if metadata is None:
            metadata = self._load_json_if_exists(
                self._find_file(model_dir, "training_set_metadata.json")
            )

        output_feature_name = self._resolve_output_feature_name(test_stats, train_stats)

        output_dir = Path(self.config.get("output_directory", "artifacts/visualizations"))
        output_dir.mkdir(parents=True, exist_ok=True)

        visualizations = self.config.get("visualizations", [])
        created = []

        for viz_type in visualizations:
            try:
                if viz_type == "learning_curves":
                    logger.info("Creating learning curves...")
                    if train_stats is None:
                        logger.warning("No training statistics available, skipping learning curves")
                        continue
                    learning_curves(
                        [train_stats],
                        output_feature_name=output_feature_name,
                        output_directory=str(output_dir),
                        file_format="png",
                    )
                    created.append("learning_curves")

                elif viz_type == "confusion_matrix":
                    logger.info("Creating confusion matrix...")
                    if test_stats is None or metadata is None:
                        logger.warning(
                            "Test statistics or training metadata not available, skipping confusion matrix"
                        )
                        continue
                    confusion_matrix(
                        [test_stats],
                        metadata,
                        output_feature_name=output_feature_name,
                        top_n_classes=[self.config.get("top_n_classes", 10)],
                        normalize=self.config.get("normalize", True),
                        output_directory=str(output_dir),
                        file_format="png",
                    )
                    created.append("confusion_matrix")

                elif viz_type == "roc_curve":
                    logger.info("Creating ROC curve...")
                    if test_stats is None or output_feature_name is None:
                        logger.warning("No binary output feature statistics available, skipping ROC curve")
                        continue
                    roc_curves_from_test_statistics(
                        [test_stats],
                        output_feature_name=output_feature_name,
                        output_directory=str(output_dir),
                        file_format="png",
                    )
                    created.append("roc_curve")

                elif viz_type in {"compare_performance", "performance_metrics"}:
                    logger.info("Creating performance metrics comparison...")
                    if test_stats is None:
                        logger.warning("No test statistics available, skipping performance metrics")
                        continue
                    compare_performance(
                        [test_stats],
                        output_feature_name=output_feature_name,
                        output_directory=str(output_dir),
                        file_format="png",
                    )
                    created.append(viz_type)

                elif viz_type == "feature_importance":
                    logger.info("Feature importance requires tree-based Ludwig models, skipping...")

            except Exception as exc:
                logger.warning(f"Failed to create {viz_type}: {exc}")

        logger.info(f"Created {len(created)} visualizations in {output_dir}")

        return {
            "output_directory": str(output_dir),
            "visualizations_created": created,
        }

    def _path_or_none(self, path_value: Any) -> Path | None:
        if path_value is None:
            return None
        return Path(path_value)

    def _find_file(self, root: Path | None, filename: str) -> Path | None:
        """Find a file directly in root or in any nested subdirectory."""
        if root is None or not root.exists():
            return None
        direct = root / filename
        if direct.exists():
            return direct
        matches = sorted(root.rglob(filename))
        return matches[0] if matches else None

    def _load_json_if_exists(self, path: Path | None) -> Dict[str, Any] | None:
        if path is None or not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _resolve_output_feature_name(
        self, test_stats: Dict[str, Any] | None, train_stats: Dict[str, Any] | None
    ) -> str | None:
        if isinstance(test_stats, dict) and test_stats:
            return next(iter(test_stats))

        if isinstance(train_stats, dict):
            for key in ("validation", "training", "test"):
                value = train_stats.get(key)
                if isinstance(value, dict) and value:
                    return next(iter(value))

        return self.config.get("output_feature_name")
