"""
Ludwig Visualization Handler
Creates visualizations from Ludwig training and evaluation results.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

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
                precision_recall_curves_from_test_statistics,
                roc_curves_from_test_statistics,
            )
        except ImportError:
            logger.warning("Ludwig visualize module not available, skipping visualizations")
            return {"skipped": True}

        train_context = self.context.get("train-ludwig", {})
        evaluate_context = self.context.get("evaluate", {})

        train_stats = train_context.get("train_stats")
        test_stats = evaluate_context.get("test_stats")
        predictions = evaluate_context.get("predictions")
        raw_data = self.context.get("data")

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

        model = train_context.get("model")
        feature_cols = train_context.get("input_features") or self._infer_feature_cols(raw_data, output_feature_name)
        y_true = self._extract_ground_truth(raw_data, output_feature_name)
        pred_frame = self._coerce_predictions_to_frame(predictions)
        y_pred = self._extract_prediction_labels(pred_frame, output_feature_name)
        y_score = self._extract_prediction_scores(pred_frame, output_feature_name)

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
                    if self._run_or_fallback_plot(
                        output_dir,
                        "confusion_matrix_*.png",
                        lambda: confusion_matrix(
                            [test_stats],
                            metadata,
                            output_feature_name=output_feature_name,
                            top_n_classes=[self.config.get("top_n_classes", 10)],
                            normalize=self.config.get("normalize", True),
                            output_directory=str(output_dir),
                            file_format="png",
                        )
                        if test_stats is not None and metadata is not None
                        else None,
                        lambda: self._plot_confusion_matrix(output_dir, y_true, y_pred),
                    ):
                        created.append("confusion_matrix")

                elif viz_type == "roc_curve":
                    logger.info("Creating ROC curve...")
                    if self._run_or_fallback_plot(
                        output_dir,
                        "roc*.png",
                        lambda: roc_curves_from_test_statistics(
                            [test_stats],
                            output_feature_name=output_feature_name,
                            output_directory=str(output_dir),
                            file_format="png",
                        )
                        if test_stats is not None and output_feature_name is not None
                        else None,
                        lambda: self._plot_roc_curve(output_dir, y_true, y_score),
                    ):
                        created.append("roc_curve")

                elif viz_type == "precision_recall_curve":
                    logger.info("Creating precision-recall curve...")
                    if self._run_or_fallback_plot(
                        output_dir,
                        "precision_recall*.png",
                        lambda: precision_recall_curves_from_test_statistics(
                            [test_stats],
                            output_feature_name=output_feature_name,
                            output_directory=str(output_dir),
                            file_format="png",
                        )
                        if test_stats is not None and output_feature_name is not None
                        else None,
                        lambda: self._plot_precision_recall_curve(output_dir, y_true, y_score),
                    ):
                        created.append("precision_recall_curve")

                elif viz_type in {"compare_performance", "performance_metrics"}:
                    logger.info("Creating performance metrics comparison...")
                    if self._run_or_fallback_plot(
                        output_dir,
                        "compare_performance*.png",
                        lambda: compare_performance(
                            [test_stats],
                            output_feature_name=output_feature_name,
                            output_directory=str(output_dir),
                            file_format="png",
                        )
                        if test_stats is not None
                        else None,
                        lambda: self._plot_performance_metrics(output_dir, y_true, y_pred, y_score),
                    ):
                        created.append(viz_type)

                elif viz_type == "feature_importance":
                    logger.info("Creating permutation feature importance...")
                    path = self._plot_permutation_importance(
                        output_dir=output_dir,
                        model=model,
                        raw_data=raw_data,
                        output_feature_name=output_feature_name,
                        feature_cols=feature_cols,
                    )
                    if path is not None:
                        created.append("feature_importance")

            except Exception as exc:
                logger.warning(f"Failed to create {viz_type}: {exc}")

        logger.info(f"Created {len(created)} visualizations in {output_dir}")
        return {
            "output_directory": str(output_dir),
            "visualizations_created": created,
        }

    def _run_or_fallback_plot(self, output_dir: Path, pattern: str, primary, fallback) -> bool:
        before = set(output_dir.glob(pattern))
        try:
            if primary is not None:
                primary()
        except Exception as exc:
            logger.warning(f"Primary Ludwig plotter failed for '{pattern}': {exc}")

        after = set(output_dir.glob(pattern))
        if after - before:
            return True

        try:
            return fallback() is not None
        except Exception as exc:
            logger.warning(f"Fallback plotter failed for '{pattern}': {exc}")
            return False

    def _plot_confusion_matrix(self, output_dir: Path, y_true, y_pred) -> str | None:
        if y_true is None or y_pred is None:
            return None

        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        labels = self._binary_labels(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(cm, cmap="Blues")
        fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        path = output_dir / "confusion_matrix_fallback.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return str(path)

    def _plot_roc_curve(self, output_dir: Path, y_true, y_score) -> str | None:
        if y_true is None or y_score is None:
            return None

        from sklearn.metrics import auc, roc_curve
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"ROC AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        path = output_dir / "roc_curve_fallback.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return str(path)

    def _plot_precision_recall_curve(self, output_dir: Path, y_true, y_score) -> str | None:
        if y_true is None or y_score is None:
            return None

        from sklearn.metrics import average_precision_score, precision_recall_curve
        import matplotlib.pyplot as plt

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recall, precision, lw=2, color="darkorange", label=f"AP = {avg_precision:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="upper right")
        path = output_dir / "precision_recall_curve_fallback.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return str(path)

    def _plot_performance_metrics(self, output_dir: Path, y_true, y_pred, y_score) -> str | None:
        if y_true is None or y_pred is None:
            return None

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        import matplotlib.pyplot as plt

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }
        if y_score is not None:
            from sklearn.metrics import average_precision_score, roc_auc_score

            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
            metrics["avg_precision"] = average_precision_score(y_true, y_score)

        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(metrics.keys())
        values = [metrics[name] for name in names]
        bars = ax.bar(names, values, color=["steelblue", "teal", "darkorange", "slateblue", "indianred", "gray"][: len(names)])
        ax.set_ylim(0, 1.05)
        ax.set_title("Performance Metrics")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        path = output_dir / "compare_performance_fallback.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return str(path)

    def _plot_permutation_importance(
        self,
        output_dir: Path,
        model,
        raw_data,
        output_feature_name: str | None,
        feature_cols: List[str],
    ) -> str | None:
        if model is None or raw_data is None or output_feature_name is None or not feature_cols:
            return None

        df = self._coerce_data_frame(raw_data)
        if df is None or output_feature_name not in df.columns:
            return None

        baseline = self._evaluate_model_predictions(model, df, output_feature_name)
        if baseline["metric_value"] is None:
            return None

        importances = []
        rng = np.random.default_rng(42)
        for feature in feature_cols:
            permuted = df.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            permuted_eval = self._evaluate_model_predictions(model, permuted, output_feature_name)
            if permuted_eval["metric_value"] is None:
                continue
            importances.append((feature, baseline["metric_value"] - permuted_eval["metric_value"]))

        if not importances:
            return None

        import matplotlib.pyplot as plt

        importances.sort(key=lambda item: item[1], reverse=True)
        labels = [item[0] for item in importances]
        values = [item[1] for item in importances]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(labels))))
        ax.barh(labels, values, color="steelblue")
        ax.invert_yaxis()
        ax.set_xlabel(f"Drop in {baseline['metric_name']}")
        ax.set_title("Permutation Feature Importance")
        path = output_dir / "feature_importance_permutation.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return str(path)

    def _evaluate_model_predictions(self, model, dataset, output_feature_name: str) -> Dict[str, Any]:
        predictions, _ = model.predict(
            dataset=dataset,
            skip_save_unprocessed_output=True,
            skip_save_predictions=True,
            output_directory=str(Path("artifacts") / "tmp_predictions"),
        )
        pred_frame = self._coerce_predictions_to_frame(predictions)
        y_true = self._extract_ground_truth(dataset, output_feature_name)
        y_pred = self._extract_prediction_labels(pred_frame, output_feature_name)
        y_score = self._extract_prediction_scores(pred_frame, output_feature_name)
        if y_true is None or y_pred is None:
            return {"metric_name": None, "metric_value": None}

        from sklearn.metrics import f1_score

        if y_score is not None:
            from sklearn.metrics import roc_auc_score

            return {"metric_name": "roc_auc", "metric_value": roc_auc_score(y_true, y_score)}
        return {"metric_name": "f1_score", "metric_value": f1_score(y_true, y_pred, zero_division=0)}

    def _coerce_predictions_to_frame(self, predictions) -> pd.DataFrame | None:
        if predictions is None:
            return None
        if isinstance(predictions, pd.DataFrame):
            return predictions
        if isinstance(predictions, dict):
            try:
                return pd.DataFrame(predictions)
            except Exception:
                return None
        return None

    def _coerce_data_frame(self, data) -> pd.DataFrame | None:
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            return data.copy()
        try:
            return pd.DataFrame(data)
        except Exception:
            return None

    def _extract_ground_truth(self, raw_data, output_feature_name: str | None):
        df = self._coerce_data_frame(raw_data)
        if df is None or output_feature_name is None or output_feature_name not in df.columns:
            return None
        return self._coerce_binary_series(df[output_feature_name])

    def _extract_prediction_labels(self, pred_frame: pd.DataFrame | None, output_feature_name: str | None):
        if pred_frame is None or output_feature_name is None:
            return None
        candidate_columns = [
            f"{output_feature_name}_predictions",
            output_feature_name,
        ]
        for column in candidate_columns:
            if column in pred_frame.columns:
                return self._coerce_binary_series(pred_frame[column])
        return None

    def _extract_prediction_scores(self, pred_frame: pd.DataFrame | None, output_feature_name: str | None):
        if pred_frame is None or output_feature_name is None:
            return None
        candidate_columns = [
            f"{output_feature_name}_probabilities_True",
            f"{output_feature_name}_probability",
            f"{output_feature_name}_probabilities_1",
        ]
        for column in candidate_columns:
            if column in pred_frame.columns:
                return pred_frame[column].astype(float).to_numpy()
        return None

    def _coerce_binary_series(self, values: Iterable[Any]):
        series = pd.Series(values)
        if pd.api.types.is_bool_dtype(series):
            return series.astype(int).to_numpy()
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(int).to_numpy()
        normalized = series.astype(str).str.strip().str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
        )
        if normalized.isna().any():
            return None
        return normalized.astype(int).to_numpy()

    def _binary_labels(self, *arrays) -> List[int]:
        labels = sorted({int(value) for array in arrays if array is not None for value in np.unique(array)})
        return labels or [0, 1]

    def _infer_feature_cols(self, raw_data, output_feature_name: str | None) -> List[str]:
        df = self._coerce_data_frame(raw_data)
        if df is None:
            return []
        return [col for col in df.columns if col != output_feature_name]

    def _path_or_none(self, path_value: Any) -> Path | None:
        if path_value is None:
            return None
        return Path(path_value)

    def _find_file(self, root: Path | None, filename: str) -> Path | None:
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
