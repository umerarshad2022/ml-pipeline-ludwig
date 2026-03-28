"""
Sklearn Pipeline Handlers
Handlers for scikit-learn and XGBoost models within the YAML pipeline engine.

Supports:
- SklearnTrainHandler: Train Random Forest, XGBoost, Logistic Regression
- SklearnEvaluateHandler: Compute metrics and produce evaluation artefacts
- SklearnVisualizeHandler: Generate plots (confusion matrix, ROC, PR, feature importance)
- SklearnXAIHandler: SHAP and LIME explanations
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_model(model_type: str, params: Dict[str, Any]):
    """Instantiate a sklearn / XGBoost model from a type name and param dict."""
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)

    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        # eval_metric can be provided in params; default to 'logloss' if absent
        xgb_params = dict(params)
        xgb_params.setdefault("eval_metric", "logloss")
        return XGBClassifier(**xgb_params, verbosity=0)

    elif model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)

    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. "
                         f"Choose from: random_forest, xgboost, logistic_regression")


def _get_split_data(context: Dict[str, Any]):
    """Return (X_train, X_test, y_train, y_test, feature_cols, target_col) from context."""
    if "X_train" in context:
        return (
            context["X_train"],
            context["X_test"],
            context["y_train"],
            context["y_test"],
            context.get("feature_cols", []),
            context.get("target_col", "target"),
        )
    raise ValueError(
        "No train/test split found in context. "
        "Ensure DataLoaderHandler has run with preprocessing.test_size configured."
    )


# ---------------------------------------------------------------------------
# SklearnTrainHandler
# ---------------------------------------------------------------------------

class SklearnTrainHandler:
    """
    Train a scikit-learn or XGBoost classifier.

    Reads model type and hyper-parameters from the top-level ``sklearn_model``
    section of the pipeline YAML (passed as ``sklearn_config``).

    Context consumed:
        - ``X_train``, ``y_train``  (produced by DataLoaderHandler with a test_size)

    Context produced:
        - ``sklearn_model``  – dict with keys: model, model_type, feature_cols,
                               target_col, output_directory
    """

    def __init__(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
        sklearn_config: Dict[str, Any],
    ):
        self.config = config
        self.context = context
        self.sklearn_config = sklearn_config

    def execute(self) -> Dict[str, Any]:
        logger.info("🌲 Starting sklearn model training…")

        model_type = self.sklearn_config.get("type")
        raw_params = dict(self.sklearn_config.get("params", {}))
        input_features: List[str] = self.sklearn_config.get("input_features", [])
        target_col: str = self.sklearn_config.get("output_feature", "is_anomaly")
        output_directory = self.config.get("output_directory", "artifacts/sklearn_model")
        experiment_name = self.config.get("experiment_name", "sklearn_experiment")

        if not model_type:
            raise ValueError("sklearn_model.type is required in the YAML config.")

        # Obtain data --------------------------------------------------------
        df = self.context.get("data")
        if df is None:
            raise ValueError("No data in context. Run data_loader step first.")

        if not input_features:
            # Fallback: use all numeric columns except the target
            input_features = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c != target_col
            ]
            logger.warning(
                f"sklearn_model.input_features not set; using: {input_features}"
            )

        # Prefer pre-split data when available (from DataLoaderHandler)
        if "X_train" in self.context:
            X_train = self.context["X_train"][input_features]
            y_train = self.context["y_train"]
            X_test = self.context["X_test"][input_features]
            y_test = self.context["y_test"]
            logger.info("📂 Using pre-split train/test data from context.")
        else:
            from sklearn.model_selection import train_test_split
            X = df[input_features]
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logger.info("✂️  Created 80/20 train/test split.")
            self.context["X_train"] = X_train
            self.context["X_test"] = X_test
            self.context["y_train"] = y_train
            self.context["y_test"] = y_test

        self.context["feature_cols"] = input_features
        self.context["target_col"] = target_col

        logger.info(f"   Model type : {model_type}")
        logger.info(f"   Features   : {input_features}")
        logger.info(f"   Target     : {target_col}")
        logger.info(f"   Train size : {len(X_train)}")
        logger.info(f"   Test size  : {len(X_test)}")

        # Build & train model ------------------------------------------------
        model = _build_model(model_type, raw_params)

        from datetime import datetime
        t0 = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - t0).total_seconds()
        logger.info(f"✅ Training complete in {training_time:.2f}s")

        # Persist model ------------------------------------------------------
        os.makedirs(output_directory, exist_ok=True)
        model_path = os.path.join(output_directory, f"{model_type}_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"💾 Model saved → {model_path}")

        # MLflow tracking ----------------------------------------------------
        self._log_mlflow(experiment_name, model_type, raw_params, training_time)

        result = {
            "model": model,
            "model_type": model_type,
            "model_path": model_path,
            "feature_cols": input_features,
            "target_col": target_col,
            "output_directory": output_directory,
        }
        self.context["sklearn_model"] = result
        return result

    @staticmethod
    def _log_mlflow(experiment_name, model_type, params, training_time):
        try:
            import mlflow
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run():
                mlflow.log_param("model_type", model_type)
                for k, v in params.items():
                    mlflow.log_param(k, v)
                mlflow.log_metric("training_time_seconds", training_time)
            logger.info("📊 MLflow: parameters logged.")
        except Exception as exc:
            logger.warning(f"MLflow logging skipped: {exc}")


# ---------------------------------------------------------------------------
# SklearnEvaluateHandler
# ---------------------------------------------------------------------------

class SklearnEvaluateHandler:
    """
    Evaluate a trained sklearn / XGBoost model.

    Context consumed:
        - ``sklearn_model``  (from SklearnTrainHandler)
        - ``X_test``, ``y_test``

    Context produced:
        - ``sklearn_evaluate``  – dict with metrics and predictions
    """

    def __init__(self, config: Dict[str, Any], context: Dict[str, Any]):
        self.config = config
        self.context = context

    def execute(self) -> Dict[str, Any]:
        logger.info("📈 Evaluating sklearn model…")

        model_info = self.context.get("sklearn_model")
        if model_info is None:
            raise ValueError("No trained model in context. Run sklearn_train first.")

        model = model_info["model"]
        feature_cols = model_info["feature_cols"]
        output_directory = self.config.get(
            "output_directory", model_info.get("output_directory", "artifacts/evaluation")
        )
        experiment_name = self.config.get("experiment_name", "sklearn_experiment")

        X_test = self.context.get("X_test")
        y_test = self.context.get("y_test")
        if X_test is None or y_test is None:
            raise ValueError("No test data in context.")

        X_test = X_test[feature_cols]

        # Predictions --------------------------------------------------------
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Metrics ------------------------------------------------------------
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            classification_report,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        if y_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            metrics["avg_precision"] = float(
                average_precision_score(y_test, y_proba)
            )

        report = classification_report(y_test, y_pred)

        logger.info("📊 Evaluation Results:")
        for k, v in metrics.items():
            logger.info(f"   {k:20s}: {v:.4f}")

        # Save metrics -------------------------------------------------------
        os.makedirs(output_directory, exist_ok=True)
        metrics_path = os.path.join(output_directory, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("=== Sklearn Model Evaluation ===\n\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\n=== Classification Report ===\n\n")
            f.write(report)
        logger.info(f"💾 Metrics saved → {metrics_path}")

        # MLflow tracking ----------------------------------------------------
        self._log_mlflow(experiment_name, metrics)

        result = {
            "metrics": metrics,
            "classification_report": report,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "output_directory": output_directory,
        }
        self.context["sklearn_evaluate"] = result
        return result

    @staticmethod
    def _log_mlflow(experiment_name, metrics):
        try:
            import mlflow
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run():
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
            logger.info("📊 MLflow: metrics logged.")
        except Exception as exc:
            logger.warning(f"MLflow logging skipped: {exc}")


# ---------------------------------------------------------------------------
# SklearnVisualizeHandler
# ---------------------------------------------------------------------------

class SklearnVisualizeHandler:
    """
    Generate evaluation visualisations for a trained sklearn model.

    Supported visualisations (configured via ``visualizations`` list in YAML):
        - confusion_matrix
        - roc_curve
        - precision_recall_curve
        - feature_importance   (tree-based models)
        - coefficients         (linear models)
        - learning_curve

    Context consumed:
        - ``sklearn_model``, ``sklearn_evaluate``, ``X_train``, ``y_train``
    """

    def __init__(self, config: Dict[str, Any], context: Dict[str, Any]):
        self.config = config
        self.context = context

    def execute(self) -> Dict[str, Any]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        logger.info("🎨 Generating sklearn visualisations…")

        model_info = self.context.get("sklearn_model")
        eval_info = self.context.get("sklearn_evaluate")

        if model_info is None:
            raise ValueError("No trained model in context.")
        if eval_info is None:
            raise ValueError("No evaluation results in context.")

        model = model_info["model"]
        feature_cols = model_info["feature_cols"]
        output_directory = self.config.get("output_directory", "artifacts/visualizations")
        visualizations = self.config.get(
            "visualizations",
            ["confusion_matrix", "roc_curve", "precision_recall_curve"],
        )

        os.makedirs(output_directory, exist_ok=True)

        y_test = self.context["y_test"]
        X_test = self.context["X_test"][feature_cols]
        y_pred = eval_info["y_pred"]
        y_proba = eval_info.get("y_proba")

        created: List[str] = []

        for vis in visualizations:
            try:
                if vis == "confusion_matrix":
                    path = self._plot_confusion_matrix(
                        y_test, y_pred, output_directory, plt, sns
                    )
                    created.append(path)

                elif vis == "roc_curve" and y_proba is not None:
                    path = self._plot_roc_curve(
                        y_test, y_proba, output_directory, plt
                    )
                    created.append(path)

                elif vis == "precision_recall_curve" and y_proba is not None:
                    path = self._plot_pr_curve(
                        y_test, y_proba, output_directory, plt
                    )
                    created.append(path)

                elif vis == "feature_importance":
                    path = self._plot_feature_importance(
                        model, feature_cols, output_directory, plt
                    )
                    if path:
                        created.append(path)

                elif vis == "coefficients":
                    path = self._plot_coefficients(
                        model, feature_cols, output_directory, plt
                    )
                    if path:
                        created.append(path)

                elif vis == "learning_curve":
                    X_train = self.context.get("X_train")
                    y_train = self.context.get("y_train")
                    if X_train is not None and y_train is not None:
                        path = self._plot_learning_curve(
                            model,
                            X_train[feature_cols],
                            y_train,
                            output_directory,
                            plt,
                            n_jobs=self.config.get("n_jobs", -1),
                        )
                        created.append(path)
                    else:
                        logger.warning("learning_curve: no training data in context.")

                else:
                    logger.warning(f"Skipping visualisation '{vis}' (not applicable).")

            except Exception as exc:
                logger.warning(f"Visualisation '{vis}' failed: {exc}")

        logger.info(f"✅ {len(created)} visualisation(s) saved to {output_directory}")
        result = {"visualizations": created, "output_directory": output_directory}
        self.context["sklearn_visualize"] = result
        return result

    # -- individual plot helpers ---------------------------------------------

    @staticmethod
    def _plot_confusion_matrix(y_test, y_pred, out_dir, plt, sns):
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        path = os.path.join(out_dir, "confusion_matrix.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"   💾 {path}")
        return path

    @staticmethod
    def _plot_roc_curve(y_test, y_proba, out_dir, plt):
        from sklearn.metrics import auc, roc_curve

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        path = os.path.join(out_dir, "roc_curve.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"   💾 {path}")
        return path

    @staticmethod
    def _plot_pr_curve(y_test, y_proba, out_dir, plt):
        from sklearn.metrics import average_precision_score, precision_recall_curve

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recall, precision, color="darkorange", lw=2, label=f"AP = {ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="upper right")
        path = os.path.join(out_dir, "precision_recall_curve.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"   💾 {path}")
        return path

    @staticmethod
    def _plot_feature_importance(model, feature_cols, out_dir, plt):
        if not hasattr(model, "feature_importances_"):
            logger.warning("Model has no feature_importances_; skipping.")
            return None
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.5)))
        ax.barh(
            [feature_cols[i] for i in indices],
            importances[indices],
            color="steelblue",
        )
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importances")
        ax.invert_yaxis()
        path = os.path.join(out_dir, "feature_importance.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"   💾 {path}")
        return path

    @staticmethod
    def _plot_coefficients(model, feature_cols, out_dir, plt):
        if not hasattr(model, "coef_"):
            logger.warning("Model has no coef_; skipping coefficients plot.")
            return None
        coef = model.coef_.ravel()
        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.5)))
        colors = ["steelblue" if c >= 0 else "tomato" for c in coef]
        ax.barh(feature_cols, coef, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient Value")
        ax.set_title("Logistic Regression Coefficients")
        path = os.path.join(out_dir, "coefficients.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"   💾 {path}")
        return path

    @staticmethod
    def _plot_learning_curve(model, X_train, y_train, out_dir, plt, n_jobs=-1):
        from sklearn.model_selection import learning_curve

        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=3,
            scoring="roc_auc",
            train_sizes=np.linspace(0.1, 1.0, 8),
            n_jobs=n_jobs,
        )
        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, train_mean, "o-", color="steelblue", label="Training AUC")
        ax.plot(train_sizes, val_mean, "o--", color="darkorange", label="Validation AUC")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("ROC AUC")
        ax.set_title("Learning Curve")
        ax.legend()
        path = os.path.join(out_dir, "learning_curve.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"   💾 {path}")
        return path


# ---------------------------------------------------------------------------
# SklearnXAIHandler
# ---------------------------------------------------------------------------

class SklearnXAIHandler:
    """
    Generate SHAP and LIME explanations for a trained sklearn / XGBoost model.

    Configuration keys (all optional):
        methods     : list of "shap" and/or "lime"  (default: ["shap", "lime"])
        n_lime      : number of LIME instances to explain  (default: 3)
        output_directory : where to save plots

    Context consumed:
        - ``sklearn_model``  (from SklearnTrainHandler)
        - ``X_train``, ``X_test``, ``y_test``
    """

    def __init__(self, config: Dict[str, Any], context: Dict[str, Any]):
        self.config = config
        self.context = context

    def execute(self) -> Dict[str, Any]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        logger.info("🔍 Generating XAI explanations…")

        model_info = self.context.get("sklearn_model")
        if model_info is None:
            raise ValueError("No trained model in context. Run sklearn_train first.")

        model = model_info["model"]
        model_type = model_info["model_type"]
        feature_cols = model_info["feature_cols"]
        output_directory = self.config.get(
            "output_directory",
            os.path.join(model_info.get("output_directory", "artifacts"), "xai"),
        )
        methods = self.config.get("methods", ["shap", "lime"])
        n_lime = self.config.get("n_lime", 3)

        X_train = self.context.get("X_train")
        X_test = self.context.get("X_test")
        y_test = self.context.get("y_test")

        if X_train is None or X_test is None:
            raise ValueError("No train/test data in context.")

        X_train = X_train[feature_cols]
        X_test = X_test[feature_cols]

        os.makedirs(output_directory, exist_ok=True)
        created: List[str] = []

        if "shap" in methods:
            shap_paths = self._run_shap(
                model, model_type, X_train, X_test, feature_cols, output_directory, plt,
                kernel_background=self.config.get("kernel_background_samples", 50),
                kernel_explain=self.config.get("kernel_explain_samples", 50),
            )
            created.extend(shap_paths)

        if "lime" in methods:
            lime_paths = self._run_lime(
                model, X_train, X_test, y_test, feature_cols, n_lime, output_directory, plt
            )
            created.extend(lime_paths)

        logger.info(f"✅ {len(created)} XAI artefact(s) saved to {output_directory}")
        result = {"xai_artifacts": created, "output_directory": output_directory}
        self.context["sklearn_xai"] = result
        return result

    # -- SHAP ----------------------------------------------------------------

    @staticmethod
    def _run_shap(model, model_type, X_train, X_test, feature_cols, out_dir, plt,
                  kernel_background=50, kernel_explain=50):
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Run: pip install shap")

        logger.info("   🔷 Running SHAP analysis…")
        created: List[str] = []

        # Choose explainer
        if model_type in ("random_forest", "xgboost"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            # SHAP ≥ 0.40 returns ndarray of shape (n_samples, n_features, n_classes)
            # Older versions return a list [neg_class, pos_class]
            if isinstance(shap_values, list):
                sv = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                sv = shap_values[:, :, 1]
            else:
                sv = shap_values
        else:
            n_bg = min(kernel_background, len(X_train))
            n_ex = min(kernel_explain, len(X_test))
            explainer = shap.KernelExplainer(
                model.predict_proba, shap.sample(X_train, n_bg)
            )
            shap_values = explainer.shap_values(X_test.iloc[:n_ex])
            if isinstance(shap_values, list):
                sv = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                sv = shap_values[:, :, 1]
            else:
                sv = shap_values

        X_plot = X_test if len(X_test) <= 200 else X_test.iloc[:200]
        sv_plot = sv if len(sv) <= 200 else sv[:200]

        # Summary beeswarm
        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.6)))
        shap.summary_plot(sv_plot, X_plot, feature_names=feature_cols, show=False)
        path = os.path.join(out_dir, "shap_summary.png")
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        created.append(path)
        logger.info(f"   💾 {path}")

        # Bar chart of mean |SHAP|
        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.6)))
        shap.summary_plot(sv_plot, X_plot, feature_names=feature_cols, plot_type="bar", show=False)
        path = os.path.join(out_dir, "shap_importance.png")
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        created.append(path)
        logger.info(f"   💾 {path}")

        return created

    # -- LIME ----------------------------------------------------------------

    @staticmethod
    def _run_lime(model, X_train, X_test, y_test, feature_cols, n_lime, out_dir, plt):
        try:
            import lime
            import lime.lime_tabular
        except ImportError:
            raise ImportError("LIME not installed. Run: pip install lime")

        logger.info("   🟡 Running LIME analysis…")
        created: List[str] = []

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_cols,
            class_names=["Normal", "Anomaly"],
            mode="classification",
            random_state=42,
        )

        predict_fn = (
            model.predict_proba
            if hasattr(model, "predict_proba")
            else lambda x: np.column_stack([1 - model.predict(x), model.predict(x)])
        )

        n_instances = min(n_lime, len(X_test))
        for i in range(n_instances):
            exp = explainer.explain_instance(
                X_test.values[i], predict_fn, num_features=len(feature_cols)
            )
            fig = exp.as_pyplot_figure()
            path = os.path.join(out_dir, f"lime_instance_{i}.png")
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            created.append(path)
            logger.info(f"   💾 {path}")

        return created
