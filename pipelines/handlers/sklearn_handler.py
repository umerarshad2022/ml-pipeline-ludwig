"""
Scikit-learn handlers for train/evaluate/visualize/xai pipeline steps.
"""
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve, train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def _build_run_timestamp(config: Dict[str, Any]) -> str:
    """Build a run-specific timestamp suffix for artifact names."""
    return str(config.get("run_id") or time.strftime("%Y%m%d_%H%M%S"))


def _timestamped_png_path(output_directory: str, base_name: str, run_ts: str) -> str:
    """Create a timestamped PNG path to avoid overwriting prior runs."""
    return os.path.join(output_directory, f"{base_name}_{run_ts}.png")


def _build_model(model_type: str, params: Dict[str, Any]):
    """Build a classifier from model type + params."""
    model_type = (model_type or "").lower()
    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**params)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _get_split_data(
    context: Dict[str, Any],
    feature_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Extract split data from context or create it from context['data']."""
    if all(k in context for k in ("X_train", "X_test", "y_train", "y_test")):
        return context["X_train"], context["X_test"], context["y_train"], context["y_test"]

    df = context.get("data")
    if df is None:
        raise ValueError("No data in context")

    target_col = target_col or context.get("target_col") or "is_anomaly"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    if not feature_cols:
        feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    context["X_train"] = X_train
    context["X_test"] = X_test
    context["y_train"] = y_train
    context["y_test"] = y_test
    return X_train, X_test, y_train, y_test


class SklearnTrainHandler:
    """Train and persist a scikit-learn compatible model."""

    def __init__(self, config: Dict[str, Any], context: Dict[str, Any], sklearn_config: Dict[str, Any]):
        self.config = config
        self.context = context
        self.sklearn_config = sklearn_config

    def execute(self) -> Dict[str, Any]:
        logger.info("🚀 Training sklearn model...")
        if "data" not in self.context:
            raise ValueError("No data in context")

        model_type = self.sklearn_config.get("type")
        params = self.sklearn_config.get("params", {})
        feature_cols = self.sklearn_config.get("input_features")
        target_col = self.sklearn_config.get("output_feature", "is_anomaly")

        X_train, X_test, y_train, y_test = _get_split_data(
            self.context, feature_cols=feature_cols, target_col=target_col
        )

        if not feature_cols:
            feature_cols = list(X_train.columns)

        model = _build_model(model_type, params)
        start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start

        output_directory = self.config.get("output_directory", f"artifacts/{model_type}_model")
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(output_directory, f"{model_type}_model.pkl")
        joblib.dump(model, model_path)

        self.context["feature_cols"] = feature_cols
        self.context["target_col"] = target_col
        self.context["X_train"] = X_train
        self.context["X_test"] = X_test
        self.context["y_train"] = y_train
        self.context["y_test"] = y_test

        result = {
            "model": model,
            "model_type": model_type,
            "model_path": model_path,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "output_directory": output_directory,
            "training_time_seconds": training_time,
        }
        self.context["sklearn_model"] = result
        return result


class SklearnEvaluateHandler:
    """Evaluate trained sklearn model and persist metrics."""

    def __init__(self, config: Dict[str, Any], context: Dict[str, Any]):
        self.config = config
        self.context = context

    def execute(self) -> Dict[str, Any]:
        model_info = self.context.get("sklearn_model")
        if not model_info:
            raise ValueError("No trained model in context")

        model = model_info["model"] if isinstance(model_info, dict) else model_info
        X_test = self.context.get("X_test")
        y_test = self.context.get("y_test")
        if X_test is None or y_test is None:
            raise ValueError("Missing test split in context")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        if y_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            metrics["avg_precision"] = float(average_precision_score(y_test, y_proba))

        output_directory = self.config.get("output_directory", "artifacts/evaluation")
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        metrics_path = os.path.join(output_directory, "metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value:.6f}\n")

        result = {
            "metrics": metrics,
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
            "y_pred": y_pred,
            "y_proba": y_proba,
            "output_directory": output_directory,
        }
        self.context["sklearn_evaluate"] = result
        return result


class SklearnVisualizeHandler:
    """Generate sklearn evaluation visualizations."""

    def __init__(self, config: Dict[str, Any], context: Dict[str, Any]):
        self.config = config
        self.context = context

    def execute(self) -> Dict[str, Any]:
        model_info = self.context.get("sklearn_model")
        eval_info = self.context.get("sklearn_evaluate", {})
        if not model_info:
            raise ValueError("No trained model in context")

        model = model_info["model"] if isinstance(model_info, dict) else model_info
        model_type = model_info.get("model_type", "") if isinstance(model_info, dict) else ""
        X_test = self.context.get("X_test")
        y_test = self.context.get("y_test")
        if X_test is None or y_test is None:
            raise ValueError("Missing X_test/y_test in context")

        visualizations = self.config.get(
            "visualizations",
            ["confusion_matrix", "roc_curve", "precision_recall_curve"],
        )
        output_directory = self.config.get(
            "output_directory", f"artifacts/{model_type}_model/visualizations"
        )
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        run_ts = _build_run_timestamp(self.config)
        created: List[str] = []

        y_pred = eval_info.get("y_pred")
        if y_pred is None:
            y_pred = model.predict(X_test)
        y_proba = eval_info.get("y_proba")
        if y_proba is None and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        if "confusion_matrix" in visualizations:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            path = _timestamped_png_path(output_directory, "confusion_matrix", run_ts)
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            created.append(path)

        if "roc_curve" in visualizations and y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = roc_auc_score(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(loc="lower right")
            path = _timestamped_png_path(output_directory, "roc_curve", run_ts)
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            created.append(path)

        if "precision_recall_curve" in visualizations and y_proba is not None:
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(recall, precision, label=f"AP={ap:.3f}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            ax.legend(loc="best")
            path = _timestamped_png_path(output_directory, "precision_recall_curve", run_ts)
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            created.append(path)

        if "feature_importance" in visualizations and hasattr(model, "feature_importances_"):
            feature_cols = self.context.get("feature_cols") or list(X_test.columns)
            importances = np.asarray(model.feature_importances_)
            order = np.argsort(importances)
            fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.5)))
            ax.barh(np.array(feature_cols)[order], importances[order])
            ax.set_title("Feature Importance")
            path = _timestamped_png_path(output_directory, "feature_importance", run_ts)
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            created.append(path)

        if "coefficients" in visualizations and hasattr(model, "coef_"):
            feature_cols = self.context.get("feature_cols") or list(X_test.columns)
            coefs = np.asarray(model.coef_).ravel()
            order = np.argsort(np.abs(coefs))
            fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.5)))
            ax.barh(np.array(feature_cols)[order], coefs[order])
            ax.set_title("Model Coefficients")
            path = _timestamped_png_path(output_directory, "coefficients", run_ts)
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            created.append(path)

        if "learning_curve" in visualizations:
            X_train = self.context.get("X_train")
            y_train = self.context.get("y_train")
            if X_train is not None and y_train is not None:
                train_sizes, train_scores, val_scores = learning_curve(
                    model,
                    X_train,
                    y_train,
                    cv=3,
                    scoring="roc_auc",
                    n_jobs=self.config.get("n_jobs", None),
                    train_sizes=np.linspace(0.2, 1.0, 5),
                )
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(train_sizes, train_scores.mean(axis=1), label="Train")
                ax.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
                ax.set_xlabel("Training Set Size")
                ax.set_ylabel("ROC AUC")
                ax.set_title("Learning Curve")
                ax.legend(loc="best")
                path = _timestamped_png_path(output_directory, "learning_curve", run_ts)
                fig.savefig(path, bbox_inches="tight", dpi=150)
                plt.close(fig)
                created.append(path)

        result = {"visualizations": created, "output_directory": output_directory}
        self.context["sklearn_visualize"] = result
        return result


class SklearnXAIHandler:
    """Generate SHAP and LIME artifacts for sklearn models."""

    def __init__(self, config: Dict[str, Any], context: Dict[str, Any]):
        self.config = config
        self.context = context

    def execute(self) -> Dict[str, Any]:
        model_info = self.context.get("sklearn_model")
        if not model_info:
            raise ValueError("No trained model in context")
        model = model_info["model"] if isinstance(model_info, dict) else model_info
        model_type = model_info.get("model_type", "") if isinstance(model_info, dict) else ""

        X_train = self.context.get("X_train")
        X_test = self.context.get("X_test")
        if X_train is None or X_test is None:
            raise ValueError("Missing X_train/X_test in context")

        output_directory = self.config.get(
            "output_directory", f"artifacts/{model_type}_model/xai"
        )
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        run_ts = _build_run_timestamp(self.config)
        methods = self.config.get("methods", ["shap"])
        n_lime = int(self.config.get("n_lime", 3))
        created: List[str] = []
        feature_cols = self.context.get("feature_cols") or list(X_train.columns)

        if "shap" in methods:
            try:
                import shap

                if model_type in ("random_forest", "xgboost"):
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(X_test)
                else:
                    bg_n = min(int(self.config.get("kernel_background_samples", 50)), len(X_train))
                    ex_n = min(int(self.config.get("kernel_explain_samples", 50)), len(X_test))
                    background = X_train.iloc[:bg_n]
                    explain_set = X_test.iloc[:ex_n]
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    sv = explainer.shap_values(explain_set)
                    X_test = explain_set

                X_plot = X_test if len(X_test) <= 200 else X_test.iloc[:200]
                sv_plot = sv if len(sv) <= 200 else sv[:200]

                if isinstance(sv_plot, list):
                    cls_idx = 1 if len(sv_plot) > 1 else 0
                    sv_plot = sv_plot[cls_idx]

                sv_plot = np.asarray(sv_plot)
                if sv_plot.ndim == 3:
                    cls_idx = 1 if sv_plot.shape[2] > 1 else 0
                    sv_plot = sv_plot[:, :, cls_idx]
                elif sv_plot.ndim == 1:
                    sv_plot = sv_plot.reshape(-1, 1)

                if not isinstance(X_plot, pd.DataFrame):
                    X_plot = pd.DataFrame(X_plot, columns=feature_cols)
                else:
                    X_plot = X_plot.copy()
                X_plot = X_plot.reset_index(drop=True)

                n_rows = min(len(X_plot), len(sv_plot))
                if n_rows > 0:
                    X_plot = X_plot.iloc[:n_rows].copy()
                    sv_plot = sv_plot[:n_rows]
                    n_features = min(X_plot.shape[1], sv_plot.shape[1])
                    X_plot = X_plot.iloc[:, :n_features]
                    sv_plot = sv_plot[:, :n_features]
                    plot_feature_cols = list(X_plot.columns)

                    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_feature_cols) * 0.6)))
                    shap.summary_plot(sv_plot, X_plot, feature_names=plot_feature_cols, show=False)
                    path = _timestamped_png_path(output_directory, "shap_summary", run_ts)
                    plt.savefig(path, bbox_inches="tight", dpi=150)
                    plt.close(fig)
                    created.append(path)

                    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_feature_cols) * 0.6)))
                    shap.summary_plot(
                        sv_plot, X_plot, feature_names=plot_feature_cols, plot_type="bar", show=False
                    )
                    path = _timestamped_png_path(output_directory, "shap_importance", run_ts)
                    plt.savefig(path, bbox_inches="tight", dpi=150)
                    plt.close(fig)
                    created.append(path)
            except ImportError:
                logger.warning("SHAP not installed; skipping SHAP artifacts.")
            except Exception:
                logger.exception("SHAP generation failed")

        if "lime" in methods and n_lime > 0:
            try:
                from lime.lime_tabular import LimeTabularExplainer

                explainer = LimeTabularExplainer(
                    training_data=np.asarray(X_train),
                    feature_names=feature_cols,
                    class_names=["normal", "anomaly"],
                    mode="classification",
                )

                fn_raw = model.predict_proba if hasattr(model, "predict_proba") else None
                if fn_raw:
                    _fc = feature_cols

                    def fn(X_arr, _predict=fn_raw, _cols=_fc):
                        return _predict(pd.DataFrame(X_arr, columns=_cols))

                    max_n = min(n_lime, len(X_test))
                    for i in range(max_n):
                        exp = explainer.explain_instance(np.asarray(X_test.iloc[i]), fn, num_features=min(10, len(feature_cols)))
                        fig = exp.as_pyplot_figure()
                        path = _timestamped_png_path(output_directory, f"lime_instance_{i}", run_ts)
                        fig.savefig(path, bbox_inches="tight", dpi=150)
                        plt.close(fig)
                        created.append(path)
            except ImportError:
                logger.warning("LIME not installed; skipping LIME artifacts.")
            except Exception:
                logger.exception("LIME generation failed")

        result = {"xai_artifacts": created, "output_directory": output_directory}
        self.context["sklearn_xai"] = result
        return result
