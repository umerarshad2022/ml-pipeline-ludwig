"""
Unit tests for SklearnTrainHandler, SklearnEvaluateHandler,
SklearnVisualizeHandler, and SklearnXAIHandler.
"""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_context():
    """Build a minimal context with pre-split data matching the pipeline output."""
    np.random.seed(42)
    n = 300
    X = pd.DataFrame({"active_users": np.random.randint(0, 100, n)})
    y = pd.Series(np.random.randint(0, 2, n), name="is_anomaly")

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    full_df = X.copy()
    full_df["is_anomaly"] = y

    return {
        "data": full_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": ["active_users"],
        "target_col": "is_anomaly",
    }


# ---------------------------------------------------------------------------
# SklearnTrainHandler
# ---------------------------------------------------------------------------

class TestSklearnTrainHandler:
    def _run_train(self, context, model_type, tmp_path, extra_params=None):
        from pipelines.handlers.sklearn_handler import SklearnTrainHandler

        params = {"random_state": 42, "n_estimators": 10} if model_type == "random_forest" else {"random_state": 42}
        if model_type == "logistic_regression":
            params = {"random_state": 42, "max_iter": 200}
        if extra_params:
            params.update(extra_params)

        sklearn_config = {
            "type": model_type,
            "params": params,
            "input_features": ["active_users"],
            "output_feature": "is_anomaly",
        }
        step_config = {
            "output_directory": str(tmp_path / model_type),
            "experiment_name": "test_experiment",
        }
        handler = SklearnTrainHandler(step_config, context, sklearn_config)
        return handler.execute()

    def test_random_forest_train(self, simple_context, tmp_path):
        result = self._run_train(simple_context, "random_forest", tmp_path)
        assert "model" in result
        assert result["model_type"] == "random_forest"
        assert "model_path" in result
        import os
        assert os.path.exists(result["model_path"])

    def test_logistic_regression_train(self, simple_context, tmp_path):
        result = self._run_train(simple_context, "logistic_regression", tmp_path)
        assert result["model_type"] == "logistic_regression"

    def test_unknown_model_type_raises(self, simple_context, tmp_path):
        from pipelines.handlers.sklearn_handler import SklearnTrainHandler

        handler = SklearnTrainHandler(
            {"output_directory": str(tmp_path)},
            simple_context,
            {"type": "neural_network", "params": {}},
        )
        with pytest.raises(ValueError, match="Unsupported model_type"):
            handler.execute()

    def test_no_data_raises(self):
        from pipelines.handlers.sklearn_handler import SklearnTrainHandler

        handler = SklearnTrainHandler(
            {},
            {},
            {"type": "random_forest", "params": {}, "input_features": ["x"]},
        )
        with pytest.raises(ValueError, match="No data in context"):
            handler.execute()

    def test_model_stored_in_context(self, simple_context, tmp_path):
        self._run_train(simple_context, "random_forest", tmp_path)
        assert "sklearn_model" in simple_context
        assert simple_context["sklearn_model"]["model"] is not None


# ---------------------------------------------------------------------------
# SklearnEvaluateHandler
# ---------------------------------------------------------------------------

class TestSklearnEvaluateHandler:
    def _setup(self, context, tmp_path):
        from pipelines.handlers.sklearn_handler import SklearnTrainHandler

        sklearn_config = {
            "type": "random_forest",
            "params": {"random_state": 42, "n_estimators": 10},
            "input_features": ["active_users"],
            "output_feature": "is_anomaly",
        }
        SklearnTrainHandler(
            {"output_directory": str(tmp_path / "rf")},
            context,
            sklearn_config,
        ).execute()

    def test_metrics_produced(self, simple_context, tmp_path):
        from pipelines.handlers.sklearn_handler import SklearnEvaluateHandler

        self._setup(simple_context, tmp_path)
        handler = SklearnEvaluateHandler(
            {"output_directory": str(tmp_path / "eval")}, simple_context
        )
        result = handler.execute()

        assert "metrics" in result
        for key in ("accuracy", "precision", "recall", "f1_score"):
            assert key in result["metrics"]
            assert 0.0 <= result["metrics"][key] <= 1.0

    def test_metrics_file_saved(self, simple_context, tmp_path):
        from pipelines.handlers.sklearn_handler import SklearnEvaluateHandler
        import os

        self._setup(simple_context, tmp_path)
        eval_dir = str(tmp_path / "eval")
        SklearnEvaluateHandler({"output_directory": eval_dir}, simple_context).execute()
        assert os.path.exists(os.path.join(eval_dir, "metrics.txt"))

    def test_no_model_raises(self):
        from pipelines.handlers.sklearn_handler import SklearnEvaluateHandler

        with pytest.raises(ValueError, match="No trained model in context"):
            SklearnEvaluateHandler({}, {}).execute()


# ---------------------------------------------------------------------------
# SklearnVisualizeHandler
# ---------------------------------------------------------------------------

class TestSklearnVisualizeHandler:
    def _setup(self, context, tmp_path):
        from pipelines.handlers.sklearn_handler import (
            SklearnEvaluateHandler,
            SklearnTrainHandler,
        )

        sklearn_config = {
            "type": "random_forest",
            "params": {"random_state": 42, "n_estimators": 10},
            "input_features": ["active_users"],
            "output_feature": "is_anomaly",
        }
        SklearnTrainHandler(
            {"output_directory": str(tmp_path / "rf")}, context, sklearn_config
        ).execute()
        SklearnEvaluateHandler(
            {"output_directory": str(tmp_path / "eval")}, context
        ).execute()

    def test_basic_visualizations(self, simple_context, tmp_path):
        from pipelines.handlers.sklearn_handler import SklearnVisualizeHandler
        import os

        self._setup(simple_context, tmp_path)
        vis_dir = str(tmp_path / "vis")
        handler = SklearnVisualizeHandler(
            {
                "output_directory": vis_dir,
                "visualizations": ["confusion_matrix", "roc_curve", "precision_recall_curve"],
            },
            simple_context,
        )
        result = handler.execute()

        assert len(result["visualizations"]) == 3
        for path in result["visualizations"]:
            assert os.path.exists(path)

    def test_feature_importance(self, simple_context, tmp_path):
        from pipelines.handlers.sklearn_handler import SklearnVisualizeHandler
        import os

        self._setup(simple_context, tmp_path)
        vis_dir = str(tmp_path / "vis2")
        result = SklearnVisualizeHandler(
            {"output_directory": vis_dir, "visualizations": ["feature_importance"]},
            simple_context,
        ).execute()
        assert any("feature_importance" in p for p in result["visualizations"])

    def test_visualization_uses_timestamped_filenames(self, simple_context, tmp_path):
        from pipelines.handlers.sklearn_handler import SklearnVisualizeHandler
        import os

        self._setup(simple_context, tmp_path)
        vis_dir = str(tmp_path / "vis_ts")
        run_id = "20260427_171400"
        result = SklearnVisualizeHandler(
            {
                "output_directory": vis_dir,
                "run_id": run_id,
                "visualizations": ["confusion_matrix", "roc_curve"],
            },
            simple_context,
        ).execute()

        assert all(path.endswith(f"_{run_id}.png") for path in result["visualizations"])
        for path in result["visualizations"]:
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# SklearnXAIHandler
# ---------------------------------------------------------------------------

class TestSklearnXAIHandler:
    def _setup(self, context, tmp_path):
        from pipelines.handlers.sklearn_handler import (
            SklearnEvaluateHandler,
            SklearnTrainHandler,
        )

        sklearn_config = {
            "type": "random_forest",
            "params": {"random_state": 42, "n_estimators": 10},
            "input_features": ["active_users"],
            "output_feature": "is_anomaly",
        }
        SklearnTrainHandler(
            {"output_directory": str(tmp_path / "rf")}, context, sklearn_config
        ).execute()
        SklearnEvaluateHandler(
            {"output_directory": str(tmp_path / "eval")}, context
        ).execute()

    def test_shap_creates_artifacts(self, simple_context, tmp_path):
        import os

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        from pipelines.handlers.sklearn_handler import SklearnXAIHandler

        self._setup(simple_context, tmp_path)
        xai_dir = str(tmp_path / "xai")
        result = SklearnXAIHandler(
            {"output_directory": xai_dir, "methods": ["shap"], "n_lime": 0},
            simple_context,
        ).execute()

        assert len(result["xai_artifacts"]) > 0
        for path in result["xai_artifacts"]:
            assert os.path.exists(path)

    def test_xai_uses_timestamped_filenames(self, simple_context, tmp_path):
        import os

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        from pipelines.handlers.sklearn_handler import SklearnXAIHandler

        self._setup(simple_context, tmp_path)
        xai_dir = str(tmp_path / "xai_ts")
        run_id = "20260427_171400"
        result = SklearnXAIHandler(
            {"output_directory": xai_dir, "methods": ["shap"], "n_lime": 0, "run_id": run_id},
            simple_context,
        ).execute()

        assert len(result["xai_artifacts"]) > 0
        assert all(path.endswith(f"_{run_id}.png") for path in result["xai_artifacts"])
        for path in result["xai_artifacts"]:
            assert os.path.exists(path)

    def test_no_model_raises(self):
        from pipelines.handlers.sklearn_handler import SklearnXAIHandler

        with pytest.raises(ValueError, match="No trained model in context"):
            SklearnXAIHandler({}, {}).execute()
