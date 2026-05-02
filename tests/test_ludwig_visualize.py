"""
Unit tests for LudwigVisualizeHandler.
"""
import json
import sys
import types

import pandas as pd


def _install_fake_ludwig(monkeypatch, calls):
    ludwig_module = types.ModuleType("ludwig")
    visualize_module = types.ModuleType("ludwig.visualize")

    def learning_curves(*args, **kwargs):
        calls["learning_curves"] = (args, kwargs)

    def confusion_matrix(*args, **kwargs):
        calls["confusion_matrix"] = (args, kwargs)

    def roc_curves_from_test_statistics(*args, **kwargs):
        calls["roc_curves_from_test_statistics"] = (args, kwargs)

    def precision_recall_curves_from_test_statistics(*args, **kwargs):
        calls["precision_recall_curves_from_test_statistics"] = (args, kwargs)

    def compare_performance(*args, **kwargs):
        calls["compare_performance"] = (args, kwargs)

    visualize_module.learning_curves = learning_curves
    visualize_module.confusion_matrix = confusion_matrix
    visualize_module.roc_curves_from_test_statistics = roc_curves_from_test_statistics
    visualize_module.precision_recall_curves_from_test_statistics = (
        precision_recall_curves_from_test_statistics
    )
    visualize_module.compare_performance = compare_performance
    ludwig_module.visualize = visualize_module

    monkeypatch.setitem(sys.modules, "ludwig", ludwig_module)
    monkeypatch.setitem(sys.modules, "ludwig.visualize", visualize_module)


class _FakeLudwigModel:
    def __init__(self, predictions_frame, metadata):
        self._predictions_frame = predictions_frame
        self.training_set_metadata = metadata

    def predict(self, dataset, **kwargs):
        return self._predictions_frame.copy(), kwargs.get("output_directory", "results")


class TestLudwigVisualizeHandler:
    def test_uses_in_memory_stats_and_prediction_fallbacks(self, monkeypatch, tmp_path):
        from pipelines.handlers.ludwig_visualize import LudwigVisualizeHandler

        calls = {}
        _install_fake_ludwig(monkeypatch, calls)

        df = pd.DataFrame(
            {
                "active_users": [10, 20, 30, 40],
                "is_anomaly": [0, 1, 0, 1],
            }
        )
        predictions = pd.DataFrame(
            {
                "is_anomaly_predictions": [0, 1, 0, 1],
                "is_anomaly_probability": [0.1, 0.9, 0.2, 0.8],
                "is_anomaly_probabilities_True": [0.1, 0.9, 0.2, 0.8],
            }
        )
        fake_model = _FakeLudwigModel(
            predictions,
            {"is_anomaly": {"type": "binary", "bool2str": ["False", "True"]}},
        )

        context = {
            "data": df,
            "train-ludwig": {
                "train_stats": {"training": {"is_anomaly": {"loss": [1.0, 0.5]}}},
                "model": fake_model,
                "output_directory": str(tmp_path / "model"),
                "input_features": ["active_users"],
            },
            "evaluate": {
                "test_stats": {"is_anomaly": {"overall_stats": {"f1_score": 0.8}}},
                "predictions": predictions,
                "output_directory": str(tmp_path / "eval"),
            },
        }

        result = LudwigVisualizeHandler(
            {
                "output_directory": str(tmp_path / "vis"),
                "visualizations": [
                    "learning_curves",
                    "confusion_matrix",
                    "roc_curve",
                    "precision_recall_curve",
                    "compare_performance",
                    "feature_importance",
                ],
            },
            context,
        ).execute()

        assert result["visualizations_created"] == [
            "learning_curves",
            "confusion_matrix",
            "roc_curve",
            "precision_recall_curve",
            "compare_performance",
            "feature_importance",
        ]
        assert calls["learning_curves"][0][0] == [context["train-ludwig"]["train_stats"]]
        assert calls["learning_curves"][1]["file_format"] == "png"
        assert calls["confusion_matrix"][0][0] == [context["evaluate"]["test_stats"]]
        assert calls["roc_curves_from_test_statistics"][1]["output_feature_name"] == "is_anomaly"
        assert calls["precision_recall_curves_from_test_statistics"][1]["file_format"] == "png"
        assert calls["compare_performance"][1]["file_format"] == "png"

        vis_dir = tmp_path / "vis"
        expected_files = [
            "confusion_matrix_fallback.png",
            "roc_curve_fallback.png",
            "precision_recall_curve_fallback.png",
            "compare_performance_fallback.png",
            "feature_importance_permutation.png",
        ]
        for name in expected_files:
            assert (vis_dir / name).exists(), name

    def test_falls_back_to_saved_json_artifacts(self, monkeypatch, tmp_path):
        from pipelines.handlers.ludwig_visualize import LudwigVisualizeHandler

        calls = {}
        _install_fake_ludwig(monkeypatch, calls)

        model_dir = tmp_path / "model"
        eval_dir = tmp_path / "eval"
        model_dir.mkdir()
        eval_dir.mkdir()

        training_stats = {"training": {"is_anomaly": {"loss": [1.0, 0.5]}}}
        test_stats = {"is_anomaly": {"overall_stats": {"precision": 0.7}}}
        metadata = {"is_anomaly": {"type": "binary"}}

        (model_dir / "training_statistics.json").write_text(json.dumps(training_stats), encoding="utf-8")
        (model_dir / "training_set_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (eval_dir / "test_statistics.json").write_text(json.dumps(test_stats), encoding="utf-8")

        context = {
            "train-ludwig": {"output_directory": str(model_dir)},
            "evaluate": {"output_directory": str(eval_dir)},
        }

        result = LudwigVisualizeHandler(
            {
                "output_directory": str(tmp_path / "vis"),
                "visualizations": ["learning_curves", "confusion_matrix", "roc_curve"],
            },
            context,
        ).execute()

        assert result["visualizations_created"] == ["learning_curves"]
        assert calls["learning_curves"][0][0] == [training_stats]
        assert calls["confusion_matrix"][0][0] == [test_stats]
        assert calls["confusion_matrix"][0][1] == metadata
        assert calls["roc_curves_from_test_statistics"][1]["output_feature_name"] == "is_anomaly"
