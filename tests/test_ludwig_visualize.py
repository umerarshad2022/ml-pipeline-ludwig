"""
Unit tests for LudwigVisualizeHandler.
"""
import json
import sys
import types
from types import SimpleNamespace


def _install_fake_ludwig(monkeypatch, calls):
    ludwig_module = types.ModuleType("ludwig")
    visualize_module = types.ModuleType("ludwig.visualize")

    def learning_curves(*args, **kwargs):
        calls["learning_curves"] = (args, kwargs)

    def confusion_matrix(*args, **kwargs):
        calls["confusion_matrix"] = (args, kwargs)

    def roc_curves_from_test_statistics(*args, **kwargs):
        calls["roc_curves_from_test_statistics"] = (args, kwargs)

    def compare_performance(*args, **kwargs):
        calls["compare_performance"] = (args, kwargs)

    visualize_module.learning_curves = learning_curves
    visualize_module.confusion_matrix = confusion_matrix
    visualize_module.roc_curves_from_test_statistics = roc_curves_from_test_statistics
    visualize_module.compare_performance = compare_performance
    ludwig_module.visualize = visualize_module

    monkeypatch.setitem(sys.modules, "ludwig", ludwig_module)
    monkeypatch.setitem(sys.modules, "ludwig.visualize", visualize_module)


class TestLudwigVisualizeHandler:
    def test_uses_in_memory_stats_and_saves_pngs(self, monkeypatch, tmp_path):
        from pipelines.handlers.ludwig_visualize import LudwigVisualizeHandler

        calls = {}
        _install_fake_ludwig(monkeypatch, calls)

        context = {
            "train-ludwig": {
                "train_stats": {"training": {"is_anomaly": {"loss": [1.0, 0.5]}}},
                "model": SimpleNamespace(training_set_metadata={"is_anomaly": {"type": "binary"}}),
                "output_directory": str(tmp_path / "model"),
            },
            "evaluate": {
                "test_stats": {"is_anomaly": {"overall_stats": {"f1_score": 0.8}}},
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
                    "compare_performance",
                ],
            },
            context,
        ).execute()

        assert result["visualizations_created"] == [
            "learning_curves",
            "confusion_matrix",
            "roc_curve",
            "compare_performance",
        ]
        assert calls["learning_curves"][0][0] == [context["train-ludwig"]["train_stats"]]
        assert calls["learning_curves"][1]["file_format"] == "png"
        assert calls["confusion_matrix"][0][0] == [context["evaluate"]["test_stats"]]
        assert calls["confusion_matrix"][0][1] == context["train-ludwig"]["model"].training_set_metadata
        assert calls["confusion_matrix"][1]["output_feature_name"] == "is_anomaly"
        assert calls["confusion_matrix"][1]["file_format"] == "png"
        assert calls["roc_curves_from_test_statistics"][0][0] == [context["evaluate"]["test_stats"]]
        assert calls["roc_curves_from_test_statistics"][1]["output_feature_name"] == "is_anomaly"
        assert calls["roc_curves_from_test_statistics"][1]["file_format"] == "png"
        assert calls["compare_performance"][0][0] == [context["evaluate"]["test_stats"]]
        assert calls["compare_performance"][1]["file_format"] == "png"

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

        assert result["visualizations_created"] == [
            "learning_curves",
            "confusion_matrix",
            "roc_curve",
        ]
        assert calls["learning_curves"][0][0] == [training_stats]
        assert calls["confusion_matrix"][0][0] == [test_stats]
        assert calls["confusion_matrix"][0][1] == metadata
        assert calls["roc_curves_from_test_statistics"][1]["output_feature_name"] == "is_anomaly"
