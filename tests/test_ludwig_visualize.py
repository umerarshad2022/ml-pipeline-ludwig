"""
Unit tests for LudwigVisualizeHandler.
"""
import sys
import types


def _register_fake_ludwig_visualize(monkeypatch, calls):
    module = types.ModuleType("ludwig.visualize")

    def learning_curves_cli(training_statistics, **kwargs):
        calls.append(("learning_curves", training_statistics, kwargs))

    def confusion_matrix_cli(test_statistics, ground_truth_metadata, **kwargs):
        calls.append(("confusion_matrix", test_statistics, ground_truth_metadata, kwargs))

    def roc_curves_from_test_statistics_cli(test_statistics, **kwargs):
        calls.append(("roc_curve", test_statistics, kwargs))

    module.learning_curves_cli = learning_curves_cli
    module.confusion_matrix_cli = confusion_matrix_cli
    module.roc_curves_from_test_statistics_cli = roc_curves_from_test_statistics_cli

    ludwig_module = types.ModuleType("ludwig")
    ludwig_module.visualize = module

    monkeypatch.setitem(sys.modules, "ludwig", ludwig_module)
    monkeypatch.setitem(sys.modules, "ludwig.visualize", module)


def test_ludwig_visualize_uses_cli_with_artifact_files(tmp_path, monkeypatch):
    from pipelines.handlers.ludwig_visualize import LudwigVisualizeHandler

    train_dir = tmp_path / "artifacts" / "ludwig_model" / "experiment_0"
    eval_dir = tmp_path / "artifacts" / "evaluation"
    vis_dir = tmp_path / "artifacts" / "visualizations"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    training_stats = train_dir / "training_statistics.json"
    metadata = train_dir / "training_set_metadata.json"
    test_stats = eval_dir / "test_statistics.json"
    training_stats.write_text("{}")
    metadata.write_text("{}")
    test_stats.write_text("{}")

    calls = []
    _register_fake_ludwig_visualize(monkeypatch, calls)

    context = {
        "train-ludwig": {"output_directory": str(train_dir)},
        "evaluate": {"test_stats": {"is_anomaly": {}}, "output_directory": str(eval_dir)},
    }
    config = {
        "visualizations": ["learning_curves", "confusion_matrix", "roc_curve"],
        "output_directory": str(vis_dir),
    }

    result = LudwigVisualizeHandler(config, context).execute()

    assert result["visualizations_created"] == ["learning_curves", "confusion_matrix", "roc_curve"]
    assert len(calls) == 3
    assert calls[0][0] == "learning_curves"
    assert calls[0][1] == str(training_stats)
    assert calls[1][0] == "confusion_matrix"
    assert calls[1][1] == str(test_stats)
    assert calls[1][2] == str(metadata)
    assert calls[2][0] == "roc_curve"
    assert calls[2][1] == str(test_stats)


def test_ludwig_visualize_skips_when_required_files_missing(tmp_path, monkeypatch):
    from pipelines.handlers.ludwig_visualize import LudwigVisualizeHandler

    train_dir = tmp_path / "artifacts" / "ludwig_model" / "experiment_0"
    train_dir.mkdir(parents=True)
    (train_dir / "training_set_metadata.json").write_text("{}")

    calls = []
    _register_fake_ludwig_visualize(monkeypatch, calls)

    context = {"train-ludwig": {"output_directory": str(train_dir)}}
    config = {
        "visualizations": ["learning_curves", "confusion_matrix", "roc_curve"],
        "output_directory": str(tmp_path / "artifacts" / "visualizations"),
    }

    result = LudwigVisualizeHandler(config, context).execute()

    assert result["visualizations_created"] == []
    assert calls == []
