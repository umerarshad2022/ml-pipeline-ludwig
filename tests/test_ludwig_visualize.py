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
        "run_id": "unit_test_run",
    }

    result = LudwigVisualizeHandler(config, context).execute()

    assert result["output_directory"].endswith("run_unit_test_run")
    assert result["visualizations_created"] == ["learning_curves", "confusion_matrix", "roc_curve"]
    assert len(calls) == 3
    assert calls[0][0] == "learning_curves"
    assert calls[0][1] == str(training_stats)
    assert calls[1][0] == "confusion_matrix"
    assert calls[1][1] == str(test_stats)
    assert calls[1][2] == str(metadata)
    assert calls[1][3]["top_n_classes"] == 10
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

    assert result["skipped"] is True
    assert calls == []


def test_ludwig_visualize_supports_legacy_roc_function_name(tmp_path, monkeypatch):
    from pipelines.handlers.ludwig_visualize import LudwigVisualizeHandler

    module = types.ModuleType("ludwig.visualize")
    calls = []

    def roc_curves_cli(test_statistics, **kwargs):
        calls.append(("roc_curve", test_statistics, kwargs))

    module.roc_curves_cli = roc_curves_cli
    ludwig_module = types.ModuleType("ludwig")
    ludwig_module.visualize = module

    monkeypatch.setitem(sys.modules, "ludwig", ludwig_module)
    monkeypatch.setitem(sys.modules, "ludwig.visualize", module)

    train_dir = tmp_path / "artifacts" / "ludwig_model" / "experiment_0"
    eval_dir = tmp_path / "artifacts" / "evaluation"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (eval_dir / "test_statistics.json").write_text("{}")

    context = {
        "train-ludwig": {"output_directory": str(train_dir)},
        "evaluate": {"test_stats": {"is_anomaly": {}}, "output_directory": str(eval_dir)},
    }
    config = {
        "visualizations": ["roc_curve"],
        "output_directory": str(tmp_path / "artifacts" / "visualizations"),
    }

    result = LudwigVisualizeHandler(config, context).execute()

    assert result["visualizations_created"] == ["roc_curve"]
    assert calls[0][1].endswith("test_statistics.json")


def test_ludwig_visualize_defaults_to_model_scoped_visualization_dir(tmp_path, monkeypatch):
    from pipelines.handlers.ludwig_visualize import LudwigVisualizeHandler

    train_dir = tmp_path / "artifacts" / "ludwig_model" / "experiment_0"
    eval_dir = tmp_path / "artifacts" / "evaluation"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (train_dir / "training_statistics.json").write_text("{}")
    (eval_dir / "test_statistics.json").write_text("{}")

    calls = []
    _register_fake_ludwig_visualize(monkeypatch, calls)

    context = {
        "train-ludwig": {"output_directory": str(train_dir)},
        "evaluate": {"test_stats": {"is_anomaly": {}}, "output_directory": str(eval_dir)},
    }
    config = {
        "visualizations": ["roc_curve"],
        "run_id": "unit_test_run",
    }

    result = LudwigVisualizeHandler(config, context).execute()

    expected_prefix = str(train_dir / "visualizations" / "run_unit_test_run")
    assert result["output_directory"] == expected_prefix
    assert result["visualizations_created"] == ["roc_curve"]
