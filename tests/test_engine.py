"""
Unit tests for LudwigPipelineEngine configuration loading and validation.
"""
import textwrap
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_yaml(tmp_path, content: str) -> str:
    path = tmp_path / "pipeline.yaml"
    path.write_text(textwrap.dedent(content))
    return str(path)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestEngineLoadConfig:
    def test_load_sklearn_pipeline(self, tmp_path):
        from pipelines.engine import LudwigPipelineEngine

        cfg_path = write_yaml(
            tmp_path,
            """
            name: Test RF Pipeline
            description: Unit test pipeline
            metadata: {}
            data: {}
            sklearn_model:
              type: random_forest
              params:
                n_estimators: 10
                random_state: 42
              input_features:
                - active_users
              output_feature: is_anomaly
            steps:
              - name: load-data
                type: data_loader
                config:
                  source:
                    type: csv
                    path: data/sample_data.csv
            """,
        )

        engine = LudwigPipelineEngine(cfg_path)
        config = engine.load_config()

        assert config.name == "Test RF Pipeline"
        assert config.sklearn_model["type"] == "random_forest"
        assert len(config.steps) == 1

    def test_sklearn_config_stored_on_engine(self, tmp_path):
        from pipelines.engine import LudwigPipelineEngine

        cfg_path = write_yaml(
            tmp_path,
            """
            name: Pipeline
            description: ""
            metadata: {}
            data: {}
            sklearn_model:
              type: xgboost
              params: {}
            steps: []
            """,
        )
        engine = LudwigPipelineEngine(cfg_path)
        engine.load_config()
        assert engine.config.sklearn_model["type"] == "xgboost"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestEngineValidation:
    def test_sklearn_validation_passes(self, tmp_path):
        from pipelines.engine import LudwigPipelineEngine

        cfg_path = write_yaml(
            tmp_path,
            """
            name: Test
            description: ""
            metadata: {}
            data: {}
            sklearn_model:
              type: random_forest
              params: {}
            steps:
              - name: train
                type: sklearn_train
                config: {}
            """,
        )
        engine = LudwigPipelineEngine(cfg_path)
        engine.load_config()
        engine.validate_config()  # should not raise

    def test_sklearn_validation_fails_when_missing_sklearn_model(self, tmp_path):
        from pipelines.engine import LudwigPipelineEngine

        cfg_path = write_yaml(
            tmp_path,
            """
            name: Test
            description: ""
            metadata: {}
            data: {}
            steps:
              - name: train
                type: sklearn_train
                config: {}
            """,
        )
        engine = LudwigPipelineEngine(cfg_path)
        engine.load_config()
        with pytest.raises(ValueError, match="sklearn_model configuration is required"):
            engine.validate_config()

    def test_ludwig_validation_skipped_for_sklearn_only_pipeline(self, tmp_path):
        """A sklearn-only pipeline should NOT require a ludwig: block."""
        from pipelines.engine import LudwigPipelineEngine

        cfg_path = write_yaml(
            tmp_path,
            """
            name: Test
            description: ""
            metadata: {}
            data: {}
            sklearn_model:
              type: random_forest
              params: {}
            steps:
              - name: train
                type: sklearn_train
                config: {}
            """,
        )
        engine = LudwigPipelineEngine(cfg_path)
        engine.load_config()
        engine.validate_config()  # must not raise ValueError about ludwig config

    def test_unknown_step_type_raises(self, tmp_path):
        from pipelines.engine import LudwigPipelineEngine

        cfg_path = write_yaml(
            tmp_path,
            """
            name: Test
            description: ""
            metadata: {}
            data: {}
            steps:
              - name: unknown_step
                type: nonexistent_handler
                config: {}
            """,
        )
        engine = LudwigPipelineEngine(cfg_path)
        engine.load_config()
        with pytest.raises(ValueError, match="Unknown step type"):
            engine._execute_step(engine.config.steps[0])
