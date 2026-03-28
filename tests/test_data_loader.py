"""
Unit tests for DataLoaderHandler and UnsupervisedLabelingHandler.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_csv(tmp_path):
    """Write a tiny CSV and return its path."""
    df = pd.DataFrame(
        {
            "active_users": np.random.randint(0, 100, 200),
            "throughput_dl": np.random.uniform(0, 200, 200),
            "is_anomaly": np.random.randint(0, 2, 200),
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def sample_parquet(tmp_path):
    pytest.importorskip("pyarrow", reason="pyarrow not installed")
    df = pd.DataFrame(
        {
            "active_users": np.random.randint(0, 100, 50),
            "is_anomaly": np.random.randint(0, 2, 50),
        }
    )
    path = tmp_path / "sample.parquet"
    df.to_parquet(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# DataLoaderHandler
# ---------------------------------------------------------------------------

class TestDataLoaderHandler:
    def test_load_csv(self, sample_csv):
        from pipelines.handlers.data_loader import DataLoaderHandler

        context = {}
        handler = DataLoaderHandler(
            {"source": {"type": "csv", "path": sample_csv}}, context
        )
        result = handler.execute()

        assert "data" in result
        assert context["data"].shape[0] == 200
        assert context["data"].shape[1] == 3

    def test_load_parquet(self, sample_parquet):
        from pipelines.handlers.data_loader import DataLoaderHandler

        context = {}
        handler = DataLoaderHandler(
            {"source": {"type": "parquet", "path": sample_parquet}}, context
        )
        result = handler.execute()

        assert result["shape"][0] == 50

    def test_missing_path_raises(self):
        from pipelines.handlers.data_loader import DataLoaderHandler

        with pytest.raises(ValueError, match="path is required"):
            DataLoaderHandler({"source": {"type": "csv"}}, {}).execute()

    def test_unknown_type_raises(self, sample_csv):
        from pipelines.handlers.data_loader import DataLoaderHandler

        with pytest.raises(ValueError, match="Unsupported source type"):
            DataLoaderHandler(
                {"source": {"type": "xlsx", "path": sample_csv}}, {}
            ).execute()

    def test_drop_columns(self, sample_csv):
        from pipelines.handlers.data_loader import DataLoaderHandler

        context = {}
        handler = DataLoaderHandler(
            {
                "source": {"type": "csv", "path": sample_csv},
                "preprocessing": {"drop_columns": ["throughput_dl"]},
            },
            context,
        )
        handler.execute()
        assert "throughput_dl" not in context["data"].columns

    def test_standard_scaling(self, sample_csv):
        from pipelines.handlers.data_loader import DataLoaderHandler

        context = {}
        handler = DataLoaderHandler(
            {
                "source": {"type": "csv", "path": sample_csv},
                "preprocessing": {
                    "scaling": "standard",
                    "target_col": "is_anomaly",
                },
            },
            context,
        )
        handler.execute()
        # After standard scaling the mean of numeric feature columns ≈ 0
        mean_active = context["data"]["active_users"].mean()
        assert abs(mean_active) < 0.1

    def test_minmax_scaling(self, sample_csv):
        from pipelines.handlers.data_loader import DataLoaderHandler

        context = {}
        handler = DataLoaderHandler(
            {
                "source": {"type": "csv", "path": sample_csv},
                "preprocessing": {
                    "scaling": "minmax",
                    "target_col": "is_anomaly",
                },
            },
            context,
        )
        handler.execute()
        assert context["data"]["active_users"].min() >= 0.0
        assert context["data"]["active_users"].max() <= 1.0

    def test_train_test_split(self, sample_csv):
        from pipelines.handlers.data_loader import DataLoaderHandler

        context = {}
        handler = DataLoaderHandler(
            {
                "source": {"type": "csv", "path": sample_csv},
                "preprocessing": {
                    "test_size": 0.2,
                    "target_col": "is_anomaly",
                },
            },
            context,
        )
        handler.execute()
        assert "X_train" in context
        assert "X_test" in context
        assert len(context["X_train"]) == pytest.approx(160, abs=5)
        assert len(context["X_test"]) == pytest.approx(40, abs=5)


# ---------------------------------------------------------------------------
# UnsupervisedLabelingHandler
# ---------------------------------------------------------------------------

class TestUnsupervisedLabelingHandler:
    def test_isolation_forest_labeling(self, sample_csv):
        from pipelines.handlers.data_loader import (
            DataLoaderHandler,
            UnsupervisedLabelingHandler,
        )

        context = {}
        DataLoaderHandler(
            {"source": {"type": "csv", "path": sample_csv}}, context
        ).execute()

        # Remove existing labels so handler creates them
        context["data"] = context["data"].drop(columns=["is_anomaly"])

        handler = UnsupervisedLabelingHandler(
            {
                "method": "isolation_forest",
                "contamination": 0.10,
                "feature": "active_users",
                "output_column": "is_anomaly",
            },
            context,
        )
        result = handler.execute()

        assert "is_anomaly" in context["data"].columns
        assert set(context["data"]["is_anomaly"].unique()).issubset({0, 1})
        assert result["n_anomalies"] > 0

    def test_no_data_raises(self):
        from pipelines.handlers.data_loader import UnsupervisedLabelingHandler

        with pytest.raises(ValueError, match="No data in context"):
            UnsupervisedLabelingHandler({}, {}).execute()

    def test_unknown_method_raises(self, sample_csv):
        from pipelines.handlers.data_loader import (
            DataLoaderHandler,
            UnsupervisedLabelingHandler,
        )

        context = {}
        DataLoaderHandler(
            {"source": {"type": "csv", "path": sample_csv}}, context
        ).execute()

        with pytest.raises(ValueError, match="Unsupported labeling method"):
            UnsupervisedLabelingHandler(
                {"method": "unsupported", "feature": "active_users"}, context
            ).execute()
