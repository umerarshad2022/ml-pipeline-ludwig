# Testing

The project uses [pytest](https://docs.pytest.org/) as its test framework.
Tests are located in the `tests/` directory and cover the core pipeline modules.

---

## Test Suite Overview

| Test file | What it tests |
|---|---|
| `tests/test_engine.py` | `LudwigPipelineEngine` — config loading, validation, unknown step types |
| `tests/test_data_loader.py` | `DataLoaderHandler` and `UnsupervisedLabelingHandler` |
| `tests/test_sklearn_handler.py` | `SklearnTrainHandler`, `SklearnEvaluateHandler`, `SklearnVisualizeHandler`, `SklearnXAIHandler` |

---

## Running the Tests

### Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Install pytest (if not already installed)
pip install pytest
```

### Run All Tests

```bash
pytest tests/ -v
```

The `-v` flag ("verbose") prints each test name as it runs, making it easy to
see what passed and what failed.

### Run a Single Test File

```bash
pytest tests/test_engine.py -v
pytest tests/test_data_loader.py -v
pytest tests/test_sklearn_handler.py -v
```

### Run a Single Test Class

```bash
pytest tests/test_engine.py::TestEngineValidation -v
```

### Run a Single Test Function

```bash
pytest tests/test_engine.py::TestEngineValidation::test_sklearn_validation_passes -v
```

### Run Tests Matching a Pattern

```bash
pytest tests/ -k "validation" -v     # All tests with "validation" in the name
pytest tests/ -k "sklearn" -v        # All sklearn-related tests
```

---

## Continuous Integration

Tests run automatically on every push via GitHub Actions.
See `.github/workflows/ci.yml` for the CI configuration.

---

## Understanding the Tests

### `test_engine.py`

#### `TestEngineLoadConfig`

Tests that the engine correctly reads and parses YAML files.

```python
def test_load_sklearn_pipeline(self, tmp_path):
    # Creates a temporary YAML file
    # Loads it with LudwigPipelineEngine
    # Asserts the parsed config has the right values
```

The `tmp_path` fixture is built into pytest — it provides a temporary directory
that is cleaned up after each test.

#### `TestEngineValidation`

Tests the `validate_config()` method:

| Test | What it checks |
|---|---|
| `test_sklearn_validation_passes` | A valid sklearn config does not raise an error |
| `test_sklearn_validation_fails_when_missing_sklearn_model` | Raises `ValueError` if `sklearn_model` block is missing |
| `test_ludwig_validation_skipped_for_sklearn_only_pipeline` | A sklearn-only pipeline does NOT require a `ludwig:` block |
| `test_unknown_step_type_raises` | Raises `ValueError` for unrecognised step types |

---

### `test_data_loader.py`

#### Fixtures

The test file defines two pytest fixtures:

```python
@pytest.fixture()
def sample_csv(tmp_path):
    """Creates a 200-row CSV with active_users, throughput_dl, is_anomaly columns."""
    ...

@pytest.fixture()
def sample_parquet(tmp_path):
    """Creates a 50-row Parquet file. Skipped if pyarrow not installed."""
    ...
```

> **What is a pytest fixture?**  
> A function decorated with `@pytest.fixture()` that provides shared setup for
> tests. When a test function lists it as a parameter, pytest automatically calls
> the fixture and passes the result as the argument.

#### `TestDataLoaderHandler`

| Test | What it checks |
|---|---|
| `test_load_csv` | CSV loaded, 200 rows, stored in context |
| `test_load_parquet` | Parquet loaded (skipped if pyarrow missing) |
| `test_missing_path_raises` | `ValueError` when `source.path` not provided |
| `test_unknown_type_raises` | `ValueError` for unsupported format like `xlsx` |
| `test_drop_columns` | Specified columns are removed from the DataFrame |
| `test_standard_scaling` | After scaling, mean of numeric columns ≈ 0 |
| `test_minmax_scaling` | After scaling, all values are in [0, 1] |
| `test_train_test_split` | Correct proportions (~160/~40 for 200 rows with 20% test) |

#### `TestUnsupervisedLabelingHandler`

| Test | What it checks |
|---|---|
| `test_isolation_forest_labeling` | Adds `is_anomaly` column with values in {0, 1}, at least one anomaly |
| `test_no_data_raises` | `ValueError` when context has no data |
| `test_unknown_method_raises` | `ValueError` for unsupported method |

---

### `test_sklearn_handler.py`

#### Fixture: `simple_context`

Pre-built context with 300-row DataFrame, pre-split into train (80%) and
test (20%) sets. All sklearn handler tests start from this context.

```python
@pytest.fixture()
def simple_context():
    # Creates X (active_users), y (is_anomaly)
    # Performs train/test split
    # Returns context dict with data, X_train, X_test, y_train, y_test
    ...
```

#### `TestSklearnTrainHandler`

| Test | What it checks |
|---|---|
| `test_random_forest_train` | Model trained, `.pkl` file exists on disk |
| `test_logistic_regression_train` | Model type is `logistic_regression` |
| `test_unknown_model_type_raises` | `ValueError` for `neural_network` |
| `test_no_data_raises` | `ValueError` when context is empty |
| `test_model_stored_in_context` | `context["sklearn_model"]` is populated |

#### `TestSklearnEvaluateHandler`

| Test | What it checks |
|---|---|
| `test_metrics_produced` | accuracy, precision, recall, f1_score all in [0, 1] |
| `test_metrics_file_saved` | `metrics.txt` file exists after evaluation |
| `test_no_model_raises` | `ValueError` when no model in context |

#### `TestSklearnVisualizeHandler`

| Test | What it checks |
|---|---|
| `test_basic_visualizations` | 3 PNG files created (confusion, ROC, PR) |
| `test_feature_importance` | `feature_importance.png` created for Random Forest |

#### `TestSklearnXAIHandler`

| Test | What it checks |
|---|---|
| `test_shap_creates_artifacts` | At least one SHAP file created (skipped if shap not installed) |
| `test_no_model_raises` | `ValueError` when no model in context |

---

## Writing New Tests

### Conventions

- Tests go in `tests/` directory.
- Test files are named `test_<module_name>.py`.
- Test classes are named `Test<ClassName>`.
- Test functions are named `test_<what_is_being_tested>`.
- Use `pytest.raises(ExceptionType, match="substring")` to test that errors are raised.
- Use `tmp_path` fixture for temporary files (pytest provides this automatically).

### Example: Testing a New Handler

```python
# tests/test_my_new_handler.py
import pytest
import pandas as pd
import numpy as np


@pytest.fixture()
def context_with_data(tmp_path):
    """Context with a simple dataset."""
    df = pd.DataFrame({
        "feature_a": np.random.randn(100),
        "is_anomaly": np.random.randint(0, 2, 100),
    })
    return {"data": df}


class TestMyNewHandler:
    def test_basic_execution(self, context_with_data, tmp_path):
        from pipelines.handlers.my_module import MyNewHandler

        handler = MyNewHandler(
            config={"output_directory": str(tmp_path)},
            context=context_with_data,
        )
        result = handler.execute()

        assert "my_output" in result

    def test_missing_data_raises(self):
        from pipelines.handlers.my_module import MyNewHandler

        with pytest.raises(ValueError, match="No data in context"):
            MyNewHandler(config={}, context={}).execute()
```

### Skipping Tests Conditionally

If a test requires an optional library:

```python
def test_shap_creates_artifacts(self, ...):
    try:
        import shap
    except ImportError:
        pytest.skip("shap not installed")
    ...
```

Or using the decorator:

```python
@pytest.mark.skipif(
    not importlib.util.find_spec("shap"),
    reason="shap not installed"
)
def test_shap_creates_artifacts(self, ...):
    ...
```

---

## Test Coverage

To measure how much of the code is covered by tests:

```bash
pip install pytest-cov
pytest tests/ --cov=pipelines --cov-report=term-missing
```

This prints which lines in `pipelines/` are **not** covered by any test.

---

## Common Test Failures and Solutions

### `ModuleNotFoundError: No module named 'pipelines'`

Run pytest from the project root directory:

```bash
cd ml-pipeline-ludwig
pytest tests/ -v
```

### `ModuleNotFoundError: No module named 'shap'`

SHAP tests are skipped if SHAP is not installed. Install it:

```bash
pip install shap==0.43.0
```

### `FAILED tests/test_sklearn_handler.py::... FileNotFoundError`

The test is trying to create files in a directory that doesn't exist.
Check that `tmp_path` is being passed correctly to the handler's `output_directory`.
