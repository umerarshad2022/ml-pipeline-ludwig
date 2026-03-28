# ML Pipeline Improvements Analysis

## Current State vs. Target State

This document identifies gaps in the current implementation relative to the thesis objective:
> _"A configurable pipeline engine covering ML techniques including XAI within a single framework, configurable via YAML."_

---

## Identified Gaps

### 1. Missing Sklearn Pipeline Handlers ❌
**Problem**: `random-forest-anomaly.yaml`, `xgboost-anomaly.yaml`, and `logistic-regression-anomaly.yaml` reference step types `sklearn_train`, `sklearn_evaluate`, and `sklearn_visualize`, but these handlers **do not exist** in `pipelines/engine.py` or `pipelines/handlers/`. Running these configs via `run_pipeline.py` raises `ValueError: Unknown step type`.

**Impact**: High — three of four YAML pipelines are non-functional.

**Fix**: Implement `pipelines/handlers/sklearn_handler.py` with `SklearnTrainHandler`, `SklearnEvaluateHandler`, `SklearnVisualizeHandler`, and register them in the engine.

---

### 2. XAI Not Integrated Into YAML Pipeline ❌
**Problem**: SHAP and LIME explanations are only available via standalone scripts (`explain_random_forest.py`, `explain_xgboost.py`). There is no `xai` step type in the pipeline engine, so XAI cannot be triggered from a YAML config.

**Impact**: High — XAI is a core thesis requirement, but it's decoupled from the pipeline.

**Fix**: Add `SklearnXAIHandler` and register a `sklearn_xai` step type in the engine; update YAML configs to include an XAI step.

---

### 3. MLflow Not Used in Code ⚠️
**Problem**: `mlflow==2.10.2` is listed in `requirements.txt`, but MLflow is never imported or called anywhere in the codebase. Experiment metadata (run IDs, metrics, parameters) is not tracked.

**Impact**: Medium — experiment reproducibility and comparison are harder without tracking.

**Fix**: Integrate MLflow logging in `SklearnTrainHandler` and `SklearnEvaluateHandler` to log parameters, metrics, and artifacts per run.

---

### 4. Missing Runtime Dependencies in `requirements.txt` ❌
**Problem**: The code imports `scikit-learn`, `xgboost`, `shap`, `lime`, and `joblib`, but none of these appear in `requirements.txt`. A fresh `pip install -r requirements.txt` will fail at runtime.

**Impact**: High — the project cannot be deployed reliably.

**Fix**: Add explicit pinned versions for `scikit-learn`, `xgboost`, `shap`, `lime`, and `joblib`.

---

### 5. Engine Validates Ludwig Config For All Pipelines ❌
**Problem**: `LudwigPipelineEngine.validate_config()` raises a `ValueError` if `ludwig` config keys are absent, even for sklearn-only pipelines.

**Impact**: High — sklearn-only YAML configs cannot be executed without a dummy `ludwig:` block.

**Fix**: Make validation conditional: only enforce Ludwig config fields when a `ludwig_*` step is present in the pipeline.

---

### 6. Preprocessing Is Not Configurable via YAML ⚠️
**Problem**: `data_loader.py` loads raw data but performs no feature scaling, encoding, or train/test splitting. Scaling options (z-score, min-max) are defined in Ludwig's internal config only.

**Impact**: Medium — sklearn models see raw, unscaled features; there is no explicit train/test split handler.

**Fix**: Add optional `preprocessing` configuration in `DataLoaderHandler` to support feature scaling (`standard`, `minmax`, `none`) and explicit `test_size` train/test split, storing both splits in context.

---

### 7. No Testing Infrastructure ❌
**Problem**: There are zero test files. No `pytest` configuration, no unit tests, no integration tests.

**Impact**: Medium — regressions cannot be caught automatically; CI/CD has nothing to run.

**Fix**: Add a `tests/` directory with unit tests for `DataLoaderHandler`, `SklearnTrainHandler`, `SklearnEvaluateHandler`, and `LudwigPipelineEngine`.

---

## Improvements Implemented in This Branch

| # | Improvement | Files Changed |
|---|---|---|
| 1 | Sklearn pipeline handlers (train, evaluate, visualize) | `pipelines/handlers/sklearn_handler.py` (new) |
| 2 | XAI handler integrated into YAML pipeline | `pipelines/handlers/sklearn_handler.py`, YAML examples |
| 3 | MLflow experiment tracking in sklearn handlers | `pipelines/handlers/sklearn_handler.py` |
| 4 | Missing runtime dependencies added | `requirements.txt` |
| 5 | Engine validation made conditional | `pipelines/engine.py` |
| 6 | Configurable preprocessing (scaling, train/test split) | `pipelines/handlers/data_loader.py` |
| 7 | Unit tests added | `tests/` (new directory) |

---

## Remaining Future Work (Out of Scope for This Branch)

- **Hyperparameter tuning**: Grid search / Bayesian optimisation step type
- **Data drift detection**: Schema validation and statistical drift checks
- **Model serving**: REST API endpoint for online inference
- **Incremental / online retraining**: Partial-fit or continual learning support
- **Additional data sources**: SQL databases, S3, REST APIs
- **Cross-validation step**: k-fold evaluation as a first-class pipeline step
