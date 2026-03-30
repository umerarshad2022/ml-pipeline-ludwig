# Architecture Overview

This document explains how all the pieces of the project fit together.
It is written for a developer who wants a mental model of the system before
diving into individual modules.

---

## 30-Second Summary

```
User provides a YAML file
        │
        ▼
  run_pipeline.py  (CLI)
        │
        ▼
  LudwigPipelineEngine  (pipelines/engine.py)
   ├─ load_config()    → parses YAML into a PipelineConfig dataclass
   ├─ validate_config() → checks required fields are present
   └─ execute()        → loops through each step and calls the right Handler
              │
              ├─► DataLoaderHandler        → loads CSV/Parquet/JSON into a DataFrame
              ├─► UnsupervisedLabelingHandler → Isolation Forest pseudo-labels
              ├─► SklearnTrainHandler      → trains RF / XGBoost / LogReg
              ├─► SklearnEvaluateHandler   → computes metrics
              ├─► SklearnVisualizeHandler  → saves PNG plots
              ├─► SklearnXAIHandler        → SHAP + LIME explanations
              ├─► LudwigTrainHandler       → trains a Ludwig neural network
              ├─► LudwigEvaluateHandler    → evaluates Ludwig model
              └─► LudwigVisualizeHandler   → Ludwig learning curves, ROC, etc.
```

All handlers share a **context dictionary** to pass data between steps.

---

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. INPUT                                                        │
│     data/sample_data.csv  +  examples/*.yaml                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. DATA LOADING  (DataLoaderHandler)                           │
│     • Reads CSV/Parquet/JSON into a pandas DataFrame            │
│     • Optional: drop columns, apply StandardScaler/MinMaxScaler │
│     • Optional: stratified train/test split                     │
│     • Stores df in context["data"]                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. UNSUPERVISED LABELLING  (UnsupervisedLabelingHandler)       │
│     • Trains sklearn IsolationForest on a numeric feature       │
│     • Converts -1/+1 predictions to 0/1 binary labels          │
│     • Adds "is_anomaly" column to context["data"]              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  4a. SKLEARN TRAINING  (SklearnTrainHandler)                    │
│     • Builds model: RandomForest / XGBClassifier / LogReg       │
│     • Splits data 80/20 (or uses pre-split from step 2)        │
│     • Calls model.fit(X_train, y_train)                        │
│     • Saves model to artifacts/<type>_model/<type>_model.pkl   │
│     • Logs params to MLflow                                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐   ┌────────────────────────────────────────┐
│ 4b. LUDWIG       │   │  5. EVALUATION  (SklearnEvaluateHandler│
│  TRAINING        │   │     or LudwigEvaluateHandler)          │
│  (LudwigTrain    │   │  • Predicts on X_test                  │
│   Handler)       │   │  • Computes accuracy, F1, ROC-AUC, etc.│
│  • Configures    │   │  • Saves metrics.txt                   │
│    Ludwig model  │   │  • Logs metrics to MLflow              │
│  • Runs training │   └────────────────┬───────────────────────┘
│    loop          │                    │
└──────────────────┘                    ▼
                         ┌──────────────────────────────────────┐
                         │  6. VISUALISATION                    │
                         │  (SklearnVisualizeHandler /          │
                         │   LudwigVisualizeHandler)            │
                         │  • Confusion matrix PNG              │
                         │  • ROC curve PNG                     │
                         │  • Precision-Recall curve PNG        │
                         │  • Feature importance PNG            │
                         │  • Learning curve PNG                │
                         └────────────────┬─────────────────────┘
                                          │
                                          ▼
                         ┌──────────────────────────────────────┐
                         │  7. EXPLAINABLE AI  (SklearnXAI      │
                         │      Handler)                        │
                         │  • SHAP summary, waterfall, force,   │
                         │    dependence plots                  │
                         │  • LIME per-instance explanations    │
                         │  • All saved as PNG to artifacts/    │
                         └──────────────────────────────────────┘
```

---

## Component Map

### `pipelines/` — the Core Package

| File | Responsibility |
|---|---|
| `engine.py` | Orchestrates everything: reads YAML, validates config, loops over steps, dispatches to handlers |
| `handlers/data_loader.py` | Steps `data_loader` and `unsupervised_labeling` |
| `handlers/sklearn_handler.py` | Steps `sklearn_train`, `sklearn_evaluate`, `sklearn_visualize`, `sklearn_xai` |
| `handlers/ludwig_handler.py` | Steps `ludwig_train`, `ludwig_evaluate`, `ludwig_predict` |
| `handlers/ludwig_visualize.py` | Step `ludwig_visualize` |

### Root-Level Scripts

| File | Responsibility |
|---|---|
| `run_pipeline.py` | CLI: parses arguments and calls `LudwigPipelineEngine` |
| `train_random_forest.py` | Self-contained Random Forest training (no YAML needed) |
| `train_xgboost.py` | Self-contained XGBoost training (no YAML needed) |
| `explain_random_forest.py` | SHAP + LIME for a saved Random Forest `.pkl` |
| `explain_xgboost.py` | SHAP + LIME for a saved XGBoost `.pkl` |

### Configuration

| File | Purpose |
|---|---|
| `examples/*.yaml` | Declarative pipeline configurations |
| `requirements.txt` | Python dependency list with pinned versions |
| `Dockerfile` | Containerises the project |
| `docker-compose.yml` | Defines container services |

---

## The Context Dictionary

The **context** is a plain Python `dict` (`{}`) that the engine creates before
executing any steps. Each handler receives a reference to this same dictionary.

- Handlers **read** from `context` to get outputs from earlier steps.
- Handlers **write** to `context` to make their outputs available to later steps.

Example context after a full sklearn pipeline run:

```python
{
    "data":          <pandas DataFrame — 15000 rows>,
    "scaler":        <StandardScaler object, if scaling was configured>,
    "X_train":       <DataFrame — 12000 rows>,
    "X_test":        <DataFrame — 3000 rows>,
    "y_train":       <Series — 12000 labels>,
    "y_test":        <Series — 3000 labels>,
    "feature_cols":  ["active_users"],
    "target_col":    "is_anomaly",
    "sklearn_model": {
        "model":           <RandomForestClassifier>,
        "model_type":      "random_forest",
        "model_path":      "artifacts/random_forest_model/random_forest_model.pkl",
        "feature_cols":    ["active_users"],
        "target_col":      "is_anomaly",
        "output_directory":"artifacts/random_forest_model",
    },
    "sklearn_evaluate": {
        "metrics":    {"accuracy": 0.70, "f1_score": 0.13, ...},
        "y_pred":     <numpy array>,
        "y_proba":    <numpy array>,
        ...
    },
    # Each step's name also maps to its full result dict:
    "load-data":            { "data": ..., "shape": ... },
    "auto-label-anomalies": { "labeled_data": ..., "n_anomalies": 1500, ... },
    "train-sklearn":        { "model": ..., "model_path": ..., ... },
    ...
}
```

> **Important:** The engine stores the result of every step under the step's
> `name` key (e.g. `context["train-sklearn"]`). The `sklearn_model` and
> `sklearn_evaluate` sub-keys are also written for convenient cross-handler access.

---

## Handler Lifecycle

Every handler follows the same simple interface:

```python
class SomeHandler:
    def __init__(self, config: dict, context: dict, *extra_args):
        # Store references — no heavy work here
        self.config = config      # step-level YAML config
        self.context = context    # shared pipeline context
        ...

    def execute(self) -> dict:
        # 1. Read inputs from self.context
        # 2. Do the work
        # 3. Write outputs back to self.context
        # 4. Return a result dict (also stored in context by the engine)
        ...
```

This makes handlers:
- **Testable in isolation** — inject a mock context dict.
- **Easy to extend** — add a new handler class and one `elif` branch in `engine.py`.
- **Stateless across runs** — the engine creates a fresh context for every `execute()` call.

---

## Dependency Graph Between Steps

```
data_loader
    │
    ▼
unsupervised_labeling (needs context["data"])
    │
    ├──► sklearn_train  (needs context["data"] or context["X_train"])
    │         │
    │         ▼
    │    sklearn_evaluate  (needs context["sklearn_model"], context["X_test"])
    │         │
    │         ▼
    │    sklearn_visualize (needs context["sklearn_model"], context["sklearn_evaluate"])
    │         │
    │         ▼
    │    sklearn_xai       (needs context["sklearn_model"], context["X_train"])
    │
    └──► ludwig_train  (needs context["data"])
              │
              ▼
         ludwig_evaluate  (needs context["train-ludwig"]["model"])
              │
              ▼
         ludwig_visualize (needs context["train-ludwig"]["output_directory"])
```

The YAML `steps` list defines execution order. The engine runs steps
**sequentially** — there is no parallel execution.

---

## Technology Stack

```
┌─────────────────────────────────────────────────────┐
│  User-facing layer                                  │
│    run_pipeline.py  (argparse CLI)                  │
├─────────────────────────────────────────────────────┤
│  Orchestration layer                                │
│    LudwigPipelineEngine  (pipelines/engine.py)      │
│    PipelineConfig  (dataclass)                      │
├─────────────────────────────────────────────────────┤
│  Handler layer                                      │
│    DataLoaderHandler    UnsupervisedLabelingHandler │
│    SklearnTrainHandler  SklearnEvaluateHandler      │
│    SklearnVisualizeHandler  SklearnXAIHandler        │
│    LudwigTrainHandler  LudwigEvaluateHandler        │
│    LudwigVisualizeHandler                           │
├─────────────────────────────────────────────────────┤
│  Third-party ML libraries                           │
│    scikit-learn  XGBoost  Ludwig  SHAP  LIME        │
├─────────────────────────────────────────────────────┤
│  Infrastructure                                     │
│    pandas  numpy  matplotlib  seaborn  joblib       │
│    MLflow (experiment tracking)  pyyaml             │
└─────────────────────────────────────────────────────┘
```
