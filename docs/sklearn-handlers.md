# Scikit-learn Handlers

**File:** `pipelines/handlers/sklearn_handler.py`

This module provides four handler classes that together implement a complete
train → evaluate → visualise → explain workflow for classical ML models.

| Class | YAML step type | Responsibility |
|---|---|---|
| `SklearnTrainHandler` | `sklearn_train` | Train a classifier, save the model, log to MLflow |
| `SklearnEvaluateHandler` | `sklearn_evaluate` | Compute metrics, save `metrics.txt`, log to MLflow |
| `SklearnVisualizeHandler` | `sklearn_visualize` | Save evaluation plots as PNG files |
| `SklearnXAIHandler` | `sklearn_xai` | Generate SHAP and LIME explanations |

There are also two internal helper functions:

| Function | Purpose |
|---|---|
| `_build_model(model_type, params)` | Instantiate a model from a type string and param dict |
| `_get_split_data(context)` | Extract `X_train`, `X_test`, `y_train`, `y_test` from the context |

---

## Supported Models

| YAML `type` | Python class | Library |
|---|---|---|
| `random_forest` | `RandomForestClassifier` | scikit-learn |
| `xgboost` | `XGBClassifier` | XGBoost |
| `logistic_regression` | `LogisticRegression` | scikit-learn |

Any other value raises `ValueError: Unsupported model_type`.

---

## `SklearnTrainHandler`

### What It Does

1. Reads `sklearn_model` config (type + hyperparameters + feature/target names).
2. Gets the dataset from `context["data"]`.
3. Uses pre-split data from context (`X_train`, `X_test`) if available; otherwise
   creates an 80/20 split.
4. Builds and trains the model with `model.fit(X_train, y_train)`.
5. Saves the model to disk using `joblib.dump()`.
6. Optionally logs parameters and training time to MLflow.

### Constructor

```python
SklearnTrainHandler(config: dict, context: dict, sklearn_config: dict)
```

| Parameter | Description |
|---|---|
| `config` | The `config:` block from the YAML step |
| `context` | Shared pipeline context |
| `sklearn_config` | The top-level `sklearn_model:` block from the YAML file |

### Key Context Outputs

After `execute()`:

| Context key | Type | Description |
|---|---|---|
| `context["sklearn_model"]` | `dict` | Contains model, model_type, model_path, feature_cols, target_col, output_directory |
| `context["X_train"]` | `DataFrame` | Training features (created if not already present) |
| `context["X_test"]` | `DataFrame` | Test features |
| `context["y_train"]` | `Series` | Training labels |
| `context["y_test"]` | `Series` | Test labels |
| `context["feature_cols"]` | `list` | List of feature column names |
| `context["target_col"]` | `str` | Name of the target/label column |

### YAML Configuration Reference

Top-level `sklearn_model:` section (not inside `steps:`):

```yaml
sklearn_model:
  type: random_forest               # Required: "random_forest" | "xgboost" | "logistic_regression"
  params:
    # Random Forest hyperparameters
    n_estimators: 100               # Number of trees
    max_depth: 10                   # Max tree depth (null = unlimited)
    min_samples_split: 5            # Min samples to split a node
    min_samples_leaf: 2             # Min samples at a leaf
    max_features: sqrt              # Features considered at each split
    random_state: 42                # Reproducibility seed
    n_jobs: -1                      # Use all CPU cores
    class_weight: balanced          # Handle class imbalance

    # XGBoost hyperparameters (use instead of above)
    # n_estimators: 100
    # max_depth: 6
    # learning_rate: 0.1
    # subsample: 0.8
    # colsample_bytree: 0.8
    # scale_pos_weight: 10          # pos_weight = n_negative / n_positive

    # Logistic Regression hyperparameters (use instead of above)
    # C: 1.0                        # Inverse regularisation strength
    # penalty: l2                   # l1, l2, or elasticnet
    # solver: lbfgs
    # max_iter: 1000

  input_features:
    - active_users                  # List of column names to use as features
  output_feature: is_anomaly        # Target column name
```

Step-level config (inside `steps:`):

```yaml
- name: train-sklearn
  type: sklearn_train
  config:
    output_directory: artifacts/random_forest_model  # Where to save the .pkl
    experiment_name: du-cell-rf-anomaly              # MLflow experiment name
```

### MLflow Integration

If MLflow is installed, `SklearnTrainHandler` automatically:
- Sets/creates an MLflow experiment with the given `experiment_name`.
- Logs all model hyperparameters as MLflow params.
- Logs `training_time_seconds` as an MLflow metric.

If MLflow fails (not installed, tracking server unreachable, etc.) the error
is silently caught and logged as a warning — the pipeline continues.

---

## `SklearnEvaluateHandler`

### What It Does

1. Reads the trained model from `context["sklearn_model"]`.
2. Reads test data from `context["X_test"]` and `context["y_test"]`.
3. Runs `model.predict()` and `model.predict_proba()`.
4. Computes a comprehensive set of metrics.
5. Saves a `metrics.txt` file.
6. Logs metrics to MLflow.

### Metrics Computed

| Metric | Description |
|---|---|
| `accuracy` | Fraction of correct predictions overall |
| `precision` | Of all predicted anomalies, how many were actually anomalies? |
| `recall` | Of all actual anomalies, how many did we detect? |
| `f1_score` | Harmonic mean of precision and recall |
| `roc_auc` | Area Under the ROC Curve (measures ranking quality) |
| `avg_precision` | Area Under the Precision-Recall Curve |

> **Why care about recall for anomaly detection?**  
> In anomaly detection, missing a real anomaly (false negative) is usually worse
> than a false alarm (false positive). High **recall** means we catch most anomalies.
> Accuracy alone is misleading because a model that always predicts "normal"
> achieves ~95% accuracy on this dataset!

### YAML Configuration Reference

```yaml
- name: evaluate
  type: sklearn_evaluate
  config:
    output_directory: artifacts/evaluation  # Where to save metrics.txt
    experiment_name: du-cell-rf-anomaly     # MLflow experiment name (optional)
```

### Context Requirements & Outputs

**Requires:** `context["sklearn_model"]`, `context["X_test"]`, `context["y_test"]`

**Sets:** `context["sklearn_evaluate"]` with keys: `metrics`, `classification_report`,
`y_pred`, `y_proba`, `output_directory`

---

## `SklearnVisualizeHandler`

### What It Does

Generates matplotlib/seaborn visualisation plots and saves them as PNG files.
Which plots are created depends on the `visualizations` list in the YAML config.

### Available Visualisations

| Name | What it shows | Notes |
|---|---|---|
| `confusion_matrix` | 2×2 table of TP/FP/TN/FN counts as a heatmap | Always available |
| `roc_curve` | False Positive Rate vs True Positive Rate at different thresholds | Needs `predict_proba` |
| `precision_recall_curve` | Precision vs Recall trade-off curve | Needs `predict_proba` |
| `feature_importance` | Bar chart of feature importances | Tree-based models only (RF, XGBoost) |
| `coefficients` | Bar chart of model coefficients | Linear models only (Logistic Regression) |
| `learning_curve` | Training vs validation AUC as training set size grows | Needs `X_train`, `y_train` in context |

### How to Read the Plots

**Confusion Matrix:**

```
                Predicted Normal  Predicted Anomaly
Actual Normal         2840              160        ← False Positives
Actual Anomaly         82               68         ← False Negatives
                                       ^^^
                              True Positives (anomalies caught!)
```

**ROC Curve:** A curve from (0,0) to (1,1). The closer to the top-left corner,
the better. The AUC (Area Under Curve) summarises this: 1.0 = perfect,
0.5 = random guessing.

**Precision-Recall Curve:** Important for imbalanced datasets. AP (Average
Precision) summarises the curve. Higher = better.

**Feature Importance:** Larger bars = that feature is used more for making
decisions. In this project, `active_users` is typically the most important feature.

### YAML Configuration Reference

```yaml
- name: visualize
  type: sklearn_visualize
  config:
    visualizations:
      - confusion_matrix          # Always recommended
      - roc_curve                 # Requires predict_proba
      - precision_recall_curve    # Requires predict_proba
      - feature_importance        # Tree models only
      - coefficients              # Linear models only
      - learning_curve            # Requires X_train/y_train
    output_directory: artifacts/visualizations
    n_jobs: -1                    # For learning_curve computation (optional)
```

### Context Requirements

**Requires:** `context["sklearn_model"]`, `context["sklearn_evaluate"]`,
`context["X_test"]`, `context["y_test"]`

**Optional:** `context["X_train"]`, `context["y_train"]` (needed for `learning_curve`)

---

## `SklearnXAIHandler`

### What It Does

Generates Explainable AI (XAI) visualisations using **SHAP** and/or **LIME**.
See [Explainable AI](xai.md) for a detailed explanation of what these plots mean.

### SHAP vs LIME

| Aspect | SHAP | LIME |
|---|---|---|
| Scope | Global + per-instance | Per-instance only |
| Speed | Fast for tree models (TreeExplainer); slow for linear/kernel | Medium |
| Explainer type | Tree/Kernel depending on model | Kernel (linear proxy) |
| Output | Summary beeswarm, bar chart | Feature weights bar chart |

### SHAP Explainer Selection

The handler automatically picks the right SHAP explainer:

| Model type | Explainer used | Why |
|---|---|---|
| `random_forest` | `shap.TreeExplainer` | Efficient tree-specific algorithm |
| `xgboost` | `shap.TreeExplainer` | XGBoost is also tree-based |
| `logistic_regression` | `shap.KernelExplainer` | No tree structure; uses SHAP's model-agnostic kernel method |

> **Warning:** `KernelExplainer` (for Logistic Regression) is significantly
> slower than `TreeExplainer`. Use small `kernel_background_samples` and
> `kernel_explain_samples` to keep it manageable.

### YAML Configuration Reference

```yaml
- name: xai
  type: sklearn_xai
  config:
    methods:
      - shap                    # SHAP analysis
      - lime                    # LIME analysis
    n_lime: 3                   # Number of test instances to explain with LIME
    output_directory: artifacts/random_forest_model/xai

    # Advanced SHAP settings (for logistic_regression only)
    kernel_background_samples: 50   # Background samples for KernelExplainer
    kernel_explain_samples: 50      # Instances to explain with KernelExplainer
```

### Context Requirements

**Requires:** `context["sklearn_model"]`, `context["X_train"]`, `context["X_test"]`

**Optional:** `context["y_test"]` (used by LIME for display)

### Output Files

| File | Description |
|---|---|
| `shap_summary.png` | Beeswarm plot: each dot is one sample; x-axis is SHAP value |
| `shap_importance.png` | Bar chart: mean \|SHAP value\| per feature |
| `lime_instance_0.png` | LIME explanation for test instance 0 |
| `lime_instance_1.png` | LIME explanation for test instance 1 |
| `lime_instance_2.png` | LIME explanation for test instance 2 |

---

## Testing the Sklearn Handlers

Unit tests live in `tests/test_sklearn_handler.py`. They cover:
- `_build_model` for all three model types
- `SklearnTrainHandler.execute()` end-to-end with a small DataFrame
- Evaluation metrics are computed and saved

```bash
pytest tests/test_sklearn_handler.py -v
```
