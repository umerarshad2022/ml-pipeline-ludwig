# YAML Configuration Reference

Every pipeline is defined by a single YAML file. This document describes every
supported key with its type, default value, and a usage example.

> **What is YAML?**  
> YAML (YAML Ain't Markup Language) is a human-readable data format. Indentation
> (spaces, not tabs) defines structure. It is commonly used for configuration
> files because it is easier to read than JSON.

---

## Top-Level Structure

```yaml
name: <string>           # Required
description: <string>    # Required
metadata: <dict>         # Required (can be empty: {})
data: <dict>             # Required (can be empty: {})

# One or both of these blocks, depending on which step types you use:
ludwig: <dict>           # Required when using ludwig_* steps
sklearn_model: <dict>    # Required when using sklearn_* steps

# Optional sections:
experiment: <dict>       # MLflow experiment tracking settings
artifacts: <dict>        # Artifact storage settings
deployment: <dict>       # Deployment settings

steps: <list>            # Required — ordered list of pipeline steps
```

---

## `name`

**Type:** string  
**Required:** Yes

Human-readable name for the pipeline. Used in log output and MLflow experiment names.

```yaml
name: DU Cell Anomaly Detection with XGBoost
```

---

## `description`

**Type:** string  
**Required:** Yes (can be empty string)

Free-text description of what this pipeline does.

```yaml
description: High-performance XGBoost classifier for anomaly detection in DU cell metrics
```

---

## `metadata`

**Type:** dict  
**Required:** Yes (can be empty: `{}`)

Arbitrary key-value metadata. Not used by the engine — for documentation purposes only.

```yaml
metadata:
  author: AIML Team
  version: 1.0.0
  created: 2026-02-07
  model_type: xgboost
```

---

## `data`

**Type:** dict  
**Required:** Yes (can be empty: `{}`)

Top-level data configuration. Currently only used to document preprocessing intent;
actual preprocessing is configured per `data_loader` step.

```yaml
data:
  preprocessing:
    drop_columns:
      - Cell_ID
      - gNB_ID
      - Timestamp
```

> **Note:** The `data.preprocessing` block here is for documentation/reference
> only. The `data_loader` step has its own `preprocessing:` block that actually
> runs the preprocessing. Both can exist and they do not conflict.

---

## `ludwig`

**Type:** dict  
**Required:** Only when the pipeline contains `ludwig_*` step types

Defines the Ludwig model architecture. The engine validates this block is present
and contains `input_features` and `output_features` before running any Ludwig step.

### `ludwig.input_features`

List of feature definitions. Each entry must have:

| Key | Type | Description |
|---|---|---|
| `name` | string | Column name in the DataFrame |
| `type` | string | Ludwig feature type (`numerical`, `binary`, `category`, `text`, etc.) |
| `preprocessing` | dict | Optional feature-specific preprocessing |

```yaml
ludwig:
  input_features:
    - name: active_users
      type: numerical
      preprocessing:
        normalization: zscore    # or "minmax" or null
```

### `ludwig.output_features`

List of target/label definitions. For binary classification:

```yaml
  output_features:
    - name: is_anomaly
      type: binary
      loss:
        type: binary_weighted_cross_entropy
        class_weights: [1, 10]   # [normal_weight, anomaly_weight]
```

### `ludwig.trainer`

Controls the training loop:

| Key | Type | Default | Description |
|---|---|---|---|
| `epochs` | int | 100 | Maximum training passes |
| `batch_size` | int | 32 | Samples per gradient update |
| `learning_rate` | float | 0.001 | Initial Adam learning rate |
| `optimizer.type` | string | `adam` | Optimiser name |
| `early_stop` | int | 10 | Stop after N epochs without improvement |
| `validation_metric` | string | `accuracy` | Metric to monitor for early stopping |
| `validation_field` | string | — | Feature name to use for validation |
| `decay` | bool | false | Enable learning rate decay |
| `decay_steps` | int | — | Steps between decay |
| `decay_rate` | float | — | Multiplicative decay factor |

### `ludwig.defaults`

Fallback preprocessing for each feature type:

```yaml
  defaults:
    numerical:
      preprocessing:
        missing_value_strategy: fill_with_mean
        normalization: zscore
    binary:
      preprocessing:
        fallback_label: 0
```

---

## `sklearn_model`

**Type:** dict  
**Required:** Only when the pipeline contains `sklearn_*` step types

### `sklearn_model.type`

**Required.** One of: `random_forest`, `xgboost`, `logistic_regression`.

### `sklearn_model.params`

Dict of hyperparameters passed directly to the scikit-learn / XGBoost constructor.
See the [sklearn-handlers](sklearn-handlers.md) document for full parameter lists.

### `sklearn_model.input_features`

List of column names to use as features. If empty or missing, all numeric columns
(except the target) are used.

### `sklearn_model.output_feature`

Name of the label column. Defaults to `"is_anomaly"` if not set.

```yaml
sklearn_model:
  type: random_forest
  params:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    min_samples_leaf: 2
    max_features: sqrt
    random_state: 42
    n_jobs: -1
    class_weight: balanced
  input_features:
    - active_users
  output_feature: is_anomaly
```

---

## `steps`

**Type:** list of step definitions  
**Required:** Yes

Each item in the list is a **step definition** with this structure:

```yaml
steps:
  - name: <string>       # Required — unique name used as context key
    type: <string>       # Required — determines which handler runs
    config: <dict>       # Optional — handler-specific configuration
```

### Step Types Reference

#### `data_loader`

```yaml
- name: load-data
  type: data_loader
  config:
    source:
      type: csv                     # "csv" | "parquet" | "json"
      path: data/sample_data.csv    # Relative or absolute path
    preprocessing:                  # Optional
      drop_columns: [Cell_ID, ...]  # Columns to remove
      scaling: none                 # "standard" | "minmax" | "none"
      target_col: is_anomaly        # Exclude from scaling; used for split
      test_size: 0.2                # Train/test split ratio (0-1)
      random_state: 42              # Reproducibility seed
```

#### `unsupervised_labeling`

```yaml
- name: auto-label-anomalies
  type: unsupervised_labeling
  config:
    method: isolation_forest        # Only supported value
    feature: active_users           # Feature column for detection
    output_column: is_anomaly       # New label column name
    contamination: 0.10             # Expected anomaly fraction (0-0.5)
```

#### `sklearn_train`

```yaml
- name: train-sklearn
  type: sklearn_train
  config:
    output_directory: artifacts/random_forest_model
    experiment_name: my-experiment   # MLflow experiment name
```

> Reads model type and hyperparameters from top-level `sklearn_model:` block.

#### `sklearn_evaluate`

```yaml
- name: evaluate
  type: sklearn_evaluate
  config:
    output_directory: artifacts/evaluation
    experiment_name: my-experiment
```

#### `sklearn_visualize`

```yaml
- name: visualize
  type: sklearn_visualize
  config:
    visualizations:
      - confusion_matrix
      - roc_curve
      - precision_recall_curve
      - feature_importance         # Tree models only
      - coefficients               # Linear models only
      - learning_curve             # Slower; requires X_train/y_train
    output_directory: artifacts/visualizations
    n_jobs: -1                     # For learning_curve parallelism
```

#### `sklearn_xai`

```yaml
- name: xai
  type: sklearn_xai
  config:
    methods:
      - shap
      - lime
    n_lime: 3                      # Number of LIME instances
    output_directory: artifacts/model/xai
    kernel_background_samples: 50  # KernelExplainer only (logistic regression)
    kernel_explain_samples: 50     # KernelExplainer only
```

#### `ludwig_train`

```yaml
- name: train-ludwig
  type: ludwig_train
  config:
    output_directory: artifacts/ludwig_model
    experiment_name: du-cell-anomaly
```

> Reads model architecture from top-level `ludwig:` block.

#### `ludwig_evaluate`

```yaml
- name: evaluate
  type: ludwig_evaluate
  config:
    output_directory: artifacts/evaluation
    # model_path: artifacts/...   # Only if running without prior train step
```

#### `ludwig_visualize`

```yaml
- name: visualize
  type: ludwig_visualize
  config:
    visualizations:
      - learning_curves
      - confusion_matrix
      - roc_curve
    output_directory: artifacts/visualizations
```

---

## `experiment` (optional)

Reserved for future MLflow configuration at the pipeline level. Currently not
used by the engine (MLflow is configured per-step).

```yaml
experiment:
  tracking_uri: http://localhost:5000
  name: my-experiment
```

---

## `artifacts` (optional)

Reserved for future artifact storage configuration (e.g. AWS S3).

```yaml
artifacts:
  store: s3
  bucket: my-ml-bucket
  prefix: pipelines/du-cell
```

---

## `deployment` (optional)

Reserved for future model deployment configuration.

```yaml
deployment:
  platform: docker
  port: 8080
```

---

## Complete Example

This is the complete `random-forest-anomaly.yaml` annotated:

```yaml
# Human-readable pipeline identity
name: DU Cell Anomaly Detection with Random Forest
description: Lightweight Random Forest classifier for anomaly detection in DU cell metrics
metadata:
  author: AIML Team
  version: 1.0.0
  created: 2026-02-07
  model_type: random_forest

# Top-level data config (informational — actual preprocessing is per-step)
data:
  preprocessing:
    drop_columns:
      - Cell_ID
      - gNB_ID
      - Timestamp

# sklearn model configuration — read by SklearnTrainHandler
sklearn_model:
  type: random_forest
  params:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    min_samples_leaf: 2
    max_features: sqrt
    random_state: 42
    n_jobs: -1
    class_weight: balanced   # Gives equal weight to minority class
  input_features:
    - active_users
  output_feature: is_anomaly

# Steps run in order
steps:
  - name: load-data             # ← context["load-data"] after this runs
    type: data_loader
    config:
      source:
        type: csv
        path: data/sample_data.csv

  - name: auto-label-anomalies  # ← adds is_anomaly column to context["data"]
    type: unsupervised_labeling
    config:
      method: isolation_forest
      contamination: 0.10
      feature: active_users
      output_column: is_anomaly

  - name: train-sklearn         # ← trains RF, saves .pkl
    type: sklearn_train
    config:
      output_directory: artifacts/random_forest_model
      experiment_name: du-cell-rf-anomaly

  - name: evaluate              # ← saves metrics.txt
    type: sklearn_evaluate
    config:
      output_directory: artifacts/evaluation

  - name: visualize             # ← saves PNG plots
    type: sklearn_visualize
    config:
      visualizations:
        - feature_importance
        - confusion_matrix
        - roc_curve
        - precision_recall_curve
      output_directory: artifacts/visualizations

  - name: xai                   # ← saves SHAP + LIME plots
    type: sklearn_xai
    config:
      methods:
        - shap
        - lime
      n_lime: 3
      output_directory: artifacts/random_forest_model/xai
```
