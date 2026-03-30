# Data Loading & Preprocessing

**File:** `pipelines/handlers/data_loader.py`

This module contains two handler classes:

| Class | YAML step type | Responsibility |
|---|---|---|
| `DataLoaderHandler` | `data_loader` | Load a tabular file (CSV/Parquet/JSON), optionally apply preprocessing, optionally split into train/test sets |
| `UnsupervisedLabelingHandler` | `unsupervised_labeling` | Use Isolation Forest to automatically generate binary anomaly labels |

---

## `DataLoaderHandler`

### What It Does

1. Reads a tabular dataset from a file (CSV, Parquet, or JSON) into a
   [pandas DataFrame](https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe).
2. Optionally drops columns you don't need.
3. Optionally scales numeric features (StandardScaler or MinMaxScaler).
4. Optionally performs a stratified train/test split.
5. Stores the DataFrame (and split arrays) in the shared pipeline context.

> **What is a DataFrame?**  
> Think of it as a spreadsheet in memory: rows are samples/observations,
> columns are features/variables. pandas makes it easy to filter, transform,
> and compute statistics on tabular data.

---

### Constructor

```python
DataLoaderHandler(config: dict, context: dict)
```

| Parameter | Description |
|---|---|
| `config` | The `config:` block from the YAML step definition |
| `context` | The shared pipeline context dictionary |

---

### `execute()` — Step-by-Step

```
1. Read config["source"]["type"]  → "csv" | "parquet" | "json"
2. Read config["source"]["path"]  → file path string
3. Load the file:
      csv     → pd.read_csv(path)
      parquet → pd.read_parquet(path)
      json    → pd.read_json(path)
4. Log shape and column names.
5. If config["preprocessing"] is set → call _apply_preprocessing(df, cfg)
6. Store df in context["data"]
7. If preprocessing["test_size"] is set → call _split(df, cfg)
8. Return {"data": df, "shape": df.shape, ...}
```

**Raises:**
- `ValueError` if `source.path` is missing.
- `ValueError` if `source.type` is not one of `csv`, `parquet`, `json`.

---

### `_apply_preprocessing(df, cfg)` — Internal

Applies column dropping and feature scaling in place.

#### Column Dropping

```yaml
preprocessing:
  drop_columns:
    - Cell_ID
    - gNB_ID
    - Timestamp
```

Drops the listed columns if they exist. Columns that do not exist are silently
ignored (no error).

#### Feature Scaling

```yaml
preprocessing:
  scaling: standard       # or "minmax" or "none"
  target_col: is_anomaly  # column to EXCLUDE from scaling
```

| `scaling` value | Behaviour | When to use |
|---|---|---|
| `standard` | `(x - mean) / std` — centres each feature at 0 with unit variance | When you need normally-distributed features (e.g. Logistic Regression, SVM) |
| `minmax` | `(x - min) / (max - min)` — scales each feature to [0, 1] | When you need bounded values (e.g. neural networks) |
| `none` (default) | No scaling | Tree-based models (Random Forest, XGBoost) — they are scale-invariant |

The fitted `scaler` object is stored in `context["scaler"]` so it can be reused
for inference on new data.

> **Why exclude `target_col` from scaling?**  
> The target column (the label we are trying to predict) should never be scaled —
> it would change the meaning of the labels.

---

### `_split(df, cfg)` — Internal

Performs a **stratified** train/test split using scikit-learn's
`train_test_split`.

```yaml
preprocessing:
  test_size: 0.2          # 20% goes to test, 80% to train
  target_col: is_anomaly  # used for stratification
  random_state: 42        # reproducible split
```

> **What is stratified splitting?**  
> It ensures the class ratio (e.g. 95% normal / 5% anomaly) is the same in
> both the train and test sets. Without stratification, you might end up with a
> test set that has no anomalies at all.

If stratification fails (e.g. a class has too few samples), it falls back to
a non-stratified split with a warning.

Stores in context:

| Key | Type | Description |
|---|---|---|
| `context["X_train"]` | `DataFrame` | Training feature matrix |
| `context["X_test"]` | `DataFrame` | Test feature matrix |
| `context["y_train"]` | `Series` | Training labels |
| `context["y_test"]` | `Series` | Test labels |

---

### YAML Configuration Reference

```yaml
- name: load-data
  type: data_loader
  config:
    source:
      type: csv              # Required: "csv" | "parquet" | "json"
      path: data/sample_data.csv  # Required: relative or absolute path

    preprocessing:           # Optional block
      drop_columns:          # Optional list of columns to remove
        - Cell_ID
        - gNB_ID
        - Timestamp

      scaling: none          # Optional: "standard" | "minmax" | "none" (default)
      target_col: is_anomaly # Required if scaling != "none" or test_size is set

      test_size: 0.2         # Optional: float (0-1) — activates train/test split
      random_state: 42       # Optional: integer for reproducibility (default: 42)
```

---

### Context Outputs

After `execute()`, the following keys are available in `context`:

| Key | Set when | Type | Description |
|---|---|---|---|
| `context["data"]` | Always | `DataFrame` | The full loaded (and preprocessed) dataset |
| `context["scaler"]` | scaling ≠ "none" | Scaler object | Fitted scaler for reuse at inference |
| `context["X_train"]` | test_size is set | `DataFrame` | Training features |
| `context["X_test"]` | test_size is set | `DataFrame` | Test features |
| `context["y_train"]` | test_size + target_col set | `Series` | Training labels |
| `context["y_test"]` | test_size + target_col set | `Series` | Test labels |

---

## `UnsupervisedLabelingHandler`

### What It Does

Many real-world datasets do not come with ground-truth anomaly labels — you have
raw sensor readings but nobody has manually marked which ones are anomalies.
This handler solves that by running **unsupervised anomaly detection** to
*create* labels automatically.

The only supported method is **Isolation Forest** from scikit-learn.

> **What is Isolation Forest?**  
> The algorithm builds random decision trees. Anomalous points are easier to
> "isolate" (they end up in short branches) because they are rare and different.
> The algorithm outputs -1 for anomalies and +1 for normal points. We convert
> this to 1 and 0 respectively.

### Constructor

```python
UnsupervisedLabelingHandler(config: dict, context: dict)
```

### `execute()` — Step-by-Step

```
1. Read context["data"]  → DataFrame (must have been loaded by data_loader step)
2. Read config["method"]         → "isolation_forest" (only supported value)
3. Read config["feature"]        → name of the column to use for detection
4. Read config["output_column"]  → name for the new label column (default: "is_anomaly")
5. Read config["contamination"]  → expected fraction of anomalies (default: 0.10)
6. Train IsolationForest on the selected feature
7. Predict: -1 = anomaly, +1 = normal
8. Convert to 0/1 and add as new column to the DataFrame
9. Update context["data"] with the labelled DataFrame
10. Return {"labeled_data": df, "n_anomalies": int, "anomaly_rate": float}
```

**Raises:**
- `ValueError` if `context["data"]` is not set (data_loader must run first).
- `ValueError` if `method` is not `"isolation_forest"`.

---

### `contamination` Parameter

The `contamination` parameter tells Isolation Forest what fraction of the
dataset to label as anomalies. Choose it based on your domain knowledge:

| Scenario | Suggested value |
|---|---|
| "Something is clearly wrong 10% of the time" | `0.10` |
| "Very rare failures, maybe 1-2%" | `0.01` to `0.05` |
| "High noise environment, ~20% anomalies" | `0.20` |

> The default is `0.10` (10%). Increasing it labels more points as anomalies;
> decreasing it labels fewer.

---

### YAML Configuration Reference

```yaml
- name: auto-label-anomalies
  type: unsupervised_labeling
  config:
    method: isolation_forest    # Required (only supported value)
    feature: active_users       # Required: column used for anomaly detection
    output_column: is_anomaly   # Optional: name of the new label column (default: "is_anomaly")
    contamination: 0.10         # Optional: expected anomaly fraction (default: 0.10)
```

---

### Context Requirements & Outputs

**Requires:** `context["data"]` (set by `data_loader` step)

**Modifies:** `context["data"]` — adds the `output_column` to the DataFrame

**Returns:**

```python
{
    "labeled_data": df,           # DataFrame with the new label column
    "n_anomalies": 1500,          # integer count of anomalous rows
    "anomaly_rate": 0.10          # float fraction of anomalous rows
}
```

---

## Testing Data Loading

Unit tests live in `tests/test_data_loader.py`. They cover:

- CSV loading
- Parquet loading (skipped if `pyarrow` not installed)
- Missing path raises `ValueError`
- Unknown source type raises `ValueError`
- Column dropping
- StandardScaler applied correctly (mean ≈ 0)
- MinMaxScaler applied correctly (values in [0, 1])
- Train/test split produces correct sizes
- Isolation Forest labelling produces binary {0, 1} labels
- Empty context raises `ValueError` in `UnsupervisedLabelingHandler`
- Unknown labelling method raises `ValueError`

```bash
pytest tests/test_data_loader.py -v
```
