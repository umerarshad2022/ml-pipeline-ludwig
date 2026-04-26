# Ludwig Handlers

**Files:**
- `pipelines/handlers/ludwig_handler.py` — training, evaluation, prediction
- `pipelines/handlers/ludwig_visualize.py` — visualisations

These handlers integrate [Ludwig AI](https://ludwig.ai/), an open-source
declarative deep-learning framework. Ludwig lets you train neural networks by
specifying a configuration dict — no custom training loops, no PyTorch/TensorFlow
boilerplate required.

| Class | YAML step type | Responsibility |
|---|---|---|
| `LudwigTrainHandler` | `ludwig_train` | Train a Ludwig neural network |
| `LudwigEvaluateHandler` | `ludwig_evaluate` | Evaluate the model and log metrics |
| `LudwigPredictHandler` | *(not in engine yet)* | Make predictions with a saved model |
| `LudwigVisualizeHandler` | `ludwig_visualize` | Generate learning-curve, confusion-matrix, ROC plots |

---

## Prerequisites

Ludwig has heavy dependencies (PyTorch, Ray, etc.). Install with:

```bash
pip install ludwig==0.8.5
pip install --force-reinstall marshmallow-dataclass==8.5.4
pip install "pydantic<2.0"
```

> **Note:** Ludwig steps will raise `ImportError` with a helpful message if
> Ludwig is not installed. This means you can run sklearn pipelines without
> installing Ludwig.

---

## `LudwigTrainHandler`

### What It Does

1. Reads the dataset from `context["data"]`.
2. Constructs a Ludwig configuration dict from the pipeline's `ludwig:` YAML block.
3. Creates a `LudwigModel` instance and calls `model.train(dataset=df)`.
4. Saves the model, training statistics, and preprocessed data to disk.
5. Stores model reference and output directory in the context.

### Constructor

```python
LudwigTrainHandler(config: dict, context: dict, ludwig_config: dict)
```

| Parameter | Description |
|---|---|
| `config` | Step-level `config:` block from the YAML |
| `context` | Shared pipeline context |
| `ludwig_config` | The top-level `ludwig:` block from the YAML file |

### `_prepare_ludwig_config()` — Internal

Assembles the Ludwig configuration dict by pulling:
- `input_features` — list of input feature definitions
- `output_features` — list of output feature definitions
- `trainer` — training loop configuration (optional)
- `defaults` — default preprocessing per feature type (optional)

### YAML Configuration Reference

Top-level `ludwig:` section:

```yaml
ludwig:
  input_features:
    - name: active_users          # Column name
      type: numerical             # Feature type: numerical, binary, category, text, etc.
      preprocessing:
        normalization: zscore     # Normalise values (zscore, minmax, or null)

  output_features:
    - name: is_anomaly            # Column name (the label)
      type: binary                # Task type: binary classification
      loss:
        type: binary_weighted_cross_entropy  # Use weighted loss for class imbalance
        class_weights: [1, 10]    # Weight class 0 = 1, class 1 (anomaly) = 10

  trainer:
    epochs: 100                   # Number of training passes over the data
    batch_size: 32                # Samples per gradient update
    learning_rate: 0.001          # Step size for Adam optimiser
    optimizer:
      type: adam
    decay: true                   # Enable learning rate decay
    decay_steps: 10000
    decay_rate: 0.96
    early_stop: 10                # Stop if validation metric doesn't improve for 10 epochs
    validation_metric: accuracy
    validation_field: is_anomaly
    learning_rate_scheduler:
      type: reduce_on_plateau
      factor: 0.5                 # Multiply LR by this on plateau
      patience: 5                 # Wait 5 epochs before reducing

  defaults:
    numerical:
      preprocessing:
        missing_value_strategy: fill_with_mean
        normalization: zscore
    binary:
      preprocessing:
        fallback_label: 0
```

Step-level config:

```yaml
- name: train-ludwig
  type: ludwig_train
  config:
    output_directory: artifacts/ludwig_model   # Where Ludwig saves the model
    experiment_name: du-cell-anomaly           # Subdirectory within output_directory
```

### Context Requirements & Outputs

**Requires:** `context["data"]` — the full dataset DataFrame

**Returns (and stores in context under the step name):**

```python
{
    "model": <LudwigModel>,              # In-memory model object
    "train_stats": <dict>,               # Training metrics per epoch
    "output_directory": "artifacts/...", # Path to saved model files
    "preprocessed_data": <tuple>         # Ludwig's internal preprocessed data
}
```

---

## `LudwigEvaluateHandler`

### What It Does

Evaluates a trained Ludwig model on the full dataset and logs per-feature metrics.

### Model Source

The handler looks for the model in two places (in order):

1. `context["train-ludwig"]["model"]` — the in-memory model object from the
   training step (step must be named `"train-ludwig"` in the YAML).
2. `config["model_path"]` — a path to a previously saved Ludwig model directory
   (if training was done in a separate run).

### YAML Configuration Reference

```yaml
- name: evaluate
  type: ludwig_evaluate
  config:
    output_directory: artifacts/evaluation
    # model_path: artifacts/ludwig_model/...  # Only needed if running without train step
```

### Context Requirements

**Requires:** `context["train-ludwig"]` (from training step) **or** `config["model_path"]`

**Requires:** `context["data"]` — the evaluation dataset

**Returns:**

```python
{
    "test_stats": {
        "is_anomaly": {
            "accuracy": 0.91,
            "loss": 0.23,
            ...
        }
    },
    "predictions": <DataFrame>,
    "output_directory": "artifacts/evaluation"
}
```

---

## `LudwigPredictHandler`

> **Note:** This class exists in `ludwig_handler.py` but is not yet wired into
> the pipeline engine's `_execute_step()` method. It can be used programmatically
> or added to the engine with a new `elif` branch.

### What It Does

Runs inference on new data using a trained Ludwig model.

```python
from pipelines.handlers.ludwig_handler import LudwigPredictHandler

context = {"train-ludwig": {"model": my_model}}
handler = LudwigPredictHandler(
    config={"output_directory": "artifacts/predictions"},
    context=context
)
result = handler.execute()
# result["predictions"] = DataFrame of predictions
```

---

## `LudwigVisualizeHandler`

### What It Does

Calls Ludwig's built-in `ludwig.visualize` functions to create training and
evaluation plots. Skips gracefully if Ludwig's visualize module is not available.

Recent behavior improvements:
- Uses compatible Ludwig ROC function names across versions (`roc_curves_from_test_statistics_cli` and fallback `roc_curves_cli`).
- Generates any plot whose required artifacts exist (instead of failing/skipping all plots together).
- Logs full exceptions for failed visualizations so pipeline logs are readable and actionable.

### Available Visualisations

| Name | What it shows |
|---|---|
| `learning_curves` | Training vs validation loss/metric per epoch |
| `confusion_matrix` | Predicted vs actual class distribution |
| `roc_curve` | ROC curve from Ludwig's evaluation output |

> **Note:** `feature_importance` is listed in the handler but currently unsupported
> for Ludwig and logs a warning. Ludwig's built-in feature importances require
> tree-based decoders.

### YAML Configuration Reference

```yaml
- name: visualize
  type: ludwig_visualize
  config:
    visualizations:
      - learning_curves
      - confusion_matrix
      - roc_curve
    output_directory: artifacts/visualizations
    # model_path: artifacts/...  # Only if not running after train step
```

### Context Requirements

**Requires:** `context["train-ludwig"]["output_directory"]` (from train step)
**or** `config["model_path"]`

Artifact expectations per visualisation:
- `learning_curves` needs `training_statistics.json`
- `confusion_matrix` needs `test_statistics.json` and `training_set_metadata.json`
- `roc_curve` needs `test_statistics.json`

If one artifact is missing, only that visualisation is skipped; others still run.

---

## Performance Notes

Based on experiments with the sample dataset (15 000 rows, 1 feature,
binary classification):

| Metric | Ludwig Neural Net | Random Forest | XGBoost |
|---|---|---|---|
| Accuracy | 90.97% | 70.00% | 67.00% |
| ROC AUC | 0.38 ⚠️ | **0.61** | **0.61** |
| Recall (Anomaly) | 0% ⚠️ | 45% | **50%** |
| Precision (Anomaly) | 0% ⚠️ | 8% | 7% |
| Training Time | ~150 s | ~1.3 s | ~1.3 s |

> **Why does Ludwig perform poorly here?**  
> Ludwig's neural network predicts "Normal" for all samples — it achieves high
> accuracy by exploiting the class imbalance (95% normal). The binary weighted
> cross-entropy loss with `class_weights: [1, 10]` is supposed to mitigate this,
> but the single-feature dataset does not provide enough signal for the neural
> network to learn the pattern.  
> 
> **Traditional ML (XGBoost) is the better choice for this task.**

---

## Ludwig Feature Types Reference

| Ludwig type | When to use |
|---|---|
| `numerical` | Continuous numbers (throughput, user counts, etc.) |
| `binary` | True/False or 0/1 target columns |
| `category` | Low-cardinality string columns (e.g., cell type) |
| `text` | Free-text strings |
| `sequence` | Ordered sequences |
| `timeseries` | Time-series data |
| `image` | Image file paths |

For the DU cell dataset, only `numerical` (input) and `binary` (output) are used.
