# Example Pipelines

The `examples/` directory contains four ready-to-run YAML pipeline configurations.
This document walks through each one, explaining every section and why it is
configured the way it is.

---

## 1. Random Forest Anomaly Detection

**File:** `examples/random-forest-anomaly.yaml`  
**Run:** `python run_pipeline.py examples/random-forest-anomaly.yaml`

### What it does

Trains a **Random Forest classifier** to detect anomalies in DU cell metrics.
Random Forests work by building many decision trees and aggregating their votes.
They are robust to noise, require minimal hyperparameter tuning, and provide
feature importances out of the box.

### Pipeline Steps

| Step | Type | What happens |
|---|---|---|
| `load-data` | `data_loader` | Reads `data/sample_data.csv` into memory |
| `auto-label-anomalies` | `unsupervised_labeling` | Isolation Forest creates binary labels |
| `train-sklearn` | `sklearn_train` | Trains `RandomForestClassifier` |
| `evaluate` | `sklearn_evaluate` | Computes accuracy, F1, ROC-AUC |
| `visualize` | `sklearn_visualize` | Saves feature_importance, confusion_matrix, roc_curve, pr_curve PNGs |
| `xai` | `sklearn_xai` | Runs SHAP (TreeExplainer) + LIME on 3 instances |

### Key Model Settings

```yaml
sklearn_model:
  type: random_forest
  params:
    n_estimators: 100        # 100 trees — good balance of accuracy vs speed
    max_depth: 10            # Prevents overfitting on noisy data
    class_weight: balanced   # Automatically adjusts for 95%/5% class imbalance
    n_jobs: -1               # Uses all CPU cores
```

### Expected Output

```
artifacts/random_forest_model/
├── random_forest_model.pkl    # Trained model (~500 KB)
├── metrics.txt                # accuracy=0.70, roc_auc=0.61, recall=0.45, ...
├── confusion_matrix.png
├── roc_curve.png
├── precision_recall_curve.png
├── feature_importance.png
└── xai/
    ├── shap_summary.png
    ├── shap_importance.png
    ├── lime_instance_0.png
    ├── lime_instance_1.png
    └── lime_instance_2.png
```

### Expected Performance

| Metric | Value |
|---|---|
| Accuracy | ~70% |
| ROC AUC | ~0.61 |
| Recall (anomaly) | ~45% |
| Precision (anomaly) | ~8% |
| Training time | ~1.3 s |

---

## 2. XGBoost Anomaly Detection

**File:** `examples/xgboost-anomaly.yaml`  
**Run:** `python run_pipeline.py examples/xgboost-anomaly.yaml`

### What it does

Trains an **XGBoost classifier** (eXtreme Gradient Boosting). XGBoost builds
trees sequentially, where each new tree corrects the errors of the previous ones.
It is typically faster and more accurate than Random Forest for structured/tabular data.

### Pipeline Steps

Same structure as the Random Forest pipeline, with an additional `learning_curve` visualisation.

| Step | Type | What happens |
|---|---|---|
| `load-data` | `data_loader` | Reads CSV |
| `auto-label-anomalies` | `unsupervised_labeling` | Creates pseudo-labels |
| `train-sklearn` | `sklearn_train` | Trains `XGBClassifier` |
| `evaluate` | `sklearn_evaluate` | Computes metrics |
| `visualize` | `sklearn_visualize` | Plus `learning_curve` (vs RF which doesn't include it) |
| `xai` | `sklearn_xai` | SHAP + LIME |

### Key Model Settings

```yaml
sklearn_model:
  type: xgboost
  params:
    n_estimators: 100
    max_depth: 6               # Shallower trees than RF to prevent overfitting
    learning_rate: 0.1         # Step size — smaller = more conservative
    subsample: 0.8             # Use 80% of data per tree (reduces overfitting)
    colsample_bytree: 0.8      # Use 80% of features per tree
    scale_pos_weight: 10       # Gives anomaly class 10× more weight
    tree_method: hist          # Faster histogram-based splitting algorithm
```

> **`scale_pos_weight` explained:**  
> With ~5% anomalies and ~95% normals, `scale_pos_weight = 95/5 = 19` would give
> equal weight. Using 10 is a deliberate compromise — it increases recall without
> generating too many false positives.

### Expected Output

```
artifacts/xgboost_model/
├── xgboost_model.pkl
├── metrics.txt                # accuracy=0.67, roc_auc=0.61, recall=0.50, ...
├── confusion_matrix.png
├── roc_curve.png
├── precision_recall_curve.png
├── feature_importance.png
├── learning_curve.png         # Extra vs Random Forest
└── xai/
    ├── shap_summary.png
    ├── shap_importance.png
    ├── lime_instance_0.png
    ├── lime_instance_1.png
    └── lime_instance_2.png
```

### Expected Performance

| Metric | Value |
|---|---|
| Accuracy | ~67% |
| ROC AUC | ~0.61 |
| Recall (anomaly) | **~50%** ← best among all models |
| Precision (anomaly) | ~7% |
| Training time | ~1.3 s |

> **Why is XGBoost the best choice here?**  
> It achieves the highest recall (50%) while maintaining similar ROC-AUC to
> Random Forest. In anomaly detection, missing an anomaly is usually more costly
> than a false alarm, so recall is the most important metric.

---

## 3. Logistic Regression Anomaly Detection

**File:** `examples/logistic-regression-anomaly.yaml`  
**Run:** `python run_pipeline.py examples/logistic-regression-anomaly.yaml`

### What it does

Trains a **Logistic Regression** model — the simplest supervised classification
algorithm. It learns a linear decision boundary: anomaly if a weighted sum of
features exceeds a threshold.

This is useful as a **baseline** to check whether more complex models (RF, XGBoost,
Ludwig) are actually doing better than a trivially simple model.

### Key Model Settings

```yaml
sklearn_model:
  type: logistic_regression
  params:
    C: 1.0                   # Regularisation strength (lower = more regularised)
    penalty: l2              # L2 (Ridge) regularisation
    solver: lbfgs            # Good default for small datasets
    max_iter: 1000           # Increase from default 100 to ensure convergence
    class_weight: balanced   # Compensate for class imbalance
```

### Unique Visualisations

This pipeline uses `coefficients` instead of `feature_importance`:

```yaml
    visualizations:
      - confusion_matrix
      - roc_curve
      - precision_recall_curve
      - coefficients           # ← logistic regression specific
```

The coefficients plot shows the **weight** assigned to each feature:
- Positive coefficient → feature increases anomaly probability
- Negative coefficient → feature decreases anomaly probability
- Larger absolute value → feature has more impact

### Expected Output

```
artifacts/logistic_regression_model/
├── logistic_regression_model.pkl
├── metrics.txt
├── confusion_matrix.png
├── roc_curve.png
├── precision_recall_curve.png
├── coefficients.png           # ← instead of feature_importance
└── xai/
    ├── shap_summary.png       # Uses KernelExplainer (slower!)
    ├── shap_importance.png
    ├── lime_instance_0.png
    ├── lime_instance_1.png
    └── lime_instance_2.png
```

> **SHAP note:** Logistic Regression uses `shap.KernelExplainer` which is
> significantly slower than `TreeExplainer`. Expect 1-5 minutes for the XAI step.
> Reduce `kernel_background_samples` and `kernel_explain_samples` to speed it up.

---

## 4. Ludwig Deep Learning

**File:** `examples/du-cell-ludwig.yaml`  
**Run:** `python run_pipeline.py examples/du-cell-ludwig.yaml`

### What it does

Trains a **Ludwig neural network** for binary classification. Ludwig automatically
constructs a deep feed-forward network with the specified input and output features.

### Requirements

Ludwig must be installed:
```bash
pip install ludwig==0.8.5
```

### Key Configuration

```yaml
ludwig:
  input_features:
    - name: active_users
      type: numerical
      preprocessing:
        normalization: zscore      # Normalise to mean=0, std=1

  output_features:
    - name: is_anomaly
      type: binary
      loss:
        type: binary_weighted_cross_entropy
        class_weights: [1, 10]     # Weight anomaly class 10×

  trainer:
    epochs: 100
    batch_size: 32
    learning_rate: 0.001
    early_stop: 10                 # Stop early if no improvement
    learning_rate_scheduler:
      type: reduce_on_plateau
      patience: 5
```

### Pipeline Steps

| Step | Type | What happens |
|---|---|---|
| `load-data` | `data_loader` | Reads CSV |
| `auto-label-anomalies` | `unsupervised_labeling` | Creates pseudo-labels |
| `train-ludwig` | `ludwig_train` | Trains neural network (~150 s) |
| `evaluate` | `ludwig_evaluate` | Computes Ludwig metrics |
| `visualize` | `ludwig_visualize` | Learning curves, confusion matrix, ROC curve |

### Expected Output

```
artifacts/
├── ludwig_model/
│   └── experiment/
│       ├── model/                 # Saved neural network weights
│       └── training_statistics.json
└── evaluation/
    └── ...
artifacts/visualizations/
├── learning_curves*.png
├── confusion_matrix*.png
└── roc_curves*.png
```

### Expected Performance

| Metric | Value |
|---|---|
| Accuracy | ~91% (**misleading!**) |
| ROC AUC | ~0.38 (⚠️ worse than random) |
| Recall (anomaly) | ~0% (**predicts "normal" for everything**) |
| Training time | ~150 s |

> **Warning:** The Ludwig model exploits class imbalance to achieve high accuracy
> by always predicting "Normal". This is a well-known pitfall for imbalanced
> classification. The ROC AUC of 0.38 (below 0.5 = random) is a clear indicator
> that the model is not useful for anomaly detection.

---

## Comparison Summary

| Pipeline | Accuracy | ROC AUC | Recall (anomaly) | Training Time | Best For |
|---|---|---|---|---|---|
| Ludwig | 90.97% | 0.38 ⚠️ | 0% ⚠️ | ~150 s | Demonstrating deep learning config |
| Random Forest | 70% | 0.61 | 45% | ~1.3 s | Interpretable baseline |
| **XGBoost** | 67% | **0.61** | **50%** | ~1.3 s | **Best anomaly detection** |
| Logistic Regression | varies | varies | varies | ~0.5 s | Fastest baseline |

---

## Creating Your Own Pipeline

Copy one of the example files and modify it:

```bash
cp examples/xgboost-anomaly.yaml my-pipeline.yaml
# Edit my-pipeline.yaml
python run_pipeline.py my-pipeline.yaml
```

Refer to the [YAML Configuration Reference](yaml-configuration.md) for all
available options.
