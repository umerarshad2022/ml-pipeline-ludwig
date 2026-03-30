# Standalone Scripts

The project includes standalone Python scripts that can be run independently of
the YAML pipeline engine. These are useful when you want to:
- Quickly train a model without writing a YAML config
- Run a full XAI analysis with more plot types than the pipeline provides
- Understand each step individually (good for learning)

---

## `train_random_forest.py`

**Purpose:** Train a Random Forest classifier from scratch.

**Run:**
```bash
python train_random_forest.py
```

### What It Does (Step by Step)

1. **Load data** from `data/sample_data.csv`
2. **Auto-label anomalies** using `IsolationForest(contamination=0.1)` on the
   `active_users` column — if the dataset does not already have an `is_anomaly` column
3. **Prepare features:** Uses only `active_users` as the feature
4. **Split data:** 80% train / 20% test (stratified by `is_anomaly`)
5. **Train model:** `RandomForestClassifier` with these settings:
   - 100 trees, max depth 10, balanced class weights
   - Uses all CPU cores (`n_jobs=-1`)
6. **Evaluate:** Prints classification report, confusion matrix, ROC-AUC, Average Precision
7. **Save model** to `artifacts/random_forest_model/random_forest_model.pkl`
8. **Save metrics** to `artifacts/random_forest_model/metrics.txt`
9. **Save plots:** Confusion matrix, ROC curve, Precision-Recall curve PNGs

### Configuration

Edit these constants at the top of the file:

```python
DATA_PATH = 'data/sample_data.csv'          # Input data
MODEL_OUTPUT_DIR = 'artifacts/random_forest_model'  # Output directory
RANDOM_STATE = 42                           # Reproducibility seed
```

### Key Difference from Pipeline

The standalone script hard-codes the model hyperparameters. The pipeline version
reads them from the YAML `sklearn_model.params` block, giving you more flexibility.

---

## `train_xgboost.py`

**Purpose:** Train an XGBoost classifier from scratch.

**Run:**
```bash
python train_xgboost.py
```

### What It Does

Same structure as `train_random_forest.py`, with two differences:

1. **Checks for XGBoost installation** and auto-installs if missing (using
   `subprocess.check_call(['pip', 'install', 'xgboost'])`).
2. **Calculates `scale_pos_weight` dynamically:**
   ```python
   scale_pos_weight = (y == 0).sum() / (y == 1).sum()
   # ≈ 18-19 for this dataset (95% normal / 5% anomaly)
   ```
   This automatically compensates for class imbalance without hardcoding a value.

### XGBoost Settings Used

```python
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=scale_pos_weight,  # Auto-calculated
    tree_method='hist',                 # Faster algorithm
    verbosity=1
)
```

### Extra Visualisation

Saves an additional **XGBoost Feature Importance** plot using XGBoost's built-in
`xgb.plot_importance()` function.

### Output

```
artifacts/xgboost_model/
├── xgboost_model.pkl
├── metrics.txt
├── confusion_matrix.png
├── roc_curve.png
├── precision_recall_curve.png
└── feature_importance.png
```

---

## `explain_random_forest.py`

**Purpose:** Generate comprehensive SHAP and LIME explanations for a trained
Random Forest model.

**Prerequisites:** Run `train_random_forest.py` first (or the RF pipeline).

**Run:**
```bash
python explain_random_forest.py
```

### Configuration

```python
DATA_PATH  = 'data/sample_data.csv'
MODEL_PATH = 'artifacts/random_forest_model/random_forest_model.pkl'
OUTPUT_DIR = 'artifacts/random_forest_model/xai'
RANDOM_STATE = 42
```

### SHAP Plots Generated

| File | Description |
|---|---|
| `shap_summary_plot.png` | Beeswarm: every sample's SHAP value for every feature |
| `shap_feature_importance.png` | Bar chart: mean \|SHAP\| per feature |
| `shap_waterfall_anomaly.png` | Waterfall: step-by-step contribution for one anomaly |
| `shap_force_plot.png` | Force: push/pull visualisation for first sample |
| `shap_dependence_active_users.png` | Dependence: feature value vs SHAP value scatter |

### LIME Plots Generated

| File | Description |
|---|---|
| `lime_explanation_normal.png` | Explanation for a normal sample |
| `lime_explanation_anomaly.png` | Explanation for an anomaly sample |
| `lime_multiple_instances.png` | 5-panel chart showing multiple instances |

### Summary Report

`xai_summary.txt` — A text file containing:
- Dataset info
- SHAP feature importances (numerical)
- List of all generated files

### Sample Size

For performance, only 1000 samples are used for SHAP computation:

```python
sample_size = min(1000, len(X))
X_sample = X.sample(n=sample_size, random_state=RANDOM_STATE)
```

To use more samples (slower but more accurate):
```python
sample_size = len(X)  # Use all data
```

---

## `explain_xgboost.py`

**Purpose:** Generate comprehensive SHAP and LIME explanations for a trained
XGBoost model.

**Prerequisites:** Run `train_xgboost.py` first (or the XGBoost pipeline).

**Run:**
```bash
python explain_xgboost.py
```

Identical structure to `explain_random_forest.py`, with:
- `MODEL_PATH = 'artifacts/xgboost_model/xgboost_model.pkl'`
- `OUTPUT_DIR = 'artifacts/xgboost_model/xai'`

The SHAP explainer used is `shap.TreeExplainer` for both RF and XGBoost (both
are tree-based models).

---

## Comparison: Standalone Scripts vs Pipeline

| Feature | Standalone Scripts | YAML Pipeline |
|---|---|---|
| Configuration | Hardcoded constants | YAML file — easy to change |
| Flexibility | Limited | High — change model type, features, steps |
| SHAP plot types | 5 (summary, bar, waterfall, force, dependence) | 2 (summary, bar) |
| LIME instances | Smart selection (normal + anomaly) | First N instances |
| Reproducibility | Good (RANDOM_STATE) | Good |
| Extensibility | Must modify Python code | Add/change YAML steps |
| Best for | Learning, debugging, quick experiments | Production, repeatable workflows |

---

## Running Everything

To run all standalone scripts in sequence:

```bash
# Activate venv
source venv/bin/activate

# Train models
python train_random_forest.py
python train_xgboost.py

# Generate XAI explanations
python explain_random_forest.py
python explain_xgboost.py

# All artifacts are now in artifacts/
```

Or use the pipeline engine for a single command:

```bash
python run_pipeline.py examples/random-forest-anomaly.yaml
python run_pipeline.py examples/xgboost-anomaly.yaml
```
