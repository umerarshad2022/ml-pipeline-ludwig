# ML Pipeline for Open RAN Anomaly Detection

YAML-based ML pipeline engine with support for Ludwig, scikit-learn models, and XAI (Explainable AI) techniques.

## 🎯 Project Overview

This project implements anomaly detection for Open RAN DU (Distributed Unit) cell metrics using multiple machine learning approaches:
- **Ludwig** - Declarative deep learning framework
- **Random Forest** - Traditional ensemble method  
- **XGBoost** - Gradient boosting for better performance
- **XAI** - SHAP and LIME for model interpretability

## 📁 Project Structure

```
ml-pipeline-ludwig/
├── pipelines/               # Core pipeline engine
│   ├── engine.py           # Main pipeline orchestrator
│   └── handlers/           # Step handlers (Ludwig, preprocessing, etc.)
├── examples/               # YAML pipeline configurations
│   ├── du-cell-ludwig.yaml           # Ludwig neural network
│   ├── random-forest-anomaly.yaml    # Random Forest classifier
│   └── xgboost-anomaly.yaml          # XGBoost classifier
├── data/
│   └── sample_data.csv     # Sample telecom data (15k rows)
├── train_random_forest.py  # Standalone RF training script
├── train_xgboost.py        # Standalone XGBoost training script
├── explain_random_forest.py # SHAP & LIME for Random Forest
├── explain_xgboost.py      # SHAP & LIME for XGBoost
├── run_pipeline.py         # Execute YAML pipelines
└── artifacts/              # Generated models and visualizations (gitignored)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ml-pipeline-ludwig

# Install dependencies
chmod +x install.sh
./install.sh

# Activate virtual environment
source venv/bin/activate
```

### Training Models

#### Option 1: Using YAML Pipelines (Ludwig)
```bash
python run_pipeline.py examples/du-cell-ludwig.yaml
```

#### Option 2: Standalone Training Scripts
```bash
# Random Forest
python train_random_forest.py

# XGBoost
python train_xgboost.py
```

### Running XAI Analysis

After training models, generate explainability visualizations:

```bash
# SHAP & LIME for Random Forest
python explain_random_forest.py

# SHAP & LIME for XGBoost
python explain_xgboost.py
```

Results saved to `artifacts/<model_name>/xai/`

## 📊 Dataset

**Source:** Synthetic Open RAN DU cell metrics  
**Size:** 15,000 samples  
**Features:**
- `timestamp` - Measurement timestamp
- `cell_id` - Cell identifier
- `throughput_dl` - Downlink throughput (Mbps)
- `throughput_ul` - Uplink throughput (Mbps)
- `prb_used_dl` - Downlink PRB utilization (%)
- `prb_used_ul` - Uplink PRB utilization (%)
- `active_users` - Number of connected users
- `is_anomaly` - Binary label (0=Normal, 1=Anomaly)

**Class Distribution:** ~95% Normal, ~5% Anomalies

## 🤖 Models

### 1. Ludwig (Neural Network)
- **Config:** `examples/du-cell-ludwig.yaml`
- **Architecture:** Deep feed-forward network
- **Training:** 100 epochs with early stopping
- **Loss:** Binary weighted cross-entropy

### 2. Random Forest
- **Script:** `train_random_forest.py`
- **Trees:** 100
- **Training Time:** ~1.3 seconds
- **ROC AUC:** 0.61

### 3. XGBoost (Best Performance)
- **Script:** `train_xgboost.py`
- **Estimators:** 100
- **Training Time:** ~1.3 seconds
- **ROC AUC:** 0.61
- **Recall:** 50% (best anomaly detection)

## 🔍 Explainable AI (XAI)

Both Random Forest and XGBoost models include comprehensive interpretability analysis:

### SHAP (SHapley Additive exPlanations)
- **Summary plots** - Feature impact distribution
- **Feature importance** - Global feature rankings
- **Waterfall plots** - Single prediction breakdown
- **Force plots** - Push/pull visualization
- **Dependence plots** - Feature value vs impact

### LIME (Local Interpretable Model-agnostic Explanations)
- **Instance explanations** - Why specific predictions were made
- **Multi-instance comparison** - Side-by-side analysis

**View XAI results:**
```bash
# Using image viewer
eog artifacts/random_forest_model/xai/*.png
eog artifacts/xgboost_model/xai/*.png

# Using file manager
nautilus artifacts/xgboost_model/xai/
```

## 📈 Model Performance Comparison

| Model | Accuracy | ROC AUC | Recall (Anomaly) | Precision (Anomaly) | Training Time |
|-------|----------|---------|------------------|---------------------|---------------|
| Ludwig Neural Net | 90.97% | 0.38 ⚠️ | 0% | 0% | ~150s |
| Random Forest | 70.00% | **0.61** | 45% | 8% | **1.32s** |
| XGBoost | 67.00% | **0.61** | **50%** | 7% | **1.31s** |

**Winner:** XGBoost (best anomaly detection with 50% recall)

> ⚠️ Ludwig's high accuracy is misleading - it predicts "normal" for all samples!

## 🛠️ Dependencies

- Python 3.11+
- Ludwig 0.8.5 (Deep learning)
- XGBoost 3.1.3 (Gradient boosting)
- scikit-learn 1.8.0 (Machine learning)
- SHAP 0.50.0 (Explainability)
- LIME 0.2.0.1 (Explainability)
- pandas, numpy, matplotlib

See `requirements.txt` for full list.

## 📝 Usage Examples

### Custom Pipeline Configuration

Create a YAML file following this structure:

```yaml
name: My Anomaly Detection Pipeline
data:
  preprocessing:
    drop_columns: [Cell_ID, gNB_ID, Timestamp]
ludwig:
  input_features:
    - name: Active_UEs_DL
      type: numerical
      preprocessing:
        normalization: zscore
  output_features:
    - name: is_anomaly
      type: binary
      loss:
        type: binary_weighted_cross_entropy
  trainer:
    epochs: 100
    batch_size: 64
```

### Programmatic Model Training

```python
# Load and train Random Forest
from train_random_forest import main
main()  # Trains model and saves to artifacts/

# Load and run XAI
from explain_xgboost import *
# Generates all SHAP and LIME visualizations
```

## 🎓 For Thesis/Research

This repository demonstrates:
1. **Multi-model comparison** for telecom anomaly detection
2. **Explainable AI** techniques for production ML systems
3. **Open RAN metrics** analysis
4. **Pipeline automation** with YAML configurations

**Key findings:**
- Traditional ML (XGBoost) outperforms neural networks for this task
- SHAP reveals which features drive anomaly predictions
- Fast training enables real-time model updates
- Explainability is critical for telecom operator trust

## 📄 Generated Artifacts

After training and XAI analysis:

```
artifacts/
├── random_forest_model/
│   ├── random_forest_model.pkl        # Trained model
│   ├── metrics.txt                    # Performance metrics
│   ├── visualizations/
│   │   ├── confusion_matrix_<timestamp>.png
│   │   ├── roc_curve_<timestamp>.png
│   │   └── precision_recall_curve_<timestamp>.png
│   └── xai/                           # XAI visualizations with timestamped PNG names
└── xgboost_model/
    ├── xgboost_model.pkl              # Trained model
    ├── metrics.txt                    # Performance metrics
    ├── visualizations/
    │   ├── confusion_matrix_<timestamp>.png
    │   ├── roc_curve_<timestamp>.png
    │   ├── precision_recall_curve_<timestamp>.png
    │   └── feature_importance_<timestamp>.png
    └── xai/                           # XAI visualizations with timestamped PNG names
```

## 🔧 Configuration

### Environment Variables
- `LUDWIG_CACHE_DIR` - Ludwig cache directory (auto-set)
- `PYTHONPATH` - Python path (auto-set by venv)

### Pipeline Settings
Edit YAML files in `examples/` to customize:
- Input/output features
- Model architecture
- Training hyperparameters
- Loss functions
- Preprocessing steps

## 🐛 Troubleshooting

**Ludwig fails to import:**
```bash
# Reinstall dependencies
pip install --force-reinstall marshmallow-dataclass==8.5.4
pip install pydantic<2.0
```

**SHAP visualization errors:**
```bash
# Ensure matplotlib backend is set
export MPLBACKEND=Agg
```

**Out of memory:**
```bash
# Reduce sample size in XAI scripts
# Edit explain_*.py: change sample_size = min(1000, len(X))
```

## 📚 References

- Ludwig: https://ludwig.ai/
- SHAP: https://shap.readthedocs.io/
- LIME: https://github.com/marcotcr/lime
- XGBoost: https://xgboost.readthedocs.io/

## 📧 Contact

For questions about this thesis project, contact [your email/info]

## 📄 License

[Your license - e.g., MIT, Academic Use Only, etc.]

---

**Last Updated:** February 2026  
**Project:** Open RAN Anomaly Detection with Explainable AI  
**Course/Thesis:** [Your course/thesis info]
