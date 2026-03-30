# Getting Started

This guide walks you through everything you need to go from a freshly cloned
repository to a running pipeline. It is written for someone who may be new to
Python or machine learning — every step is explained in plain English.

---

## Prerequisites

| Requirement | Why it is needed |
|---|---|
| **Python 3.11 or newer** | The project uses modern Python features and is tested on 3.11+ |
| **pip** | Python's package installer — comes bundled with Python |
| **git** | To clone the repository |
| **4 GB free disk space** | For virtual environment, dependencies, and model artifacts |
| **Internet connection** | To download packages during installation |

> **What is Python?**  
> Python is a programming language. When you "run a Python script" you are asking
> the Python interpreter (`python`) to execute instructions written in a `.py` file.

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/umerarshad2022/ml-pipeline-ludwig.git
cd ml-pipeline-ludwig
```

> `git clone` downloads a copy of the project code to your computer.  
> `cd` ("change directory") moves you inside that folder.

---

## Step 2 — Create a Virtual Environment

A **virtual environment** is an isolated folder that holds *this project's* Python
packages without interfering with other Python projects on your machine.

```bash
# Create the virtual environment (only needed once)
python -m venv venv

# Activate it — Linux / macOS
source venv/bin/activate

# Activate it — Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

After activation your terminal prompt will start with `(venv)`.

---

## Step 3 — Install Dependencies

### Option A — Automated (recommended)

```bash
chmod +x install.sh
./install.sh
```

The `install.sh` script:
1. Creates/activates the virtual environment.
2. Upgrades `pip`.
3. Installs all packages listed in `requirements.txt`.

### Option B — Manual

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **What is `requirements.txt`?**  
> It is a plain text file listing every Python package the project needs, with
> the exact version number. `pip install -r requirements.txt` reads that file and
> installs everything automatically.

### What Gets Installed

| Package | Purpose |
|---|---|
| `pandas` | Tabular data manipulation (like Excel in code) |
| `numpy` | Fast numerical arrays and math |
| `scikit-learn` | Traditional ML algorithms (Random Forest, Logistic Regression, preprocessing) |
| `xgboost` | Gradient-boosted tree algorithm |
| `ludwig` | Declarative deep-learning framework |
| `shap` | SHAP explainability |
| `lime` | LIME explainability |
| `matplotlib` / `seaborn` | Plotting and visualization |
| `pyyaml` | Read YAML configuration files |
| `mlflow` | Experiment tracking and metric logging |
| `joblib` | Save/load trained model files |
| `boto3` | AWS S3 storage (optional) |

---

## Step 4 — Verify Installation

```bash
python -c "import pandas, sklearn, xgboost, yaml; print('All core packages OK')"
```

Expected output:
```
All core packages OK
```

---

## Step 5 — Run Your First Pipeline

The fastest way to see the project working end-to-end:

```bash
python run_pipeline.py examples/random-forest-anomaly.yaml
```

You will see detailed logs like:

```
================================================================================
  LUDWIG ML PIPELINE
================================================================================
Config: examples/random-forest-anomaly.yaml
...
[1/6] load-data        ← loads data/sample_data.csv
[2/6] auto-label-anomalies  ← uses Isolation Forest to create labels
[3/6] train-sklearn    ← trains a Random Forest
[4/6] evaluate         ← computes accuracy, F1, ROC-AUC
[5/6] visualize        ← saves confusion matrix, ROC curve, etc.
[6/6] xai              ← generates SHAP and LIME explanations
================================================================================
  ✅ PIPELINE COMPLETED SUCCESSFULLY
================================================================================
```

### What Was Created?

After the run you will find new files under `artifacts/`:

```
artifacts/
├── random_forest_model/
│   ├── random_forest_model.pkl    # The trained model (binary file)
│   ├── metrics.txt                # Accuracy, F1, ROC-AUC, etc.
│   ├── confusion_matrix.png       # Visualisation
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── xai/
│       ├── shap_summary.png
│       ├── shap_waterfall_*.png
│       └── lime_explanation_*.png
```

---

## Step 6 — Try Other Pipelines

```bash
# XGBoost (usually best anomaly recall)
python run_pipeline.py examples/xgboost-anomaly.yaml

# Logistic Regression (fastest, good baseline)
python run_pipeline.py examples/logistic-regression-anomaly.yaml

# Ludwig deep learning (requires Ludwig to be installed correctly)
python run_pipeline.py examples/du-cell-ludwig.yaml
```

---

## Step 7 — Run the Tests

```bash
pip install pytest
pytest tests/ -v
```

All tests should pass. See [Testing](testing.md) for details.

---

## Command-Line Interface Reference

`run_pipeline.py` accepts these arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `config` | Yes | — | Path to the YAML pipeline config file |
| `--output` | No | `artifacts` | Directory where artifacts are written |
| `--gpu` | No | disabled | Enable GPU training (if CUDA is available) |

**Examples:**

```bash
# Run with a custom output directory
python run_pipeline.py examples/xgboost-anomaly.yaml --output /tmp/my-run

# Enable GPU
python run_pipeline.py examples/du-cell-ludwig.yaml --gpu
```

---

## Running with Docker

If you prefer not to install Python locally:

```bash
# Build the image
docker build -t ml-pipeline-ludwig .

# Run a pipeline inside the container
docker run --rm -v $(pwd)/artifacts:/app/artifacts \
  ml-pipeline-ludwig python run_pipeline.py examples/random-forest-anomaly.yaml
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'ludwig'"

Ludwig has heavy dependencies. If the automated install failed:

```bash
pip install ludwig==0.8.5
pip install --force-reinstall marshmallow-dataclass==8.5.4
pip install "pydantic<2.0"
```

### "ModuleNotFoundError: No module named 'xgboost'"

```bash
pip install xgboost==2.0.3
```

### SHAP visualization errors or blank plots

```bash
export MPLBACKEND=Agg
python run_pipeline.py examples/random-forest-anomaly.yaml
```

### Out of Memory during XAI

Edit the relevant `explain_*.py` script and reduce the sample size:

```python
sample_size = min(500, len(X))   # reduce from 1000 to 500
```

### Pipeline fails with "Config file not found"

Make sure you are running commands from the **project root directory**
(`ml-pipeline-ludwig/`), not from a subdirectory.

```bash
# Wrong
cd examples
python ../run_pipeline.py random-forest-anomaly.yaml  # path will break

# Correct
cd ml-pipeline-ludwig
python run_pipeline.py examples/random-forest-anomaly.yaml
```

---

## Next Steps

Once the pipeline is running:

- Understand the YAML config → [YAML Configuration Reference](yaml-configuration.md)
- See how the engine works → [Pipeline Engine](pipeline-engine.md)
- Learn about SHAP/LIME plots → [Explainable AI](xai.md)
- Add your own data → [Dataset](dataset.md)
