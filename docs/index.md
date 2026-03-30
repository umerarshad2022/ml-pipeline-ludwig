# ML Pipeline Ludwig — Documentation

Welcome to the complete developer documentation for the **ML Pipeline Ludwig** project.

This project is a **YAML-driven machine learning pipeline engine** designed for
anomaly detection in Open RAN (Radio Access Network) DU (Distributed Unit) cell metrics.
It supports multiple ML backends — Ludwig (deep learning), scikit-learn (Random Forest,
Logistic Regression), and XGBoost — and bundles Explainable AI (XAI) tooling via
SHAP and LIME.

> **New to Python or machine learning?**  
> Start with [Getting Started](getting-started.md). It explains every step from
> installing Python to running your first pipeline, assuming zero prior knowledge.

---

## Table of Contents

| Document | What you will learn |
|---|---|
| [Getting Started](getting-started.md) | Install the project, set up a virtual environment, run your first pipeline in 5 minutes |
| [Architecture Overview](architecture.md) | How all the pieces fit together; bird's-eye data-flow diagram |
| [Pipeline Engine](pipeline-engine.md) | Deep dive into `pipelines/engine.py` — the orchestrator that reads YAML and runs steps |
| [Data Loading & Preprocessing](data-loading.md) | `DataLoaderHandler` and `UnsupervisedLabelingHandler` explained line-by-line |
| [Scikit-learn Handlers](sklearn-handlers.md) | Train, evaluate, visualise, and explain Random Forest / XGBoost / Logistic Regression models |
| [Ludwig Handlers](ludwig-handlers.md) | Train, evaluate, predict, and visualise deep-learning models with Ludwig AI |
| [YAML Configuration Reference](yaml-configuration.md) | Every YAML key documented with type, default, and examples |
| [Example Pipelines](examples.md) | Annotated walk-through of all four ready-to-run example configs |
| [Explainable AI (XAI)](xai.md) | SHAP and LIME — what they are, how to configure them, how to read the plots |
| [Standalone Scripts](standalone-scripts.md) | `train_random_forest.py`, `train_xgboost.py`, `explain_*.py` — when and how to use them |
| [Dataset](dataset.md) | Column descriptions, class distribution, and how to swap in your own data |
| [Testing](testing.md) | Run existing tests, understand what each test covers, write new ones |

---

## Repository Layout at a Glance

```
ml-pipeline-ludwig/
│
├── docs/                        ← You are here — all documentation
│
├── pipelines/                   ← Core Python package (the "engine")
│   ├── engine.py                  Main orchestrator
│   └── handlers/
│       ├── data_loader.py         Load CSV/Parquet/JSON + unsupervised labelling
│       ├── sklearn_handler.py     Train/evaluate/visualise/XAI for sklearn models
│       ├── ludwig_handler.py      Train/evaluate/predict with Ludwig
│       └── ludwig_visualize.py    Learning-curve, confusion-matrix, ROC plots for Ludwig
│
├── examples/                    ← Ready-to-run YAML pipeline configs
│   ├── du-cell-ludwig.yaml        Ludwig deep-learning pipeline
│   ├── random-forest-anomaly.yaml Random Forest pipeline
│   ├── xgboost-anomaly.yaml       XGBoost pipeline
│   └── logistic-regression-anomaly.yaml Logistic Regression pipeline
│
├── data/
│   └── sample_data.csv          ← 15 000-row synthetic Open RAN dataset
│
├── tests/                       ← pytest test suite
│   ├── test_engine.py
│   ├── test_data_loader.py
│   └── test_sklearn_handler.py
│
├── run_pipeline.py              ← CLI entry point for YAML pipelines
├── train_random_forest.py       ← Standalone RF training script
├── train_xgboost.py             ← Standalone XGBoost training script
├── explain_random_forest.py     ← SHAP + LIME for Random Forest
├── explain_xgboost.py           ← SHAP + LIME for XGBoost
│
├── requirements.txt             ← Python dependency list
├── Dockerfile                   ← Container build file
├── docker-compose.yml           ← Multi-container setup
└── install.sh                   ← Automated install script (Linux/macOS)
```

---

## Key Concepts (30-second summary)

| Term | Plain-English meaning |
|---|---|
| **Pipeline** | A sequence of steps (load data → label → train → evaluate → visualise) driven by a single YAML file |
| **Engine** | The Python class (`LudwigPipelineEngine`) that reads the YAML and runs each step |
| **Handler** | A Python class responsible for *one* pipeline step (e.g. `SklearnTrainHandler`) |
| **Context** | A shared Python dictionary that handlers use to pass data to each other |
| **Ludwig** | An open-source framework that trains deep neural networks from a config file — no custom training loop needed |
| **SHAP / LIME** | Explainability libraries that show *why* a model made a specific prediction |
| **Artifact** | Any file saved to disk after a step: trained model `.pkl`, metric `.txt`, plot `.png` |
| **Anomaly** | In this project, a DU cell reading that deviates significantly from normal operating conditions |

---

## Quick Reference — Running Pipelines

```bash
# Random Forest anomaly detection
python run_pipeline.py examples/random-forest-anomaly.yaml

# XGBoost anomaly detection
python run_pipeline.py examples/xgboost-anomaly.yaml

# Logistic Regression anomaly detection
python run_pipeline.py examples/logistic-regression-anomaly.yaml

# Ludwig deep learning (requires Ludwig installed)
python run_pipeline.py examples/du-cell-ludwig.yaml
```

All output artifacts are written to the `artifacts/` directory.
