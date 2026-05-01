# Model Selection Rationale

## Why Ludwig, XGBoost, and Random Forest?

The choice of models in this project was deliberate and motivated by three complementary goals: demonstrating a **comparative study**, satisfying the **YAML-configurable pipeline** thesis requirement, and fulfilling the **Explainable AI (XAI)** objective.

---

## 1. Comparative Study by Design

The project is explicitly structured as a multi-model comparison for telecom anomaly detection on Open RAN DU cell metrics. Using three fundamentally different approaches — deep learning, gradient boosting, and ensemble trees — enables objective trade-off analysis rather than committing to a single technique. This is a recognised best practice in applied ML research.

---

## 2. Ludwig (Deep Learning / Neural Network)

**What it is:** A declarative deep learning framework that configures deep feed-forward networks entirely via YAML, without writing training loops.

**Why it was chosen:**
- Directly satisfies the thesis objective: *"a configurable pipeline engine … configurable via YAML."* Ludwig is the only framework that natively treats model configuration as a YAML document.
- Supports advanced architectures (MLP, CNN, RNN, Transformers) and GPU training, making the pipeline future-proof as data volume grows.
- Serves as the **deep learning baseline** in the comparison.

**What the results show:**
- ROC AUC: **0.38** — the model predicts "Normal" for all samples (accuracy 90.97% is misleading on the imbalanced dataset).
- This is itself a meaningful finding: deep learning is not always superior, especially on small, heavily imbalanced datasets (~15 k samples, 95 % / 5 % class split).

---

## 3. XGBoost (Gradient Boosting)

**What it is:** A gradient-boosted decision tree algorithm, widely regarded as the state-of-the-art for structured/tabular data.

**Why it was chosen:**
- Telecom DU cell metrics are structured, numerical, tabular data — exactly the domain where XGBoost consistently excels in industry benchmarks and Kaggle competitions.
- Achieves the **best anomaly detection recall (50 %)** and the best ROC AUC (0.61) while training in ~1.3 seconds, which is critical for real-time telecom monitoring.
- Has **native SHAP support**, directly enabling the XAI component of the project.
- Comparing a boosting method against a bagging method (Random Forest) is a standard and well-recognised evaluation in the literature.

---

## 4. Random Forest (Ensemble / Bagging)

**What it is:** A classical ensemble method that builds many decision trees and aggregates their predictions.

**Why it was chosen:**
- Provides the **standard ensemble baseline** — robust to overfitting on modest-sized datasets, interpretable at the feature level, and well-accepted in academic benchmarks.
- Enables a direct **bagging vs. boosting** comparison against XGBoost, which is a common evaluation axis in anomaly detection literature.
- Fully compatible with SHAP and LIME, satisfying the explainability requirement without additional integration work.

---

## 5. Why Not Other Techniques?

| Technique | Reason Not Used |
|-----------|----------------|
| **Logistic Regression / SVM** | Too simple for the non-linear patterns expected in network anomalies; suitable as a sanity-check baseline but not a primary model. |
| **Isolation Forest / One-Class SVM** | Unsupervised; the dataset carries ground-truth labels (`is_anomaly`), so supervised methods are more appropriate and yield stronger performance. |
| **Standalone LSTM / Transformer** | Ludwig already exposes these architectures internally via YAML config; adding them as separate models would be redundant. |
| **AutoML (AutoSklearn, TPOT, etc.)** | Lack the configurability and transparency required by the thesis; XAI integration would be significantly harder. |
| **k-NN / Naive Bayes** | Poor fit for high-dimensional, continuous telecom features; no meaningful SHAP integration for explainability. |

---

## 6. Summary

| Model | Type | ROC AUC | Anomaly Recall | Training Time | XAI Support |
|-------|------|---------|----------------|---------------|-------------|
| Ludwig Neural Net | Deep Learning | 0.38 ⚠️ | 0 % | ~150 s | Medium |
| Random Forest | Ensemble (Bagging) | **0.61** | 45 % | **1.32 s** | ✅ SHAP + LIME |
| XGBoost | Ensemble (Boosting) | **0.61** | **50 %** | **1.31 s** | ✅ SHAP + LIME |

**Key finding:** Traditional ML models (XGBoost, Random Forest) outperform the neural network on this dataset, demonstrating that model selection should be driven by data characteristics — not by algorithm popularity. This is a core insight of the project.

---

## References

- Ludwig: <https://ludwig.ai/>
- XGBoost: <https://xgboost.readthedocs.io/>
- scikit-learn Random Forest: <https://scikit-learn.org/stable/modules/ensemble.html#forest>
- SHAP: <https://shap.readthedocs.io/>
- LIME: <https://github.com/marcotcr/lime>
