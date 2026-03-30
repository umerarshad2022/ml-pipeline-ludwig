# Explainable AI (XAI)

Machine learning models — especially ensemble methods like Random Forest and
XGBoost — are often called "black boxes" because it is hard to understand *why*
they made a specific prediction. **Explainable AI (XAI)** techniques address this
by providing human-interpretable explanations.

This project integrates two industry-standard XAI libraries:
- **SHAP** (SHapley Additive exPlanations)
- **LIME** (Local Interpretable Model-agnostic Explanations)

---

## Why XAI Matters

Consider a DU cell anomaly detector deployed in a telecom network operations centre:

- The model says **"anomaly detected"** on cell 0042.
- The operator asks: **why?**
- Without XAI, there is no answer.
- With XAI, you can say: *"The active_users value (2.3) is far above the normal
  range, and that is driving 94% of the anomaly probability."*

This is critical for:
- **Operator trust** — people trust systems they can understand.
- **Debugging** — identify when the model is using spurious correlations.
- **Compliance** — regulations may require explainability for automated decisions.

---

## SHAP (SHapley Additive exPlanations)

### Concept

SHAP is based on **Shapley values** from cooperative game theory. The idea:
*how much does each feature contribute to this prediction, compared to the average prediction?*

For a prediction `f(x)`:
```
f(x) = base_value + shap(feature_1) + shap(feature_2) + ... + shap(feature_n)
```

Where:
- `base_value` = average model output across all training samples
- `shap(feature_i)` = how much feature `i` pushed the prediction up or down

SHAP values are **consistent** (unlike simple feature importances) and can be
both positive (push towards anomaly) or negative (push towards normal).

### SHAP Plots Generated

#### 1. Summary Beeswarm Plot (`shap_summary.png`)

```
Feature: active_users
    ●●●●●●●●●●●●●     ●●●●
    ─────────────────────────
    ◄───── Normal ─── Anomaly ──────►
    SHAP value: negative    positive
```

- Each dot is one sample from the test set.
- **Position on x-axis** = SHAP value (positive = pushes towards anomaly).
- **Colour** = actual feature value (red = high, blue = low).
- **Spreading** = multiple samples with similar SHAP values.

**How to read:** If high `active_users` values (red dots) cluster on the right
(positive SHAP), that means high user counts increase the anomaly probability.

#### 2. Feature Importance Bar Chart (`shap_importance.png`)

Shows the **mean absolute SHAP value** for each feature — a global measure of
feature importance. Features are sorted from most to least important.

In this project, `active_users` typically appears as the dominant (and often only)
feature.

### SHAP Explainers

The handler automatically selects the right explainer:

| Model | Explainer | Notes |
|---|---|---|
| Random Forest | `shap.TreeExplainer` | Exact, very fast (milliseconds) |
| XGBoost | `shap.TreeExplainer` | Exact, very fast |
| Logistic Regression | `shap.KernelExplainer` | Approximate, slower (minutes) |

`TreeExplainer` uses the tree structure directly for exact computation.
`KernelExplainer` is model-agnostic and approximates SHAP values by sampling
from the training data.

### SHAP in the Pipeline (YAML)

```yaml
- name: xai
  type: sklearn_xai
  config:
    methods:
      - shap
    output_directory: artifacts/model/xai

    # Only needed for logistic_regression:
    kernel_background_samples: 50   # Fewer = faster but less accurate
    kernel_explain_samples: 50      # Fewer instances = faster
```

---

## LIME (Local Interpretable Model-agnostic Explanations)

### Concept

LIME explains **individual predictions** by building a simple (linear) proxy
model in the local neighbourhood of a specific instance.

For a single prediction:
1. Perturb the input slightly (add noise to feature values).
2. Get predictions from the original model for all perturbed inputs.
3. Fit a linear model that approximates the original model's behaviour near this point.
4. The linear model's coefficients are the "explanation" for this instance.

### Key Difference from SHAP

| | SHAP | LIME |
|---|---|---|
| Scope | Can be global (all samples) or local | Only local (one instance at a time) |
| Faithfulness | Exact for trees | Approximate (local linear proxy) |
| Speed | Fast (TreeExplainer) | Medium (always needs sampling) |
| Consistency | Mathematically consistent | Depends on sampling randomness |

**When to prefer LIME:** When you need per-instance explanations and the user
wants a simpler "which features matter for *this specific* case" answer.

### LIME Plots Generated

For each instance, LIME produces a bar chart (`lime_instance_N.png`):

```
Prediction: Anomaly (probability 0.83)

Feature weights:
  active_users > 85     |████████████| +0.34
  active_users > 60     |████         | +0.12
  active_users <= 90    ████|          | -0.05
```

- **Positive bars** (rightward) = feature is pushing the prediction towards anomaly.
- **Negative bars** (leftward) = feature is pushing towards normal.
- **Bar width** = magnitude of influence.

### LIME in the Pipeline (YAML)

```yaml
- name: xai
  type: sklearn_xai
  config:
    methods:
      - lime
    n_lime: 3                    # How many test instances to explain
    output_directory: artifacts/model/xai
```

Choosing `n_lime`:
- **3-5** is enough to see variety in explanations.
- **More instances** gives better coverage but takes more time and storage.
- LIME always picks the first N instances from `X_test`.

---

## Standalone XAI Scripts

For more advanced analysis, the project includes standalone scripts that offer
more SHAP plot types than the pipeline handler:

### `explain_random_forest.py`

```bash
# First train the model
python train_random_forest.py

# Then explain it
python explain_random_forest.py
```

Generates:
- `shap_summary.png` — beeswarm plot
- `shap_importance.png` — bar chart
- `shap_waterfall_sample_0.png` — waterfall plot for instance 0
- `shap_force_sample_0.png` — force plot for instance 0
- `shap_dependence_active_users.png` — feature dependence plot
- `lime_explanation_0.png` through `lime_explanation_2.png`

### `explain_xgboost.py`

```bash
# First train the model
python train_xgboost.py

# Then explain it
python explain_xgboost.py
```

Same output structure as the Random Forest explainer.

---

## Additional SHAP Plot Types (Standalone Scripts)

### Waterfall Plot

Shows the contribution of each feature to a **single prediction**, starting
from the base value and adding each feature's SHAP contribution:

```
base_value = 0.05 (average anomaly probability)
+ active_users: +0.71
─────────────────────
final prediction = 0.76  ← anomaly probability for this instance
```

### Force Plot

A horizontal version of the waterfall plot. Features are shown as forces
pushing the prediction left (towards normal) or right (towards anomaly).

### Dependence Plot

Shows the relationship between a feature's **value** and its **SHAP value** for
all test instances. Reveals non-linear effects and interactions:

```
active_users value
0   20   40   60   80   100
 ●●●●●●●●●●●                 ← low values → low SHAP (normal)
              ●●●●●●●●●●●    ← high values → high SHAP (anomaly)
```

---

## Interpreting Results in This Project

Given the dataset has only one feature (`active_users`), the XAI outputs show:

- **High `active_users`** → positive SHAP value → predicts anomaly
- **Low `active_users`** → negative SHAP value → predicts normal
- Both abnormally high AND abnormally low user counts can be anomalies
  (Isolation Forest labels both extremes)

This makes intuitive sense: a cell with unusually many or unusually few active
users compared to normal hours is likely experiencing an issue.

---

## Performance Tips

| Tip | When to apply |
|---|---|
| Reduce `n_lime` to 1-2 | When LIME is too slow |
| Reduce `kernel_background_samples` to 20 | When Logistic Regression SHAP is too slow |
| Use `random_forest` or `xgboost` for fast SHAP | If speed is critical |
| Increase `n_lime` to 10+ | For thorough per-instance analysis |
