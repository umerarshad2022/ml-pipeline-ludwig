# Dataset

**File:** `data/sample_data.csv`

---

## Overview

| Property | Value |
|---|---|
| **Format** | CSV (comma-separated values) |
| **Total rows** | 15 000 |
| **Columns** | 8 |
| **Domain** | Open RAN DU (Distributed Unit) cell metrics — Telecom |
| **Source** | Synthetic — generated to simulate realistic Open RAN behaviour |
| **Anomaly rate** | ~4.9% (733 anomalies out of 15 000 rows) |

---

## Column Descriptions

| Column | Type | Unit | Description |
|---|---|---|---|
| `timestamp` | string (ISO-8601) | — | Measurement timestamp (e.g. `2026-01-17T11:47:14.568474`) |
| `cell_id` | string | — | Cell identifier (e.g. `cell_001`, `cell_002`, `cell_003`) |
| `throughput_dl` | float | Mbps | Downlink throughput — data transferred from network to UE per second |
| `throughput_ul` | float | Mbps | Uplink throughput — data transferred from UE to network per second |
| `prb_used_dl` | float | % | Downlink Physical Resource Block utilisation (0–100%) |
| `prb_used_ul` | float | % | Uplink Physical Resource Block utilisation (0–100%) |
| `active_users` | integer | count | Number of User Equipment (UE) devices actively connected to this cell |
| `is_anomaly` | integer | 0 or 1 | Ground-truth label: **0 = Normal**, **1 = Anomaly** |

---

## Glossary

> **New to telecom terminology?** Here are plain-English explanations:

| Term | Explanation |
|---|---|
| **Open RAN** | An open, standardised Radio Access Network architecture. Instead of proprietary hardware, it uses standard interfaces that allow different vendors' equipment to interoperate. |
| **DU (Distributed Unit)** | A component of the O-RAN architecture responsible for real-time radio signal processing. It connects radio antennas to the network core. |
| **Cell** | A geographic area covered by one radio antenna. In this dataset, `cell_001`, `cell_002`, `cell_003` represent three cells measured over time. |
| **UE (User Equipment)** | Any device that connects to the network — smartphone, laptop, IoT sensor, etc. |
| **Throughput (DL/UL)** | The actual amount of data transferred. "DL" = downlink (network → device), "UL" = uplink (device → network). Measured in megabits per second (Mbps). |
| **PRB (Physical Resource Block)** | The smallest unit of radio frequency resource that can be allocated. PRB utilisation measures how busy the radio spectrum is (0–100%). |
| **Anomaly** | In this context, a measurement that deviates significantly from normal operating conditions — could indicate equipment failure, congestion, interference, or a cyberattack. |

---

## Class Distribution

```
Normal  (is_anomaly = 0):  14 267 rows  (95.1%)
Anomaly (is_anomaly = 1):    733 rows   ( 4.9%)
```

This is a **highly imbalanced** dataset. A naive model that always predicts
"Normal" achieves 95.1% accuracy — which is why accuracy alone is a poor metric.
Use **ROC-AUC**, **Recall (anomaly class)**, and **F1-score** instead.

---

## Feature Statistics (approximate)

| Feature | Normal min | Normal max | Anomaly min | Anomaly max |
|---|---|---|---|---|
| `throughput_dl` (Mbps) | ~10 | ~200 | ~5 | ~200 |
| `throughput_ul` (Mbps) | ~5 | ~100 | ~2 | ~100 |
| `prb_used_dl` (%) | ~20 | ~100 | ~10 | ~100 |
| `prb_used_ul` (%) | ~20 | ~100 | ~10 | ~100 |
| `active_users` | ~10 | ~90 | ~0 | ~100+ |

> Anomalies in this dataset are characterised by abnormal `active_users` values
> (either very low or very high), combined with degraded throughput or very high/
> low PRB utilisation.

---

## Temporal Structure

Measurements are taken approximately **every minute** for each cell.
The dataset covers multiple cells over several thousand time steps.

The data is **not sorted** — timestamps from different cells are interleaved.
The models in this project treat each row independently (no time-series modelling).

---

## Using a Different Dataset

To swap in your own data:

### Step 1 — Match the column format

Your CSV must have:
- At least one numeric feature column.
- An `is_anomaly` column with values 0 (normal) and 1 (anomaly), **OR** you can
  skip the label column and let `UnsupervisedLabelingHandler` create it.

### Step 2 — Update the YAML config

```yaml
- name: load-data
  type: data_loader
  config:
    source:
      type: csv
      path: data/your_data.csv     # ← change this

    preprocessing:
      drop_columns:
        - your_id_column           # Drop non-feature columns
        - your_timestamp_column
```

### Step 3 — Update feature names

```yaml
sklearn_model:
  input_features:
    - your_feature_1              # ← list your numeric features
    - your_feature_2
  output_feature: your_label_col  # ← name of your label column
```

### Step 4 — Adjust unsupervised labelling (if no labels)

If your dataset has no labels:

```yaml
- name: auto-label
  type: unsupervised_labeling
  config:
    method: isolation_forest
    feature: your_most_informative_feature
    contamination: 0.05           # ← adjust based on expected anomaly rate
    output_column: is_anomaly
```

### Notes on Data Size

| Dataset size | Recommendation |
|---|---|
| < 1 000 rows | May not have enough data for reliable training; results will be noisy |
| 1 000 – 50 000 rows | Ideal for this pipeline |
| > 50 000 rows | Works fine; SHAP/LIME may be slow — reduce `n_lime` and `sample_size` |
| > 1 000 000 rows | Use `test_size: 0.05` or sample a subset before training |

---

## The `du_cell_report.csv` File

There is a second data file: `data/du_cell_report.csv`. This appears to be an
alternative or extended version of the dataset. It has the same column structure
as `sample_data.csv` and can be used as an alternative input by changing the
`path` in your YAML config.
