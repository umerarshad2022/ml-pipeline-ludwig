#!/usr/bin/env python3
"""
Random Forest Anomaly Detection for DU Cell Metrics
Lightweight CPU-friendly alternative to neural networks
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

# Configuration
DATA_PATH = 'data/sample_data.csv'
MODEL_OUTPUT_DIR = 'artifacts/random_forest_model'
RANDOM_STATE = 42

# Create output directory
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("  RANDOM FOREST ANOMALY DETECTION")
print("=" * 80)
print(f"Data: {DATA_PATH}")
print(f"Output: {MODEL_OUTPUT_DIR}")
print("=" * 80)

# Load data
print("\n📥 Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"   Columns: {list(df.columns)}")

# Auto-label anomalies using Isolation Forest if not already labeled
if 'is_anomaly' in df.columns:
    print(f"\n🏷️  Using existing labels...")
    existing_anomalies = df['is_anomaly'].sum()
    print(f"   Found {existing_anomalies} anomalies ({existing_anomalies/len(df)*100:.1f}%)")
else:
    print(f"\n🏷️  Auto-labeling with Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.1, random_state=RANDOM_STATE)
    df['is_anomaly'] = iso_forest.fit_predict(df[['active_users']])
    df['is_anomaly'] = (df['is_anomaly'] == -1).astype(int)
    print(f"✅ Labeled {df['is_anomaly'].sum()} anomalies ({df['is_anomaly'].sum()/len(df)*100:.1f}%)")

# Prepare features and target
feature_cols = ['active_users']  # You can add more features like throughput_dl, prb_used_dl, etc.
X = df[feature_cols]
y = df['is_anomaly']

print(f"\n📊 Dataset statistics:")
print(f"   Features: {feature_cols}")
print(f"   Normal samples: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"   Anomaly samples: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\n✂️  Train/test split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# Train Random Forest
print(f"\n🌲 Training Random Forest...")
start_time = datetime.now()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced',
    verbose=1
)

rf_model.fit(X_train, y_train)
training_time = (datetime.now() - start_time).total_seconds()
print(f"✅ Training completed in {training_time:.2f} seconds")

# Make predictions
print(f"\n🔮 Making predictions...")
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluate
print(f"\n📈 Model Performance:")
print("=" * 80)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"                 Predicted Normal  Predicted Anomaly")
print(f"Actual Normal         {cm[0,0]:6d}            {cm[0,1]:6d}")
print(f"Actual Anomaly        {cm[1,0]:6d}            {cm[1,1]:6d}")

# ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Average Precision
avg_precision = average_precision_score(y_test, y_pred_proba)
print(f"Average Precision: {avg_precision:.4f}")

# Feature importance
print(f"\n🎯 Feature Importance:")
for feature, importance in zip(feature_cols, rf_model.feature_importances_):
    print(f"   {feature}: {importance:.4f}")

# Save model
model_path = os.path.join(MODEL_OUTPUT_DIR, 'random_forest_model.pkl')
joblib.dump(rf_model, model_path)
print(f"\n💾 Model saved to: {model_path}")

# Save metrics
metrics = {
    'model': 'Random Forest',
    'training_time': training_time,
    'test_samples': len(X_test),
    'roc_auc': roc_auc,
    'average_precision': avg_precision,
    'confusion_matrix': cm.tolist()
}
metrics_path = os.path.join(MODEL_OUTPUT_DIR, 'metrics.txt')
with open(metrics_path, 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")
print(f"📊 Metrics saved to: {metrics_path}")

# Visualizations
print(f"\n📉 Generating visualizations...")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
print(f"   ✅ Confusion matrix saved")

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'roc_curve.png'), dpi=150)
print(f"   ✅ ROC curve saved")

# 3. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, linewidth=2, label=f'AP = {avg_precision:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Random Forest')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'precision_recall_curve.png'), dpi=150)
print(f"   ✅ Precision-recall curve saved")

print("\n" + "=" * 80)
print("  ✅ RANDOM FOREST TRAINING COMPLETE")
print("=" * 80)
print(f"\n📁 All artifacts saved to: {MODEL_OUTPUT_DIR}/")
print(f"   - Model: random_forest_model.pkl")
print(f"   - Metrics: metrics.txt")
print(f"   - Visualizations: *.png")
