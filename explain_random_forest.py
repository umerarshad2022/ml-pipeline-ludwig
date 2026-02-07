#!/usr/bin/env python3
"""
XAI Analysis for Random Forest Model using SHAP and LIME
Generates explainability visualizations for anomaly detection model
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular

# Configuration
DATA_PATH = 'data/sample_data.csv'
MODEL_PATH = 'artifacts/random_forest_model/random_forest_model.pkl'
OUTPUT_DIR = 'artifacts/random_forest_model/xai'
RANDOM_STATE = 42

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("  RANDOM FOREST - XAI ANALYSIS (SHAP & LIME)")
print("=" * 80)
print(f"Model: {MODEL_PATH}")
print(f"Data: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 80)

# Load data
print("\n📥 Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df)} rows")

# Prepare features
feature_cols = ['active_users']
X = df[feature_cols]
y = df['is_anomaly'] if 'is_anomaly' in df.columns else None

# Load model
print(f"\n📦 Loading Random Forest model...")
rf_model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded successfully")
print(f"   Features: {feature_cols}")
print(f"   Classes: {rf_model.classes_}")

# Sample data for faster computation
sample_size = min(1000, len(X))
X_sample = X.sample(n=sample_size, random_state=RANDOM_STATE)
print(f"\n📊 Using {sample_size} samples for XAI analysis")

#============================================================================
# SHAP ANALYSIS
#============================================================================
print("\n" + "=" * 80)
print("  SHAP (SHapley Additive exPlanations) ANALYSIS")
print("=" * 80)

print("\n🔍 Creating SHAP explainer...")
# Use TreeExplainer for tree-based models (much faster)
explainer = shap.TreeExplainer(rf_model)
print("✅ SHAP explainer created")

print("\n🔮 Computing SHAP values...")
shap_values = explainer(X_sample)
print(f"✅ SHAP values computed for {len(X_sample)} samples")

print("\n📉 Generating SHAP visualizations...")

# 1. SHAP Summary Plot (Beeswarm)
plt.figure(figsize=(10, 6))
# For binary classification, use values for anomaly class (index 1)
shap.summary_plot(shap_values[:,:,1], X_sample, show=False)
plt.title('SHAP Summary Plot - Random Forest\nFeature Impact on Anomaly Prediction', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ SHAP summary plot saved")

# 2. SHAP Summary Plot (Bar)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[:,:,1], X_sample, plot_type="bar", show=False)
plt.title('SHAP Feature Importance - Random Forest\nAverage Impact on Anomaly Prediction', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ SHAP feature importance saved")

# 3. SHAP Waterfall Plot (for first anomaly prediction)
if y is not None and y.sum() > 0:
    # Find first anomaly in sample
    anomaly_indices = X_sample.index[X_sample.index.isin(df[df['is_anomaly']==1].index)]
    if len(anomaly_indices) > 0:
        anomaly_idx = list(X_sample.index).index(anomaly_indices[0])
        plt.figure(figsize=(10, 6))
        # For binary classification, select anomaly class (index 1)
        shap.waterfall_plot(shap_values[anomaly_idx,:,1], show=False)
        plt.title('SHAP Waterfall Plot - Anomaly Prediction\nHow Features Contribute to Anomaly Detection', fontsize=12, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'shap_waterfall_anomaly.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ SHAP waterfall plot (anomaly) saved")

# 4. SHAP Force Plot (interactive HTML) - First prediction
shap.force_plot(explainer.expected_value[1], 
                shap_values.values[0,:,1], 
                X_sample.iloc[0,:],
                matplotlib=True,
                show=False)
plt.title('SHAP Force Plot - First Prediction\nPush and Pull of Features', fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_force_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ SHAP force plot saved")

# 5. SHAP Dependence Plot
for feature in feature_cols:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature, shap_values.values[:,:,1], X_sample, show=False)
    plt.title(f'SHAP Dependence Plot - {feature}\nRelationship between Feature Value and Impact', fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'shap_dependence_{feature}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ SHAP dependence plot for {feature} saved")

#============================================================================
# LIME ANALYSIS
#============================================================================
print("\n" + "=" * 80)
print("  LIME (Local Interpretable Model-agnostic Explanations) ANALYSIS")
print("=" * 80)

print("\n🔍 Creating LIME explainer...")
lime_explainer = lime_tabular.LimeTabularExplainer(
    X_sample.values,
    feature_names=feature_cols,
    class_names=['Normal', 'Anomaly'],
    mode='classification',
    random_state=RANDOM_STATE
)
print("✅ LIME explainer created")

print("\n🔮 Generating LIME explanations...")

# Explain 3 different instances: normal, anomaly, and borderline
instances_to_explain = []

if y is not None:
    # Get one normal sample
    normal_idx = X_sample.index[X_sample.index.isin(df[df['is_anomaly']==0].index)]
    if len(normal_idx) > 0:
        instances_to_explain.append(('normal', list(X_sample.index).index(normal_idx[0])))
    
    # Get one anomaly sample
    anomaly_idx = X_sample.index[X_sample.index.isin(df[df['is_anomaly']==1].index)]
    if len(anomaly_idx) > 0:
        instances_to_explain.append(('anomaly', list(X_sample.index).index(anomaly_idx[0])))
else:
    # Just explain first 2 samples if no labels
    instances_to_explain = [('sample_0', 0), ('sample_1', 1)]

for instance_name, idx in instances_to_explain:
    print(f"\n   Explaining {instance_name} (index {idx})...")
    
    # Get LIME explanation
    exp = lime_explainer.explain_instance(
        X_sample.iloc[idx].values,
        rf_model.predict_proba,
        num_features=len(feature_cols),
        top_labels=2
    )
    
    # Save as matplotlib figure
    fig = exp.as_pyplot_figure(label=1)  # Label 1 for anomaly class
    plt.title(f'LIME Explanation - {instance_name.upper()}\nLocal Feature Importance for Anomaly Prediction', 
              fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'lime_explanation_{instance_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ LIME explanation for {instance_name} saved")

# Create comprehensive LIME visualization (multiple instances)
print("\n📊 Creating comprehensive LIME visualization...")
n_instances = min(5, len(X_sample))
fig, axes = plt.subplots(n_instances, 1, figsize=(12, 4*n_instances))
if n_instances == 1:
    axes = [axes]

for i in range(n_instances):
    exp = lime_explainer.explain_instance(
        X_sample.iloc[i].values,
        rf_model.predict_proba,
        num_features=len(feature_cols)
    )
    
    # Get feature importance
    exp_list = exp.as_list(label=1)
    features = [f[0] for f in exp_list]
    importances = [f[1] for f in exp_list]
    
    # Plot
    axes[i].barh(features, importances, color=['red' if x < 0 else 'green' for x in importances])
    axes[i].set_xlabel('Feature Importance')
    axes[i].set_title(f'Sample {i+1} - Predicted: {rf_model.predict([X_sample.iloc[i]])[0]}')
    axes[i].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    axes[i].grid(alpha=0.3)

plt.suptitle('LIME Explanations - Multiple Instances\nRandom Forest Anomaly Detection', 
             fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lime_multiple_instances.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Multiple instances LIME plot saved")

#============================================================================
# SUMMARY REPORT
#============================================================================
print("\n" + "=" * 80)
print("  GENERATING SUMMARY REPORT")
print("=" * 80)

summary_path = os.path.join(OUTPUT_DIR, 'xai_summary.txt')
with open(summary_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("  XAI ANALYSIS SUMMARY - RANDOM FOREST\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Model: Random Forest Classifier\n")
    f.write(f"Dataset: {DATA_PATH}\n")
    f.write(f"Samples analyzed: {sample_size}\n")
    f.write(f"Features: {feature_cols}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("  SHAP ANALYSIS RESULTS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("SHAP Feature Importance (Mean Absolute SHAP Values):\n")
    mean_shap = np.abs(shap_values.values[:,:,1]).mean(axis=0)
    for feature, importance in zip(feature_cols, mean_shap):
        f.write(f"  {feature}: {importance:.4f}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("  VISUALIZATIONS GENERATED\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("SHAP Plots:\n")
    f.write("  ✓ shap_summary_plot.png - Overall feature impact distribution\n")
    f.write("  ✓ shap_feature_importance.png - Average feature importance\n")
    f.write("  ✓ shap_waterfall_anomaly.png - Single anomaly prediction breakdown\n")
    f.write("  ✓ shap_force_plot.png - Push/pull visualization\n")
    f.write("  ✓ shap_dependence_*.png - Feature value vs impact relationship\n\n")
    
    f.write("LIME Plots:\n")
    f.write("  ✓ lime_explanation_*.png - Individual instance explanations\n")
    f.write("  ✓ lime_multiple_instances.png - Comparison across samples\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("  KEY INSIGHTS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("SHAP provides global model interpretability:\n")
    f.write("  - Shows which features are most important overall\n")
    f.write("  - Reveals how feature values affect predictions\n")
    f.write("  - Consistent across all predictions\n\n")
    
    f.write("LIME provides local instance interpretability:\n")
    f.write("  - Explains why THIS specific prediction was made\n")
    f.write("  - Creates simple linear approximation locally\n")
    f.write("  - Different explanations for different instances\n")

print(f"✅ Summary report saved: {summary_path}")

print("\n" + "=" * 80)
print("  ✅ XAI ANALYSIS COMPLETE - RANDOM FOREST")
print("=" * 80)
print(f"\n📁 All XAI artifacts saved to: {OUTPUT_DIR}/")
print("\nGenerated Files:")
print("  SHAP:")
print("    - shap_summary_plot.png")
print("    - shap_feature_importance.png")
print("    - shap_waterfall_anomaly.png")
print("    - shap_force_plot.png")
print("    - shap_dependence_*.png")
print("  LIME:")
print("    - lime_explanation_*.png")
print("    - lime_multiple_instances.png")
print("  REPORT:")
print("    - xai_summary.txt")
