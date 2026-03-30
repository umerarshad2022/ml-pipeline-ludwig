X_plot = X_test if len(X_test) <= 200 else X_test.iloc[:200]
sv_plot = sv if len(sv) <= 200 else sv[:200]

# Normalize SHAP output shape and enforce SHAP/X alignment
if isinstance(sv_plot, list):
    cls_idx = 1 if len(sv_plot) > 1 else 0
    sv_plot = sv_plot[cls_idx]

sv_plot = np.asarray(sv_plot)

if sv_plot.ndim == 3:
    cls_idx = 1 if sv_plot.shape[2] > 1 else 0
    sv_plot = sv_plot[:, :, cls_idx]
elif sv_plot.ndim == 1:
    sv_plot = sv_plot.reshape(-1, 1)

if not isinstance(X_plot, pd.DataFrame):
    X_plot = pd.DataFrame(X_plot, columns=feature_cols)
else:
    X_plot = X_plot.copy()
X_plot = X_plot.reset_index(drop=True)

# Align row counts to avoid SHAP assertion: features.shape[0] == len(shaps)
n_rows = min(len(X_plot), len(sv_plot))
if n_rows == 0:
    raise ValueError("No rows available for SHAP plotting after alignment.")
if len(X_plot) != len(sv_plot):
    logger.warning(
        "SHAP/X row mismatch detected (X=%d, SHAP=%d). Truncating both to %d rows.",
        len(X_plot), len(sv_plot), n_rows
    )
X_plot = X_plot.iloc[:n_rows].copy()
sv_plot = sv_plot[:n_rows]

# Align feature counts defensively
n_features = min(X_plot.shape[1], sv_plot.shape[1])
if X_plot.shape[1] != sv_plot.shape[1]:
    logger.warning(
        "SHAP/X feature mismatch detected (X=%d, SHAP=%d). Truncating both to %d features.",
        X_plot.shape[1], sv_plot.shape[1], n_features
    )
X_plot = X_plot.iloc[:, :n_features]
sv_plot = sv_plot[:, :n_features]
plot_feature_cols = list(X_plot.columns)

# Summary beeswarm
fig, ax = plt.subplots(figsize=(8, max(4, len(plot_feature_cols) * 0.6)))
shap.summary_plot(sv_plot, X_plot, feature_names=plot_feature_cols, show=False)
path = os.path.join(out_dir, "shap_summary.png")
plt.savefig(path, bbox_inches="tight", dpi=150)
plt.close()
created.append(path)
logger.info(f"   💾 {path}")

# Bar chart of mean |SHAP|
fig, ax = plt.subplots(figsize=(8, max(4, len(plot_feature_cols) * 0.6)))
shap.summary_plot(sv_plot, X_plot, feature_names=plot_feature_cols, plot_type="bar", show=False)
path = os.path.join(out_dir, "shap_importance.png")
plt.savefig(path, bbox_inches="tight", dpi=150)
plt.close()
created.append(path)
logger.info(f"   💾 {path}")