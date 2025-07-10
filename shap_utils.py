import shap
import numpy as np

def compute_tree_shap_local(model, features, index):
    """
    Compute SHAP values for a single sample (row) using a tree-based model (e.g., Isolation Forest).
    Returns a list of SHAP values for the selected row.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        # For Isolation Forest, shap_values is usually a 2D array (samples x features)
        return shap_values[index]
    except Exception as e:
        # Fallback: return zeros if SHAP fails
        return [0.0] * features.shape[1]

def compute_mlp_shap_local(model, X_scaled, index):
    """
    Compute SHAP values for a single sample (row) using a neural network (MLP) model.
    Uses KernelExplainer for general models.
    """
    try:
        # Use a small background set for efficiency
        background = X_scaled[np.random.choice(X_scaled.shape[0], min(50, X_scaled.shape[0]), replace=False)]
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_scaled[index:index+1], nsamples=100)
        # shap_values is a list for multi-output, take the first if so
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        return shap_values[0]
    except Exception as e:
        return [0.0] * X_scaled.shape[1]

def compute_mlp_shap_global(model, X_scaled):
    """
    Compute global SHAP values for an MLP model (mean absolute SHAP value per feature).
    """
    try:
        background = X_scaled[np.random.choice(X_scaled.shape[0], min(50, X_scaled.shape[0]), replace=False)]
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_scaled, nsamples=100)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        global_importance = np.abs(shap_values).mean(axis=0)
        return global_importance
    except Exception as e:
        return np.zeros(X_scaled.shape[1])
