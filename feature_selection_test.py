"""
Usage example for the featureselection package.
"""

import numpy as np
import matplotlib.pyplot as plt
from GRLT_analysis.feature_selection import forward_subset_selection, lasso_feature_selection

# Generate sample data
np.random.seed(42)
n_samples = 100
n_features = 10

# Create a dataset with only 3 relevant features
X = np.random.randn(n_samples, n_features)
true_coefs = np.zeros(n_features)
true_coefs[0] = 1.0  # Feature 0 is relevant
true_coefs[2] = -0.5  # Feature 2 is relevant
true_coefs[5] = 0.25  # Feature 5 is relevant

# Generate target with some noise
y = X @ true_coefs + np.random.randn(n_samples) * 0.1

# Forward selection
print("Running forward selection...")
selected_features, theta, preds, mse, r2 = forward_subset_selection(
    X, y, val_ratio=0.2, epsilon=1e-4, max_features=5
)

# Lasso selection
print("\nRunning lasso feature selection...")
lasso_features, coefs, intercept, preds, alpha, val_mse, r_sq = lasso_feature_selection(
    X, y, val_ratio=0.2, alpha_min=1e-4, alpha_max=1.0, n_alphas=20
)

print("\nTrue relevant features: [0, 2, 5]")
print(f"Forward selection found: {selected_features}")
print(f"Lasso selection found: {list(lasso_features)}")