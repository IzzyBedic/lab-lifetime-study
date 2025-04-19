"""
Forward feature selection implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import mse, r_squared, train_val_split, standardize, fit_linear_regression

def forward_subset_selection(X, y, val_ratio=0.2, epsilon=1e-4, max_features=None, verbose=True, plot=True, scale=True):
    """
    Forward subset selection using validation MSE.

    Parameters:
    - X: (n_samples, n_features) input data
    - y: target values
    - val_ratio: proportion of data to use for validation
    - epsilon: minimum improvement in validation MSE required to add a feature
    - max_features: maximum number of features to select (None = no limit)
    - verbose: print progress
    - plot: show MSE and coefficient plots
    - scale: standardize features before selection

    Returns:
    - selected_features: list of selected feature indices
    - theta: final model parameters (with intercept)
    - preds: predictions on the full dataset
    - final_mse: MSE on the full dataset
    - final_r2: R^2 on the full dataset
    """

    X_train, X_val, y_train, y_val = train_val_split(X, y, val_ratio)

    if scale:
        X_train, X_val, mean, std = standardize(X_train, X_val)
    else:
        mean, std = np.zeros(X.shape[1]), np.ones(X.shape[1])

    n_samples, n_features = X.shape
    remaining_features = set(range(n_features))
    selected_features = []
    best_val_mse = np.inf
    train_mse_history = []
    val_mse_history = []
    r_squared_history = []

    while remaining_features and (max_features is None or len(selected_features) < max_features):
        best_candidate = None
        best_candidate_mse = np.inf
        best_train_mse = None

        for feature in remaining_features:
            trial_features = selected_features + [feature]
            theta, train_preds, train_mse = fit_linear_regression(X_train[:, trial_features], y_train)
            _, val_preds, val_mse = fit_linear_regression(X_val[:, trial_features], y_val)

            if val_mse < best_candidate_mse:
                best_candidate_mse = val_mse
                best_candidate = feature
                best_train_mse = train_mse

        improvement = best_val_mse - best_candidate_mse

        if improvement > epsilon:
            selected_features.append(best_candidate)
            remaining_features.remove(best_candidate)
            best_val_mse = best_candidate_mse
            train_mse_history.append(best_train_mse)
            val_mse_history.append(best_candidate_mse)
            r_squared_history.append(r_squared(y_val, val_preds))

            if verbose:
                print(f"Added feature {best_candidate}: val MSE = {best_candidate_mse:.5f}, improvement = {improvement:.5f} (total: {selected_features})")
        else:
            if verbose:
                print("No significant improvement. Stopping.")
            break

        if max_features is not None and len(selected_features) >= max_features:
            if verbose:
                print(f"Reached max_features = {max_features}. Stopping.")
            break

    # Final model using standardized full dataset
    X_scaled = (X - mean) / std
    X_final = X_scaled[:, selected_features]
    theta, preds, final_mse = fit_linear_regression(X_final, y)
    final_r2 = r_squared(y, preds)

    # Convert coefficients back to original scale
    unscaled_coefs = np.zeros(len(selected_features))
    for i, feat_idx in enumerate(selected_features):
        unscaled_coefs[i] = theta[i + 1] / std[feat_idx]

    unscaled_intercept = theta[0] - np.sum([
        (theta[i + 1] * mean[feat_idx]) / std[feat_idx]
        for i, feat_idx in enumerate(selected_features)
    ])

    if verbose:
        # Output best model coefficients
        print(f"\nSelected Features: {selected_features}")
        print(f"Coefficients: {unscaled_coefs}")
        print(f"Best Intercept: {unscaled_intercept}")
        print(f"Best Validation MSE: {best_val_mse:.4f}")
        print(f"Best R-squared: {final_r2:.4f}")

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_mse_history)+1), train_mse_history, label="Train MSE", marker='o')
        plt.plot(range(1, len(val_mse_history)+1), val_mse_history, label="Validation MSE", marker='s')
        plt.xlabel("Number of Selected Features")
        plt.ylabel("Mean Squared Error")
        plt.title("Forward Subset Selection Performance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_mse_history)+1), r_squared_history, label="R_sq", marker='o')
        plt.xlabel("Number of Selected Features")
        plt.ylabel("R_sq")
        plt.title("Forward Subset Selection Performance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return selected_features, theta, preds, final_mse, final_r2