"""
Forward feature selection implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from .utils import mse, r_squared, train_val_split, standardize, fit_linear_regression

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

"""
Lasso feature selection implementation.
"""

class LassoNumpy:
    """
    Lasso regression implementation using only NumPy.
    Uses coordinate descent optimization algorithm.
    """
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0

    def _soft_threshold(self, x, lambda_):
        """Soft thresholding operator for coordinate descent"""
        if x > lambda_:
            return x - lambda_
        elif x < -lambda_:
            return x + lambda_
        else:
            return 0

    def fit(self, X, y):
        """
        Fit Lasso regression model using coordinate descent.

        Parameters:
        X : numpy array of shape (n_samples, n_features)
        y : numpy array of shape (n_samples,)

        Returns:
        self
        """
        n_samples, n_features = X.shape

        # Initialize coefficients
        if self.coef_ is None:
            self.coef_ = np.zeros(n_features)

        # Center y
        y_mean = np.mean(y)
        y_centered = y - y_mean

        # Compute X^T * X diagonals for faster updates
        # (used in the coordinate descent update)
        X_squared = np.sum(X ** 2, axis=0)

        # Coordinate descent
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()

            # Update each coefficient
            for j in range(n_features):
                if X_squared[j] == 0:
                    continue

                # Compute residual without feature j
                y_partial = y_centered - np.dot(X, self.coef_) + self.coef_[j] * X[:, j]

                # Compute the coordinate update
                z_j = np.dot(X[:, j], y_partial)

                # Apply soft thresholding
                self.coef_[j] = self._soft_threshold(z_j, self.alpha) / X_squared[j]

            # Check for convergence
            delta_coef = np.sum(np.abs(self.coef_ - coef_old))
            if delta_coef < self.tol:
                break

        # Set intercept to account for centering of y
        self.intercept_ = y_mean

        return self

    def predict(self, X):
        """Make predictions using the fitted model"""
        return np.dot(X, self.coef_) + self.intercept_


def lasso_regression(X_train, y_train, X_val, y_val, alpha):
    """Fit a Lasso model using NumPy and return the coefficients and MSE."""
    model = LassoNumpy(alpha=alpha)
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_mse = np.mean((y_train - train_preds) ** 2)
    val_mse = np.mean((y_val - val_preds) ** 2)
    return model.coef_, model.intercept_, train_mse, val_mse, train_preds, val_preds


def lasso_feature_selection(X, y, val_ratio=0.2, alpha_min=1e-4, alpha_max=1e2, n_alphas=100, scale=True, verbose=True, plot=True):
    """
    Subset feature selection using lasso regression with NumPy implementation.

    Parameters:
    - X: (n_samples, n_features) input data
    - y: target values
    - val_ratio: proportion of data to use for validation
    - alpha_min: minimum alpha value
    - alpha_max: maximum alpha value
    - n_alphas: number of alphas
    - scale: whether to standardize features
    - verbose: print progress
    - plot: show MSE and coefficient plots

    Returns:
    - selected_features: list of selected feature indices
    - best_coefs: final model parameters
    - best_intercept: final intercept
    - best_preds: predictions using best model
    - best_alpha: alpha used for best results
    - best_val_mse: MSE on the validation dataset
    - best_r_sq: R_sq on the validation dataset
    """
    # Split into training and validation sets
    n = X.shape[0]
    idx = np.random.permutation(n)
    split = int(n * (1 - val_ratio))
    X_train, X_val = X[idx[:split]], X[idx[split:]]
    y_train, y_val = y[idx[:split]], y[idx[split:]]

    # Standardize features
    if scale:
        train_mean = np.mean(X_train, axis=0)
        train_std = np.std(X_train, axis=0)
        train_std[train_std == 0] = 1.0  # Avoid divide-by-zero

        X_train = (X_train - train_mean) / train_std
        X_val = (X_val - train_mean) / train_std

    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alphas)
    best_alpha = None
    best_val_mse = np.inf
    best_coefs = None
    best_intercept = None
    best_preds = None
    selected_features = None
    best_r_sq = -np.inf

    train_mse_history = []
    val_mse_history = []
    r_sq_history = []

    # Loop through the alphas to find the best one
    for alpha in alphas:
        coefs, intercept, train_mse, val_mse, train_preds, val_preds = lasso_regression(
            X_train, y_train, X_val, y_val, alpha
        )

        # Calculate R-squared
        ss_res = np.sum((y_val - val_preds)**2)
        ss_tot = np.sum((y_val - np.mean(y_val))**2)
        r_sq = 1 - ss_res / ss_tot

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_r_sq = r_sq
            best_alpha = alpha

    if best_alpha is None:
        print("Lasso failed: Never found suitable features")
        return None