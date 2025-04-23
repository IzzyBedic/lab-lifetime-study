"""
Utility functions for feature selection.
"""

import numpy as np

def mse(y_true, y_pred):
    """Computes Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

def r_squared(y_true, y_pred):
    """Computes R_sq"""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot

def train_val_split(X, y, val_ratio=0.2):
    """Splits data into training and validation sets"""
    n = X.shape[0]
    idx = np.random.permutation(n)
    split = int(n * (1 - val_ratio))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

def standardize(X_train, X_val):
    """Standardize based on training set stats."""
    X_train = X_train.astype(float)
    X_val = X_val.astype(float)

    print("X_train shape:", X_train.shape, "X_val shape:", X_val.shape)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # Avoid divide-by-zero

    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std

    return X_train_std, X_val_std, mean, std

def standardize_old(X_train, X_val):
    """Standardize based on training set stats."""
    print("X_train shape:", X_train.shape)
    print("Any NaNs?", np.isnan(X_train).any())
    print("Any Infs?", np.isinf(X_train).any())
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # Avoid divide-by-zero
    return (X_train - mean) / std, (X_val - mean) / std, mean, std

def standardize_new(X_train, X_val):
    """Standardize based on training set stats."""
    # Filter out constant columns (columns with zero variance)
    print(X_train.shape, X_val.shape)
    non_constant_columns = X_train.std(axis=0) != 0  # Columns where std != 0
    X_train_filtered = X_train[:, non_constant_columns]
    X_val_filtered = X_val[:, non_constant_columns]

    print(X_train_filtered.shape, X_val_filtered.shape)  # Check filtered shapes

    # Compute the mean and std of the filtered columns
    mean = X_train_filtered.mean(axis=0)
    std = X_train_filtered.std(axis=0)

    # Handle any remaining columns with zero variance
    std[std == 0] = 1.0  # Avoid divide-by-zero

    # Return standardized data
    return (X_train_filtered - mean) / std, (X_val_filtered - mean) / std, mean, std


def one_hot_encode_np(X, categorical_columns, original_column_names):
    """
    Perform one-hot encoding on specified categorical columns.

    Parameters:
    X : np.ndarray
        The input feature matrix (n_samples, n_features).
    categorical_columns : list
        List of indices specifying which columns in X are categorical.
    original_column_names : list
        List of feature names corresponding to columns in X.

    Returns:
    np.ndarray, list
        The input data X with one-hot encoded categorical columns and
        a list of new category names.
    """
    n_samples = X.shape[0]

    # Create lists to store encoded columns and their names
    encoded_columns = []
    new_category_names = []

    # Process each column
    for col, column_name in enumerate(original_column_names):
        if col in categorical_columns:
            # Get unique categories in this column
            categories = np.unique(X[:, col])

            # Create one-hot encoded columns for this feature
            for category in categories:
                # Create a column with 1s where the category matches
                encoded_col = (X[:, col] == category).astype(int).reshape(-1, 1)
                encoded_columns.append(encoded_col)
                new_category_names.append(f'{column_name}_Category_{category}')
        else:
            # For non-categorical columns, just keep the original
            encoded_columns.append(X[:, col].reshape(-1, 1))
            new_category_names.append(column_name)

    # Combine all columns into a single matrix
    X_encoded = np.hstack(encoded_columns)

    return X_encoded, new_category_names

def fit_linear_regression(X, y):
    """Fit linear regression using normal equations (with intercept)."""
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    try:
        theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
    except np.linalg.LinAlgError:
        return None, None, np.inf
    predictions = X_bias @ theta
    return theta, predictions, mse(y, predictions)