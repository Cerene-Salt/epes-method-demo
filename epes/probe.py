import numpy as np
from numpy.polynomial import Polynomial

def fit_polynomial_coeffs(x_values, y_values, degree=5):
    """
    Fit a polynomial to (x, y) samples and return its coefficients.
    """
    poly = Polynomial.fit(x_values, y_values, deg=degree)
    return poly.convert().coef

def probe_model_function(model, feature_range, num_samples=1000, degree=5, fixed_features=None, X_ref=None):
    """
    Probe a model to extract its functional representation as polynomial coefficients,
    ensuring the probe input matches the model's expected feature count.

    Parameters
    ----------
    model : object
        Must have a .predict(X) method that accepts array-like input.
    feature_range : tuple
        (min_value, max_value) range to sample for the probing axis.
    num_samples : int
        Number of points to sample in the domain.
    degree : int
        Degree of polynomial to fit.
    fixed_features : list[int]
        Indices of features to vary (usually [0] for first feature).
    X_ref : np.ndarray
        Reference dataset (without target column) to get correct feature count.
    """
    if X_ref is None:
        raise ValueError("X_ref (reference dataset) must be provided to match feature count.")

    # Start with a baseline row: mean of each feature
    X_probe = np.tile(np.mean(X_ref, axis=0), (num_samples, 1))

    # Vary the chosen feature(s) along the given range
    for feat_idx in fixed_features or [0]:
        X_probe[:, feat_idx] = np.linspace(feature_range[0], feature_range[1], num_samples)

    # Predict with the same number of features the model was trained on
    y_samples = np.array(model.predict(X_probe))

    # Fit polynomial to the probe results
    return fit_polynomial_coeffs(
        np.linspace(feature_range[0], feature_range[1], num_samples),
        y_samples,
        degree=degree
    )

if __name__ == "__main__":
    # Quick self-test with a simple linear model
    from sklearn.linear_model import LinearRegression
    X = np.linspace(0, 10, 20).reshape(-1, 1)
    y = 2 * X.flatten() + 1
    model = LinearRegression().fit(X, y)

    coeffs = probe_model_function(
        model,
        feature_range=(0, 10),
        num_samples=50,
        degree=1,
        fixed_features=[0],
        X_ref=X
    )
    print("Fitted coefficients:", coeffs)
