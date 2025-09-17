import numpy as np
from numpy.polynomial import Polynomial

def fit_polynomial_coeffs(x_values, y_values, degree=5):
    """
    Fit a polynomial to (x, y) samples and return its coefficients.
    """
    poly = Polynomial.fit(x_values, y_values, deg=degree)
    return poly.convert().coef

def probe_model_function(model, input_domain, num_samples=1000, degree=5, fixed_features=None):
    """
    Probe a model to extract its functional representation as polynomial coefficients.

    Parameters
    ----------
    model : object
        Must have a .predict(X) method that accepts array-like input.
    input_domain : tuple
        (min_x, max_x) range to sample for the probing axis.
    num_samples : int
        Number of points to sample in the domain.
    degree : int
        Degree of polynomial to fit.
    fixed_features : list or np.ndarray, optional
        Values for the other features to keep constant while probing.
        Example: fixed_features=[0] for a 2D model means vary feature 0, keep feature 1 at 0.
    """
    x_samples = np.linspace(input_domain[0], input_domain[1], num_samples)

    if fixed_features is not None:
        fixed_array = np.tile(fixed_features, (num_samples, 1))
        X = np.column_stack([x_samples, fixed_array])
    else:
        X = x_samples.reshape(-1, 1)

    y_samples = np.array(model.predict(X))
    return fit_polynomial_coeffs(x_samples, y_samples, degree=degree)

if __name__ == "__main__":
    # Quick self-test with a simple linear model
    from sklearn.linear_model import LinearRegression
    X = np.linspace(0, 10, 20).reshape(-1, 1)
    y = 2 * X.flatten() + 1
    model = LinearRegression().fit(X, y)

    coeffs = probe_model_function(model, input_domain=(0, 10), num_samples=50, degree=1)
    print("Fitted coefficients:", coeffs)
