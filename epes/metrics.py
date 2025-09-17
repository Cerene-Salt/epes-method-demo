import numpy as np
from numpy.linalg import norm

def _pad_to_same_length(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    max_len = max(len(a), len(b))
    a_pad = np.pad(a, (0, max_len - len(a)), mode="constant")
    b_pad = np.pad(b, (0, max_len - len(b)), mode="constant")
    return a_pad, b_pad

def structural_divergence(f_coeffs, g_coeffs, norm_type='l2'):
    """
    ϝ — static structural divergence between coefficient vectors.

    norm_type:
      - 'l1'  -> L1 norm of coefficient difference
      - 'l2'  -> L2 norm of coefficient difference
      - 'l12' -> ratio ||diff||_1 / ||diff||_2 (if diff is nonzero; else 0.0)
    """
    f_pad, g_pad = _pad_to_same_length(f_coeffs, g_coeffs)
    diff = f_pad - g_pad

    if norm_type == 'l1':
        return norm(diff, 1)
    elif norm_type == 'l2':
        return norm(diff, 2)
    elif norm_type == 'l12':
        l2 = norm(diff, 2)
        return norm(diff, 1) / l2 if l2 != 0 else 0.0
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

def rate_divergence(f_prime_coeffs, g_prime_coeffs, norm_type='l2'):
    """
    δϝ — rate divergence between derivative coefficient vectors.
    Mirrors structural_divergence but applied to derivative coefficients.
    """
    return structural_divergence(f_prime_coeffs, g_prime_coeffs, norm_type=norm_type)

def fusion_metric(static_div, rate_div, alpha=0.7, beta=0.3):
    """
    ϝ* — fusion metric combining static (ϝ) and rate (δϝ).
    Default weights favor static structure slightly.
    """
    return alpha * static_div + beta * rate_div

if __name__ == "__main__":
    import numpy as np
    f = np.array([1, 2, 3])
    g = np.array([1, 2, 4])
    print(structural_divergence(f, g))
    print(rate_divergence(f, g))
    print(fusion_metric(1.0, 0.5)) 
from epes.metrics import fusion_metric
print(fusion_metric(1.0, 0.5))
