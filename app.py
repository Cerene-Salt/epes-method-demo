
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from epes.probe import probe_model_function
from epes.metrics import structural_divergence, rate_divergence, fusion_metric

st.set_page_config(page_title="Epe’s Method Audit", layout="centered")
st.title("🔍 Epe’s Method – Model Comparison Tool")

st.markdown("""
Upload two trained scikit-learn models and a dataset to compare their behavior using custom divergence metrics.
""")

# Upload section
model1_file = st.file_uploader("📁 Upload Model 1 (.pkl)", type=["pkl"])
model2_file = st.file_uploader("📁 Upload Model 2 (.pkl)", type=["pkl"])
data_file = st.file_uploader("📄 Upload Dataset (.csv)", type=["csv"])

if model1_file and model2_file and data_file:
    # Load data
    df = pd.read_csv(data_file)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Load models
    model1 = joblib.load(model1_file)
    model2 = joblib.load(model2_file)

    # Probe both models
    coeffs1 = probe_model_function(model1, (X[:, 0].min(), X[:, 0].max()), num_samples=200, degree=5, fixed_features=[0])
    coeffs2 = probe_model_function(model2, (X[:, 0].min(), X[:, 0].max()), num_samples=200, degree=5, fixed_features=[0])

    # Compute metrics
    static_div = structural_divergence(coeffs1, coeffs2)
    f_prime_coeffs = np.array([i * coeffs1[i] for i in range(1, len(coeffs1))])
    g_prime_coeffs = np.array([i * coeffs2[i] for i in range(1, len(coeffs2))])
    rate_div = rate_divergence(f_prime_coeffs, g_prime_coeffs)
    fusion = fusion_metric(static_div, rate_div)

    # Display metrics
    st.subheader("📊 Divergence Metrics")
    st.metric("Static Divergence (ϝ)", f"{static_div:.4f}")
    st.metric("Rate Divergence (δϝ)", f"{rate_div:.4f}")
    st.metric("Fusion Metric (ϝ*)", f"{fusion:.4f}")

    # Plot 1D slice
    st.subheader("📈 1D Slice Plot")
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
    y1_vals = np.polyval(coeffs1[::-1], x_vals)
    y2_vals = np.polyval(coeffs2[::-1], x_vals)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y1_vals, label="Model 1", color="blue")
    ax.plot(x_vals, y2_vals, label="Model 2", color="orange")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Predicted Output")
    ax.legend()
    st.pyplot(fig)
