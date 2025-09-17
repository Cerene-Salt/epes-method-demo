import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from epes.probe import probe_model_function
from epes.metrics import structural_divergence, rate_divergence, fusion_metric

# 1. Create a toy dataset
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

# 2. Train two different models
model1 = LogisticRegression().fit(X, y)
model2 = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X, y)

# 3. Probe both models along the first feature axis, keeping second feature fixed at 0
coeffs1 = probe_model_function(
    model1,
    input_domain=(X[:, 0].min(), X[:, 0].max()),
    num_samples=200,
    degree=5,
    fixed_features=[0]
)

coeffs2 = probe_model_function(
    model2,
    input_domain=(X[:, 0].min(), X[:, 0].max()),
    num_samples=200,
    degree=5,
    fixed_features=[0]
)

# 4. Compute divergences
static_div = structural_divergence(coeffs1, coeffs2, norm_type='l2')

# Derivative coefficients: drop the constant term and multiply by degree index
f_prime_coeffs = np.array([i * coeffs1[i] for i in range(1, len(coeffs1))])
g_prime_coeffs = np.array([i * coeffs2[i] for i in range(1, len(coeffs2))])

rate_div = rate_divergence(f_prime_coeffs, g_prime_coeffs, norm_type='l2')

fusion = fusion_metric(static_div, rate_div)

# 5. Print results
print("Static divergence (ϝ):", static_div)
print("Rate divergence (δϝ):", rate_div)
print("Fusion metric (ϝ*):", fusion)

# --- Step 4: Visualization ---


# Create a dense set of probe points for plotting
x_plot = np.linspace(X[:, 0].min(), X[:, 0].max(), 300)

# Keep second feature fixed at 0 (same as in probing)
X_plot_2d = np.column_stack([x_plot, np.zeros_like(x_plot)])

# Get model predictions
y_pred1 = model1.predict(X_plot_2d)
y_pred2 = model2.predict(X_plot_2d)

# Plot decision curves
plt.figure(figsize=(8, 5))
plt.plot(x_plot, y_pred1, label="Model 1 (Logistic Regression)", color="blue", linewidth=2)
plt.plot(x_plot, y_pred2, label="Model 2 (Decision Tree)", color="red", linewidth=2, linestyle="--")

# Annotate metrics on the plot
plt.title("Model Comparison Along Feature 1 (Feature 2 fixed at 0)")
plt.xlabel("Feature 1 value")
plt.ylabel("Predicted Class")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)

# Show metrics in the corner
metrics_text = (
    f"ϝ (Static): {static_div:.3f}\n"
    f"δϝ (Rate): {rate_div:.3f}\n"
    f"ϝ* (Fusion): {fusion:.3f}"
)
plt.text(0.02, 0.95, metrics_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.show()

# --- Step 5: Full 2D Decision Boundary Visualization ---


# Create a mesh grid over the feature space
h = 0.02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict over the grid for each model
Z1 = model1.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z2 = model2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Define color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ['#FF0000', '#0000FF']

# Plot side-by-side decision boundaries
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Model 1
axes[0].contourf(xx, yy, Z1, cmap=cmap_light, alpha=0.6)
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor='k', s=20)
axes[0].set_title("Model 1: Logistic Regression")
axes[0].set_xlim(xx.min(), xx.max())
axes[0].set_ylim(yy.min(), yy.max())

# Model 2
axes[1].contourf(xx, yy, Z2, cmap=cmap_light, alpha=0.6)
axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor='k', s=20)
axes[1].set_title("Model 2: Decision Tree")
axes[1].set_xlim(xx.min(), xx.max())
axes[1].set_ylim(yy.min(), yy.max())

plt.suptitle("Full 2D Decision Boundaries")
plt.tight_layout()
plt.show()

