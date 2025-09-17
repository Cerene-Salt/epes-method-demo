# Epe’s Method – Model Comparison & Audit Tool

## 📌 Overview
Epe’s Method is a modular Python package for quantitatively and visually comparing two machine learning models.  
It combines **custom divergence metrics**, **model probing**, and **visualization** to deliver both numeric and intuitive insights.

---

## 🚀 Features
- **Custom Metrics**: Structural Divergence (ϝ), Rate Divergence (δϝ), Fusion Metric (ϝ*)
- **Model Probing**: Polynomial fitting of model predictions along a chosen axis
- **Visualizations**:
  - 1D slice plots with metrics overlay
  - 2D decision boundary maps
- **Reusable Package Structure**: Editable install, modular code, example scripts

---

## 📂 Project Structure
 [1] DATA INPUT
    ↓
    Any dataset (synthetic or real-world)
    + Two trained models to compare

[2] PROBING (epes/probe.py)
    ↓
    - Select a probing axis (e.g., Feature 1)
    - Fix other features (e.g., Feature 2 = 0)
    - Sample model predictions along this axis
    - Fit polynomial → get coefficient vector

[3] METRICS (epes/metrics.py)
    ↓
    - Structural Divergence (ϝ): compares shape of functions
    - Rate Divergence (δϝ): compares derivatives
    - Fusion Metric (ϝ*): weighted combination of both

[4] RESULTS
    ↓
    - Numeric outputs: ϝ, δϝ, ϝ*
    - Interpretable differences between models

[5] VISUALIZATION
    ↓
    - 1D Slice Plot: prediction curves + metrics overlay
    - 2D Decision Boundaries: side-by-side classification maps

[6] INSIGHT
    ↓
    - Quantitative + visual understanding of model differences
    - Ready for reporting, auditing, or research

### Additional Files
Some `.py` files in the package (e.g., `utils.py`) are not used in the current demo but are included to illustrate potential extension points for future development.

