import os
import joblib
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 0) Make output folder
os.makedirs("demo_files", exist_ok=True)

# 1) Build a toy dataset
X, y = make_classification(
    n_samples=200,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    random_state=42
)
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 5)])
df["target"] = y
df.to_csv("demo_files/test_dataset.csv", index=False)

# 2) Train two simple models
model1 = LogisticRegression(max_iter=1000, n_jobs=None)
model1.fit(X, y)

model2 = DecisionTreeClassifier(max_depth=3, random_state=42)
model2.fit(X, y)

# 3) Save models
joblib.dump(model1, "demo_files/model1.pkl")
joblib.dump(model2, "demo_files/model2.pkl")

# 4) Write a tiny README for testers
readme = """# Epe’s Method – Demo files

Use these files to test the live app:
https://epes-method-demo-26yqqnr3rzhmhmktcrz7dd.streamlit.app

Files:
- model1.pkl — Logistic Regression
- model2.pkl — Decision Tree (max_depth=3)
- test_dataset.csv — 4 features + target (last column)

How to use:
1) Open the app link above
2) Upload model1.pkl as Model 1
3) Upload model2.pkl as Model 2
4) Upload test_dataset.csv as Dataset
5) Review metrics (ϝ, δϝ, ϝ*) and the plot
"""
with open("demo_files/README_demo.md", "w", encoding="utf-8") as f:
    f.write(readme)

print("✅ Created: demo_files/test_dataset.csv, model1.pkl, model2.pkl, README_demo.md")
