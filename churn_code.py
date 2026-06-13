# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# ==========================================

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD ORIGINAL DATASET
# ==========================================

original_df = pd.read_csv(r"C:\Users\kurov\OneDrive\Desktop\Customer-Churn-Prediction\data\Telco-Customer-Churn.csv")

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("\nDataset Shape:")
print(original_df.shape)

# ==========================================
# DATA CLEANING
# ==========================================

original_df["TotalCharges"] = pd.to_numeric(
    original_df["TotalCharges"],
    errors="coerce"
)

original_df["TotalCharges"] = original_df[
    "TotalCharges"
].fillna(
    original_df["TotalCharges"].median()
)

# ==========================================
# CREATE ML COPY
# ==========================================

ml_df = original_df.copy()

# customerID is not useful for prediction
ml_df.drop("customerID", axis=1, inplace=True)

# ==========================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================

ml_df = pd.get_dummies(
    ml_df,
    drop_first=True
)

# ==========================================
# FEATURES & TARGET
# ==========================================

X = ml_df.drop("Churn_Yes", axis=1)
y = ml_df["Churn_Yes"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=5000
)

model.fit(
    X_train,
    y_train
)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(
    "\nAccuracy:",
    round(
        accuracy_score(
            y_test,
            y_pred
        ) * 100,
        2
    ),
    "%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance["Absolute_Value"] = (
    importance["Coefficient"].abs()
)

importance = importance.sort_values(
    by="Absolute_Value",
    ascending=False
)

print("\n" + "=" * 50)
print("TOP 15 CHURN DRIVERS")
print("=" * 50)

print(
    importance[
        ["Feature", "Coefficient"]
    ].head(15)
)

# ==========================================
# PREDICT CHURN PROBABILITY
# ==========================================

all_probabilities = model.predict_proba(X)[:, 1]

# Add probabilities to ORIGINAL dataset
original_df[
    "Churn_Probability"
] = all_probabilities

# ==========================================
# CREATE RISK LEVELS
# ==========================================

original_df["Risk_Level"] = pd.cut(
    original_df["Churn_Probability"],
    bins=[0, 0.30, 0.70, 1.00],
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)

# ==========================================
# OPTIONAL: PREDICTED CHURN
# ==========================================

original_df["Predicted_Churn"] = np.where(
    original_df["Churn_Probability"] > 0.50,
    "Yes",
    "No"
)

# ==========================================
# RISK DISTRIBUTION
# ==========================================

print("\nRisk Distribution:")
print(
    original_df[
        "Risk_Level"
    ].value_counts()
)

# ==========================================
# EXPORT FOR POWER BI
# ==========================================

original_df.to_csv(
    "C:\Users\kurov\OneDrive\Desktop\Customer-Churn-Prediction\churn_dashboard.csv",
    index=False
)

print(os.getcwd())

print("\n" + "=" * 50)
print("EXPORT SUCCESSFUL")
print("=" * 50)

print(
    "\nCreated File: churn_dashboard.csv"
)

print(
    "\nColumns Added:"
)

print(
    "- Churn_Probability"
)

print(
    "- Risk_Level"
)

print(
    "- Predicted_Churn"
)