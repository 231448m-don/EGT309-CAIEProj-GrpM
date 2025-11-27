from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
import pandas as pd
import sqlite3
import seaborn as sns
import numpy as np
import re as re
from matplotlib import pyplot as plt

# ==========================================
# 1. LOAD DATA
# ==========================================
con = sqlite3.connect("C:/Users/ilhan/Desktop/EGT309-CAIEProj-GrpM/final_bmarket.db")
df = pd.read_sql_query("SELECT * FROM df_model_table", con)

TARGET_COLUMN = "subscription_status_encoded"

print("\n=== RAW DATA INFO ===")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Target value counts:\n", df[TARGET_COLUMN].value_counts())

# ==========================================
# 2. FEATURES / TARGET + PREPROCESSING
# ==========================================
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].astype(int)

# Drop ID-like columns
cols_to_drop = [c for c in X.columns if "id" in c.lower()]
print("\nDropping ID columns:", cols_to_drop)
X = X.drop(columns=cols_to_drop)

# One-hot encode object columns
obj_cols = X.select_dtypes(include=["object"]).columns
print("\nObject columns to one-hot encode:", obj_cols.tolist())

X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

# Replace infinities + fill NaNs
X = X.replace([float("inf"), float("-inf")], np.nan)
X = X.fillna(X.median(numeric_only=True))

print("\n=== AFTER ENCODING / CLEANING ===")
print("Final feature shape:", X.shape)
print("Remaining NaNs:", int(X.isna().sum().sum()))

# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n=== TRAIN / TEST SPLIT ===")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train distribution:\n", y_train.value_counts())

# ==========================================
# 4. TRAIN LOGISTIC REGRESSION
# ==========================================
log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_model.fit(X_train, y_train)

# ==========================================
# 5. EVALUATE MODEL
# ==========================================
y_pred = log_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="binary")  # assumes 0/1, positive class = 1

print("\n\n=== LOGISTIC REGRESSION PERFORMANCE ===")
print(f"Accuracy : {acc:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Pred 0", "Pred 1"]
)
print("=== CONFUSION MATRIX ===")
print(cm_df)
