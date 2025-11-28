from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
import os
from pathlib import Path
import joblib
import pandas as pd
import sqlite3
import seaborn as sns
import numpy as np
import re as re
from matplotlib import pyplot as plt

# ==========================================
# 1. LOAD DATA
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /app
DB_PATH = os.path.join(BASE_DIR, "final_bmarket.db")

con = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM df_model_table", con)

target_column = "subscription_status_encoded"

print("\n=== RAW DATA INFO ===")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nTarget distribution:\n", df[target_column].value_counts())

# ==========================================
# 2. FEATURES / TARGET + ENCODING
# ==========================================
X = df.drop(columns=[target_column])
y = df[target_column]

# Find all object / string columns
obj_cols = X.select_dtypes(include=["object"]).columns
print("\nObject columns to one-hot encode:", obj_cols.tolist())

# One-hot encode them
X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

print("\n=== AFTER ENCODING ===")
print("Feature shape:", X.shape)
print("Number of encoded features:", X.shape[1])

print("\nUnique target values:", df[target_column].unique())
print("Number of unique target values:", df[target_column].nunique())

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
print("\ny_train distribution:\n", y_train.value_counts())

# ==========================================
# 4. BASE XGBOOST MODEL
# ==========================================
xgb_clf = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",   # binary classification
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_clf.fit(X_train, y_train)

# Predictions
y_pred = xgb_clf.predict(X_test)

# ==========================================
# 5. EVALUATE BASE MODEL
# ==========================================
print("\n\n=== XGBOOST: BASE MODEL PERFORMANCE ===")
base_acc = accuracy_score(y_test, y_pred)
base_f1  = f1_score(y_test, y_pred, average="binary")

print(f"Accuracy : {base_acc:.4f}")
print(f"F1-score : {base_f1:.4f}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

cm_base = confusion_matrix(y_test, y_pred)
cm_base_df = pd.DataFrame(
    cm_base,
    index=["Actual 0", "Actual 1"],
    columns=["Pred 0", "Pred 1"]
)
print("=== CONFUSION MATRIX ===")
print(cm_base_df)

# ==========================================
# SAVE MODEL
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent.parent   # goes up from src/ to project root
SAVE_DIR = ROOT_DIR / "saved_models"
SAVE_DIR.mkdir(exist_ok=True)

model_path = SAVE_DIR / "xgboost.pkl"
joblib.dump(xgb_clf, model_path)
print(f"\n[LOG] Saved Logistic Regression model to: {model_path}")

