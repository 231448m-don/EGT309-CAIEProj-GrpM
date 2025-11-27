import warnings
warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
warnings.filterwarnings("ignore", category=FutureWarning)

import sqlite3
import pandas as pd
import seaborn as sns
import numpy as np
import re as re
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# 1. LOAD DATASET
con = sqlite3.connect("C:/Users/ilhan/Desktop/EGT309-CAIEProj-GrpM/final_bmarket.db")
df = pd.read_sql_query("SELECT * FROM df_model_table", con)

TARGET = "subscription_status_encoded"

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# 2. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n=== DATA SHAPES ===")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("\ny_train label counts:\n", y_train.value_counts())

# 3. REMOVE STRING COLUMNS
cols_with_strings = X_train.columns[
    X_train.applymap(lambda x: isinstance(x, str)).sum() > 0
]
print("\nColumns dropped (string columns):")
print(cols_with_strings.to_list())

X_train = X_train.drop(columns=cols_with_strings)
X_test = X_test.drop(columns=cols_with_strings)

# 4. TRAIN RANDOM FOREST WITH OOB
rf_clf = RandomForestClassifier(
    oob_score=True,
    random_state=42,
    warm_start=True,
    n_jobs=-1
)

oob_results = []
for n_trees in [10, 50, 100, 500, 600, 800, 1000, 1100, 1200, 1300, 1400, 1500]:
    rf_clf.set_params(n_estimators=n_trees)
    rf_clf.fit(X_train, y_train)
    oob_error = 1 - rf_clf.oob_score_
    oob_results.append(pd.Series({
        "n_trees": n_trees,
        "oob_error": oob_error
    }))

oob_df = pd.concat(oob_results, axis=1).T.set_index("n_trees")
print("\n=== OOB ERROR RESULTS ===")
print(oob_df)

# 5. FINAL MODEL (e.g. 1100 trees)
rf_final = RandomForestClassifier(
    oob_score=True,
    random_state=42,
    n_estimators=1100,
    n_jobs=-1
)

rf_final.fit(X_train, y_train)
y_pred = rf_final.predict(X_test)

print("\n\n=== RANDOM FOREST TEST PERFORMANCE ===")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))
