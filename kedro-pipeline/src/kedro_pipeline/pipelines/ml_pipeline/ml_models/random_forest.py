import warnings
warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, precision_recall_curve
)

def train_random_forest(input_data: pd.DataFrame, target_col: str):
    """
    Train a RandomForest classifier on the given modelling table.

    Parameters
    ----------
    input_data : pd.DataFrame
        Full modelling table (features + target).
    target_col : str
        Name of the target column.

    Returns
    -------
    model : RandomForestClassifier
        Fitted sklearn model.
    """
    df = input_data.copy()

    # 1. Split X / y
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # 2. Drop ID-like columns
    id_cols = [c for c in X.columns if "id" in c.lower()]
    X = X.drop(columns=id_cols)

    # 3. One-hot encode object columns
    obj_cols = X.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    # 4. Clean infinities / NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 6. Model
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,              # limit depth → less overfitting
        min_samples_leaf=5,       # leaf must have enough samples
        min_samples_split=10,
        class_weight="balanced",  # handle imbalance
        n_jobs=-1,
        random_state=42,
    )

    rf_clf.fit(X_train, y_train)
    
    # ================================
    # HIGH-PRECISION THRESHOLD TUNING 
    # ================================

    # 1) Get predicted probabilities for the positive class
    y_proba = rf_clf.predict_proba(X_test)[:, 1]

    # 2) Build Precision–Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # thresholds has length = len(precision) - 1
    precision = precision[:-1]
    recall = recall[:-1]

    # 3) Choose a minimum recall
    min_recall = 0.15  # lower → more extreme precision, higher → more balanced

    mask = recall >= min_recall

    if mask.any():
        # Among thresholds that keep recall >= min_recall, pick the one with max precision
        best_idx = precision[mask].argmax()
        best_thr = thresholds[mask][best_idx]

        print(f"\n[RF HIGH PRECISION] Chosen threshold: {best_thr:.3f}")
        print(f"Precision at this threshold: {precision[mask][best_idx]:.3f}")
        print(f"Recall at this threshold   : {recall[mask][best_idx]:.3f}")
    else:
        best_thr = 0.5
        print(f"\n[RF HIGH PRECISION] No threshold reached recall >= {min_recall}. Using 0.5.")

    # 4) Apply this stricter threshold
    y_pred_custom = (y_proba >= best_thr).astype(int)

    print("\n=== Confusion Matrix (Random Forest – custom high-precision threshold) ===")
    print(confusion_matrix(y_test, y_pred_custom))

    print("\n=== Classification Report (Random Forest – custom high-precision threshold) ===")
    print(classification_report(y_test, y_pred_custom, digits=3))

    print(f"Custom precision: {precision_score(y_test, y_pred_custom):.3f}")
    print(f"Custom recall   : {recall_score(y_test, y_pred_custom):.3f}")
    print(f"Custom F1-score : {f1_score(y_test, y_pred_custom):.3f}")

    return rf_clf

