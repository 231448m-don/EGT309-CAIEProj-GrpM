from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report, confusion_matrix, precision_recall_curve
import pandas as pd
import numpy as np

def train_xgboost(input_data: pd.DataFrame, target_col: str):
    """
    Train an XGBoost classifier on the given modelling table.

    Parameters
    ----------
    input_data : pd.DataFrame
        Full modelling table (features + target).
    target_col : str
        Name of the target column.

    Returns
    -------
    model : XGBClassifier
        Fitted XGBoost model.
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
    xgb_clf = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # ================================
    # HIGH-PRECISION THRESHOLD TUNING
    # ================================

    xgb_clf.fit(X_train, y_train)   
    # Get predicted probabilities for the positive class
    y_proba = xgb_clf.predict_proba(X_test)[:, 1]

    # Build Precision–Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    min_recall = 0.15
    precision = precision[:-1]  # last point has no corresponding threshold
    recall = recall[:-1]

    mask = recall >= min_recall
    if mask.any():
        # Among thresholds that keep recall >= min_recall, pick max precision
        best_idx = np.argmax(precision[mask])
        best_thr = thresholds[mask][best_idx]
        print(f"\n[HIGH PRECISION] Chosen threshold: {best_thr:.3f}")
        print(f"Precision at this threshold: {precision[mask][best_idx]:.3f}")
        print(f"Recall at this threshold   : {recall[mask][best_idx]:.3f}")
    else:
        # Fallback if nothing meets min_recall
        best_thr = 0.5
        print(f"\n[HIGH PRECISION] No threshold reached recall >= {min_recall}. Using 0.5.")

    # Use the chosen threshold instead of default 0.5
    y_pred_custom = (y_proba >= best_thr).astype(int)

    print("\n=== Confusion Matrix (custom high-precision threshold) ===")
    print(confusion_matrix(y_test, y_pred_custom))

    print("\n=== Classification Report (custom high-precision threshold) ===")
    print(classification_report(y_test, y_pred_custom, digits=3))

    print(f"Custom precision: {precision_score(y_test, y_pred_custom):.3f}")
    print(f"Custom F1-score : {f1_score(y_test, y_pred_custom):.3f}")


    return xgb_clf

