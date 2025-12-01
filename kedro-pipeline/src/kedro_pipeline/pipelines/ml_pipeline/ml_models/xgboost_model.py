from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
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

    xgb_clf.fit(X_train, y_train)

    # 7. Metrics
    y_pred = (xgb_clf.predict_proba(X_test)[:, 1] > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n[LOG] XGBoost performance")
    print(f"  accuracy = {acc:.4f}")
    print(f"  precision = {prec:.4f}")
    print(f"  recall    = {rec:.4f}")
    print(f"  f1       = {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return xgb_clf

