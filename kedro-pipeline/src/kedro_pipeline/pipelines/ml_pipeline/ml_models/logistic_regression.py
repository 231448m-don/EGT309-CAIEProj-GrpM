from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

def train_logistic_regression(input_data: pd.DataFrame, target_col: str):
    """
    Train a logistic regression model on the given modelling table.

    Parameters
    ----------
    input_data : pd.DataFrame
        Full modelling table (features + target).
    target_col : str
        Name of the target column, e.g. 'subscription_status_encoded'.

    Returns
    -------
    model : LogisticRegression
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

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 6. Basic metrics (for logging)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n[LOG] Logistic Regression performance")
    print(f"  accuracy = {acc:.4f}")
    print(f"  precision = {prec:.4f}")
    print(f"  recall    = {rec:.4f}")
    print(f"  f1       = {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model

