import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def load_data() -> pd.DataFrame:
    """Load exported CSV for Kedro."""
    return pd.read_csv("data/01_raw/df_model_table.csv")


def preprocess(df: pd.DataFrame):
    """Return X, y after encoding categorical features."""
    target = "subscription_status_encoded"
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    obj_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    return X, y


def train_logistic(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"[Kedro][Logistic] Accuracy={acc:.4f}, F1={f1:.4f}")
    return {"accuracy": acc, "f1": f1}


def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"[Kedro][RandomForest] Accuracy={acc:.4f}, F1={f1:.4f}")
    return {"accuracy": acc, "f1": f1}


def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"[Kedro][XGBoost] Accuracy={acc:.4f}, F1={f1:.4f}")
    return {"accuracy": acc, "f1": f1}
