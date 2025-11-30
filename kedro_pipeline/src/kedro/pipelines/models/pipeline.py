from kedro.pipeline import Pipeline, node
from .logistic_regression import (
    preprocess_data,
    train_logistic_regression,
    evaluate_model
)

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=preprocess_data,
            inputs=dict(
                df="raw_model_data",
                target_column="params:target_column"
            ),
            outputs=["X", "y"],
            name="preprocess_data"
        ),
        node(
            func=train_logistic_regression,
            inputs=["X_train", "y_train"],
            outputs="logistic_model",
            name="train_logistic_regression"
        ),
        node(
            func=evaluate_model,
            inputs=["logistic_model", "X_test", "y_test"],
            outputs="logistic_metrics",
            name="evaluate_logistic_model"
        ),
    ])
