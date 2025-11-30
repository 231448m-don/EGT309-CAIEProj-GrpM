from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_data, preprocess, train_logistic, train_random_forest, train_xgboost


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(load_data, None, "raw_df", name="load_data"),
            node(preprocess, "raw_df", ["X", "y"], name="preprocess"),

            node(train_logistic, ["X", "y"], "logistic_results", name="train_logistic"),
            node(train_random_forest, ["X", "y"], "rf_results", name="train_random_forest"),
            node(train_xgboost, ["X", "y"], "xgb_results", name="train_xgboost"),
        ]
    )
