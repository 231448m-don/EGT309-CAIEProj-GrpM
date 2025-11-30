from kedro.pipeline import Pipeline, node
from .logistic_regression import train_logistic_regression
from .random_forest import train_random_forest
from .xgboost import train_xgboost

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_logistic_regression,
                inputs="model_input_data",
                outputs="logistic_model_output",
                name="logistic_regression_node"
            ),
            node(
                func=train_random_forest,
                inputs="model_input_data",
                outputs="random_forest_model_output",
                name="random_forest_node"
            ),
            node(
                func=train_xgboost,
                inputs="model_input_data",
                outputs="xgboost_model_output",
                name="xgboost_node"
            ),
        ]
    )
